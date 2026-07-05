"""Policy Analyzer — multi-rollout session server.

Usage:
    python -m policy_analyzer [--port 8000]

Rollouts run as independent subprocesses, one per GPU, so each gets a clean CUDA
context (CUDA_VISIBLE_DEVICES is set before the child starts). A scheduler thread
assigns queued rollouts to free GPUs as VRAM allows and reaps finished processes;
an OOM exit requeues the rollout until memory frees up. The server process itself
never imports JAX.

API:
    GET    /api/policies              list available training runs
    GET    /api/checkpoints?run=NAME  list checkpoint steps for a run
    POST   /api/sessions              start a session {run, checkpoint_step, n_det, n_sto}
    GET    /api/sessions              list all known sessions (newest first)
    GET    /api/sessions/{sid}        get session status (one-shot snapshot)
    GET    /api/sessions/{sid}/stream Server-Sent Events: pushes state on change
    DELETE /api/sessions/{sid}        delete a session (stops running rollouts)
    GET    /*                         static files from analysis/
"""

from __future__ import annotations

import argparse
import collections
import http.server
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from policy_analyzer.paths import ANALYSIS_DIR, LOGS_DIR, PKG_DIR
from policy_analyzer.session import RolloutInfo, Session
from policy_analyzer.worker import _free_vram_gb, MIN_FREE_VRAM_GB

_APP_TEMPLATE = Path(__file__).parent / "analyzer_template.html"
_APP_ASSETS = Path(__file__).parent / "assets"

_MAX_OOM_RETRIES = 10
_EXIT_OOM = 75            # collect_one EX_TEMPFAIL exit code
_SCHED_POLL_SECS = 1.0    # scheduler tick interval


# ── listing helpers (no JAX import) ─────────────────────────────────────────────

def _list_policies(logs_dir: Path) -> list[dict]:
    if not logs_dir.exists():
        return []
    date_re = re.compile(r"-\d{8}-")
    result = []
    for d in logs_dir.iterdir():
        if not d.is_dir():
            continue
        ckpt_dir = d / "checkpoints"
        if not ckpt_dir.exists():
            continue
        steps = _list_checkpoints(d)
        if not steps:
            continue
        m = date_re.search(d.name)
        env = d.name[: m.start()] if m else d.name
        result.append({"name": d.name, "env": env, "n_checkpoints": len(steps)})
    return sorted(result, key=lambda r: r["name"], reverse=True)


def _list_checkpoints(log_dir: Path) -> list[str]:
    ckpt_dir = log_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(
        [d.name for d in ckpt_dir.iterdir() if d.is_dir()],
        key=lambda s: int(s),
    )


def _detect_gpus() -> list[int]:
    """Physical GPU indices via nvidia-smi; falls back to [0]."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        ids = [int(x) for x in out.stdout.split()]
        return ids or [0]
    except Exception:
        return [0]


def _tail(path: Path, n_chars: int = 600) -> str:
    try:
        return path.read_text(errors="replace")[-n_chars:].strip() or "subprocess failed"
    except Exception:
        return "subprocess failed"


# ── compare-eval cache helpers (no JAX) ─────────────────────────────────────────

def _compare_cache_path(run_name: str, mode: str, benchmark: str) -> Path:
    return ANALYSIS_DIR / "compare_cache" / run_name / benchmark / f"{mode}.npz"


def _compare_cache_done(run_name: str, mode: str, benchmark: str) -> bool:
    p = _compare_cache_path(run_name, mode, benchmark)
    return (p.parent / f"{p.stem}.DONE").exists() and p.exists()


def _load_compare_cache(run_name: str, mode: str, benchmark: str) -> dict | None:
    """Load a cached eval npz into {channels, n_rollouts, env_name, sensor_bundle}."""
    import numpy as np
    p = _compare_cache_path(run_name, mode, benchmark)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
        names = [str(n) for n in z["_channels"]]
        return {
            "channels": {n: z[n] for n in names},
            "n_rollouts": int(z["_n_rollouts"]),
            "env_name": str(z["_env_name"]),
            "sensor_bundle": str(z["_sensor_bundle"]),
            # Older caches predate _dt; None disables seconds→steps conversion.
            "dt": float(z["_dt"]) if "_dt" in z.files else None,
        }
    except Exception:
        return None


# ── HTTP handler ──────────────────────────────────────────────────────────────

def _make_handler(server: "AnalysisServer", analysis_dir: Path):
    class _Handler(http.server.SimpleHTTPRequestHandler):
        # HTTP/1.1 is required for SSE: under HTTP/1.0 the browser buffers a
        # response with no Content-Length until the connection closes, so it
        # never processes the event stream incrementally.
        protocol_version = "HTTP/1.1"

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(analysis_dir), **kw)

        def _no_body(self, code: int) -> None:
            """Send a bodyless response that's valid under HTTP/1.1 keep-alive."""
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def end_headers(self):
            # Force revalidation of the app shell so a server restart always
            # serves the latest UI instead of a stale cached copy.
            if getattr(self, "_nocache_html", False):
                self.send_header("Cache-Control", "no-cache")
                self._nocache_html = False
            super().end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            if path == "/api/policies":
                self._json(_list_policies(server.logs_dir))
            elif path == "/api/checkpoints":
                run = params.get("run", "")
                steps = _list_checkpoints(server.logs_dir / run) if run else []
                self._json(steps)
            elif path == "/api/queues":
                self._json(server.list_queues_payload())
            elif path == "/api/compare/config":
                self._json(server.compare_config(params.get("env", "")))
            elif path == "/api/compare/benchmarks":
                self._json(server.compare_benchmarks(params.get("env", "")))
            elif path.startswith("/api/compare/"):
                cid = path[len("/api/compare/"):]
                data = server.compare_status(cid)
                if data is not None:
                    self._json(data)
                else:
                    self._no_body(404)
            elif path == "/api/sessions":
                self._json(server.list_sessions())
            elif path.startswith("/api/sessions/") and path.endswith("/stream"):
                sid = path[len("/api/sessions/"):-len("/stream")]
                self._sse(sid)
            elif path.startswith("/api/sessions/"):
                sid = path[len("/api/sessions/"):]
                data = server.get_session(sid)
                if data is not None:
                    self._json(data)
                else:
                    self._no_body(404)
            else:
                pth = urllib.parse.urlsplit(self.path).path
                if pth == "/" or pth.endswith(".html"):
                    self._nocache_html = True   # see end_headers()
                super().do_GET()

        def _sse(self, sid: str) -> None:
            """Server-Sent Events stream of a session's state (pushes on change)."""
            if server.get_session(sid) is None:
                self._no_body(404)
                return
            # This response streams indefinitely with no Content-Length, so the
            # connection can't be reused — tell the framework not to keep it alive.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = server.add_listener(sid)
            try:
                self._sse_send(server.get_session(sid))   # initial snapshot
                while True:
                    try:
                        item = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")    # heartbeat / disconnect probe
                        self.wfile.flush()
                        continue
                    if item is None:                       # session deleted
                        self._sse_send({"session_id": sid, "deleted": True})
                        break
                    self._sse_send(item)
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass
            finally:
                server.remove_listener(sid, q)

        def _sse_send(self, obj: object) -> None:
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/sessions":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                sid = server.start_session(
                    run=body["run"],
                    checkpoint_step=body.get("checkpoint_step", "latest"),
                    n_det=int(body.get("n_det", 0)),
                    n_sto=int(body.get("n_sto", 0)),
                )
                self._json({"session_id": sid})
            elif path == "/api/compare":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._json(server.start_compare(body))
            else:
                self._no_body(405)

        def do_DELETE(self):
            path = urllib.parse.urlsplit(self.path).path
            if path.startswith("/api/sessions/"):
                sid = path[len("/api/sessions/"):]
                ok = server.delete_session(sid)
                self._no_body(200 if ok else 404)
            else:
                self._no_body(405)

        def _json(self, data: object) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return _Handler


# ── analysis server ───────────────────────────────────────────────────────────

class AnalysisServer:
    def __init__(self, analysis_dir: Path, logs_dir: Path):
        self.analysis_dir = analysis_dir
        self.logs_dir = logs_dir
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

        # scheduler state (guarded by _lock)
        self._gpus = _detect_gpus()
        self._pending: collections.deque[tuple] = collections.deque()
        self._running: dict[int, dict] = {}          # gpu_id -> job dict
        self._oom_attempts: dict[tuple[str, bool], int] = {}

        # compare-eval scheduler state (guarded by _lock). Compare jobs are
        # lightweight rollout evals (no rendering) that share the GPU pool with
        # session rollouts; their results are cached on disk under compare_cache/.
        self._compare_pending: collections.deque[dict] = collections.deque()
        self._compare_inflight: set[tuple[str, str, str]] = set()   # (run_name, mode, benchmark)
        self._compare_errors: dict[tuple[str, str, str], str] = {}  # (run_name, mode, benchmark) -> msg
        self._compare_oom: dict[tuple[str, str], int] = {}     # OOM retry counter
        self._compare_jobs: dict[str, dict] = {}               # compare_id -> request

        # SSE broker state (guarded by _lock)
        self._listeners: dict[str, set] = {}         # sid -> set[queue.Queue]
        self._last_sent: dict[str, str] = {}         # sid -> last broadcast json

        self._sensor_bundle_cache: dict[str, str] = {}   # run -> sensor_bundle

        analysis_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_sessions()
        (analysis_dir / "index.html").write_bytes(_APP_TEMPLATE.read_bytes())
        if _APP_ASSETS.is_dir():
            shutil.copytree(_APP_ASSETS, analysis_dir / "assets", dirs_exist_ok=True)

        self._sched = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="rollout-scheduler"
        )
        self._sched.start()
        print(f"Policy Analyzer ready — {len(self._gpus)} GPU(s): {self._gpus}", flush=True)

    # ── session loading ───────────────────────────────────────────────────────

    def _load_existing_sessions(self) -> None:
        sessions_dir = self.analysis_dir / "sessions"
        if not sessions_dir.exists():
            return
        for d in sorted(sessions_dir.iterdir()):
            if not d.is_dir():
                continue
            sj = d / "session.json"
            if not sj.exists():
                continue
            try:
                data = json.loads(sj.read_text(encoding="utf-8"))
                rollouts = [
                    RolloutInfo(
                        name=r["name"],
                        deterministic=r["deterministic"],
                        seed=r["seed"],
                        status=(
                            "error" if r["status"] in ("running", "pending") else r["status"]
                        ),
                        error=(
                            "Server restarted"
                            if r["status"] in ("running", "pending")
                            else r.get("error")
                        ),
                    )
                    for r in data["rollouts"]
                ]
                sess = Session(
                    session_id=data["session_id"],
                    session_dir=d,
                    run=data["run"],
                    checkpoint_step=data["checkpoint_step"],
                    rollouts=rollouts,
                )
                self._sessions[data["session_id"]] = sess
            except Exception:
                pass

    # ── scheduler ───────────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while True:
            try:
                self._reap()
                self._dispatch()
                self._broadcast()
            except Exception:
                import traceback
                traceback.print_exc()
            time.sleep(_SCHED_POLL_SECS)

    # ── SSE broker ──────────────────────────────────────────────────────────────

    def add_listener(self, sid: str) -> "queue.Queue":
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.setdefault(sid, set()).add(q)
        return q

    def remove_listener(self, sid: str, q: "queue.Queue") -> None:
        with self._lock:
            listeners = self._listeners.get(sid)
            if listeners:
                listeners.discard(q)
                if not listeners:
                    self._listeners.pop(sid, None)

    def _broadcast(self) -> None:
        """Push fresh snapshots to listeners of sessions whose state changed."""
        with self._lock:
            sids = list(self._listeners.keys())
            sessions = {sid: self._sessions.get(sid) for sid in sids}
        for sid in sids:
            sess = sessions[sid]
            if sess is None:
                continue
            snap = self._payload(sess)
            js = json.dumps(snap, sort_keys=True)
            with self._lock:
                if self._last_sent.get(sid) == js:
                    continue
                self._last_sent[sid] = js
                listeners = list(self._listeners.get(sid, ()))
            for q in listeners:
                q.put(snap)

    def _reap(self) -> None:
        """Mark seeds done as their DONE sentinels appear; handle group exit."""
        with self._lock:
            running = list(self._running.items())

        for gpu, job in running:
            if job.get("kind") == "compare":
                self._reap_compare(gpu, job)
                continue

            sid = job["sid"]
            sess = self._sessions.get(sid)

            # Progressive completion: a seed is done once its DONE sentinel lands.
            if sess is not None:
                for name in list(job["pending"]):
                    if (sess.session_dir / name / "DONE").exists():
                        sess.update_rollout(name, "done")
                        job["pending"].discard(name)

            rc = job["proc"].poll()
            if rc is None:
                continue

            job["logf"].close()
            tag = "det" if job["det"] else "sto"
            key = (sid, job["det"])

            if rc == _EXIT_OOM and sess is not None:
                n = self._oom_attempts.get(key, 0) + 1
                self._oom_attempts[key] = n
                if n > _MAX_OOM_RETRIES:
                    for name in job["pending"]:
                        sess.update_rollout(name, "error", "OOM: exceeded retries")
                    print(f"[GPU {gpu}] {sid}/{tag} OOM — gave up", flush=True)
                else:
                    for name in job["pending"]:
                        sess.update_rollout(name, "pending")
                        sess.update_rollout_detail(name, "OOM — waiting for VRAM to retry")
                    with self._lock:
                        self._pending.append((sid, job["det"], job["seeds"]))
                    print(f"[GPU {gpu}] {sid}/{tag} OOM — requeued ({n})", flush=True)
            elif sess is not None:
                # Process exited; resolve any seeds without a DONE sentinel.
                self._oom_attempts.pop(key, None)
                msg = _tail(job["logpath"]) if rc != 0 else "no artifacts produced"
                for name in list(job["pending"]):
                    if (sess.session_dir / name / "DONE").exists():
                        sess.update_rollout(name, "done")
                    else:
                        sess.update_rollout(name, "error", msg)
                print(f"[GPU {gpu}] {sid}/{tag} exited rc={rc}", flush=True)

            with self._lock:
                self._running.pop(gpu, None)

    def _dispatch(self) -> None:
        """Launch queued work (session rollouts, then compare evals) onto free
        GPUs that have enough VRAM. Session rollouts take priority."""
        with self._lock:
            free_gpus = [g for g in self._gpus if g not in self._running]
            has_work = bool(self._pending) or bool(self._compare_pending)
        if not has_work or not free_gpus:
            return

        for gpu in free_gpus:
            with self._lock:
                if not (self._pending or self._compare_pending):
                    break
            free = _free_vram_gb(gpu)  # shell out outside the lock
            with self._lock:
                if self._pending:
                    sid, det, seeds = self._pending[0]
                    sess = self._sessions.get(sid)
                    if sess is None:                   # session deleted
                        self._pending.popleft()
                        continue
                    if free < MIN_FREE_VRAM_GB:
                        tag = "det" if det else "sto"
                        for s in seeds:
                            sess.update_rollout_detail(
                                f"{tag}_{s}",
                                f"waiting for VRAM — {free:.1f}/{MIN_FREE_VRAM_GB} GB on GPU {gpu}",
                            )
                        continue
                    self._pending.popleft()
                    self._launch_locked(gpu, sid, det, seeds, sess)
                elif self._compare_pending:
                    if free < MIN_FREE_VRAM_GB:
                        continue
                    task = self._compare_pending.popleft()
                    self._launch_compare_locked(gpu, task)

    def _launch_locked(self, gpu: int, sid: str, det: bool, seeds, sess: Session) -> None:
        """Spawn a collect_one subprocess for a whole group on `gpu`. Caller holds _lock."""
        tag = "det" if det else "sto"
        names = [f"{tag}_{s}" for s in seeds]

        cmd = [
            sys.executable, "-m", "policy_analyzer.collect_one",
            "--log-dir", str(self.logs_dir / sess.run),
            "--out-base", str(sess.session_dir),
            "--seeds", ",".join(str(s) for s in seeds),
        ]
        if det:
            cmd.append("--deterministic")
        if sess.checkpoint_step and sess.checkpoint_step != "latest":
            cmd += ["--checkpoint-step", sess.checkpoint_step]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        logpath = sess.session_dir / f"{tag}.log"
        logf = open(logpath, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(PKG_DIR.parent), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        self._running[gpu] = {
            "proc": proc, "sid": sid, "det": det, "seeds": seeds,
            "names": names, "pending": set(names),
            "logf": logf, "logpath": logpath,
        }
        for name in names:
            sess.update_rollout(name, "running")
        print(f"[GPU {gpu}] launched {sid}/{tag} seeds={list(seeds)}", flush=True)

    def _launch_compare_locked(self, gpu: int, task: dict) -> None:
        """Spawn a compare_collect subprocess for one run+mode+benchmark. Caller holds _lock."""
        run_name, mode, benchmark = task["run_name"], task["mode"], task["benchmark"]
        out_path = _compare_cache_path(run_name, mode, benchmark)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh run invalidates a stale DONE sentinel from a prior eval.
        done = out_path.parent / f"{out_path.stem}.DONE"
        done.unlink(missing_ok=True)

        cmd = [
            sys.executable, "-m", "policy_analyzer.compare_collect",
            "--log-dir", str(self.logs_dir / run_name),
            "--out", str(out_path),
            "--mode", mode,
            "--n-rollouts", str(task["n_rollouts"]),
            "--benchmark", benchmark,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        logpath = out_path.parent / f"{out_path.stem}.collect.log"
        logf = open(logpath, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(PKG_DIR.parent), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        self._running[gpu] = {
            "kind": "compare", "proc": proc, "run_name": run_name, "mode": mode,
            "benchmark": benchmark, "n_rollouts": task["n_rollouts"],
            "logf": logf, "logpath": logpath, "out_path": out_path,
        }
        self._compare_errors.pop((run_name, mode, benchmark), None)
        print(f"[GPU {gpu}] launched compare {run_name}/{benchmark}/{mode}", flush=True)

    def _reap_compare(self, gpu: int, job: dict) -> None:
        """Resolve a finished compare-eval subprocess."""
        rc = job["proc"].poll()
        if rc is None:
            return
        job["logf"].close()
        run_name, mode, benchmark = job["run_name"], job["mode"], job["benchmark"]
        key = (run_name, mode, benchmark)
        out_path = job["out_path"]
        done = out_path.parent / f"{out_path.stem}.DONE"

        with self._lock:
            self._running.pop(gpu, None)
            self._compare_inflight.discard(key)

        if rc == _EXIT_OOM:
            n = self._compare_oom.get(key, 0) + 1
            self._compare_oom[key] = n
            if n <= _MAX_OOM_RETRIES:
                with self._lock:
                    self._compare_inflight.add(key)
                    self._compare_pending.append({
                        "run_name": run_name, "mode": mode, "benchmark": benchmark,
                        "n_rollouts": job["n_rollouts"],
                    })
                print(f"[GPU {gpu}] compare {run_name}/{benchmark}/{mode} OOM — requeued ({n})", flush=True)
                return
            self._compare_errors[key] = "OOM: exceeded retries"
        elif rc != 0 or not done.exists():
            self._compare_errors[key] = _tail(job["logpath"])
        else:
            self._compare_oom.pop(key, None)
        print(f"[GPU {gpu}] compare {run_name}/{benchmark}/{mode} exited rc={rc}", flush=True)

    # ── public API ────────────────────────────────────────────────────────────

    def start_session(self, run: str, checkpoint_step: str, n_det: int, n_sto: int) -> str:
        run_short = run.split("-")[-1] if "-" in run else run
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{run_short}"
        session_dir = self.analysis_dir / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        # Each group (det / sto) runs all its seeds in one vmapped pass on one GPU.
        rollouts: list[RolloutInfo] = []
        groups: list[tuple[bool, tuple[int, ...]]] = []
        for det, n, tag in ((True, n_det, "det"), (False, n_sto, "sto")):
            seeds = tuple(range(1, n + 1))
            if not seeds:
                continue
            for s in seeds:
                rollouts.append(RolloutInfo(name=f"{tag}_{s}", deterministic=det, seed=s))
            groups.append((det, seeds))

        sess = Session(
            session_id=sid, session_dir=session_dir,
            run=run, checkpoint_step=checkpoint_step, rollouts=rollouts,
        )
        sess._save()

        with self._lock:
            self._sessions[sid] = sess
            for det, seeds in groups:
                self._pending.append((sid, det, seeds))

        return sid

    def _sensor_bundle(self, run: str) -> str | None:
        """sensor_bundle from logs/<run>/checkpoints/config.json (cached)."""
        if run in self._sensor_bundle_cache:
            return self._sensor_bundle_cache[run]
        sb = None
        cfg_path = self.logs_dir / run / "checkpoints" / "config.json"
        try:
            sb = json.loads(cfg_path.read_text(encoding="utf-8")).get("sensor_bundle")
        except Exception:
            pass
        self._sensor_bundle_cache[run] = sb
        return sb

    def _payload(self, sess: Session) -> dict:
        """Session snapshot enriched with values derivable from the logs."""
        d = sess.to_dict()
        d["sensor_bundle"] = self._sensor_bundle(sess.run)
        return d

    def list_sessions(self) -> list[dict]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(
            [self._payload(s) for s in sessions],
            key=lambda d: d["session_id"],
            reverse=True,
        )

    def get_session(self, sid: str) -> dict | None:
        with self._lock:
            sess = self._sessions.get(sid)
        return self._payload(sess) if sess else None

    # ── compare API ─────────────────────────────────────────────────────────────

    def list_queues_payload(self) -> list[dict]:
        from policy_analyzer import queues
        return queues.list_queues()

    def compare_config(self, env: str) -> dict:
        """Default success criterion for an env, enriched with channels from any
        existing cache for that env so the UI can offer a channel dropdown."""
        from policy_analyzer import success_curve

        channels: list[str] | None = None
        cache_root = ANALYSIS_DIR / "compare_cache"
        if env and cache_root.is_dir():
            for run_dir in cache_root.iterdir():
                if not run_dir.is_dir():
                    continue
                for bench_dir in run_dir.iterdir():
                    if not bench_dir.is_dir():
                        continue
                    for mode in ("det", "sto"):
                        if not _compare_cache_done(run_dir.name, mode, bench_dir.name):
                            continue
                        cache = _load_compare_cache(run_dir.name, mode, bench_dir.name)
                        if cache and cache["env_name"] == env:
                            channels = sorted(cache["channels"].keys())
                            break
                    if channels:
                        break
                if channels:
                    break
        return success_curve.config_for_env(env, channels)

    def compare_benchmarks(self, env: str) -> list[dict]:
        """Benchmarks selectable for an env (label/description/name), default first."""
        from policy_analyzer import benchmarks
        return [
            {"name": b["name"], "label": b["label"], "description": b["description"]}
            for b in benchmarks.benchmarks_for_env(env)
        ]

    def start_compare(self, body: dict) -> dict:
        """Queue evals for any uncached runs and register a compare job."""
        mode = body.get("mode", "det")
        n_rollouts = int(body.get("n_rollouts", 50))
        policies = body.get("policies", [])
        criterion = body.get("criterion", {})
        benchmark = body.get("benchmark", "default")

        cid = f"cmp-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
        with self._lock:
            for pol in policies:
                for run_name in pol.get("run_names", []):
                    cached = _compare_cache_done(run_name, mode, benchmark)
                    cache = _load_compare_cache(run_name, mode, benchmark) if cached else None
                    enough = cache is not None and cache["n_rollouts"] >= n_rollouts
                    key = (run_name, mode, benchmark)
                    if enough or key in self._compare_inflight:
                        continue
                    self._compare_inflight.add(key)
                    self._compare_errors.pop(key, None)
                    self._compare_pending.append(
                        {"run_name": run_name, "mode": mode,
                         "benchmark": benchmark, "n_rollouts": n_rollouts}
                    )
            self._compare_jobs[cid] = {
                "mode": mode, "n_rollouts": n_rollouts, "benchmark": benchmark,
                "criterion": criterion, "policies": policies,
            }
        return {"compare_id": cid}

    def compare_status(self, cid: str) -> dict | None:
        from policy_analyzer import success_curve

        with self._lock:
            job = self._compare_jobs.get(cid)
        if job is None:
            return None

        mode = job["mode"]
        benchmark = job.get("benchmark", "default")
        runs_status = []
        all_ready = True
        for pol in job["policies"]:
            for run_name in pol.get("run_names", []):
                key = (run_name, mode, benchmark)
                if _compare_cache_done(run_name, mode, benchmark):
                    st = "ready"
                elif key in self._compare_errors:
                    st = "error"
                elif key in self._compare_inflight:
                    st = "running"
                else:
                    # Not cached, not erroring, not queued → safety net so the
                    # client doesn't poll forever for something never launched.
                    st = "error"
                    self._compare_errors[key] = "not collected"
                if st != "ready":
                    all_ready = False
                runs_status.append({
                    "run_name": run_name, "status": st,
                    "error": self._compare_errors.get(key),
                })

        if not all_ready:
            return {"status": "collecting", "runs": runs_status}

        # All caches present → load and compute curves.
        cmp_policies = []
        for pol in job["policies"]:
            seeds = []
            for run_name in pol.get("run_names", []):
                cache = _load_compare_cache(run_name, mode, benchmark)
                if cache:
                    seeds.append(cache)
            if seeds:
                cmp_policies.append({
                    "label": pol.get("label", ""),
                    "sensor_bundle": pol.get("sensor_bundle", ""),
                    "seeds": seeds,
                })
        result = success_curve.compute_curves(cmp_policies, job["criterion"])
        return {"status": "ready", "runs": runs_status, "result": result}

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(sid, None)
            if sess is None:
                return False
            # Drop queued rollouts for this session.
            self._pending = collections.deque(
                t for t in self._pending if t[0] != sid
            )
            # Kill any running rollouts for this session.
            for gpu, job in list(self._running.items()):
                if job["sid"] == sid:
                    job["proc"].terminate()
                    job["logf"].close()
                    self._running.pop(gpu, None)
            # Notify any SSE listeners that the session is gone.
            self._last_sent.pop(sid, None)
            for q in self._listeners.get(sid, ()):
                q.put(None)

        if sess.session_dir.exists():
            shutil.rmtree(sess.session_dir, ignore_errors=True)
        return True

    def serve(self, port: int) -> None:
        HandlerClass = _make_handler(self, self.analysis_dir)
        with http.server.ThreadingHTTPServer(("", port), HandlerClass) as httpd:
            print(f"\nPolicy Analyzer  →  http://localhost:{port}/")
            print(f"  SSH tunnel: ssh -L <local>:localhost:{port} <host>")
            print("  Ctrl-C to stop.\n", flush=True)
            httpd.serve_forever()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(prog="policy_analyzer")
    ap.add_argument("--port", type=int, default=8000, metavar="PORT")
    ap.add_argument("--serve", type=int, metavar="PORT", dest="port",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    server = AnalysisServer(
        analysis_dir=ANALYSIS_DIR,
        logs_dir=LOGS_DIR,
    )
    server.serve(args.port)


if __name__ == "__main__":
    main()

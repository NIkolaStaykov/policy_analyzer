"""Policy Analyzer — multi-rollout session server.

Usage:
    python -m policy_analyzer [--port 8000]

Rollouts run as independent subprocesses, one per GPU, so each gets a clean CUDA
context (CUDA_VISIBLE_DEVICES is set before the child starts). A scheduler thread
assigns queued rollouts to free GPUs as VRAM allows and reaps finished processes;
an OOM exit requeues the rollout until memory frees up. The server process itself
never imports JAX.

API:
    GET    /api/policies              training runs + per-task config facets
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
import collections.abc
import hashlib
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

from policy_analyzer import run_metrics
from policy_analyzer.paths import ANALYSIS_DIR, LOGS_DIR, PKG_DIR
from policy_analyzer.session import RolloutInfo, Session
from policy_analyzer.worker import _free_vram_gb, MIN_FREE_VRAM_GB

_APP_TEMPLATE = Path(__file__).parent / "analyzer_template.html"
_APP_ASSETS = Path(__file__).parent / "assets"

_MAX_OOM_RETRIES = 10
_EXIT_OOM = 75            # collect_one EX_TEMPFAIL exit code
_SCHED_POLL_SECS = 1.0    # scheduler tick interval
# Rollouts one analysis may hold (cells × repeats). Each keeps its full
# trajectory plus a rendered video — ~8 MB — and the point of this tab is to
# look at each one, not to average over thousands. A wide sweep belongs in the
# Multi-policy tab, which stores per-step channels only.
MAX_SESSION_ROLLOUTS = 24
# Cockpit metrics only move while a queue is training, and an incremental
# refresh costs a couple of requests, so a few minutes is plenty.
_METRICS_REFRESH_SECS = 300.0

# How many eval subprocesses share a GPU is decided from measured free VRAM, not
# a fixed count — so the pool shrinks automatically when training or anything
# else outside this server is using the card.
#
# The one wrinkle is that a job takes ~10 s to allocate (imports + checkpoint
# restore come first), during which nvidia-smi still reports its memory as free.
# Admitting on the raw reading would let a burst of launches all see an empty
# card. So a just-launched job is charged its expected footprint until it has
# had time to allocate for real. Both numbers are measured, not guessed: a grid
# job holds ~2.5-2.7 GB almost independently of batch width (2568 MiB at 280
# concurrent episodes, 2664 MiB at 4096), and its allocation lands ~9 s in.
EXPECTED_JOB_VRAM_GB = 3.0
VRAM_SETTLE_SECS = 12.0


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


def _env_of_run(run_name: str) -> str:
    """The env a run trained on, from the name (`<Env>-<date>-<suffix>`)."""
    m = re.search(r"-\d{8}-", run_name)
    return run_name[: m.start()] if m else run_name


def _rollout_ids(tag: str, axes: list, cells: list, seeds) -> list[tuple]:
    """(name, cell, seed) for one group, in the order collect_one runs them.

    Cell-major, seeds within a cell — the batch layout of
    grid_collect.run_pinned_rollouts, so the names line up with its output
    element for element. Without a sweep the name stays `<tag>_<seed>`, which is
    what the historical artifact directories are called.
    """
    if not axes:
        return [(f"{tag}_{s}", 0, s) for s in seeds]
    return [(f"{tag}_c{c}_{s}", c, s) for c in range(len(cells)) for s in seeds]


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

# The grid pseudo-benchmark: instead of sampling episode parameters from the
# training distribution (compare_collect), sweep a deterministic grid over the
# run's randomization ranges with every randomized quantity pinned per cell
# (grid_collect). The whole grid runs as one batch, so cells no longer each pay
# a compile and rollouts-per-cell is a normal UI knob (Advanced panel) rather
# than a server-side constant.
GRID_BENCHMARK = "grid"
# Default rollouts per cell; the UI may change it. Mirrors
# grid_collect.DEFAULT_N_ROLLOUTS — see the note there for why it is 64. A full
# grid pins every swept quantity, so repeats only average over what stays random
# (un-swept axes, obs noise/bias, perturbations, stochastic actions); they matter
# most for a partial sweep. 64 also keeps any grid above the ~64-concurrent-
# episode threshold below which the sim reads systematically high.
GRID_N_ROLLOUTS = 64
GRID_MAX_CELLS = 8192    # mirrors grid_collect.DEFAULT_MAX_CELLS


def _compare_cache_path(run_name: str, mode: str, benchmark: str) -> Path:
    # `benchmark` may be a composite grid id "grid/<sig>", which nests one level
    # deeper (compare_cache/<run>/grid/<sig>/<mode>.npz) — each axis selection +
    # resolution gets its own cache, so different grids never collide.
    return ANALYSIS_DIR / "compare_cache" / run_name / benchmark / f"{mode}.npz"


def _is_grid(benchmark: str) -> bool:
    """True for the grid benchmark, bare ("grid") or spec-keyed ("grid/<sig>")."""
    return benchmark == GRID_BENCHMARK or benchmark.startswith(GRID_BENCHMARK + "/")


def _grid_sig(grid_spec: dict | None) -> str:
    """Deterministic short signature of a grid axis selection + resolution.

    Computed identically in start_compare and view_score so the client only sends
    the spec, and re-scoring targets exactly the collected grid's cache.
    """
    axes = (grid_spec or {}).get("axes", [])
    canon = ",".join(
        f"{a['name']}:{int(a['points'])}"
        for a in sorted(axes, key=lambda a: a["name"])
    )
    return hashlib.sha1(canon.encode()).hexdigest()[:8]


def _compare_cache_done(run_name: str, mode: str, benchmark: str) -> bool:
    p = _compare_cache_path(run_name, mode, benchmark)
    return (p.parent / f"{p.stem}.DONE").exists() and p.exists()


def _collect_log_path(run_name: str, mode: str, benchmark: str) -> Path:
    """Log written by the collector subprocess (see _launch_compare_locked)."""
    p = _compare_cache_path(run_name, mode, benchmark)
    return p.parent / f"{p.stem}.collect.log"


def _collect_progress(run_name: str, mode: str, benchmark: str) -> str | None:
    """Sub-seed progress for a running collector, from the tail of its log.

    Grid collection prints one "[grid c/n]" line per cell, so a long sweep can
    report how far it is instead of sitting on "running" for minutes. Sampled
    benchmarks are one batched eval with nothing to report → None.
    """
    path = _collect_log_path(run_name, mode, benchmark)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    hits = re.findall(r"\[grid (\d+)/(\d+)\]", tail)
    if not hits:
        return None
    c, n = hits[-1]
    return f"cell {c}/{n}"


class _LazyChannels(collections.abc.Mapping):
    """The npz's channel arrays, decompressed only when actually asked for.

    A cache carries every recorded channel (~26 for the pinch env, ~10 MB each
    once inflated), but a view reads one — the success channel — plus at most a
    couple of binning channels. Materializing the whole file per seed cost more
    than the entire rest of the request, and it scaled with the *dataset*, not
    with the subset on screen. Keys are known up front from `_channels`, so
    membership and listing stay free; only __getitem__ touches the zip.
    """

    __slots__ = ("_path", "_names", "_memo")

    def __init__(self, path, names):
        self._path = path
        self._names = list(names)
        self._memo: dict = {}

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def __getitem__(self, name):
        if name not in self._memo:
            if name not in self._names:
                raise KeyError(name)
            import numpy as np
            # Re-opening reads only the zip's central directory (~3 ms); it beats
            # holding 130 file handles open across a request.
            with np.load(self._path, allow_pickle=False) as z:
                self._memo[name] = z[name]
        return self._memo[name]


def _load_compare_cache(run_name: str, mode: str, benchmark: str) -> dict | None:
    """Load a cached eval npz into {channels, n_rollouts, env_name, sensor_bundle}."""
    import numpy as np
    p = _compare_cache_path(run_name, mode, benchmark)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            names = [str(n) for n in z["_channels"]]
            return {
                "channels": _LazyChannels(p, names),
                "n_rollouts": int(z["_n_rollouts"]),
                "env_name": str(z["_env_name"]),
                "sensor_bundle": str(z["_sensor_bundle"]),
                # Older caches predate _dt; None disables seconds→steps conversion.
                "dt": float(z["_dt"]) if "_dt" in z.files else None,
            }
    except Exception:
        return None


def _load_grid_cache(run_name: str, mode: str, benchmark: str = GRID_BENCHMARK) -> dict | None:
    """Load a grid_collect npz: channels [n_cells, N, T] + grid metadata.

    `benchmark` is the spec-keyed id ("grid/<sig>") so different axis selections
    load from their own cache.
    """
    import numpy as np
    p = _compare_cache_path(run_name, mode, benchmark)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            names = [str(n) for n in z["_channels"]]
            return {
                "channels": _LazyChannels(p, names),
                "axes": json.loads(str(z["_grid_axes"])),
                "cell_values": z["_cell_values"],
                "n_rollouts": int(z["_n_rollouts"]),
                "env_name": str(z["_env_name"]),
                "sensor_bundle": str(z["_sensor_bundle"]),
                "dt": float(z["_dt"]),
            }
    except Exception:
        return None


def _load_compare_policies(policies: list, mode: str, benchmark: str) -> list:
    """Load each policy's cached seeds into success_curve's cmp_policies shape.

    Seeds whose cache is missing are skipped; a policy with no loadable seed is
    dropped. Used both by the poll path (compare_status) and the synchronous
    grid re-score endpoint, which reads the same on-disk caches.
    """
    cmp_policies = []
    for pol in policies:
        seeds = []
        for run_name in pol.get("run_names", []):
            cache = (
                _load_grid_cache(run_name, mode, benchmark)
                if _is_grid(benchmark)
                else _load_compare_cache(run_name, mode, benchmark)
            )
            if cache:
                seeds.append(cache)
        if seeds:
            cmp_policies.append({
                "label": pol.get("label", ""),
                "sensor_bundle": pol.get("sensor_bundle", ""),
                "attrs": _policy_attrs(pol),
                "seeds": seeds,
            })
    return cmp_policies


def _load_run_config(run_name: str) -> dict | None:
    """Read a run's authoritative checkpoints/config.json as a plain dict."""
    p = LOGS_DIR / run_name / "checkpoints" / "config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── policy attributes (the config knobs a dataset's policies differ by) ───────
#
# The policies in a dataset differ by whatever the sweep varied — PD gains, a
# sensor bundle, a reward weight. That lives in each run's checkpoints/config.json,
# so we read one config per policy, keep its scalar leaves, and offer every key
# that actually varies as a heatmap axis. Nothing is hardcoded per env: a dataset
# gets exactly the axes its own sweep turned.

POLICY_AXIS_MAX_VALUES = 24   # beyond this a key is a per-run detail, not an axis

_policy_attr_cache: dict[tuple, dict] = {}   # run_names -> attrs (configs are immutable)


def _flat_scalars(cfg: dict, prefix: str = "") -> dict:
    """Config leaves as {dotted key: scalar}; list/dict values are dropped."""
    out: dict = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(_flat_scalars(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float, str, bool)):
            out[f"{prefix}{k}"] = v
    return out


def _policy_attrs(pol: dict) -> dict:
    """One policy's config knobs, from its first seed with a readable config.

    Seeds of a policy differ only in their RNG seed, so any one of them carries
    the policy's settings.
    """
    key = tuple(pol.get("run_names", []))
    if key in _policy_attr_cache:
        return _policy_attr_cache[key]
    attrs: dict = {}
    for run_name in key:
        cfg = _load_run_config(run_name)
        if cfg:
            attrs = _flat_scalars(cfg)
            attrs.setdefault("sensor_bundle", pol.get("sensor_bundle", ""))
            break
    _policy_attr_cache[key] = attrs
    return attrs


_run_attr_cache: dict[str, dict] = {}   # run name -> config scalars (immutable)


def _run_attrs(logs_dir: Path, run_name: str) -> dict:
    """One run's config knobs as {dotted key: scalar}; empty if unreadable."""
    if run_name not in _run_attr_cache:
        path = logs_dir / run_name / "checkpoints" / "config.json"
        try:
            cfg = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — an unreadable config is just no attrs
            cfg = {}
        _run_attr_cache[run_name] = _flat_scalars(cfg)
    return _run_attr_cache[run_name]


def _policies_payload(logs_dir: Path) -> dict:
    """Runs, plus per task the config knobs that vary across that task's runs.

    The picker filters ~900 runs by task, config value and free text, so every
    run ships the knobs its own task turned. Config keys constant across a task
    are dropped: they can't narrow anything and they triple the payload. The
    facets are the same axes the multi-policy tab offers, grouped per task
    rather than per dataset.

    Each run's training result (held-success %, eval reward, status, divergence
    — see run_metrics) is merged into the same attribute namespace under a
    "perf." prefix. That is what tells the seeds of one policy apart, and it
    costs nothing extra in the client: search terms (`perf.success>0.5`) and
    facet chips work on it exactly as they do on a config knob. Unlike config
    keys, perf keys survive the trim even when they vary too finely to be a
    facet — they are the numbers on every row.
    """
    policies = _list_policies(logs_dir)
    metrics = run_metrics.load()
    by_env: dict[str, list[dict]] = {}
    for pol in policies:
        by_env.setdefault(pol["env"], []).append(pol)
    facets = {}
    for env, rows in by_env.items():
        cfg_attrs = [_run_attrs(logs_dir, r["name"]) for r in rows]
        perf_attrs = [metrics.get(r["name"], {}) for r in rows]
        axes = _policy_axes([{**c, **p} for c, p in zip(cfg_attrs, perf_attrs)])
        facets[env] = axes
        keep = {ax["name"] for ax in axes}
        for row, cfg, perf in zip(rows, cfg_attrs, perf_attrs):
            row["attrs"] = {k: v for k, v in cfg.items() if k in keep}
            row["attrs"].update(perf)
    return {"policies": policies, "facets": facets}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _attr_text(v) -> str:
    """A non-numeric value as the client sees it (JavaScript's String(v))."""
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def _policy_axes(attrs_list: list[dict]) -> list[dict]:
    """Config keys that vary across policies, as selectable heatmap axes.

    A key qualifies when at least two policies carry it and it takes 2..
    POLICY_AXIS_MAX_VALUES distinct values. The client matches a policy onto an
    axis by comparing string forms, so a key whose values mix numbers with text
    is skipped — its two sides would never format alike.
    """
    seen: dict[str, list] = {}
    for attrs in attrs_list:
        for k, v in attrs.items():
            seen.setdefault(k, []).append(v)
    axes = []
    for name, vals in sorted(seen.items()):
        if len(vals) < 2:
            continue
        numeric = all(_is_number(v) for v in vals)
        if not numeric and any(_is_number(v) for v in vals):
            continue
        uniq = sorted({(float(v) if numeric else _attr_text(v)) for v in vals})
        if not 2 <= len(uniq) <= POLICY_AXIS_MAX_VALUES:
            continue
        axes.append({"name": name, "label": name.split(".")[-1],
                     "values": uniq, "numeric": numeric})
    # Two keys can share a last segment (a.gain / b.gain) — spell those out.
    by_label: dict[str, list] = {}
    for ax in axes:
        by_label.setdefault(ax["label"], []).append(ax)
    for group in by_label.values():
        if len(group) > 1:
            for ax in group:
                ax["label"] = ax["name"]
    return axes


def _compute_view(cmp_policies: list, criterion: dict, visualization: str) -> dict:
    """Render loaded caches as the chosen view — decoupled from how they were
    gathered. Heatmap bins/pools per compute_grid_heatmap (grid cells or sampled
    bins via criterion['heatmap_axes']); curves pool every rollout."""
    from policy_analyzer import success_curve
    if visualization == "heatmap":
        result = success_curve.compute_grid_heatmap(cmp_policies, criterion)
        # Policy axes are derived from the policies that actually made it into
        # the result, so their values span exactly what the heatmap can show.
        result["policy_axes"] = _policy_axes(
            [p.get("attrs", {}) for p in result.get("policies", [])])
        return result
    return success_curve.compute_curves(cmp_policies, criterion)


# ── saved datasets (named collection identities; caches live in compare_cache) ──

DATASETS_DIR = ANALYSIS_DIR / "datasets"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "dataset"


def _dataset_cache_bench(meta: dict) -> str:
    """On-disk cache benchmark id for a dataset (grid selections are spec-keyed)."""
    benchmark = meta.get("benchmark", "default")
    if benchmark == GRID_BENCHMARK:
        return f"{GRID_BENCHMARK}/{_grid_sig(meta.get('grid_spec'))}"
    return benchmark


def _dataset_path(slug: str) -> Path:
    return DATASETS_DIR / f"{slug}.json"


def _write_dataset(meta: dict) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    _dataset_path(meta["slug"]).write_text(json.dumps(meta, indent=2))


def _load_dataset(slug: str) -> dict | None:
    p = _dataset_path(slug)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _dataset_progress(meta: dict) -> tuple[str, int, int]:
    """(status, seeds_done, seeds_total) — "ready" once every run's cache is present.

    Counted off the on-disk DONE sentinels, so progress survives a page reload or
    a server restart mid-collection.
    """
    bench, mode = _dataset_cache_bench(meta), meta.get("mode", "det")
    runs = [r for pol in meta.get("policies", []) for r in pol.get("run_names", [])]
    if not runs:
        return "empty", 0, 0
    done = sum(_compare_cache_done(r, mode, bench) for r in runs)
    return ("ready" if done == len(runs) else "collecting"), done, len(runs)


def _list_datasets() -> list[dict]:
    if not DATASETS_DIR.is_dir():
        return []
    out = []
    for p in DATASETS_DIR.glob("*.json"):
        meta = _load_dataset(p.stem)
        if not meta:
            continue
        status, seeds_done, seeds_total = _dataset_progress(meta)
        out.append({
            "name": meta.get("name", p.stem),
            "slug": meta.get("slug", p.stem),
            "env": meta.get("env", ""),
            "benchmark": meta.get("benchmark", "default"),
            "mode": meta.get("mode", "det"),
            "n_policies": len(meta.get("policies", [])),
            "created": meta.get("created", ""),
            "status": status,
            "seeds_done": seeds_done,
            "seeds_total": seeds_total,
        })
    return sorted(out, key=lambda d: d.get("created", ""), reverse=True)


# ── rendered-view cache ───────────────────────────────────────────────────────
#
# Rendering a dataset re-reads every seed's npz (a 100-policy grid dataset is
# ~300 files) and re-scores it — the wait between opening a dataset and seeing
# anything. The all-policies view is also the one asked for over and over
# unchanged, since that is what a freshly opened dataset selects, so its
# finished result is written to disk and replayed on the next request.
#
# All-selected views get one file per settings combo: that view is what a
# freshly opened dataset lands on, so it is asked for over and over unchanged.
# Every partial selection shares a single "last" slot instead — enough to come
# back to the view you left a dataset on, without a file per one-off probe.

VIEW_CACHE_DIR = ANALYSIS_DIR / "view_cache"


def _view_cache_key(mode: str, benchmark: str, visualization: str,
                    criterion: dict, labels: list[str]) -> str:
    """Everything the rendered result depends on, hashed. Mirrors the client's
    viewSignature() — same settings in, same file out."""
    canon = json.dumps(
        {"mode": mode, "benchmark": benchmark, "visualization": visualization,
         "criterion": criterion, "policies": sorted(labels)},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def _view_cache_stamp(policies: list, mode: str, benchmark: str) -> str:
    """Fingerprint of the source caches this view was rendered from.

    Re-collecting a seed rewrites its npz under the same path, which leaves the
    key untouched — so the stamp is stored alongside the result and checked on
    read. A changed (or vanished) source file misses, and the view is recomputed.
    """
    parts = []
    for pol in policies:
        for run_name in sorted(pol.get("run_names", [])):
            p = _compare_cache_path(run_name, mode, benchmark)
            try:
                st = p.stat()
                parts.append(f"{run_name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{run_name}:missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _view_cache_dir(slug: str) -> Path:
    # Per-dataset directory so deleting a dataset can drop its views wholesale.
    # Re-slugified because the slug arrives from the client.
    return VIEW_CACHE_DIR / _slugify(slug)


def _view_cache_path(slug: str, key: str, all_selected: bool) -> Path:
    # One file per settings combo for the all-selected views; one shared slot,
    # overwritten, for whatever subset was looked at last.
    return _view_cache_dir(slug) / (f"{key}.json" if all_selected else "last.json")


def _read_view_cache(slug: str, key: str, stamp: str,
                     all_selected: bool) -> dict | None:
    p = _view_cache_path(slug, key, all_selected)
    if not p.exists():
        return None
    try:
        entry = json.loads(p.read_text())
    except Exception:
        return None
    # Both checks matter for the shared slot: it holds one subset's view, so a
    # different subset (or a re-collect) has to miss rather than read it. Entries
    # written before the slot existed carry no "key" and are keyed by filename
    # alone — still valid, so they survive the upgrade rather than all missing.
    if entry.get("key", key) != key or entry.get("stamp") != stamp:
        return None
    return entry.get("result")


def _write_view_cache(slug: str, key: str, stamp: str, result: dict,
                      all_selected: bool) -> None:
    p = _view_cache_path(slug, key, all_selected)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader never sees a half-written view, and two
        # concurrent renders of the same settings can't interleave.
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": key, "stamp": stamp, "result": result}))
        tmp.replace(p)
    except Exception:
        pass   # a cache write failing must never fail the render


def _drop_view_cache(slug: str) -> None:
    shutil.rmtree(_view_cache_dir(slug), ignore_errors=True)


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

        def _error_json(self, exc: Exception) -> None:
            """Report a handler crash as a 500 + JSON body. An unhandled
            exception otherwise closes the connection with an empty reply,
            which the frontend cannot distinguish from a dead server."""
            import traceback
            traceback.print_exc()
            try:
                body = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}
                ).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass  # headers already sent (e.g. mid-SSE) — nothing to salvage

        def do_GET(self):
            try:
                self._do_GET()
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as exc:  # noqa: BLE001 — HTTP boundary
                self._error_json(exc)

        def do_POST(self):
            try:
                self._do_POST()
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as exc:  # noqa: BLE001 — HTTP boundary
                self._error_json(exc)

        def _do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            if path == "/api/policies":
                self._json(_policies_payload(server.logs_dir))
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
            elif path == "/api/compare/grid_axes":
                self._json(server.grid_axes(
                    params.get("run", ""), params.get("env", "")))
            elif path == "/api/datasets":
                self._json(server.list_datasets())
            elif path.startswith("/api/datasets/"):
                meta = server.get_dataset(path[len("/api/datasets/"):])
                if meta is not None:
                    self._json(meta)
                else:
                    self._no_body(404)
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

        def _do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/sessions":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._json(server.start_session(
                    run=body["run"],
                    checkpoint_step=body.get("checkpoint_step", "latest"),
                    n_det=int(body.get("n_det", 0)),
                    n_sto=int(body.get("n_sto", 0)),
                    sweep=body.get("sweep") or [],
                ))
            elif path == "/api/compare":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._json(server.start_compare(body))
            elif path == "/api/compare/view":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._json(server.view_score(body))
            elif path == "/api/datasets":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                self._json(server.create_dataset(body))
            else:
                self._no_body(405)

        def do_DELETE(self):
            path = urllib.parse.urlsplit(self.path).path
            if path.startswith("/api/datasets/"):
                ok = server.delete_dataset(path[len("/api/datasets/"):])
                self._no_body(200 if ok else 404)
            elif path.startswith("/api/sessions/"):
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
        self._running: dict[int, list[dict]] = {}    # gpu_id -> running job dicts
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
        self._clear_sessions()
        (analysis_dir / "index.html").write_bytes(_APP_TEMPLATE.read_bytes())
        if _APP_ASSETS.is_dir():
            shutil.copytree(_APP_ASSETS, analysis_dir / "assets", dirs_exist_ok=True)

        self._sched = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="rollout-scheduler"
        )
        self._sched.start()
        # Training metrics come from the cockpit over HTTP, which is slow enough
        # cold (~20 s for every queue) that no request may ever wait on it: the
        # picker reads the snapshot on disk, this thread keeps it fresh.
        threading.Thread(
            target=self._metrics_loop, daemon=True, name="run-metrics"
        ).start()
        print(f"Policy Analyzer ready — {len(self._gpus)} GPU(s): {self._gpus}", flush=True)

    def _metrics_loop(self) -> None:
        while True:
            try:
                idx = run_metrics.refresh()
                print(f"[metrics] {len(idx)} runs indexed from the cockpit", flush=True)
            except Exception as exc:  # noqa: BLE001 — a background refresh never dies
                print(f"[metrics] refresh failed: {exc}", flush=True)
            time.sleep(_METRICS_REFRESH_SECS)

    # ── session loading ───────────────────────────────────────────────────────

    def _clear_sessions(self) -> None:
        """Start every boot with an empty sessions dir.

        A single-policy analysis is working data, not a cache: its artifacts are
        the videos and rollout pages of one look at one policy, they are a
        gigabyte per couple of dozen runs, and nothing downstream reads them
        back. Keeping them across restarts only accumulated stale analyses
        behind a dropdown — so the directory is a scratch space for the analyses
        started in this process, wiped on the way up.

        Note this is why the sweep explorer reads compare_cache instead: those
        npz files ARE a cache (content-addressed by run + mode + grid spec, and
        shared with the Compare tab), whereas everything under sessions/ is
        per-look and disposable.
        """
        sessions_dir = self.analysis_dir / "sessions"
        if not sessions_dir.exists():
            return
        for d in sessions_dir.iterdir():
            shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)

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

    def _unsettled_reserve_locked(self, gpu: int) -> float:
        """VRAM (GB) to hold back for jobs on `gpu` that are still allocating.

        nvidia-smi reports a just-launched job's memory as free until it gets
        past imports and checkpoint restore, so without this a burst of launches
        would each see an empty card. Caller holds _lock.
        """
        now = time.time()
        young = sum(1 for j in self._running.get(gpu, ())
                    if now - j.get("started", 0.0) < VRAM_SETTLE_SECS)
        return young * EXPECTED_JOB_VRAM_GB

    def _drop_running(self, gpu: int, job: dict) -> None:
        """Remove one finished job from a GPU's slot list. Caller holds _lock."""
        jobs = self._running.get(gpu)
        if not jobs:
            return
        # Identity, not equality: several jobs on one GPU can look alike.
        self._running[gpu] = [j for j in jobs if j is not job]
        if not self._running[gpu]:
            self._running.pop(gpu, None)

    def _reap(self) -> None:
        """Mark seeds done as their DONE sentinels appear; handle group exit."""
        with self._lock:
            running = [(gpu, job)
                       for gpu, jobs in self._running.items()
                       for job in jobs]

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
                self._drop_running(gpu, job)

    def _dispatch(self) -> None:
        """Launch queued work (session rollouts, then compare evals) onto GPUs
        with a free slot and enough VRAM. Session rollouts take priority.

        There is no cap on jobs per GPU: a GPU accepts another job whenever it
        still has MIN_FREE_VRAM_GB of headroom, counting jobs too young to have
        allocated yet (see EXPECTED_JOB_VRAM_GB). A job that is admitted anyway
        and runs out of memory exits EX_TEMPFAIL and is requeued, so the gate
        only has to be roughly right.
        """
        with self._lock:
            has_work = bool(self._pending) or bool(self._compare_pending)
        if not has_work:
            return

        for gpu in self._gpus:
            with self._lock:
                if not (self._pending or self._compare_pending):
                    break
            free = _free_vram_gb(gpu)  # shell out outside the lock
            with self._lock:
                # Headroom the card would still have once the candidate job AND
                # any jobs still allocating have taken their share. Charging the
                # candidate matters: without it, a card with 7 GB free accepts
                # two 2.6 GB jobs and lands under MIN_FREE_VRAM_GB.
                free -= self._unsettled_reserve_locked(gpu) + EXPECTED_JOB_VRAM_GB
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
        names = [n for n, _, _ in _rollout_ids(tag, sess.axes, sess.cells, seeds)]

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
        if sess.axes:
            # The grid is resolved once, in the request thread; the subprocess
            # reads it rather than re-deriving values that must match the names
            # above element for element.
            pins = sess.session_dir / f"pins_{tag}.json"
            pins.write_text(json.dumps(
                {"axes": sess.axes, "cells": sess.cells, "names": names}))
            cmd += ["--pins", str(pins)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        logpath = sess.session_dir / f"{tag}.log"
        logf = open(logpath, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(PKG_DIR.parent), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        self._running.setdefault(gpu, []).append({
            "proc": proc, "sid": sid, "det": det, "seeds": seeds,
            "names": names, "pending": set(names),
            "logf": logf, "logpath": logpath, "started": time.time(),
        })
        for name in names:
            sess.update_rollout(name, "running")
        print(f"[GPU {gpu}] launched {sid}/{tag} seeds={list(seeds)} "
              f"({len(self._running[gpu])} running)", flush=True)

    def _launch_compare_locked(self, gpu: int, task: dict) -> None:
        """Spawn a compare_collect subprocess for one run+mode+benchmark. Caller holds _lock."""
        run_name, mode, benchmark = task["run_name"], task["mode"], task["benchmark"]
        out_path = _compare_cache_path(run_name, mode, benchmark)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh run invalidates a stale DONE sentinel from a prior eval.
        done = out_path.parent / f"{out_path.stem}.DONE"
        done.unlink(missing_ok=True)

        if _is_grid(benchmark):
            cmd = [
                sys.executable, "-m", "policy_analyzer.grid_collect",
                "--log-dir", str(self.logs_dir / run_name),
                "--out", str(out_path),
                "--mode", mode,
                "--n-rollouts", str(task["n_rollouts"]),
            ]
            if task.get("axes"):
                cmd += ["--axes", ",".join(task["axes"])]
            if task.get("axis_points"):
                cmd += ["--axis-points",
                        ",".join(f"{k}={v}" for k, v in task["axis_points"].items())]
        else:
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
        self._running.setdefault(gpu, []).append({
            "kind": "compare", "proc": proc, "run_name": run_name, "mode": mode,
            "benchmark": benchmark, "n_rollouts": task["n_rollouts"],
            "logf": logf, "logpath": logpath, "out_path": out_path,
            "started": time.time(),
        })
        self._compare_errors.pop((run_name, mode, benchmark), None)
        print(f"[GPU {gpu}] launched compare {run_name}/{benchmark}/{mode} "
              f"({len(self._running[gpu])} running)", flush=True)

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
            self._drop_running(gpu, job)
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

    def resolve_sweep(self, run: str, spec: list) -> dict:
        """Turn an axis selection into concrete axes + cells, or an error.

        Runs in the request thread: axis values come from the run's config via
        grid_collect.axes_from_spec, which is numpy-and-stdlib only, so the
        client can see the exact grid (and any mistake in it) before a GPU is
        touched.
        """
        if not spec:
            return {"axes": [], "cells": [[]]}
        cfg = _load_run_config(run)
        if cfg is None:
            return {"error": f"no config.json for {run}"}
        from policy_analyzer import grid_collect
        try:
            axes = grid_collect.axes_from_spec(_env_of_run(run), cfg, spec)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"axes": axes, "cells": grid_collect.cell_grid(axes)}

    def start_session(self, run: str, checkpoint_step: str, n_det: int, n_sto: int,
                      sweep: list | None = None) -> dict:
        """Queue one analysis: (cells × repeats) rollouts, rendered and analysed.

        With no sweep this is the historical behaviour — n_det + n_sto sampled
        rollouts. With one, every cell of the grid gets n_det + n_sto rollouts
        with those axes pinned, and the counts read as repeats per cell.
        """
        resolved = self.resolve_sweep(run, sweep or [])
        if "error" in resolved:
            return resolved
        axes, cells = resolved["axes"], resolved["cells"]
        total = len(cells) * (n_det + n_sto)
        if total == 0:
            return {"error": "nothing to run — set a rollout count"}
        if total > MAX_SESSION_ROLLOUTS:
            return {"error": (
                f"{len(cells)} cells × {n_det + n_sto} rollouts = {total}, over the "
                f"{MAX_SESSION_ROLLOUTS} limit for one analysis. Every rollout here "
                "keeps its full trajectory and a video (~8 MB), so drop a point or "
                "a repeat — or sweep it in the Multi-policy tab, which keeps "
                "per-step channels only and scales to thousands of cells."
            )}

        run_short = run.split("-")[-1] if "-" in run else run
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{run_short}"
        session_dir = self.analysis_dir / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        # Each group (det / sto) runs all its rollouts in one vmapped pass on one
        # GPU — cells and repeats are both batch dimensions.
        rollouts: list[RolloutInfo] = []
        groups: list[tuple[bool, tuple[int, ...]]] = []
        for det, n, tag in ((True, n_det, "det"), (False, n_sto, "sto")):
            seeds = tuple(range(1, n + 1))
            if not seeds:
                continue
            for name, cell, seed in _rollout_ids(tag, axes, cells, seeds):
                params = {ax["name"]: cells[cell][i] for i, ax in enumerate(axes)}
                rollouts.append(RolloutInfo(
                    name=name, deterministic=det, seed=seed, params=params))
            groups.append((det, seeds))

        sess = Session(
            session_id=sid, session_dir=session_dir,
            run=run, checkpoint_step=checkpoint_step, rollouts=rollouts,
            axes=axes, cells=cells,
        )
        sess._save()

        with self._lock:
            self._sessions[sid] = sess
            for det, seeds in groups:
                self._pending.append((sid, det, seeds))

        return {"session_id": sid}

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
                    # Grid caches nest one level deeper (grid/<sig>/<mode>.npz),
                    # so a benchmark maps to several ids when it's the grid dir.
                    if bench_dir.name == GRID_BENCHMARK:
                        bench_ids = [
                            f"{GRID_BENCHMARK}/{d.name}"
                            for d in bench_dir.iterdir() if d.is_dir()
                        ]
                    else:
                        bench_ids = [bench_dir.name]
                    for bid in bench_ids:
                        for mode in ("det", "sto"):
                            if not _compare_cache_done(run_dir.name, mode, bid):
                                continue
                            cache = (
                                _load_grid_cache(run_dir.name, mode, bid)
                                if _is_grid(bid)
                                else _load_compare_cache(run_dir.name, mode, bid)
                            )
                            if cache and cache["env_name"] == env:
                                channels = sorted(cache["channels"].keys())
                                break
                        if channels:
                            break
                    if channels:
                        break
                if channels:
                    break
        return success_curve.config_for_env(env, channels)

    def grid_axes(self, run: str, env: str) -> list[dict]:
        """Candidate grid axes for a run, read from its config.json (per-run
        ranges). Returns UI-safe fields for the Compare-tab checkbox selector."""
        from policy_analyzer import grid_axes as _ga
        cfg = _load_run_config(run)
        if cfg is None or not env:
            return []
        return _ga.candidates_ui(env, cfg)

    def compare_benchmarks(self, env: str) -> list[dict]:
        """Benchmarks selectable for an env (label/description/name), default first."""
        from policy_analyzer import benchmarks
        from policy_analyzer.grid_collect import SUPPORTED_ENVS as _grid_envs
        out = [
            {"name": b["name"], "label": b["label"], "description": b["description"]}
            for b in benchmarks.benchmarks_for_env(env)
        ]
        if env in _grid_envs:
            out.append({
                "name": GRID_BENCHMARK,
                "label": "Grid — pinned randomization sweep",
                "description": (
                    "Deterministic grid over the run's randomization ranges "
                    f"(episode + model DR), {GRID_N_ROLLOUTS} rollouts per cell "
                    "by default (Advanced) with every randomized quantity "
                    "pinned. Choose which axes to "
                    "sweep (checkboxes, per-axis resolution); renders a colour-coded "
                    "heatmap where you pick any two axes to plot against (the rest "
                    "are averaged out)."
                ),
            })
        return out

    def _resolve_collection(self, body: dict) -> dict:
        """Normalize a collection request → {mode, n_rollouts, benchmark(resolved),
        task_extra, grid_spec} or {"error": …}. Grid selections become the
        spec-keyed benchmark id and carry the axes to sweep."""
        mode = body.get("mode", "det")
        n_rollouts = int(body.get("n_rollouts", 50))
        benchmark = body.get("benchmark", "default")
        task_extra: dict = {}
        grid_spec = None
        if benchmark == GRID_BENCHMARK:
            # Rollouts per cell is its own knob so switching benchmark never
            # silently invalidates the pooled benchmark's caches.
            n_rollouts = max(1, int(body.get("grid_n_rollouts", GRID_N_ROLLOUTS)))
            grid_spec = body.get("grid_spec") or {}
            sel = grid_spec.get("axes", [])
            if not sel:
                return {"error": "select at least one grid axis"}
            cells = 1
            for a in sel:
                cells *= max(1, int(a["points"]))
            if cells > GRID_MAX_CELLS:
                return {"error": f"{cells} grid cells exceeds max {GRID_MAX_CELLS}; "
                                 "uncheck an axis or lower its points"}
            benchmark = f"{GRID_BENCHMARK}/{_grid_sig(grid_spec)}"
            task_extra = {
                "axes": [a["name"] for a in sel],
                "axis_points": {a["name"]: int(a["points"]) for a in sel},
            }
        return {"mode": mode, "n_rollouts": n_rollouts, "benchmark": benchmark,
                "task_extra": task_extra, "grid_spec": grid_spec}

    def _queue_and_register(self, policies: list, coll: dict, job_extra: dict) -> str:
        """Enqueue evals for any uncached runs and register a compare job."""
        mode, n_rollouts, benchmark = coll["mode"], coll["n_rollouts"], coll["benchmark"]
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
                         "benchmark": benchmark, "n_rollouts": n_rollouts,
                         **coll["task_extra"]}
                    )
            self._compare_jobs[cid] = {
                "mode": mode, "n_rollouts": n_rollouts, "benchmark": benchmark,
                "policies": policies, **job_extra,
            }
        return cid

    def start_compare(self, body: dict) -> dict:
        """Queue evals for any uncached runs and register a compare job."""
        coll = self._resolve_collection(body)
        if "error" in coll:
            return coll
        cid = self._queue_and_register(body.get("policies", []), coll, {
            "criterion": body.get("criterion", {}),
            # Visualization is independent of the collection benchmark; default
            # follows the benchmark only when the client didn't specify one.
            "visualization": body.get(
                "visualization", "heatmap" if _is_grid(coll["benchmark"]) else "curves"),
        })
        return {"compare_id": cid}

    # ── saved datasets ──
    def list_datasets(self) -> list[dict]:
        return _list_datasets()

    def get_dataset(self, slug: str) -> dict | None:
        """Dataset meta, enriched with each policy's config knobs.

        The client needs these up front — before any view is computed — to render
        the policy selector and its per-attribute bulk toggles.
        """
        meta = _load_dataset(slug)
        if meta is None:
            return None
        attrs = [_policy_attrs(p) for p in meta.get("policies", [])]
        for pol, a in zip(meta.get("policies", []), attrs):
            pol["attrs"] = a
        meta["policy_axes"] = _policy_axes(attrs)
        return meta

    def delete_dataset(self, slug: str) -> bool:
        p = _dataset_path(slug)
        if not p.exists():
            return False
        p.unlink()          # caches are content-addressed and may be shared — keep them
        _drop_view_cache(slug)   # rendered views are per-dataset, so they go with it
        return True

    def create_dataset(self, body: dict) -> dict:
        """Persist a named dataset (collection identity) and queue its collection."""
        name = (body.get("name") or "").strip()
        policies = body.get("policies", [])
        if not name:
            return {"error": "dataset name required"}
        if not policies:
            return {"error": "select at least one policy"}
        coll = self._resolve_collection(body)
        if "error" in coll:
            return coll
        slug = _slugify(name)
        meta = {
            "name": name, "slug": slug, "env": body.get("env", ""),
            "mode": coll["mode"],
            # Store the client-form benchmark + grid_spec; the on-disk cache id is
            # re-derived via _dataset_cache_bench, matching view_score.
            "benchmark": body.get("benchmark", "default"),
            "grid_spec": coll["grid_spec"],
            "grid_n_rollouts": int(body.get("grid_n_rollouts", GRID_N_ROLLOUTS)),
            "n_rollouts": coll["n_rollouts"], "policies": policies,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _write_dataset(meta)
        cid = self._queue_and_register(policies, coll, {"dataset": slug})
        return {"dataset": slug, "compare_id": cid}

    def compare_status(self, cid: str) -> dict | None:
        with self._lock:
            job = self._compare_jobs.get(cid)
        if job is None:
            return None

        mode = job["mode"]
        benchmark = job.get("benchmark", "default")
        # Which seeds actually hold a GPU right now, so "running" means running
        # and everything else in flight reports as "queued".
        with self._lock:
            on_gpu = {
                (j["run_name"], j["mode"], j["benchmark"]): gpu
                for gpu, jobs in self._running.items()
                for j in jobs if j.get("kind") == "compare"
            }
        runs_status = []
        all_ready = True
        for pol in job["policies"]:
            for run_name in pol.get("run_names", []):
                key = (run_name, mode, benchmark)
                entry = {"run_name": run_name, "policy": pol.get("label", "")}
                if _compare_cache_done(run_name, mode, benchmark):
                    st = "ready"
                elif key in self._compare_errors:
                    st = "error"
                elif key in on_gpu:
                    st = "running"
                    entry["gpu"] = on_gpu[key]
                    entry["progress"] = _collect_progress(run_name, mode, benchmark)
                elif key in self._compare_inflight:
                    st = "queued"
                else:
                    # Not cached, not erroring, not queued → safety net so the
                    # client doesn't poll forever for something never launched.
                    st = "error"
                    self._compare_errors[key] = "not collected"
                if st != "ready":
                    all_ready = False
                entry["status"] = st
                entry["error"] = self._compare_errors.get(key)
                runs_status.append(entry)

        n_done = sum(r["status"] == "ready" for r in runs_status)
        payload = {"runs": runs_status, "seeds_done": n_done,
                   "seeds_total": len(runs_status)}

        if not all_ready:
            return {"status": "collecting", **payload}

        # Dataset-collect jobs carry no visualization — collection only. The client
        # visualizes via view_score over the saved dataset once ready.
        if job.get("dataset"):
            return {"status": "ready", **payload}

        # All caches present → render the chosen view (viz independent of source).
        cmp_policies = _load_compare_policies(job["policies"], mode, benchmark)
        result = _compute_view(cmp_policies, job["criterion"], job.get("visualization"))
        return {"status": "ready", **payload, "result": result}

    def view_score(self, body: dict) -> dict:
        """Render an already-collected cache as any view — no GPU, no new rollouts.

        Collection caches all channels; switching visualization (curves↔heatmap),
        success channel, threshold, criterion, or (for a sampled heatmap) the bin
        axes is pure numpy over the cached arrays, so this is synchronous. Serves
        all four combos: {grid, sampled} × {curves, heatmap}. Returns {"error": …}
        if a run's cache is missing so the client can fall back to Generate.

        Views of a saved dataset are replayed from (and written to) the
        rendered-view cache — see VIEW_CACHE_DIR. `cache_only` asks for that
        replay alone: a hit renders, a miss returns without computing, so the
        client can offer an instant plot without silently doing the work.
        """
        mode = body.get("mode", "det")
        criterion = body.get("criterion", {})
        policies = body.get("policies", [])
        visualization = body.get("visualization", "curves")
        benchmark = body.get("benchmark", "default")
        # Resolve the on-disk cache id: grid selections are spec-keyed, sampled
        # benchmarks use their own name.
        if benchmark == GRID_BENCHMARK:
            benchmark = f"{GRID_BENCHMARK}/{_grid_sig(body.get('grid_spec'))}"
        for pol in policies:
            for run_name in pol.get("run_names", []):
                if not _compare_cache_done(run_name, mode, benchmark):
                    return {"error": "not collected"}

        # Cacheable only when the request names a saved dataset: the slug names
        # the cache dir, and the dataset's own policy list is what "all
        # selected" is measured against (the client is not trusted to say so).
        slug = body.get("dataset") or ""
        labels = [p.get("label", "") for p in policies]
        meta = _load_dataset(slug) if slug else None
        cacheable = bool(meta)
        if cacheable:
            all_selected = sorted(labels) == sorted(
                p.get("label", "") for p in meta.get("policies", []))
            key = _view_cache_key(mode, benchmark, visualization, criterion, labels)
            stamp = _view_cache_stamp(policies, mode, benchmark)
            hit = _read_view_cache(slug, key, stamp, all_selected)
            if hit is not None:
                return {"status": "ready", "result": hit, "cached": True}
        if body.get("cache_only"):
            return {"error": "no cache"}

        cmp_policies = _load_compare_policies(policies, mode, benchmark)
        if not cmp_policies:
            return {"error": "not collected"}
        result = _compute_view(cmp_policies, criterion, visualization)
        if cacheable:
            _write_view_cache(slug, key, stamp, result, all_selected)
        return {"status": "ready", "result": result}

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(sid, None)
            if sess is None:
                return False
            # Drop queued rollouts for this session.
            self._pending = collections.deque(
                t for t in self._pending if t[0] != sid
            )
            # Kill any running rollouts for this session. Compare jobs share the
            # slot lists and carry no "sid", hence .get rather than [].
            for gpu, jobs in list(self._running.items()):
                for job in list(jobs):
                    if job.get("sid") == sid:
                        job["proc"].terminate()
                        job["logf"].close()
                        self._drop_running(gpu, job)
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

    # The server itself never imports JAX, but most API endpoints (and every
    # collect subprocess, which reuses sys.executable) need numpy. Fail fast
    # with a pointer instead of 500-ing on first use: launching with a bare
    # `python` outside the venv is exactly how the Compare tab goes dead.
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit(
            f"policy_analyzer: {sys.executable} has no numpy — launch with the "
            "training venv, e.g.\n"
            "  mujoco_playground/.venv/bin/python -m policy_analyzer"
        )

    server = AnalysisServer(
        analysis_dir=ANALYSIS_DIR,
        logs_dir=LOGS_DIR,
    )
    server.serve(args.port)


if __name__ == "__main__":
    main()

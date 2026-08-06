"""Per-run training metrics, read off the experiment cockpit (no JAX, stdlib only).

The cockpit already derives what the policy picker needs to tell seeds of one
policy apart — eval/train held-success %, eval reward, the run's status and
whether it diverged — from each queue's wandb history. None of that is
recomputed here: we read its HTTP API (`/api/queues`, `/api/queues/<id>`) and
keep a local index keyed by run name.

The index is incremental. A queue's detail is re-fetched only when its
`last_activity` moves, so the first refresh costs one request per queue (~180)
and every refresh after that costs a handful — the queues that are actually
running. The index is persisted, so a server restart starts from the last
snapshot and the picker is never blocked on the network: `load()` reads the
file, `refresh()` is what talks to the cockpit, and the server calls it from a
background thread.

With the cockpit down, refresh() leaves the snapshot untouched and the picker
simply shows no performance numbers for runs it has never seen.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from policy_analyzer.paths import ANALYSIS_DIR

COCKPIT_URL = "http://127.0.0.1:8010"
CACHE_PATH = ANALYSIS_DIR / "run_metrics.json"

_TIMEOUT = 20.0   # a cold queue detail parses wandb history; be patient


def _get(path: str, timeout: float):
    url = COCKPIT_URL.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_metrics(run: dict, queue_id: str) -> dict:
    """One cockpit run row reduced to the numbers the picker shows.

    Kept flat and JSON-scalar: these values are merged into the same attribute
    namespace as a run's config knobs, where the search terms and the facet
    chips both expect scalars.
    """
    success = run.get("success") or {}
    reward = run.get("reward") or {}
    div = run.get("divergence") or {}
    out = {
        "perf.status": run.get("status") or "unknown",
        "perf.queue": queue_id,
    }
    if success.get("eval") is not None:
        out["perf.success"] = round(float(success["eval"]), 4)
    if success.get("train") is not None:
        out["perf.success_train"] = round(float(success["train"]), 4)
    if reward.get("eval") is not None:
        out["perf.reward"] = round(float(reward["eval"]), 3)
    if run.get("seed") is not None:
        out["perf.seed"] = run["seed"]
    if run.get("final_step") is not None:
        out["perf.steps"] = int(run["final_step"])
    if div.get("flag"):
        out["perf.divergence"] = div["flag"]
    return out


def _read_cache() -> dict:
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing or half-written cache is just empty
        return {"queues": {}}
    return data if isinstance(data, dict) and "queues" in data else {"queues": {}}


def _flatten(cache: dict) -> dict:
    """{run_name: metrics} across every cached queue.

    A run can appear in more than one queue snapshot (a resumed queue keeps the
    original exp_name); later queues win, which is also the fresher reading.
    """
    out: dict = {}
    for entry in cache.get("queues", {}).values():
        out.update(entry.get("runs", {}))
    return out


def load() -> dict:
    """The last persisted index — never touches the network."""
    return _flatten(_read_cache())


def refresh(timeout: float = _TIMEOUT) -> dict:
    """Bring the index up to date from the cockpit and persist it.

    Returns the flattened index. On a connection failure the snapshot on disk is
    returned unchanged — a missing cockpit degrades the picker, it doesn't break
    it.
    """
    cache = _read_cache()
    try:
        queues = _get("/api/queues", timeout)
    except Exception:  # noqa: BLE001 — cockpit down / not installed
        return _flatten(cache)

    known = cache["queues"]
    for q in queues:
        qid = q.get("id")
        if not qid:
            continue
        stamp = str(q.get("last_activity") or q.get("started_at") or "")
        cached = known.get(qid)
        if cached is not None and cached.get("stamp") == stamp:
            continue
        try:
            detail = _get("/api/queues/" + urllib.parse.quote(qid), timeout)
        except Exception:  # noqa: BLE001 — skip this queue, keep the rest
            continue
        runs = {}
        for run in detail.get("runs", []):
            name = run.get("exp_name")
            if name:
                runs[name] = _run_metrics(run, qid)
        known[qid] = {"stamp": stamp, "runs": runs}

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass
    return _flatten(cache)

"""Queue-oriented discovery of trained policies for the Compare tab (no JAX).

Training runs are launched in *queues*: logs/_queue/<queue>/status.json records
one entry per run, with its env, suffix, training seed (flags.seed), result, the
resulting log dir name (exp_name), and the overrides file it was launched with.

Training-seed replicates of one policy share a byte-identical overrides file
(only flags.seed differs, and the suffix carries a trailing _s<N>). We therefore
group a queue's entries by overrides-file content into *policies*, each exposing
its individual seed runs. The UI renders each policy as a stack of sheets that
expands to the per-seed runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from policy_analyzer.paths import LOGS_DIR, MJPG_ROOT

_QUEUE_DIR = LOGS_DIR / "_queue"
_SEED_SUFFIX_RE = re.compile(r"_s\d+$")


def _strip_seed_suffix(suffix: str) -> str:
    return _SEED_SUFFIX_RE.sub("", suffix or "")


def _has_checkpoints(run_name: str) -> bool:
    ckpt_dir = LOGS_DIR / run_name / "checkpoints"
    return ckpt_dir.is_dir() and any(d.is_dir() for d in ckpt_dir.iterdir())


def _read_overrides(rel_path: str | None) -> str | None:
    """Overrides-file text (the grouping key). Path is relative to the repo root."""
    if not rel_path:
        return None
    p = MJPG_ROOT / rel_path
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _sensor_bundle(overrides_text: str | None) -> str:
    if not overrides_text:
        return ""
    for line in overrides_text.splitlines():
        if line.startswith("sensor_bundle:"):
            return line.split(":", 1)[1].strip()
    return ""


def _policies_for_queue(entries: list[dict]) -> list[dict]:
    """Group a queue's status entries into policies keyed by overrides content."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for e in entries:
        run_name = e.get("exp_name")
        if not run_name:
            continue  # never produced a run dir (launch failed early)
        suffix = e.get("suffix", "")
        overrides_text = _read_overrides(e.get("overrides_file"))
        # Key on overrides content when available; else fall back to the
        # seed-stripped suffix so replicates still collapse.
        key = overrides_text if overrides_text is not None else f"sfx:{_strip_seed_suffix(suffix)}"

        g = groups.get(key)
        if g is None:
            g = groups[key] = {
                "label": _strip_seed_suffix(suffix) or run_name,
                "env_name": e.get("env_name") or run_name.split("-")[0],
                "sensor_bundle": _sensor_bundle(overrides_text),
                "seeds": [],
            }
            order.append(key)

        runnable = e.get("result") == "ok" and _has_checkpoints(run_name)
        g["seeds"].append({
            "run_name": run_name,
            "seed": e.get("flags", {}).get("seed"),
            "result": e.get("result", "unknown"),
            "runnable": runnable,
        })

    policies = []
    for i, key in enumerate(order):
        g = groups[key]
        g["seeds"].sort(key=lambda s: (s["seed"] is None, s["seed"]))
        policies.append({
            "id": f"{i}",
            "n_seeds": len(g["seeds"]),
            "runnable_seeds": sum(s["runnable"] for s in g["seeds"]),
            **g,
        })
    return policies


def list_queues(limit: int = 60) -> list[dict]:
    """Recent queues (newest first), each with its grouped policies."""
    if not _QUEUE_DIR.is_dir():
        return []
    queue_dirs = sorted(
        (d for d in _QUEUE_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:limit]

    out = []
    for d in queue_dirs:
        sj = d / "status.json"
        if not sj.exists():
            continue
        try:
            entries = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        policies = _policies_for_queue(entries)
        if not policies:
            continue
        out.append({
            "queue": d.name,
            "mtime": d.stat().st_mtime,
            "policies": policies,
        })
    return out

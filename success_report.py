"""Headless queue → success-rate-table driver (no JAX in this process).

The Compare tab in the server already turns training queues into success-vs-
threshold curves, but only interactively over HTTP. This is the batch path: give
it one or more queue names (or substrings), it

  1. resolves every runnable seed run across the matching queues (queues.py),
  2. ensures a compare_collect eval cache exists for each (run the subprocess on
     a GPU if the .DONE sentinel is missing; reuse it otherwise),
  3. groups the caches by sensor_bundle (the authoritative field stored in each
     cache, so a policy pools its seeds even when split across queues), and
  4. prints a success-rate table via success_curve.compute_curves — the rate at
     the env's reference tolerance plus the peak rate.

Like the server, this process never imports JAX; the per-run eval runs in a
fresh compare_collect subprocess with CUDA_VISIBLE_DEVICES pinned.

Usage (from inside a dev container, on a free GPU):
    python -m policy_analyzer.success_report pinch_sweep_size_rand \
        --mode det --n-rollouts 50 --gpu 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from policy_analyzer import success_curve
from policy_analyzer.paths import ANALYSIS_DIR, LOGS_DIR, PKG_DIR
from policy_analyzer.queues import list_queues
from policy_analyzer.worker import _free_vram_gb


def _compare_cache_path(run_name: str, mode: str) -> Path:
    return ANALYSIS_DIR / "compare_cache" / run_name / f"{mode}.npz"


def _cache_done(run_name: str, mode: str) -> bool:
    p = _compare_cache_path(run_name, mode)
    return (p.parent / f"{p.stem}.DONE").exists() and p.exists()


def _load_cache(run_name: str, mode: str) -> dict | None:
    """Load a compare_collect npz into {channels, n_rollouts, env_name, sensor_bundle}."""
    import numpy as np

    p = _compare_cache_path(run_name, mode)
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
        }
    except Exception:
        return None


def _pick_gpu(requested: int | None) -> int:
    """Use the requested GPU, else the one with the most free VRAM (0..7)."""
    if requested is not None:
        return requested
    best, best_free = 0, -1.0
    for g in range(8):
        free = _free_vram_gb(g)
        if free > best_free:
            best, best_free = g, free
    print(f"auto-selected GPU {best} ({best_free:.1f} GB free)", flush=True)
    return best


def _run_compare(run_name: str, mode: str, n_rollouts: int, gpu: int) -> bool:
    """Run compare_collect for one run on `gpu`; return True on success."""
    out_path = _compare_cache_path(run_name, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{out_path.stem}.DONE").unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "policy_analyzer.compare_collect",
        "--log-dir", str(LOGS_DIR / run_name),
        "--out", str(out_path),
        "--mode", mode,
        "--n-rollouts", str(n_rollouts),
    ]
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"  [GPU {gpu}] eval {run_name} ({mode}, N={n_rollouts}) …", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(PKG_DIR.parent), env=env).returncode
    if rc == 0 and _cache_done(run_name, mode):
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
        return True
    print(f"    FAILED (rc={rc})", flush=True)
    return False


def _resolve_seed_runs(queue_filters: list[str]) -> list[dict]:
    """All runnable seed runs from queues matching any filter substring."""
    runs: list[dict] = []
    seen: set[str] = set()
    for q in list_queues(limit=200):
        if queue_filters and not any(f in q["queue"] for f in queue_filters):
            continue
        for pol in q["policies"]:
            for s in pol["seeds"]:
                if not s["runnable"] or s["run_name"] in seen:
                    continue
                seen.add(s["run_name"])
                runs.append({
                    "run_name": s["run_name"],
                    "seed": s["seed"],
                    "sensor_bundle": pol["sensor_bundle"],
                    "env_name": pol["env_name"],
                    "queue": q["queue"],
                })
    return runs


def _success_at(curve_x: list[float], series: list[float], ref: float) -> float:
    """Linear-interpolate a success series at threshold `ref`."""
    import numpy as np

    return float(np.interp(ref, curve_x, series))


def main() -> int:
    ap = argparse.ArgumentParser(prog="success_report")
    ap.add_argument("queues", nargs="+",
                    help="queue name(s) or substring(s) to pool")
    ap.add_argument("--mode", choices=("det", "sto"), default="det")
    ap.add_argument("--n-rollouts", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index for evals (default: most-free)")
    ap.add_argument("--reuse", action="store_true", default=True,
                    help="reuse existing caches (default)")
    ap.add_argument("--force", dest="reuse", action="store_false",
                    help="re-run evals even if a cache exists")
    ap.add_argument("--tol", type=float, default=None,
                    help="override the success tolerance (display units)")
    args = ap.parse_args()

    runs = _resolve_seed_runs(args.queues)
    if not runs:
        print(f"No runnable runs match {args.queues}", file=sys.stderr)
        return 1

    env_name = runs[0]["env_name"]
    bundles = sorted({r["sensor_bundle"] or "?" for r in runs})
    print(f"{len(runs)} runnable seed runs across {len(bundles)} bundles "
          f"({env_name}): {', '.join(bundles)}\n")

    gpu = None  # picked lazily, only if an eval is actually needed

    # 1) ensure a cache exists for each run
    caches: dict[str, dict] = {}
    for r in runs:
        rn = r["run_name"]
        if args.reuse and _cache_done(rn, args.mode):
            c = _load_cache(rn, args.mode)
            if c is not None:
                caches[rn] = c
                print(f"  cached  {rn}", flush=True)
                continue
        if gpu is None:
            gpu = _pick_gpu(args.gpu)
        if _run_compare(rn, args.mode, args.n_rollouts, gpu):
            c = _load_cache(rn, args.mode)
            if c is not None:
                caches[rn] = c

    # 2) group caches by sensor_bundle (authoritative field from the cache)
    by_bundle: dict[str, list[dict]] = {}
    for r in runs:
        c = caches.get(r["run_name"])
        if c is None:
            continue
        by_bundle.setdefault(c["sensor_bundle"], []).append(c)

    if not by_bundle:
        print("No caches available — nothing to report.", file=sys.stderr)
        return 1

    # 3) compute success curves
    any_cache = next(iter(caches.values()))
    criterion = success_curve.config_for_env(
        env_name, list(any_cache["channels"].keys())
    )
    if args.tol is not None:
        criterion["ref_value"] = args.tol
    ref = criterion.get("ref_value")

    policies = [
        {"label": b, "sensor_bundle": b,
         "seeds": [{"channels": c["channels"]} for c in seeds_]}
        for b, seeds_ in sorted(by_bundle.items())
    ]
    result = success_curve.compute_curves(policies, criterion)

    # 4) print the table
    chan = result["channel"]
    print(f"\nSuccess criterion: |{chan}| < thresh held {result['hold_steps']} "
          f"steps;  ref tol = {ref} ({criterion.get('xlabel','')})")
    print(f"Mode={args.mode}  N={args.n_rollouts} rollouts/seed\n")

    header = f"{'bundle':<18}{'seeds':>6}{'N':>5}"
    if ref is not None:
        header += f"{'succ@ref %':>22}"
    header += f"{'peak %':>22}"
    print(header)
    print("-" * len(header))
    x = result["x"]
    for pol in result["policies"]:
        line = f"{pol['label']:<18}{pol['n_seeds']:>6}{pol['n_rollouts']:>5}"
        if ref is not None:
            m = _success_at(x, pol["mean"], ref)
            lo = _success_at(x, pol["min"], ref)
            hi = _success_at(x, pol["max"], ref)
            line += f"{f'{m:5.1f} [{lo:4.1f},{hi:5.1f}]':>22}"
        import numpy as np

        pm = float(np.max(pol["mean"]))
        plo = float(np.max(pol["min"]))
        phi = float(np.max(pol["max"]))
        line += f"{f'{pm:5.1f} [{plo:4.1f},{phi:5.1f}]':>22}"
        print(line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

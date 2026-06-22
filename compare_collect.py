"""Lightweight eval runner for the Compare tab — one process, one GPU, one policy.

Launched as a subprocess by the analyzer server (one run+mode per free GPU), so
each gets a clean CUDA context with CUDA_VISIBLE_DEVICES pinned before Python
starts. Unlike collect_one (which renders frames + exports a frontend per seed),
this only runs many rollouts and records per-step scalar channels — the cheap
path that produces the data behind success-vs-threshold curves.

Reuses collect.restore_policy + collect.run_eval_rollouts; no rendering, no
mujoco_playground experimentation scripts.

Output cache (under analysis/compare_cache/<run_name>/<mode>.npz):
    one float32 [N, T] array per channel, plus reserved meta keys, plus a DONE
    sentinel touched once the npz is complete.

Exit codes:
    0   cache written
    75  out-of-memory (EX_TEMPFAIL) — parent should requeue after VRAM frees up
    1   any other failure (stderr carries the traceback)

Usage:
    python -m policy_analyzer.compare_collect \
        --log-dir logs/<run> --out analysis/compare_cache/<run>/det.npz \
        --mode det --n-rollouts 50
"""

from __future__ import annotations

import argparse
import sys
import traceback

EXIT_OOM = 75  # EX_TEMPFAIL


def main() -> int:
    ap = argparse.ArgumentParser(prog="compare_collect")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("det", "sto"), default="det")
    ap.add_argument("--n-rollouts", type=int, default=50)
    args = ap.parse_args()

    from pathlib import Path

    # Heavy imports (JAX etc.) happen here, after CUDA_VISIBLE_DEVICES is set.
    import numpy as np
    from policy_analyzer import collect
    from policy_analyzer.worker import _is_oom

    log_dir = Path(args.log_dir)
    out = Path(args.out)
    deterministic = args.mode == "det"

    try:
        handles = collect.restore_policy(log_dir)
        result = collect.run_eval_rollouts(
            handles, n_rollouts=args.n_rollouts, deterministic=deterministic
        )
        channels = result["channels"]

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            _channels=np.array(sorted(channels.keys())),
            _n_rollouts=np.array(result["n_rollouts"]),
            _episode_length=np.array(result["episode_length"]),
            _mode=np.array(args.mode),
            _env_name=np.array(handles["env_name"]),
            _sensor_bundle=np.array(str(handles["env_cfg"].sensor_bundle)),
            **channels,
        )
        (out.parent / f"{out.stem}.DONE").touch()
        print(f"Wrote {out}  ({len(channels)} channels, N={result['n_rollouts']})")
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level boundary
        traceback.print_exc()
        return EXIT_OOM if _is_oom(exc) else 1


if __name__ == "__main__":
    sys.exit(main())

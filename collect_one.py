"""Batched rollout runner — one process, one GPU, all seeds of one group.

Launched as a subprocess by the analyzer server (one group per free GPU) so each
group gets a clean CUDA context. The parent sets CUDA_VISIBLE_DEVICES in this
process's environment *before* Python starts, which is the only reliable way to
pin JAX to a specific GPU.

All seeds of the group run in a single vmapped pass (shared JIT compile), then
each seed's artifacts are written to its own directory `<out-base>/<tag>_<seed>`
where tag is "det" or "sto". A `DONE` sentinel is touched once a seed's artifacts
are complete, so the server can mark seeds done progressively.

Exit codes:
    0   all seeds completed, artifacts written
    75  out-of-memory (EX_TEMPFAIL) — parent should requeue after VRAM frees up
    1   any other failure (stderr carries the traceback)

Usage:
    python -m policy_analyzer.collect_one \
        --log-dir logs/<run> --out-base analysis/sessions/<sid> \
        --seeds 1,2,3 [--deterministic] [--checkpoint-step 1000000]
"""

from __future__ import annotations

import argparse
import sys
import traceback

EXIT_OOM = 75  # EX_TEMPFAIL


def main() -> int:
    ap = argparse.ArgumentParser(prog="collect_one")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seed ints")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--checkpoint-step", default=None)
    args = ap.parse_args()

    from pathlib import Path

    # Heavy imports (JAX etc.) happen here, after CUDA_VISIBLE_DEVICES is set.
    from policy_analyzer import collect, frontend, visualize
    from policy_analyzer.worker import _is_oom

    log_dir = Path(args.log_dir)
    out_base = Path(args.out_base)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tag = "det" if args.deterministic else "sto"

    try:
        ckpt = args.checkpoint_step
        if ckpt in (None, "", "latest"):
            ckpt = None

        handles = collect.restore_policy(log_dir, checkpoint_step=ckpt)
        rollouts = collect.run_batched_rollout(
            handles, seeds, deterministic=args.deterministic
        )

        # Load the render env once and reuse it across all seeds.
        env_name, cfg, env = collect.load_env_from_checkpoint(str(handles["restore_path"]))

        for seed, rollout in zip(seeds, rollouts):
            out_dir = out_base / f"{tag}_{seed}"
            collect.write_artifacts(out_dir, rollout)
            visualize.visualize_input_distributions(out_dir, schema=rollout["schema"])
            visualize.visualize_dof_evolution(out_dir, schema=rollout["schema"])
            frontend.export_frontend(
                out_dir, schema=rollout["schema"], update_index=False,
                env=env, env_name=env_name, cfg=cfg,
            )
            (out_dir / "DONE").touch()
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level boundary
        traceback.print_exc()
        return EXIT_OOM if _is_oom(exc) else 1


if __name__ == "__main__":
    sys.exit(main())

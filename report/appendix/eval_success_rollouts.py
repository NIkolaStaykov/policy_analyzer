"""Evaluate final-checkpoint policies and record per-rollout successful-steps counts.

For each run (one training seed of one sensor-bundle variant) we run N rollouts
from the final checkpoint and store, per rollout, the number of steps the env's
success condition held (sum of the per-step `reward/success_per_step` channel;
cross-checked against the cumulative `success_count`). Raw per-rollout rows — no
per-policy aggregation — so downstream success-vs-threshold plots (Compare-tab
style) can be built from this.

Usage (one GPU, a slice of the run list):
    CUDA_VISIBLE_DEVICES=0 python eval_success_rollouts.py \
        --gpu 0 --shard 0 --n-shards 2 --n-rollouts 100 --out out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

LOGS = Path("/local/home/nstaykov/workspace/mujoco_playground/logs")
QUEUE_DIR = LOGS / "_queue"

QUEUES = {
    "pinch_sweep_size_rand-20260622-195014": "pinch",
    "pinch_sweep_size_rand_sinusoid-20260624-142357": "pinch_sinusoid",
    "pinch_sweep_size_rand_sinusoid-20260623-133237": "pinch_sinusoid",
    "downwards_sensor_sweep_120_cubesizeDR-20260626-215239": "downwards_rotate",
}

FIELDS = [
    "task", "queue", "env_name", "exp_name", "sensor_bundle", "suffix", "seed",
    "eval_mode", "rollout_idx", "successful_steps", "episode_length",
    "success_fraction", "result",
]


def run_list():
    runs = []
    for q, task in QUEUES.items():
        status = json.load(open(QUEUE_DIR / q / "status.json"))
        for r in status:
            runs.append(dict(
                task=task, queue=q, exp_name=r["exp_name"],
                env_name=r["env_name"], suffix=r.get("suffix", ""),
                seed=r.get("flags", {}).get("seed"), result=r.get("result"),
            ))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--mode", choices=("det", "sto"), default="det")
    ap.add_argument("--exps", default="",
                    help="comma-separated exp_names to process (overrides sharding)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import numpy as np
    import mujoco_playground
    from policy_analyzer import collect
    print(f"[gpu{args.gpu}] mujoco_playground from: {mujoco_playground.__file__}", flush=True)

    all_runs = run_list()
    if args.exps:
        want = set(args.exps.split(","))
        runs = [r for r in all_runs if r["exp_name"] in want]
    else:
        runs = [r for i, r in enumerate(all_runs) if i % args.n_shards == args.shard]
    deterministic = args.mode == "det"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    f.flush()

    print(f"[gpu{args.gpu}] shard {args.shard}/{args.n_shards}: {len(runs)} runs, "
          f"N={args.n_rollouts}, mode={args.mode}", flush=True)

    for j, r in enumerate(runs):
        exp = r["exp_name"]
        log_dir = LOGS / exp
        try:
            handles = collect.restore_policy(log_dir, config_overrides={})
            sensor_bundle = str(handles["env_cfg"].sensor_bundle)
            res = collect.run_eval_rollouts(
                handles, n_rollouts=args.n_rollouts, deterministic=deterministic
            )
            ch = res["channels"]
            sps = ch["reward/success_per_step"]            # [N, T] 0/1
            succ_steps = sps.sum(axis=1).astype(int)        # [N]
            T = int(res["episode_length"])
            # cross-check against cumulative success_count final value
            if "success_count" in ch:
                cc = ch["success_count"][:, -1].astype(int)
                nmis = int((cc != succ_steps).sum())
                if nmis:
                    print(f"  [warn] {exp}: {nmis}/{len(cc)} rollouts "
                          f"success_count != sum(success_per_step)", flush=True)
            for i in range(len(succ_steps)):
                w.writerow({
                    "task": r["task"], "queue": r["queue"], "env_name": r["env_name"],
                    "exp_name": exp, "sensor_bundle": sensor_bundle,
                    "suffix": r["suffix"], "seed": r["seed"],
                    "eval_mode": args.mode, "rollout_idx": i,
                    "successful_steps": int(succ_steps[i]), "episode_length": T,
                    "success_fraction": round(float(succ_steps[i]) / T, 6),
                    "result": r["result"],
                })
            f.flush()
            print(f"  [{j+1}/{len(runs)}] {exp}: mean_succ_steps="
                  f"{succ_steps.mean():.1f}/{T}  (succ_frac={succ_steps.mean()/T:.3f})",
                  flush=True)
        except Exception:
            print(f"  [ERROR] {exp}", flush=True)
            traceback.print_exc()

    f.close()
    print(f"[gpu{args.gpu}] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

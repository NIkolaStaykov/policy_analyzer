"""Per-timestep detailed trajectories for a few exemplary rollouts per policy/seed.

Complements the aggregate success datasets: instead of one number per rollout,
this dumps the full per-step trajectory for a handful of exemplary rollouts, so
single-rollout behaviour can be plotted.

- pinch / pinch_sinusoid: per step -> force_target, finger_force_sum (= f_thumb +
  f_index), effective_force, f_thumb, f_index, success flag, total reward.
  Exemplary rollouts = fixed indices (shared eval RNG => same initial conditions
  across every policy and seed, so trajectories are directly comparable).
- downwards_rotate: per step -> orientation error (rad + deg) and every raw reward
  component + total reward. Exemplary rollouts = the most challenging ones, goal
  rotation >= 90 deg (top-N by goal angle).

Rollout indices match the aggregate datasets (same shared RNG stream).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_detailed_trajectories.py \
        --gpu 0 --tag head --out-dir <dir> [--exps a,b,c]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    "downwards_sensor_sweep_120-20260625-124815": "downwards_rotate",
}

N_ROLLOUTS = 100
PINCH_IDX = [0, 1, 2]          # fixed exemplary rollouts for the pinch tasks
DOWN_MIN_ANGLE = 90.0          # deg; challenging-rollout threshold
DOWN_N = 3                     # number of challenging rollouts to keep

PINCH_COLS = [
    "task", "queue", "env_name", "exp_name", "sensor_bundle", "seed",
    "rollout_idx", "step", "time_s",
    "force_target", "finger_force_sum", "effective_force", "f_thumb", "f_index",
    "success_per_step", "reward", "env_code_commit",
]
DOWN_COLS = [
    "task", "queue", "env_name", "exp_name", "sensor_bundle", "seed",
    "rollout_idx", "step", "time_s", "goal_angle_deg",
    "ori_error_rad", "ori_error_deg", "reward",
    "rew_action_rate", "rew_cube_on_floor", "rew_cube_ori", "rew_fingertip_pos",
    "rew_joint_vel", "rew_wrist_vel", "rew_success", "env_code_commit",
]


def run_list():
    runs = []
    for q, task in QUEUES.items():
        for r in json.load(open(QUEUE_DIR / q / "status.json")):
            runs.append(dict(task=task, queue=q, exp_name=r["exp_name"],
                             env_name=r["env_name"],
                             seed=r.get("flags", {}).get("seed"),
                             result=r.get("result")))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True, help="output filename tag, e.g. head / oldfm")
    ap.add_argument("--exps", default="", help="comma-separated exp_names (default: all)")
    ap.add_argument("--commit", default="3682863", help="env_code_commit label for these rows")
    ap.add_argument("--min-target-angle-deg", type=float, default=None,
                    help="override min goal angle (deg); downwards only")
    ap.add_argument("--max-target-angle-deg", type=float, default=None,
                    help="override max goal angle (deg); downwards only")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import numpy as np
    import mujoco_playground
    from policy_analyzer import collect
    print(f"[gpu{args.gpu}] mujoco_playground from: {mujoco_playground.__file__}", flush=True)

    overrides = {}
    if args.min_target_angle_deg is not None:
        overrides["min_target_angle"] = math.radians(args.min_target_angle_deg)
    if args.max_target_angle_deg is not None:
        overrides["max_target_angle"] = math.radians(args.max_target_angle_deg)
    if overrides:
        print(f"[gpu{args.gpu}] target-angle band override: "
              f"[{args.min_target_angle_deg}, {args.max_target_angle_deg}] deg", flush=True)

    runs = run_list()
    if args.exps:
        want = set(args.exps.split(","))
        runs = [r for r in runs if r["exp_name"] in want]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers, files = {}, {}

    def writer_for(task):
        if task not in writers:
            cols = DOWN_COLS if task == "downwards_rotate" else PINCH_COLS
            p = out_dir / f"detailed_{task}__{args.tag}.csv"
            f = open(p, "w", newline="")
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            writers[task] = w
            files[task] = f
        return writers[task]

    down_idx = None  # computed once from the first downwards run (shared RNG)

    for j, r in enumerate(runs):
        exp = r["exp_name"]
        task = r["task"]
        try:
            handles = collect.restore_policy(LOGS / exp, config_overrides=overrides)
            sensor_bundle = str(handles["env_cfg"].sensor_bundle)
            res = collect.run_eval_rollouts(handles, n_rollouts=N_ROLLOUTS, deterministic=True)
            ch = res["channels"]
            T = int(res["episode_length"])
            dt = float(res["dt"])
            w = writer_for(task)

            if task == "downwards_rotate":
                ga = ch["curriculum/goal_angle_per_step"][:, 0]  # deg, per rollout
                if down_idx is None:
                    cand = [i for i in range(len(ga)) if ga[i] >= DOWN_MIN_ANGLE]
                    cand.sort(key=lambda i: -ga[i])
                    down_idx = cand[:DOWN_N]
                    print(f"  downwards challenge indices (deg): "
                          f"{[(i, round(float(ga[i]),1)) for i in down_idx]}", flush=True)
                idxs = down_idx
                for ri in idxs:
                    for t in range(T):
                        w.writerow({
                            "task": task, "queue": r["queue"], "env_name": r["env_name"],
                            "exp_name": exp, "sensor_bundle": sensor_bundle, "seed": r["seed"],
                            "rollout_idx": ri, "step": t, "time_s": round(t * dt, 4),
                            "goal_angle_deg": round(float(ga[ri]), 3),
                            "ori_error_rad": round(float(ch["info.ori_error"][ri, t]), 6),
                            "ori_error_deg": round(float(ch["info.ori_error"][ri, t]) * 180.0 / math.pi, 4),
                            "reward": round(float(ch["reward"][ri, t]), 6),
                            "rew_action_rate": round(float(ch["reward/action_rate_per_step"][ri, t]), 6),
                            "rew_cube_on_floor": round(float(ch["reward/cube_on_floor_per_step"][ri, t]), 6),
                            "rew_cube_ori": round(float(ch["reward/cube_ori_per_step"][ri, t]), 6),
                            "rew_fingertip_pos": round(float(ch["reward/fingertip_pos_per_step"][ri, t]), 6),
                            "rew_joint_vel": round(float(ch["reward/joint_vel_per_step"][ri, t]), 6),
                            "rew_wrist_vel": round(float(ch["reward/wrist_vel_per_step"][ri, t]), 6),
                            "rew_success": round(float(ch["reward/success_per_step"][ri, t]), 6),
                            "env_code_commit": args.commit,
                        })
            else:
                fsum = ch["f_thumb"] + ch["f_index"]
                for ri in PINCH_IDX:
                    for t in range(T):
                        w.writerow({
                            "task": task, "queue": r["queue"], "env_name": r["env_name"],
                            "exp_name": exp, "sensor_bundle": sensor_bundle, "seed": r["seed"],
                            "rollout_idx": ri, "step": t, "time_s": round(t * dt, 4),
                            "force_target": round(float(ch["force_target"][ri, t]), 6),
                            "finger_force_sum": round(float(fsum[ri, t]), 6),
                            "effective_force": round(float(ch["effective_force"][ri, t]), 6),
                            "f_thumb": round(float(ch["f_thumb"][ri, t]), 6),
                            "f_index": round(float(ch["f_index"][ri, t]), 6),
                            "success_per_step": int(ch["reward/success_per_step"][ri, t]),
                            "reward": round(float(ch["reward"][ri, t]), 6),
                            "env_code_commit": args.commit,
                        })
            files[task].flush()
            print(f"  [{j+1}/{len(runs)}] {exp} ({task}, bundle={sensor_bundle})", flush=True)
        except Exception:
            print(f"  [ERROR] {exp}", flush=True)
            traceback.print_exc()

    for f in files.values():
        f.close()
    print(f"[gpu{args.gpu}] done -> {out_dir} (tag={args.tag})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

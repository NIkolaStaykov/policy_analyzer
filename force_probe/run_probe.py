"""Driver: for each selected policy type, roll out, probe every representation.

Reads selected_seeds.json + the policy checkpoints; writes:
  data/rollout_<exp>.npz     — captured (obs_state, forces) rollout data
  probe_results.csv / .json  — one row per (policy, representation)

Usage (one free GPU):
    CUDA_VISIBLE_DEVICES=0 python run_probe.py --gpu 0 --n-rollouts 64
    # old-code force.magnitude run, under the 0ac79f5 worktree PYTHONPATH:
    CUDA_VISIBLE_DEVICES=0 python run_probe.py --gpu 0 --only-old
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).parent
LOGS = Path("/local/home/nstaykov/workspace/mujoco_playground/logs")

FIELDS = [
    "task", "sensor_bundle", "exp_name", "seed", "mean_eval_reward", "old_code",
    "k_forces", "n_samples", "representation", "width",
    "r2_mean", "r2_per_channel", "rmse_per_channel_N", "val_loc_err",
]
REPR_ORDER = ["input", "hidden_0", "hidden_1", "hidden_2", "hidden_3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--exps", default="", help="comma-separated exp_names (default: all selected)")
    ap.add_argument("--only-old", action="store_true", help="only old-code runs (need worktree PYTHONPATH)")
    ap.add_argument("--skip-old", action="store_true", help="skip old-code runs")
    ap.add_argument("--out", default=str(HERE / "probe_results.csv"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import numpy as np
    import mujoco_playground  # noqa: F401
    from policy_analyzer import collect
    from policy_analyzer import eval_runs
    from policy_analyzer.force_probe import probe_lib
    print(f"[gpu{args.gpu}] mujoco_playground: {mujoco_playground.__file__}", flush=True)

    # restore_policy spins up ppo.train (num_timesteps=0) only to load params; the
    # config's 8192 training envs then allocate multiple GB and OOM on a shared
    # GPU. Params come from the checkpoint and are independent of env count, so
    # shrink the training scaffold to the bare minimum for the restore.
    _orig_get = eval_runs._get_ppo_params

    def _lean_get(env_name, impl):
        p = _orig_get(env_name, impl)
        p.num_envs = 64
        p.num_eval_envs = 16
        p.batch_size = 16
        p.num_minibatches = 4
        p.num_evals = 0
        return p

    eval_runs._get_ppo_params = _lean_get

    selected = json.load(open(HERE / "selected_seeds.json"))
    if args.exps:
        want = set(args.exps.split(","))
        selected = [s for s in selected if s["exp_name"] in want]
    if args.only_old:
        selected = [s for s in selected if s["old_code"]]
    if args.skip_old:
        selected = [s for s in selected if not s["old_code"]]

    (HERE / "data").mkdir(exist_ok=True)
    out = Path(args.out)
    # append mode: preserve rows from a prior (e.g. HEAD) pass
    exists = out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
        f.flush()

    json_rows = []
    jpath = HERE / "probe_results.json"
    if jpath.exists():
        json_rows = json.load(open(jpath))

    print(f"[gpu{args.gpu}] {len(selected)} policy types, N={args.n_rollouts}", flush=True)
    for j, s in enumerate(selected):
        exp = s["exp_name"]
        task = s["task"]
        try:
            handles = collect.restore_policy(LOGS / exp)
            obs, forces = probe_lib.capture_rollouts(handles, task, n_rollouts=args.n_rollouts)
            np.savez_compressed(HERE / "data" / f"rollout_{exp}.npz",
                                obs=obs, forces=forces, task=task,
                                sensor_bundle=s["sensor_bundle"])
            acts, val_err = probe_lib.extract_activations(handles, obs)
            assert val_err < 1e-3, f"activation validation failed: max|loc_manual-loc_net|={val_err}"
            k = forces.shape[1]
            print(f"  [{j+1}/{len(selected)}] {task}/{s['sensor_bundle']} {exp}"
                  f"  n={len(obs)} k={k} val_err={val_err:.2e}", flush=True)
            for rep in [r for r in REPR_ORDER if r in acts]:
                res = probe_lib.train_probe(acts[rep], forces)
                row = {
                    "task": task, "sensor_bundle": s["sensor_bundle"], "exp_name": exp,
                    "seed": s["seed"], "mean_eval_reward": s["mean_eval_reward"],
                    "old_code": s["old_code"], "k_forces": k, "n_samples": len(obs),
                    "representation": rep, "width": acts[rep].shape[1],
                    "r2_mean": res["r2_mean"],
                    "r2_per_channel": ";".join(str(v) for v in res["r2_per_channel"]),
                    "rmse_per_channel_N": ";".join(str(v) for v in res["rmse_per_channel_N"]),
                    "val_loc_err": f"{val_err:.2e}",
                }
                w.writerow(row)
                json_rows.append({**row, "r2_per_channel": res["r2_per_channel"],
                                  "rmse_per_channel_N": res["rmse_per_channel_N"]})
                print(f"      {rep:9} (w={acts[rep].shape[1]:4}) "
                      f"R2_mean={res['r2_mean']:.3f}  per-ch={res['r2_per_channel']}", flush=True)
            f.flush()
            json.dump(json_rows, open(jpath, "w"), indent=2)
        except Exception:
            print(f"  [ERROR] {exp}", flush=True)
            traceback.print_exc()

    f.close()
    print(f"[gpu{args.gpu}] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

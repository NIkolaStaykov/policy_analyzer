"""Select the best training seed per (task, sensor_bundle) by mean eval reward.

Isolated from the main codebase: reads only the training logs' local wandb
`output.log` files (which contain per-eval `` <step>: reward=<x> `` lines whose
final value matches wandb-summary's eval/episode_reward) and the report appendix
network CSV (exp_name -> sensor_bundle). Writes selected_seeds.json.

"mean eval reward during training" = mean over the eval points (the `reward=`
lines), NOT the training-batch `mean episode reward=` lines.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

WS = Path("/local/home/nstaykov/workspace")
LOGS = WS / "mujoco_playground" / "logs"
WANDB = WS / "mujoco_playground" / "wandb"
QUEUE_DIR = LOGS / "_queue"
NETWORK_CSV = WS / "policy_analyzer" / "report" / "appendix" / "run_parameters_network.csv"
OUT = Path(__file__).parent / "selected_seeds.json"

QUEUES = {
    "pinch_sweep_size_rand-20260622-195014": "pinch",
    "pinch_sweep_size_rand-20260622-222627": "pinch",
    "pinch_sweep_size_rand_sinusoid-20260624-142357": "pinch_sinusoid",
    "pinch_sweep_size_rand_sinusoid-20260623-133237": "pinch_sinusoid",
    "downwards_sensor_sweep_120-20260625-124815": "downwards_rotate",
}

# force.magnitude runs predate commit 0e297c3 (Jun 24) which grew the
# fingertip_forces obs 2->4, so their `state` obs (and checkpoint) is smaller
# than current HEAD -> they must be restored under the old env code (0ac79f5).
OLD_FORCE_QUEUES = {
    "pinch_sweep_size_rand-20260622-195014",
    "pinch_sweep_size_rand_sinusoid-20260623-133237",
}

EVAL_LINE = re.compile(r"^(\d+):\s*reward=([-\d.]+)\s*$")
CKPT_LINE = re.compile(r"logs/([^/]+)/checkpoints")


def bundle_map() -> dict[str, str]:
    m = {}
    with open(NETWORK_CSV) as f:
        for row in csv.DictReader(f):
            m[row["exp_name"]] = row["sensor_bundle"]
    return m


def wandb_index() -> dict[str, list[Path]]:
    """exp_name -> [wandb run dirs], via the 'Checkpoint path: .../logs/<exp>' line."""
    idx: dict[str, list[Path]] = {}
    for log in WANDB.glob("run-*/files/output.log"):
        try:
            head = "".join(log.open().readlines()[:6])
        except Exception:
            continue
        m = CKPT_LINE.search(head)
        if m:
            idx.setdefault(m.group(1), []).append(log.parent.parent)
    return idx


def mean_eval_reward(run_dirs: list[Path]) -> tuple[float | None, int]:
    """Mean of all `<step>: reward=<x>` eval points across the run's wandb dirs."""
    run_dirs = sorted(run_dirs, key=lambda d: d.name)  # chronological
    vals: list[float] = []
    for d in run_dirs:
        for ln in (d / "files" / "output.log").open():
            m = EVAL_LINE.match(ln.strip())
            if m:
                vals.append(float(m.group(2)))
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def run_list():
    runs = []
    for q, task in QUEUES.items():
        for r in json.load(open(QUEUE_DIR / q / "status.json")):
            runs.append(dict(
                task=task, queue=q, exp_name=r["exp_name"],
                env_name=r["env_name"], result=r.get("result"),
                seed=r.get("flags", {}).get("seed"),
            ))
    return runs


def main():
    bundles = bundle_map()
    widx = wandb_index()
    runs = run_list()

    per_run = []
    for r in runs:
        exp = r["exp_name"]
        r["sensor_bundle"] = bundles.get(exp, "?")
        r["old_code"] = r["queue"] in OLD_FORCE_QUEUES and r["sensor_bundle"] == "force.magnitude"
        mer, n = mean_eval_reward(widx.get(exp, []))
        r["mean_eval_reward"] = mer
        r["n_evals"] = n
        per_run.append(r)

    # group by (task, sensor_bundle); pick max mean_eval_reward among result==ok
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in per_run:
        groups.setdefault((r["task"], r["sensor_bundle"]), []).append(r)

    selected = []
    print(f"{'task':16} {'bundle':16} {'seed':4} {'mean_eval':>9} {'n':>3}  exp")
    for (task, bundle), rs in sorted(groups.items()):
        ok = [r for r in rs if r["result"] == "ok" and r["mean_eval_reward"] is not None]
        if not ok:
            print(f"[skip] {task}/{bundle}: no ok runs with eval reward "
                  f"({[r['result'] for r in rs]})")
            continue
        best = max(ok, key=lambda r: r["mean_eval_reward"])
        selected.append({
            "task": task, "sensor_bundle": bundle, "exp_name": best["exp_name"],
            "queue": best["queue"], "seed": best["seed"], "env_name": best["env_name"],
            "mean_eval_reward": round(best["mean_eval_reward"], 4),
            "n_evals": best["n_evals"], "old_code": best["old_code"],
            "n_seeds_considered": len(ok),
        })
        print(f"{task:16} {bundle:16} {str(best['seed']):4} "
              f"{best['mean_eval_reward']:9.3f} {best['n_evals']:3d}  {best['exp_name']}"
              f"{'  [OLD-CODE]' if best['old_code'] else ''}")

    OUT.write_text(json.dumps(selected, indent=2))
    print(f"\n{len(selected)} policy types selected -> {OUT}")


if __name__ == "__main__":
    main()

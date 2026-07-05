#!/usr/bin/env bash
# Run the force probe over all selected policy types, ONE process per policy so
# GPU memory is released between policies (the shared GPU is tight).
#   ./run_all.sh <gpu>            # all HEAD-restorable policies
#   ./run_all.sh <gpu> old        # only the old-code (0ac79f5) force.magnitude run
set -u
GPU="${1:-0}"
MODE="${2:-head}"

WS=/local/home/nstaykov/workspace
HERE="$WS/policy_analyzer/force_probe"
PY="$WS/mujoco_playground/.venv/bin/python"
export PYTHONPATH="$WS"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28

if [ "$MODE" = "old" ]; then
  FILTER='[s["exp_name"] for s in J if s["old_code"]]'
  RUNDIR="$WS/mujoco_playground/.claude/worktrees/force-probe-oldcode"
else
  FILTER='[s["exp_name"] for s in J if not s["old_code"]]'
  RUNDIR="$WS/mujoco_playground"
fi

EXPS=$("$PY" -c "import json;J=json.load(open('$HERE/selected_seeds.json'));print(' '.join($FILTER))")
echo "MODE=$MODE  GPU=$GPU  RUNDIR=$RUNDIR"
echo "policies: $EXPS"

cd "$RUNDIR" || exit 1
for exp in $EXPS; do
  echo "=================== $exp ==================="
  "$PY" -m policy_analyzer.force_probe.run_probe --gpu "$GPU" --n-rollouts 64 --exps "$exp"
done
echo "ALL DONE ($MODE)"

# Appendix / report data

Raw data extracts for the four report queues, all keyed on `exp_name` (the
training run = one sensor-bundle variant at one seed). Everything here is **raw,
per-seed / per-rollout data** — no per-policy aggregation — so plots can be built
downstream.

Runs covered: 51 across 4 queues.

| queue | task | env | runs |
|-------|------|-----|-----:|
| `pinch_sweep_size_rand-20260622-195014` | pinch | TesolloCubePinch | 12 |
| `pinch_sweep_size_rand_sinusoid-20260623-133237` | pinch_sinusoid | TesolloCubePinch | 15 |
| `pinch_sweep_size_rand_sinusoid-20260624-142357` | pinch_sinusoid | TesolloCubePinch | 20 |
| `downwards_sensor_sweep_120_cubesizeDR-20260626-215239` | downwards_rotate | TesolloDownwardsRotateZ | 4 |

## 1. Run parameters (appendix table)

Per-run network / domain-randomization / training-procedure parameters.

- `run_parameters_network.csv`
- `run_parameters_domain_randomization.csv`
- `run_parameters_training.csv`
- `run_parameters_data_dictionary.csv` — column descriptions, units, categories.
- `extract_run_parameters.py` — regenerates the three CSVs from the training logs.

Within any single queue every parameter is constant; only `sensor_bundle` and
`seed` vary. See that dictionary for details.

## 2. Success rollouts — global view (policy comparison plot)

Split by task into 3 datasets — **one row per evaluation rollout** (51 runs × 100
rollouts = 5100 rows total):

- `success_rollouts_pinch.csv` — 1200 rows (12 runs)
- `success_rollouts_pinch_sinusoid.csv` — 3500 rows (35 runs, both sinusoid queues)
- `success_rollouts_downwards_rotate.csv` — 400 rows (4 runs)

Column `successful_steps` is the number of steps the env's success condition held
in that rollout (Compare-tab-style metric, unified across all three tasks). Build a
success-vs-threshold curve by sweeping a threshold on `successful_steps` (or
`success_fraction`) and plotting, per policy, the % of rollouts above it — pooling
the 100 eval rollouts, keeping training seeds as separate lines/samples.

- `success_rollouts_data_dictionary.csv` — column descriptions.
- `eval_success_rollouts.py` — regenerates the data (restores each run's final
  checkpoint, runs 100 deterministic rollouts, records per-rollout successful_steps).

### How it was produced
- Final checkpoint of each run; 100 rollouts; `deterministic` policy (distribution
  mean). Eval RNG stream shared across runs for comparability. No benchmark
  overrides (each checkpoint's own env config).
- `successful_steps` = sum of the per-step `reward/success_per_step` channel,
  cross-checked against the cumulative `success_count` (they matched for every run).

### Env-code provenance (important)
Evaluation must use each policy's training-time observation spec. Commit `0e297c3`
(Jun 24) expanded the force.magnitude observation (`fingertip_forces` 2→4; policy
`state` 19→21). The 6 force.magnitude runs from queues `...195014` and `...133237`
predate it and were evaluated against their training-time code (`0ac79f5`) via an
isolated git worktree; all other runs use current HEAD (`3682863`). The
`env_code_commit` column records this per row. Only the policy network (the
`state` obs) is used at inference, so the parallel `privileged_state` change does
not affect these rollouts; the 45 HEAD-evaluated runs loaded without dimension
error, which confirms their `state` obs matches their checkpoints.

The one `failed` training run (force.magnitude, queue `...142357`, seed 7) is kept
with `result=failed` so it can be filtered.

## 3. Detailed rollouts — single-trajectory view (per-timestep)

Full per-step trajectories for a few exemplary rollouts per policy/seed
(num_seeds × num_exemplary_rollouts trajectories per policy, 3 exemplary each):

- `detailed_rollouts_pinch.csv` — 2880 rows (12 runs × 3 rollouts × 80 steps)
- `detailed_rollouts_pinch_sinusoid.csv` — 8400 rows (35 runs × 3 × 80)
- `detailed_rollouts_downwards_rotate.csv` — 1440 rows (4 runs × 3 × 120)

Contents per step:
- pinch / pinch_sinusoid: `force_target`, `finger_force_sum` (= f_thumb + f_index),
  plus `effective_force`, `f_thumb`, `f_index`, `success_per_step`, `reward`.
- downwards_rotate: `ori_error_rad` / `ori_error_deg`, every raw reward component
  (`rew_*`), total `reward`, and `goal_angle_deg`.

Exemplary-rollout selection:
- pinch tasks: fixed indices 0,1,2. Because the eval RNG stream is shared, a given
  index is the SAME initial condition (cube size, force target) across every policy
  and seed — so these trajectories are directly comparable across policies.
- downwards: the 3 most challenging rollouts, goal rotation >= 90 deg (indices
  82/13/74 → 115.0/114.7/114.4 deg; again shared across the 4 policies).

Files: `detailed_rollouts_data_dictionary.csv`, `eval_detailed_trajectories.py`
(regenerator). Same env-code provenance handling as §2 (`env_code_commit` column;
the 6 old force.magnitude runs evaluated under `0ac79f5`).

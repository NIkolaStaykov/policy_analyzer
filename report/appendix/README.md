# Appendix / report data

Raw data extracts for the report queues, all keyed on `exp_name` (the training run =
one sensor-bundle variant at one seed). Everything here is **raw, per-seed /
per-rollout data** — no per-policy aggregation — so plots can be built downstream.

Runs covered: 62 across 5 queues.

| queue | task | env | runs |
|-------|------|-----|-----:|
| `pinch_sweep_size_rand-20260622-195014` | pinch | TesolloCubePinch | 12 |
| `pinch_sweep_size_rand-20260622-222627` | pinch | TesolloCubePinch | 3 |
| `pinch_sweep_size_rand_sinusoid-20260623-133237` | pinch_sinusoid | TesolloCubePinch | 15 |
| `pinch_sweep_size_rand_sinusoid-20260624-142357` | pinch_sinusoid | TesolloCubePinch | 20 |
| `downwards_sensor_sweep_120-20260625-124815` | downwards_rotate | TesolloDownwardsRotateZ | 12 |

The static `pinch` task pools two queues from the same Jun-22 session: the 4-bundle
sweep (none, baseline, proprio.target, force.magnitude × 3 seeds) plus a
proprio.delta-only continuation (3 seeds), giving 5 bundles for the static
set-and-hold force target. The
downwards queue is the **no-DR control sweep** (4 bundles × 3 seeds, no cube-size
domain randomization).

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

**One row per evaluation rollout** (100 rollouts per run):

- `success_rollouts_pinch.csv` — 1500 rows (15 runs; 5 bundles incl. proprio.delta)
- `success_rollouts_pinch_sinusoid.csv` — 3500 rows (35 runs, both sinusoid queues)
- `success_rollouts_downwards_rotate.csv` — 1200 rows (12 runs), goal rotation
  uniform over the **full 0–120° training range**.

The downwards success eval samples the full 0–120° training range (env
`min/max_target_angle` set to 0 and 120). Each downwards row carries `goal_angle_deg`
(the rollout's actual target) plus `target_min_deg`/`target_max_deg`, so success can
be binned by difficulty. The pinch files have those columns empty. (The 120–150°
out-of-distribution study is a **detailed** dataset, not a success dataset — see §3.)

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
  mean). Eval RNG stream shared across runs for comparability. Each checkpoint's own
  env config, downwards sampling the full 0–120° goal range (`min_target_angle` /
  `max_target_angle` set to 0 and 120).
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

Three sinusoid-queue runs are marked `failed` and kept with `result=failed` so they
can be filtered: force.magnitude s7 and proprio.delta s4 (queue `...142357`), and
baseline s1 (queue `...133237`).

## 3. Detailed rollouts — single-trajectory view (per-timestep)

Full per-step trajectories for a few exemplary rollouts per policy/seed
(num_seeds × num_exemplary_rollouts trajectories per policy, 3 exemplary each):

- `detailed_rollouts_pinch.csv` — 3600 rows (15 runs × 3 rollouts × 80 steps)
- `detailed_rollouts_pinch_sinusoid.csv` — 8400 rows (35 runs × 3 × 80)
- `detailed_rollouts_downwards_rotate.csv` — 4320 rows (12 runs × 3 × 120),
  in-distribution goal rotations (~114–115°).
- `detailed_rollouts_downwards_rotate_ood_120-150.csv` — 4320 rows (12 runs × 3 ×
  120), **out-of-distribution** goal rotations (~148.7°, beyond the 0–120° training
  range). Same schema; policies fail to reach the target — final `ori_error_deg`
  stays ~67–104° across bundles (force.magnitude gets closest).

Contents per step:
- pinch / pinch_sinusoid: `force_target`, `finger_force_sum` (= f_thumb + f_index),
  plus `effective_force`, `f_thumb`, `f_index`, `success_per_step`, `reward`.
- downwards_rotate (both files): `ori_error_rad` / `ori_error_deg`, every raw reward
  component (`rew_*`), total `reward`, and `goal_angle_deg`.

Exemplary-rollout selection:
- pinch tasks: fixed indices 0,1,2. Because the eval RNG stream is shared, a given
  index is the SAME initial condition (cube size, force target) across every policy
  and seed — so these trajectories are directly comparable across policies.
- downwards: the 3 most challenging rollouts (largest goal angle) within the eval
  band — indices 82/13/74, at ~114.4–115.0° for the in-distribution file and
  ~148.6–148.7° for the OOD (120–150°) file; shared across all policies and seeds.

Files: `detailed_rollouts_data_dictionary.csv`, `eval_detailed_trajectories.py`
(regenerator). Same env-code provenance handling as §2 (`env_code_commit` column;
the 6 old force.magnitude runs evaluated under `0ac79f5`).

## 4. Critic privileged state

- `critic_privileged_state.csv` — per-task breakdown of the `privileged_state`
  observation the value network (critic) sees, one row per component.

The policies are asymmetric actor-critic: the policy reads the noisy `state`
obs (`policy_obs_key=state`), while the critic reads a ground-truth,
noise-free `privileged_state` (`value_obs_key=privileged_state`). This file
lists, per task, each component of that vector in concatenation order, its
dimensionality, the source expression in the env code, and a description.
Columns: `task`, `env_name`, `order`, `component`, `dims`, `source_expr`,
`description`.

Totals: **pinch / pinch_sinusoid = 43 dims** (identical structure — both are
`TesolloCubePinch`; only the `force_target` component differs, static vs.
sinusoidal), **downwards_rotate = 114 dims** (`TesolloDownwardsRotateZ`, full
26-DoF hand). Both expose the DR'd cube-size / cube-pose latents the policy
cannot observe, which is the point of the privileged critic.

Env-code provenance (as in §2): the 6 old force.magnitude runs predate commit
`0e297c3`, which grew `fingertip_forces` 2→4, so their `privileged_state` was
41 dims (fingertip_forces = 2) rather than 43. The table above reflects current
HEAD. This affects training only — the critic is not used at eval, so it has no
bearing on the rollout datasets in §2–§3.

Source: `pinch.py::_obs_privileged`, `downwards_rotate_z.py::_obs_privileged`.

# Training setup for saved runs

Extracted from the `checkpoints` branch: per-run training logs
(`<queue>/_queue/*.log`), override YAMLs, queue YAMLs, and per-checkpoint
`ppo_network_config.json`. Two task families are stored: **CubePinch** (force
control) and **DownwardsRotateZ** (in-hand Z rotation). Values below are the
common settings; where the two tasks differ, both are given.

## Parameters needed

### Global

**Policy / value architecture** (`ppo_network_config.json`, PPO `network_factory`)
- Both actor and critic are MLPs with identical hidden layer sizes (3 layers each):
  - **Pinch:** `[256, 128, 64]`
  - **Rotation:** `[512, 256, 128]`
- Activation `silu`; distribution `tanh_normal`; scalar noise, `init_noise_std = 1.0`;
  kernel init `lecun_uniform`; `normalize_observations = true`.
- Policy reads obs key `state`, value reads `privileged_state`.
  - Pinch sizes: state 1 / privileged 33 (for `sb_none`); action size 8.
  - Rotation sizes: state 61 / privileged 114; action size 26.
  - (state size varies with the sensor bundle.)

**Learning rate / EMA / batch / envs** (PPO Training Parameters block)
- **Learning rate:** `1e-4` (both tasks).
- **EMA (action smoothing `ema_alpha`):** Pinch `0.15`, Rotation `0.2`.
  (Pinch 0.15 chosen from an EMA sweep at 8192 envs; noted as an env-config field, not a CLI flag.)
- **Batch size:** `256`; `num_minibatches = 64`; `num_updates_per_batch = 4`.
- **Num envs:** `8192`.

**Rollout length**
- `unroll_length = 40` (both); `action_repeat = 1`.

**Domain randomization**
- **Cube size:** scale range `[0.85, 1.15]` (Pinch runs, `domain_rand.cube_size`).
  The **Rotation** (`downwards_sensor_sweep_120`) queue is the **no-DR control**
  sweep — DR is off, only the observation sensor bundle varies.
- **Starting cube position:** `domain_rand.cube_pos = [0.0, 0.0]` and
  `cube_pos_offset = [0.0, 0.0]` → **not randomized** in these runs.
  (`domain_rand.cube_mass = [1.0, 1.0]` → mass not randomized either.)

**Other training config (for reference)**
| | Pinch | Rotation |
|---|---|---|
| `num_timesteps` | 300M (500M for the sinusoid queue) | 1B |
| `episode_length` | 80 | 120 |
| `discounting` | 0.995 | 0.99 |
| `entropy_cost` | 0.01 | 0.005 |
| `reward_scaling` | 1.0 | 1.0 |
| `max_grad_norm` | 1.0 | (default) |
| `ctrl_dt` / `sim_dt` | 0.05 / 0.01 | 0.05 / 0.01 |

### Pinch env

- **Target force:** `force_target_range = [2.0, 5.0]` N.
  - Static queues: a fixed target is sampled per-episode within this range.
  - Sinusoid queue: the target is swept over this range (see below).
- **Tolerance:** `force_tolerance = 0.75` N.
- **Hold steps needed for success:** `success_hold_time = 0.5` s → **10 control
  steps** (0.5 / ctrl_dt 0.05). `success_reward = 10.0`.

**For the sinusoid** (`pinch_sweep_size_rand_sinusoid-*` queues)
- Enabled via `force_target_sinusoid: true`, `force_target_period: 2.0` s.
- **Amplitude:** the sinusoid sweeps the full `force_target_range [2, 5]` N once
  per period → mean ≈ 3.5 N, amplitude ≈ 1.5 N.
- **Phase randomization range:** phase is sampled **per-episode on reset over the
  full cycle** (a uniform 0…2π / 0…`force_target_period` offset), so the policy
  cannot memorize a fixed schedule and must track force via feedback.
  (Note: only `force_target_sinusoid`/`period`/`range` are logged in the env
  config; the explicit amplitude and phase are derived from these + the queue
  YAML description, not stored as separate numeric fields.)

### Rotation env

- **Target angle range:** `max_target_angle = 2.0943951` rad = **120°**. With
  `curriculum.enable = false`, goals are sampled **full-range in [−120°, +120°]**
  from step 0 (no curriculum ramp).
- Success tolerance: `ori_tolerance_rad = 0.05236` rad = **3°**;
  `target_hold_time = 0.05` s → **1 control step**.

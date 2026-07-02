# Policy checkpoints backup

Trained PPO checkpoints from four experiment queues (mujoco_playground / Tesollo hand tasks).
Backed up from `mujoco_playground/logs` (which is gitignored in that repo).

## Layout

```
<queue-name>/
  _queue/            queue metadata: per-run logs, override yamls, status.json, queue.yaml
  runs/<exp_name>/checkpoints/<step>/   orbax checkpoint (+ ppo_network_config.json)
```

Each run's `checkpoints/config.json` and per-step `ppo_network_config.json` describe the network.
Step directories are named by cumulative env steps; the largest is the final policy.

The downwards queue is the no-DR control sweep (downwards_sensor_sweep_120), 4 bundles x 3 seeds.

## Queues

### `pinch_sweep_size_rand-20260622-195014`

12 runs (12 ok).

| idx | suffix | seed | result | #ckpts | final step |
|----:|--------|-----:|--------|-------:|-----------:|
| 0 | cubepinch_sb_none_s1 | 1 | ok | 9 | 300810240 |
| 1 | cubepinch_sb_none_s2 | 2 | ok | 9 | 300810240 |
| 2 | cubepinch_sb_none_s3 | 3 | ok | 9 | 300810240 |
| 3 | cubepinch_sb_baseline_s1 | 1 | ok | 9 | 300810240 |
| 4 | cubepinch_sb_baseline_s2 | 2 | ok | 9 | 300810240 |
| 5 | cubepinch_sb_baseline_s3 | 3 | ok | 9 | 300810240 |
| 6 | cubepinch_sb_propriotarget_s1 | 1 | ok | 9 | 300810240 |
| 7 | cubepinch_sb_propriotarget_s2 | 2 | ok | 9 | 300810240 |
| 8 | cubepinch_sb_propriotarget_s3 | 3 | ok | 9 | 300810240 |
| 9 | cubepinch_sb_forcemagnitude_s1 | 1 | ok | 9 | 300810240 |
| 10 | cubepinch_sb_forcemagnitude_s2 | 2 | ok | 9 | 300810240 |
| 11 | cubepinch_sb_forcemagnitude_s3 | 3 | ok | 9 | 300810240 |

### `pinch_sweep_size_rand_sinusoid-20260624-142357`

20 runs (19 ok).

| idx | suffix | seed | result | #ckpts | final step |
|----:|--------|-----:|--------|-------:|-----------:|
| 0 | cubepinch_sb_baseline_s6 | 6 | ok | 9 | 501350400 |
| 1 | cubepinch_sb_baseline_s7 | 7 | ok | 9 | 501350400 |
| 2 | cubepinch_sb_baseline_s8 | 8 | ok | 9 | 501350400 |
| 3 | cubepinch_sb_baseline_s4 | 4 | ok | 9 | 501350400 |
| 4 | cubepinch_sb_baseline_s5 | 5 | ok | 9 | 501350400 |
| 5 | cubepinch_sb_propriotarget_s6 | 6 | ok | 9 | 501350400 |
| 6 | cubepinch_sb_propriotarget_s7 | 7 | ok | 9 | 501350400 |
| 7 | cubepinch_sb_propriotarget_s8 | 8 | ok | 9 | 501350400 |
| 8 | cubepinch_sb_propriotarget_s4 | 4 | ok | 9 | 501350400 |
| 9 | cubepinch_sb_propriotarget_s5 | 5 | ok | 9 | 501350400 |
| 10 | cubepinch_sb_forcemagnitude_s6 | 6 | ok | 9 | 501350400 |
| 11 | cubepinch_sb_forcemagnitude_s7 | 7 | failed | 2 | 111411200 |
| 12 | cubepinch_sb_forcemagnitude_s8 | 8 | ok | 9 | 501350400 |
| 13 | cubepinch_sb_forcemagnitude_s4 | 4 | ok | 9 | 501350400 |
| 14 | cubepinch_sb_forcemagnitude_s5 | 5 | ok | 9 | 501350400 |
| 15 | cubepinch_sb_propriodelta_s6 | 6 | ok | 9 | 501350400 |
| 16 | cubepinch_sb_propriodelta_s7 | 7 | ok | 9 | 501350400 |
| 17 | cubepinch_sb_propriodelta_s8 | 8 | ok | 9 | 501350400 |
| 18 | cubepinch_sb_propriodelta_s4 | 4 | ok | 9 | 501350400 |
| 19 | cubepinch_sb_propriodelta_s5 | 5 | ok | 9 | 501350400 |

### `pinch_sweep_size_rand_sinusoid-20260623-133237`

15 runs (15 ok).

| idx | suffix | seed | result | #ckpts | final step |
|----:|--------|-----:|--------|-------:|-----------:|
| 0 | cubepinch_sb_none_s1 | 1 | ok | 9 | 300810240 |
| 1 | cubepinch_sb_none_s2 | 2 | ok | 9 | 300810240 |
| 2 | cubepinch_sb_none_s3 | 3 | ok | 9 | 300810240 |
| 3 | cubepinch_sb_baseline_s1 | 1 | ok | 9 | 300810240 |
| 4 | cubepinch_sb_baseline_s2 | 2 | ok | 9 | 300810240 |
| 5 | cubepinch_sb_baseline_s3 | 3 | ok | 9 | 300810240 |
| 6 | cubepinch_sb_propriotarget_s1 | 1 | ok | 9 | 300810240 |
| 7 | cubepinch_sb_propriotarget_s2 | 2 | ok | 9 | 300810240 |
| 8 | cubepinch_sb_propriotarget_s3 | 3 | ok | 9 | 300810240 |
| 9 | cubepinch_sb_forcemagnitude_s1 | 1 | ok | 9 | 300810240 |
| 10 | cubepinch_sb_forcemagnitude_s2 | 2 | ok | 9 | 300810240 |
| 11 | cubepinch_sb_forcemagnitude_s3 | 3 | ok | 9 | 300810240 |
| 12 | cubepinch_sb_propriodelta_s1 | 1 | ok | 9 | 300810240 |
| 13 | cubepinch_sb_propriodelta_s2 | 2 | ok | 9 | 300810240 |
| 14 | cubepinch_sb_propriodelta_s3 | 3 | ok | 9 | 300810240 |

### `downwards_sensor_sweep_120-20260625-124815`

12 runs (12 ok).

| idx | suffix | seed | result | #ckpts | final step |
|----:|--------|-----:|--------|-------:|-----------:|
| 0 | downwardsrotatez_sb_forcemagnitude_s0 | 0 | ok | 19 | 1008599040 |
| 1 | downwardsrotatez_sb_forcemagnitude_s1 | 1 | ok | 19 | 1008599040 |
| 2 | downwardsrotatez_sb_forcemagnitude_s2 | 2 | ok | 19 | 1008599040 |
| 3 | downwardsrotatez_sb_baseline_s0 | 0 | ok | 19 | 1008599040 |
| 4 | downwardsrotatez_sb_baseline_s1 | 1 | ok | 19 | 1008599040 |
| 5 | downwardsrotatez_sb_baseline_s2 | 2 | ok | 19 | 1008599040 |
| 6 | downwardsrotatez_sb_propriodelta_s0 | 0 | ok | 19 | 1008599040 |
| 7 | downwardsrotatez_sb_propriodelta_s1 | 1 | ok | 19 | 1008599040 |
| 8 | downwardsrotatez_sb_propriodelta_s2 | 2 | ok | 19 | 1008599040 |
| 9 | downwardsrotatez_sb_propriotarget_s0 | 0 | ok | 19 | 1008599040 |
| 10 | downwardsrotatez_sb_propriotarget_s1 | 1 | ok | 19 | 1008599040 |
| 11 | downwardsrotatez_sb_propriotarget_s2 | 2 | ok | 19 | 1008599040 |

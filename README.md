<p align="center">
  <img src="assets/logo.png" alt="Policy Analyzer" width="120">
</p>

<h1 align="center">Policy Analyzer</h1>

<p align="center">
  A browser dashboard for inspecting and comparing trained dexterous-manipulation policies.
</p>

---

Training tells you the reward went up. It doesn't tell you *what the policy actually
does*. Policy Analyzer answers that: point it at a
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) log directory
and it rolls out checkpoints across your GPUs, records every observation and action, and
serves the result as an interactive dashboard.

## Single Policy Analysis

Pick a run from a searchable policy picker — filter by task, by config field
(`finger_kp=8`, `kv>=0.2`), or by training performance (`perf.success>0.5`), with numbers
pulled live from the experiment cockpit. Choose a checkpoint, how many deterministic and
stochastic rollouts to launch, and optionally **pin** randomization axes so every
combination becomes its own rollout with that quantity held at a known value instead of
sampled.

Rollouts land live over Server-Sent Events. A video grid fills in as each finishes,
alongside a rollout table and aggregate reward and per-term reward curves. Click any
rollout for its own page: input-distribution histograms, per-DOF trajectories, and a 2×2
grid of motor targets, action deltas, tracking error and joint state — all grouped by the
policy's real I/O schema, rebuilt from the environment so the labels can never drift out
of sync with the network.

## Multi-policy Analysis

The ablation view, split cleanly between *gathering* rollouts and *looking* at them.

**Collect a dataset.** Training runs launched as a *queue* are grouped into policies by
their overrides file, with training-seed replicates pooled automatically. Evaluate them
against their training config, against a benchmark preset, or over a deterministic grid
that sweeps the run's own randomization ranges — cube size, force target, goal angle,
friction — with each cell pinned.

**Then plot it.** Success-rate-vs-threshold curves (mean line, min–max band over seeds),
or heatmaps over any two swept axes, faceted rows and columns by a policy knob, with any
panel usable as a reference the others are shown as a difference from. Export the chart
as PNG or the underlying data as CSV.

## Install

Policy Analyzer lives beside the training repo:

```
workspace/
  mujoco_playground/    # your training repo — logs, env code
  policy_analyzer/      # this package
```

Set `MUJOCO_PLAYGROUND_ROOT` if your layout differs. Everything the analyzer writes stays
under `policy_analyzer/analysis/`.

It needs the same environment your training runs use — `jax`, `brax`, `mujoco`,
`mujoco_playground` — plus `numpy`, `matplotlib`, `mediapy` and `pyyaml`. Launch it with
that interpreter, since rollout subprocesses inherit it:

```bash
.venv/bin/python -m policy_analyzer --port 8000
```

Then open `http://localhost:8000`. On a remote box, tunnel it:
`ssh -L 8000:localhost:8000 <host>`.

## Under the hood

The server process never imports JAX. Every rollout runs in its own subprocess with
`CUDA_VISIBLE_DEVICES` pinned before Python starts — the only reliable way to give each
one a clean CUDA context. A scheduler thread hands queued work to free GPUs, reaps
finished processes, and retries on OOM. Results are cached on disk behind a `DONE`
sentinel, so re-opening a dataset is instant. A whole grid sweep is one vmapped scan: the
environment is built once and the rollout compiles once, however many cells it has.

```
__main__.py         HTTP + SSE server, GPU scheduler
collect_one.py      batched rollout runner (one process, one GPU, all seeds)
compare_collect.py  cheap eval runner — scalar channels, no rendering
grid_collect.py     grid sweep with every randomization axis pinned
grid_axes.py        which axes are sweepable per env, and how to pin each
success_curve.py    success-vs-threshold curves and heatmaps (numpy only)
queues.py           training queues → policies → seed runs
run_metrics.py      per-run training metrics, read off the experiment cockpit
io_schema.py        observation/action layout, derived from the env
visualize.py        matplotlib figures; frontend.py exports the per-rollout page
```

## Benchmarks

A benchmark is a YAML file in `benchmarks/` describing environment overrides to evaluate
against — a harder goal range, domain randomization switched off. Drop a file in and it
appears for every matching environment.

```yaml
name: downwards_rotate_symmetric_band_90_120
label: "Downwards rotate — symmetric band (90–120°)"
envs: ["TesolloDownwardsRotateZ"]
env_overrides:
  curriculum.enable: false
  min_target_angle: 1.5707963267948966
  max_target_angle: 2.0943951023931953
```

## Headless use

Success tables without the browser:

```bash
python -m policy_analyzer.success_report pinch_sweep_size_rand \
    --mode det --n-rollouts 50 --gpu 0
```

## Studies and report data

- [`force_probe/`](force_probe/README.md) — can fingertip forces be recovered *linearly*
  from a frozen policy's hidden state? They can, in every policy we tested, including
  those that never observe force.
- [`report/appendix/`](report/appendix/README.md) — the raw per-rollout and per-timestep
  datasets behind our sensor-ablation study, with data dictionaries and the scripts that
  regenerate them.

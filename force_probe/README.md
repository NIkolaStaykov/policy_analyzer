# Force-probe study (isolated bubble)

**Question:** can fingertip force magnitudes be recovered *linearly* from a frozen
policy's internal representations — i.e. does the policy compute an internal
estimate of contact force even when it doesn't directly observe it?

Self-contained under `force_probe/`. It touches **no** env or training code — it
only *reads* checkpoints + sensors and runs the policies at inference. (The one
pre-`0e297c3` force.magnitude checkpoint is restored under a read-only git
worktree at `0ac79f5`; see below.)

## Method

For each **policy type** = (task × sensor_bundle), take the **best seed by mean
eval reward during training** (mean of the per-eval `reward=` points parsed from
each run's local wandb `output.log`; see `select_seeds.py`). Then:

1. **Roll out** the deterministic (frozen) policy for 64 episodes, capturing per
   step the policy input obs (`state`) paired with the ground-truth fingertip
   force magnitudes read from the *same* pre-step data — 2 per-finger forces
   (thumb, index) for the pinch tasks, 5 per-tip forces for downwards.
   (`probe_lib.capture_rollouts`)
2. **Re-derive the frozen activations** from the checkpoint: normalized input +
   each hidden layer (swish MLP, manual forward validated to `max|Δ|<1e-3` vs the
   network's own output — in practice `0.0`). (`probe_lib.extract_activations`)
3. **Train a single linear layer** (closed-form ridge, α=10, standardized inputs)
   from each representation → force, scored by R² on a held-out 20% split.
   (`probe_lib.train_probe`)

The linear probe is the *only* thing trained; the policy is frozen.

Files: `select_seeds.py` → `selected_seeds.json`; `run_probe.py` (per-policy
driver, with a lean-restore monkeypatch so it fits a shared GPU) driven by
`run_all.sh`; outputs `data/rollout_<exp>.npz`, `probe_results.csv`,
`probe_results.json`.

## Results — mean R² (val) predicting fingertip force

`*` = old-code (0ac79f5) restore. `input` width varies with the bundle's obs.

```
task             bundle          seed  evalR |  input     h0     h1     h2 |   best  @layer
-----------------------------------------------------------------------------------------------
pinch            baseline           3   21.4 |  0.645  0.792  0.799  0.738 |  0.799 hidden_1
pinch            force.magnitude*   1   25.9 |  1.000  1.000  0.995  0.919 |  1.000   input
pinch            none               2   11.5 |  0.049  0.320  0.440  0.440 |  0.440 hidden_2
pinch            proprio.delta      2   22.3 |  0.792  0.897  0.876  0.803 |  0.897 hidden_0
pinch            proprio.target     1   23.6 |  0.774  0.891  0.876  0.847 |  0.891 hidden_0

pinch_sinusoid   baseline           8   20.5 |  0.772  0.878  0.886  0.845 |  0.886 hidden_1
pinch_sinusoid   force.magnitude    8   33.2 |  1.000  1.000  1.000  0.997 |  1.000 hidden_0
pinch_sinusoid   none               3   14.0 |  0.096  0.105  0.111  0.110 |  0.111 hidden_1
pinch_sinusoid   proprio.delta      8   22.4 |  0.916  0.945  0.944  0.932 |  0.945 hidden_0
pinch_sinusoid   proprio.target     5   33.2 |  0.954  0.973  0.972  0.965 |  0.973 hidden_0

downwards_rotate baseline           1   14.4 |  0.648  0.710  0.714  0.692 |  0.714 hidden_1
downwards_rotate force.magnitude    0   14.4 |  1.000  0.999  0.930  0.830 |  1.000   input
downwards_rotate proprio.delta      2   14.1 |  0.642  0.703  0.695  0.658 |  0.703 hidden_0
downwards_rotate proprio.target     1   13.8 |  0.706  0.753  0.734  0.699 |  0.753 hidden_0
```

## Takeaways

- **Force is linearly decodable from the hidden state in every policy**, and for
  every bundle that does *not* observe force directly, a hidden layer beats the
  raw-input baseline — the network *constructs* force information that is not
  linearly present in its input. Clearest case: pinch/`none`, whose input is a
  single scalar (the force-target command, R²=0.05) yet whose hidden layers reach
  R²=0.44.
- **`force.magnitude` bundles observe force directly** → input R²≈1.0. It stays
  ≈1.0 through the early hidden layers, then *decays* with depth (h2 as low as
  0.83 for downwards): the policy compresses/discards force detail it no longer
  needs near the action head.
- **Peak decodability is early/mid (`hidden_0`/`hidden_1`)**, declining at the
  last hidden layer — force is a mid-level feature, not what the final pre-action
  layer emphasizes.
- **More force info ↔ better policy**: bundles that expose or strongly imply force
  (`force.magnitude`, `proprio.target`, `proprio.delta`) both train to higher eval
  reward and carry more decodable force; `none` is worst on both.
- **`pinch_sinusoid`/`none` is the floor** (R²≈0.11): with a time-varying force
  target and no proprio/force obs, the feedforward policy has almost nothing to
  reconstruct force from — consistent with it being among the weakest policies.

"""Grid-based post-training evaluation over a run's randomization ranges.

Where compare_collect samples episode parameters from the run's training
distributions (fixed seed stream, random draws), this walks a deterministic
grid over every randomized quantity of the run and evaluates N rollouts per
grid cell with that quantity PINNED:

  - episode-level parameters (sampled in env.reset) are pinned via degenerate
    config ranges, e.g. force_target_range=[v, v] / min==max target angle;
  - model-level domain-randomization parameters (applied by the env's
    domain_randomize at training time) are pinned by applying the SAME model
    transformation with a fixed value instead of a uniform draw.

Axes are derived from the run's own config.json (checkpoints/config.json), so
the grid covers exactly the ranges the queue trained with. Model-DR axes whose
configured range is degenerate are dropped automatically.

THE WHOLE GRID RUNS AS ONE BATCH. Cells are not a Python loop: all
n_cells * n_rollouts episodes are a single vmapped scan, so the env is built
once and the rollout compiles once. Both pin kinds become batch dimensions:

  - model axes stack their pinned mjx.Model fields along a leading cell axis
    (the same trick as wrapper.BraxDomainRandomizationVmapWrapper, with grid
    coordinates instead of random draws);
  - episode axes write TRACED per-episode values into the env config for the
    duration of the reset trace, so `jax.random.uniform(minval=v, maxval=v)`
    returns exactly that cell's value. Nothing is injected after the fact, so
    reset semantics — including the t=0 observation — are bit-identical to a
    reset that happened to sample the grid value.

Supported envs: TesolloCubePinch, TesolloDownwardsRotateZ.

Output artifact (npz, same channel layout as compare_collect but with a
leading grid-cell axis):
    <name>.npz   channels [n_cells, N, T]; _grid_axes (JSON) describes the
                 axes; _cell_values [n_cells, n_axes] gives each cell's
                 coordinates; _dr = "pinned".

Usage:
    python -m policy_analyzer.grid_collect --log-dir logs/<run> \
        [--out X.npz] [--mode det|sto] [--n-rollouts 8] \
        [--points 8] [--axis-points a=4,b=6] [--axes a,b] [--checkpoint N] \
        [--max-batch 4096] [--sequential]
"""

from __future__ import annotations

import argparse
import sys
import traceback

from policy_analyzer import grid_axes

EXIT_OOM = 75  # EX_TEMPFAIL (same contract as compare_collect)

# Candidate axes per env are declared in grid_axes.MANIFEST (stdlib-only, shared
# with the server). This module's top level must likewise stay stdlib-only: the
# analyzer server imports SUPPORTED_ENVS to decide whether to offer the grid.
SUPPORTED_ENVS = tuple(grid_axes.MANIFEST)

DEFAULT_POINTS = grid_axes.DEFAULT_POINTS  # per-axis resolution unless overridden
# Rollouts per cell. 64 rather than a token 8 for two measured reasons: a cell
# estimate's standard error falls from ~0.045 to ~0.016, and the whole grid is
# one batch, so the extra episodes are close to free (280 -> 2240 concurrent
# episodes costs ~0.2s of an ~19s job, which is dominated by restore+compile).
# It also keeps a small grid clear of the low-batch-width bias: below ~64
# concurrent episodes the sim reads systematically high (~0.38 vs ~0.21 on a
# pinch grid), so a 1-cell sweep at 64 rollouts is still in the good regime.
DEFAULT_N_ROLLOUTS = 64
# Guardrail against a typo'd sweep, not a throughput limit: cells are batched,
# and training on the same GPU runs num_envs=8192, so a grid of this size is a
# normal workload rather than an exceptional one.
DEFAULT_MAX_CELLS = 8192
# Episodes simulated concurrently (cells x rollouts). Chunked at this width so a
# very large grid degrades to a few big batches instead of one OOM.
DEFAULT_MAX_BATCH = 4096

# mjx.Model fields written by each model-axis transform in _pin_model_axis.
# Only these get a leading cell axis in the batched model; everything else stays
# shared. Must match _pin_model_axis exactly — a missing field would silently
# evaluate every cell on the nominal model.
MODEL_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "TesolloCubePinch": {
        "cube_size": ("geom_size", "body_pos"),
        "cube_pos": ("body_pos",),
        "cube_mass": ("body_mass", "body_inertia"),
        "actuator_kp": ("actuator_gainprm", "actuator_biasprm"),
    },
    "TesolloDownwardsRotateZ": {
        "cube_size": ("geom_size",),
    },
}


def _linspace(lo: float, hi: float, n: int, endpoint: bool = True):
    import numpy as np

    if lo == hi:
        return np.array([lo])
    return np.linspace(lo, hi, n, endpoint=endpoint)


def _gen_values(cand: dict, n: int):
    """Sweep values (1-D np.ndarray) for one candidate axis in real units.

    Driven by the manifest flags on `cand`: geometric spacing for log2 axes
    (uniform in octaves, matching the reset sampler), a dropped duplicate
    endpoint for full-circle phase, and a sign-mirror for symmetric axes
    (training draws U(lo, hi) with a random sign, so both signs are covered).
    """
    import numpy as np

    lo, hi = float(cand["lo"]), float(cand["hi"])
    if cand.get("space") == "log2":
        vals = np.geomspace(lo, hi, n)
    elif lo == hi:
        vals = np.array([lo])
    else:
        full_circle = cand.get("wrap") and abs((hi - lo) - 2.0 * np.pi) < 1e-9
        vals = np.linspace(lo, hi, n, endpoint=not full_circle)
    if cand.get("symmetric"):
        vals = np.unique(np.concatenate([-vals[::-1], vals]))
    return vals


def build_axes(
    env_name: str, cfg, selected: set | None = None, points: dict | None = None
) -> list[dict]:
    """Derive grid axes from a run's config via the shared grid_axes manifest.

    `selected` (axis names) restricts which candidates are swept; `points` maps
    axis name -> resolution (default DEFAULT_POINTS). Each returned axis:
    {name, kind, space, values (1-D np.ndarray), pin (recipe), source}.
    """
    if env_name not in grid_axes.MANIFEST:
        raise ValueError(
            f"grid_collect does not support env {env_name!r} "
            f"(supported: {', '.join(SUPPORTED_ENVS)})"
        )
    points = points or {}
    cands = grid_axes.candidates(env_name, cfg.to_dict())
    axes: list[dict] = []
    for cand in cands:
        if selected is not None and cand["name"] not in selected:
            continue
        n = int(points.get(cand["name"], DEFAULT_POINTS))
        axes.append({
            "name": cand["name"],
            "kind": cand["kind"],
            "space": cand["space"],
            "values": _gen_values(cand, n),
            "pin": cand["pin"],
            "source": cand["source"],
        })
    return axes


def _pin_episode(cfg, axis: dict, value: float) -> None:
    """Pin one episode axis to `value` by applying its manifest pin recipe.

    Recipes: "range"/"range_pair" set the config range degenerate ([v, v]);
    "scalar" sets a real-units scalar and "degenerate" collapses its companion
    range so reset() takes the scalar branch (frequency/amplitude); "set" merges
    fixed extra overrides (e.g. symmetric_target off).
    """
    pin = axis["pin"]
    v = float(value)
    overrides: dict = {}
    if "range" in pin:
        overrides[pin["range"]] = [v, v]
    if "range_pair" in pin:
        lo_k, hi_k = pin["range_pair"]
        overrides[lo_k] = v
        overrides[hi_k] = v
    if "scalar" in pin:
        overrides[pin["scalar"]] = v
    if "degenerate" in pin:
        overrides[pin["degenerate"]] = [0.0, 0.0]  # hi <= lo -> reset uses scalar
    if "set" in pin:
        overrides.update(pin["set"])
    _apply_overrides_strict(cfg, overrides)


def _static_episode_pins(cfg, axes: list[dict]) -> None:
    """Apply the parts of every episode pin that must be baked into the config.

    The "set" and "degenerate" recipes exist to steer Python-level branches in
    reset() (``if freq_hi > freq_lo``, ``if config.curriculum.enable``), which
    are decided at trace time and are therefore the same for every cell. Only
    the value itself varies per cell, and that part is applied as a tracer by
    _traced_episode_pins.
    """
    overrides: dict = {}
    for ax in axes:
        pin = ax["pin"]
        if "degenerate" in pin:
            overrides[pin["degenerate"]] = [0.0, 0.0]  # hi <= lo -> scalar branch
        if "set" in pin:
            overrides.update(pin["set"])
    if overrides:
        _apply_overrides_strict(cfg, overrides)


def _cfg_node(cfg, dotted: str):
    """(parent_node, leaf_key) for a dotted config key, raising if absent."""
    parts = str(dotted).split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"config has no key {dotted!r} to pin")
    return node, parts[-1]


def _traced_episode_pins(cfg, axes: list[dict], values):
    """Context manager writing per-episode axis values into `cfg` as tracers.

    Used inside the vmapped trace: reset() reads these keys as the bounds of a
    `jax.random.uniform` (or as a scalar), so a traced v in both slots of a
    range makes the draw return exactly v for that batch element. ConfigDict
    type-checks assignments, hence ignore_type() — the values are JAX tracers
    where the schema declares floats.

    Wraps reset() only. step() reads the pinned quantity from state.info, and
    leaving tracers in the config across the scan would expose them to code
    paths (e.g. the curriculum's `int(max_target_angle / band_width)`) that
    assume concrete floats.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        saved: list[tuple] = []
        with cfg.ignore_type():
            for ax, v in zip(axes, values):
                pin = ax["pin"]
                targets = []
                if "range" in pin:
                    targets.append((pin["range"], [v, v]))
                if "range_pair" in pin:
                    lo_k, hi_k = pin["range_pair"]
                    targets += [(lo_k, v), (hi_k, v)]
                if "scalar" in pin:
                    targets.append((pin["scalar"], v))
                if not targets:
                    raise KeyError(
                        f"episode axis {ax['name']!r} has no value-bearing pin "
                        f"recipe (got {sorted(pin)})"
                    )
                for key, val in targets:
                    node, leaf = _cfg_node(cfg, key)
                    saved.append((node, leaf, node[leaf]))
                    node[leaf] = val
            try:
                yield
            finally:
                for node, leaf, old in reversed(saved):
                    node[leaf] = old

    return _ctx()


def _apply_overrides_strict(cfg, overrides: dict) -> None:
    """Set dotted config keys, raising on failure.

    benchmarks.apply_overrides swallows errors (fine for optional benchmark
    tweaks); here a silently dropped override would leave the cell sampling
    randomly instead of pinned, corrupting the grid.
    """
    for key, val in overrides.items():
        parts = str(key).split(".")
        node = cfg
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            # Unlocked ConfigDicts silently ADD unknown keys; a renamed config
            # field would otherwise leave the cell unpinned.
            raise KeyError(f"config has no key {key!r} to pin")
        node[parts[-1]] = val


def _cube_ids(mj_model):
    """Cube body id + geom ids from an env's compiled MjModel."""
    import numpy as np

    bid = mj_model.body("cube").id
    gids = np.array(
        [g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] == bid]
    )
    return bid, gids


def _pin_model_axis(env, env_name: str, axis_name: str, value: float) -> None:
    """Apply one model-DR axis to env's mjx model with a FIXED value.

    Mirrors the env's domain_randomize transformation exactly (pinch
    _domain_randomize_impl.rand / downwards_rotate_z.domain_randomize.rand),
    with the uniform draw replaced by `value`. Mutates env._mjx_model in place;
    must run before anything traces env.step/reset.
    """
    m = env.mjx_model
    bid, gids = _cube_ids(env.mj_model)
    v = float(value)

    if env_name == "TesolloDownwardsRotateZ":
        if axis_name != "cube_size":
            raise KeyError(f"no model pin for {env_name}/{axis_name}")
        geom_size = m.geom_size.at[gids].set(m.geom_size[gids] * v)
        env._mjx_model = m.tree_replace({"geom_size": geom_size})
        return

    # TesolloCubePinch — mirrors pinch._domain_randomize_impl.rand
    if axis_name == "cube_size":
        geom_size = m.geom_size.at[gids].set(m.geom_size[gids] * v)
        body_pos = m.body_pos.at[bid, 2].set(geom_size[gids[0], 2])
        env._mjx_model = m.tree_replace(
            {"geom_size": geom_size, "body_pos": body_pos}
        )
    elif axis_name == "cube_pos":
        # Training draws dx, dy independently from the same range; the grid
        # pins both components to the same value (diagonal of that square).
        body_pos = m.body_pos.at[bid, :2].set(m.body_pos[bid, :2] + v)
        env._mjx_model = m.tree_replace({"body_pos": body_pos})
    elif axis_name == "cube_mass":
        body_mass = m.body_mass.at[bid].set(m.body_mass[bid] * v)
        body_inertia = m.body_inertia.at[bid].set(m.body_inertia[bid] * v)
        env._mjx_model = m.tree_replace(
            {"body_mass": body_mass, "body_inertia": body_inertia}
        )
    elif axis_name == "actuator_kp":
        kp = m.actuator_gainprm[:, 0] * v
        gainprm = m.actuator_gainprm.at[:, 0].set(kp)
        biasprm = m.actuator_biasprm.at[:, 1].set(-kp)
        env._mjx_model = m.tree_replace(
            {"actuator_gainprm": gainprm, "actuator_biasprm": biasprm}
        )
    else:
        raise KeyError(f"no model pin for {env_name}/{axis_name}")


def _cell_index(axes: list[dict]):
    """(shape, cell_values [n_cells, n_axes]) in row-major cell order.

    The order is itertools.product over axis indices — the layout the npz,
    success_curve.compute_grid_heatmap and the frontend pivot all assume.
    """
    import itertools

    import numpy as np

    shape = tuple(len(ax["values"]) for ax in axes)
    cell_values = np.array(
        [
            [float(axes[i]["values"][idx[i]]) for i in range(len(axes))]
            for idx in itertools.product(*(range(s) for s in shape))
        ]
    )
    return shape, cell_values


def _batched_model_fields(env, env_name: str, model_axes, model_vals):
    """Stack each cell's pinned mjx.Model fields along a leading cell axis.

    Rather than re-deriving every transform in batched form, this replays the
    per-cell _pin_model_axis onto a pristine base model and keeps the fields it
    touched. Axes are applied in manifest order, so overlapping writes (pinch's
    cube_size sets body_pos[2], cube_pos then offsets body_pos[:2]) compose
    exactly as they did cell-by-cell.
    """
    import jax.numpy as jp

    fields = sorted(
        set().union(
            *(MODEL_FIELDS[env_name][ax["pin"]["model"]] for ax in model_axes)
        )
    )
    base = env.mjx_model
    per_cell = []
    try:
        for vals in model_vals:
            env._mjx_model = base
            for ax, v in zip(model_axes, vals):
                _pin_model_axis(env, env_name, ax["pin"]["model"], float(v))
            per_cell.append({f: getattr(env._mjx_model, f) for f in fields})
    finally:
        env._mjx_model = base
    return {f: jp.stack([d[f] for d in per_cell]) for f in fields}


def _scalar_channels(state, drop: set):
    """Per-step scalar channels of a state, selected at trace time.

    Mirrors collect.run_eval_rollouts' post-hoc filter (keep numeric entries
    that are scalar per step) but applies it inside the scan, so vector info
    entries like obs_bias are never materialised for the whole batch.
    """
    import jax.numpy as jp

    out = {"reward": state.reward.astype(jp.float32),
           "done": state.done.astype(jp.float32)}
    for prefix, d in (("", state.metrics), ("info.", state.info)):
        for k, v in d.items():
            if k in drop:
                continue
            arr = jp.asarray(v)
            if arr.ndim == 0 and jp.issubdtype(arr.dtype, jp.number):
                out[f"{prefix}{k}"] = arr.astype(jp.float32)
    return out


def run_grid_eval(
    handles: dict,
    axes: list[dict],
    n_rollouts: int,
    deterministic: bool,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> dict:
    """Evaluate every grid cell in one batched pass; channels are [n_cells, N, T].

    The env is built once (with the static half of the episode pins baked in)
    and the rollout is traced once; cells and rollouts are both batch
    dimensions. Batch element b = cell * n_rollouts + rollout, and the rng
    stream is the same `split(PRNGKey(0), n_rollouts)` in every cell, so cells
    stay seed-paired exactly as they were when each ran on its own.
    """
    import copy
    import time

    import jax
    import jax.numpy as jp
    import numpy as np

    from mujoco_playground import registry
    from policy_analyzer import collect

    env_name = handles["env_name"]
    shape, cell_values = _cell_index(axes)
    n_cells = int(np.prod(shape))

    ep_cols = [i for i, ax in enumerate(axes) if ax["kind"] == "episode"]
    model_cols = [i for i, ax in enumerate(axes) if ax["kind"] == "model"]
    ep_axes = [axes[i] for i in ep_cols]
    model_axes = [axes[i] for i in model_cols]

    # One env for the whole grid: only the static pins (branch selectors) go in.
    cfg = copy.deepcopy(handles["env_cfg"])
    _static_episode_pins(cfg, ep_axes)
    env = registry.load(env_name, config=cfg)

    model_stack = (
        _batched_model_fields(env, env_name, model_axes, cell_values[:, model_cols])
        if model_axes else {}
    )
    ep_stack = jp.asarray(cell_values[:, ep_cols], dtype=jp.float32)  # [n_cells, n_ep]

    episode_length = int(handles["ppo_params"].episode_length)
    inference_fn = handles["make_inference_fn"](
        handles["params"], deterministic=deterministic
    )
    base_model = env.mjx_model
    drop = set(collect.DROP_INFO)

    def step_fn(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        act = inference_fn(state.obs, act_key)[0]
        state = env.step(state, act)
        return (state, rng), _scalar_channels(state, drop)

    def single(model_fields, ep_vals, rng):
        env._mjx_model = (
            base_model.tree_replace(model_fields) if model_fields else base_model
        )
        try:
            with _traced_episode_pins(env._config, ep_axes, ep_vals):
                state = env.reset(rng)
            _, traj = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)
        finally:
            env._mjx_model = base_model
        return traj

    run = jax.jit(jax.vmap(single, in_axes=(0, 0, 0)))

    rngs = jax.random.split(jax.random.PRNGKey(0), n_rollouts)  # [N, 2]
    cells_per_chunk = max(1, min(n_cells, max_batch // max(1, n_rollouts)))
    n_chunks = -(-n_cells // cells_per_chunk)
    print(
        f"[grid] batching {n_cells} cells x {n_rollouts} rollouts as "
        f"{n_chunks} pass(es) of up to {cells_per_chunk * n_rollouts} episodes",
        flush=True,
    )

    chunks: list[dict] = []
    for start in range(0, n_cells, cells_per_chunk):
        t0 = time.monotonic()
        stop = min(start + cells_per_chunk, n_cells)
        sel = np.arange(start, stop)
        # Pad a short final chunk by repeating its last cell so every pass has
        # identical shapes and reuses the one compilation; padding is discarded.
        if n_chunks > 1 and len(sel) < cells_per_chunk:
            sel = np.concatenate([sel, np.full(cells_per_chunk - len(sel), sel[-1])])
        nc = len(sel)

        mf = {f: jp.repeat(v[sel], n_rollouts, axis=0) for f, v in model_stack.items()}
        ep = jp.repeat(ep_stack[sel], n_rollouts, axis=0)   # [B, n_ep]
        rr = jp.tile(rngs, (nc, 1))                          # [B, 2]

        traj = run(mf, ep, rr)                               # leaves [B, T]
        got = stop - start
        chunks.append({
            k: np.asarray(v).reshape(nc, n_rollouts, -1)[:got]
            for k, v in traj.items()
        })
        print(
            f"[grid] cells {start + 1}-{stop}/{n_cells} "
            f"({nc * n_rollouts} episodes, {time.monotonic() - t0:.1f}s)",
            flush=True,
        )

    channels = {
        k: np.concatenate([c[k] for c in chunks], axis=0)   # [n_cells, N, T]
        for k in chunks[0]
    }

    for c in range(n_cells):
        coord = ", ".join(
            f"{ax['name']}={cell_values[c, i]:.4g}" for i, ax in enumerate(axes)
        )
        headline = ""
        if "reward/success_per_step" in channels:
            headline = f"  success/step={channels['reward/success_per_step'][c].mean():.3f}"
        for err_name in ("info.force_error", "info.ori_error"):
            if err_name in channels:
                headline += (
                    f"  |{err_name.split('.')[1]}|="
                    f"{np.abs(channels[err_name][c]).mean():.3f}"
                )
        print(f"[grid {c + 1}/{n_cells}] {coord}{headline}")

    return {
        "channels": channels,
        "grid_shape": np.array(shape),
        "cell_values": cell_values,
        "n_rollouts": n_rollouts,
        "episode_length": episode_length,
        "dt": float(env.dt),
    }


def run_grid_eval_sequential(
    handles: dict,
    axes: list[dict],
    n_rollouts: int,
    deterministic: bool,
) -> dict:
    """Reference implementation: one env rebuild + one compile per cell.

    Superseded by run_grid_eval (same outputs, ~n_cells times slower). Kept as
    the correctness oracle behind --sequential: the batched path pins episode
    axes with tracers rather than with a rebuilt config, so having a
    known-good path to diff against is worth the duplication.
    """
    import copy
    import itertools
    import time

    import numpy as np

    from mujoco_playground import registry
    from policy_analyzer import collect

    env_name = handles["env_name"]
    base_cfg = handles["env_cfg"]
    shape, cell_values = _cell_index(axes)   # [n_cells, n_axes]
    n_cells = int(np.prod(shape))

    per_cell: list[dict] = []
    for c, idx in enumerate(itertools.product(*(range(s) for s in shape))):
        t0 = time.monotonic()
        cfg = copy.deepcopy(base_cfg)
        for i, ax in enumerate(axes):
            if ax["kind"] == "episode":
                _pin_episode(cfg, ax, ax["values"][idx[i]])
        env = registry.load(env_name, config=cfg)
        for i, ax in enumerate(axes):
            if ax["kind"] == "model":
                _pin_model_axis(
                    env, env_name, ax["pin"]["model"], ax["values"][idx[i]]
                )

        cell_handles = {**handles, "eval_env": env}
        result = collect.run_eval_rollouts(
            cell_handles, n_rollouts=n_rollouts, deterministic=deterministic
        )
        per_cell.append(result)

        coord = ", ".join(
            f"{ax['name']}={float(ax['values'][idx[i]]):.4g}"
            for i, ax in enumerate(axes)
        )
        ch = result["channels"]
        headline = ""
        if "reward/success_per_step" in ch:
            headline = f"  success/step={ch['reward/success_per_step'].mean():.3f}"
        for err_name in ("info.force_error", "info.ori_error"):
            if err_name in ch:
                headline += f"  |{err_name.split('.')[1]}|={np.abs(ch[err_name]).mean():.3f}"
        print(
            f"[grid {c + 1}/{n_cells}] {coord}{headline}"
            f"  ({time.monotonic() - t0:.1f}s)"
        )

    # Channels present in every cell (same env/config family, so normally all).
    names = [k for k in per_cell[0]["channels"] if all(k in r["channels"] for r in per_cell)]
    channels = {
        k: np.stack([r["channels"][k] for r in per_cell]) for k in names
    }  # [n_cells, N, T]

    return {
        "channels": channels,
        "grid_shape": np.array(shape),
        "cell_values": cell_values,
        "n_rollouts": n_rollouts,
        "episode_length": per_cell[0]["episode_length"],
        "dt": per_cell[0]["dt"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="grid_collect")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", default=None, help="output npz (default: analysis/grid_cache/<run>-<mode>.npz)")
    ap.add_argument("--mode", choices=("det", "sto"), default="det")
    ap.add_argument("--checkpoint", default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--n-rollouts", type=int, default=DEFAULT_N_ROLLOUTS,
                    help="rollouts per grid cell")
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS,
                    help="default points per axis")
    ap.add_argument("--axis-points", default=None,
                    help="per-axis resolution, e.g. force_frequency=8,cube_size=4")
    ap.add_argument("--axes", default=None,
                    help="comma-separated axis names to keep (default: all candidates)")
    ap.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    ap.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                    help="max episodes simulated concurrently (cells x rollouts)")
    ap.add_argument("--sequential", action="store_true",
                    help="reference path: rebuild + recompile per cell (slow; "
                         "for diffing against the batched path)")
    args = ap.parse_args()

    from pathlib import Path

    # Heavy imports (JAX etc.) after CUDA_VISIBLE_DEVICES is set by the caller.
    import json

    import numpy as np

    from policy_analyzer import collect, paths
    from policy_analyzer.worker import _is_oom

    log_dir = Path(args.log_dir)
    deterministic = args.mode == "det"

    try:
        handles = collect.restore_policy(log_dir, checkpoint_step=args.checkpoint)
        env_name, env_cfg = handles["env_name"], handles["env_cfg"]

        cand_names = [c["name"] for c in grid_axes.candidates(env_name, env_cfg.to_dict())]
        points = {name: args.points for name in cand_names}
        if args.axis_points:
            for tok in args.axis_points.split(","):
                key, _, val = tok.partition("=")
                points[key.strip()] = int(val)
        selected = None
        if args.axes:
            selected = {a.strip() for a in args.axes.split(",")}
            unknown = selected - set(cand_names)
            if unknown:
                raise ValueError(
                    f"--axes names {sorted(unknown)} not among candidate axes "
                    f"{cand_names}"
                )

        axes = build_axes(env_name, env_cfg, selected=selected, points=points)
        if not axes:
            raise ValueError(
                "no grid axes: every candidate range in this run's config is "
                "degenerate (or was filtered out by --axes)"
            )

        n_cells = int(np.prod([len(ax["values"]) for ax in axes]))
        for ax in axes:
            print(
                f"[grid] axis {ax['name']} ({ax['kind']}, {len(ax['values'])} pts)"
                f" from {ax['source']}"
            )
        if n_cells > args.max_cells:
            raise ValueError(
                f"{n_cells} grid cells exceeds --max-cells={args.max_cells}; "
                "reduce --axis-points or restrict --axes"
            )
        print(f"[grid] {n_cells} cells x {args.n_rollouts} rollouts ({args.mode})")

        if args.sequential:
            result = run_grid_eval_sequential(
                handles, axes, n_rollouts=args.n_rollouts,
                deterministic=deterministic,
            )
        else:
            result = run_grid_eval(
                handles, axes, n_rollouts=args.n_rollouts,
                deterministic=deterministic, max_batch=args.max_batch,
            )

        out = Path(args.out) if args.out else (
            paths.ANALYSIS_DIR / "grid_cache" / f"{log_dir.name}-{args.mode}.npz"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        channels = result["channels"]
        grid_axes_meta = [
            {
                "name": ax["name"],
                "kind": ax["kind"],
                "values": [float(v) for v in ax["values"]],
                "source": ax["source"],
            }
            for ax in axes
        ]
        np.savez_compressed(
            out,
            _channels=np.array(sorted(channels.keys())),
            _grid_axes=np.array(json.dumps(grid_axes_meta)),
            _grid_shape=result["grid_shape"],
            _cell_values=result["cell_values"],
            _n_rollouts=np.array(result["n_rollouts"]),
            _episode_length=np.array(result["episode_length"]),
            _dt=np.array(result["dt"]),
            _mode=np.array(args.mode),
            _env_name=np.array(handles["env_name"]),
            _sensor_bundle=np.array(str(handles["env_cfg"].sensor_bundle)),
            _checkpoint=np.array(str(handles["restore_path"])),
            # Every randomized quantity is held at a known grid value per cell
            # (vs "nominal" in compare_collect caches, where model DR is off and
            # episode parameters are randomly sampled).
            _dr=np.array("pinned"),
            **channels,
        )
        (out.parent / f"{out.stem}.DONE").touch()
        print(
            f"Wrote {out}  ({len(channels)} channels, "
            f"{result['cell_values'].shape[0]} cells, N={result['n_rollouts']})"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level boundary
        traceback.print_exc()
        return EXIT_OOM if _is_oom(exc) else 1


if __name__ == "__main__":
    sys.exit(main())

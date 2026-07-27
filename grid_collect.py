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

Supported envs: TesolloCubePinch, TesolloDownwardsRotateZ.

Output artifact (npz, same channel layout as compare_collect but with a
leading grid-cell axis):
    <name>.npz   channels [n_cells, N, T]; _grid_axes (JSON) describes the
                 axes; _cell_values [n_cells, n_axes] gives each cell's
                 coordinates; _dr = "pinned".

Usage:
    python -m policy_analyzer.grid_collect --log-dir logs/<run> \
        [--out X.npz] [--mode det|sto] [--n-rollouts 8] \
        [--points-episode 7] [--points-model 5] [--axes a,b] [--checkpoint N]
"""

from __future__ import annotations

import argparse
import sys
import traceback

EXIT_OOM = 75  # EX_TEMPFAIL (same contract as compare_collect)

# Envs with grid-axis derivations below. Imported by the analyzer server (which
# never imports numpy/JAX) to decide whether to offer the grid benchmark, so
# keep this module's top level stdlib-only.
SUPPORTED_ENVS = ("TesolloCubePinch", "TesolloDownwardsRotateZ")

# Defaults: episode axes are the quantity under study (finer), model-DR axes
# are robustness context (coarser). Endpoints are always included so the grid
# covers the full trained range, extremes included.
DEFAULT_EPISODE_POINTS = 7
DEFAULT_MODEL_POINTS = 5
DEFAULT_N_ROLLOUTS = 8
DEFAULT_MAX_CELLS = 200


def _linspace(lo: float, hi: float, n: int, endpoint: bool = True):
    import numpy as np

    if lo == hi:
        return np.array([lo])
    return np.linspace(lo, hi, n, endpoint=endpoint)


def build_axes(
    env_name: str, cfg, n_episode: int, n_model: int
) -> list[dict]:
    """Derive grid axes from a run's env config.

    Each axis: {name, kind ("episode"|"model"), values (1-D np.ndarray),
    source (human-readable origin of the range)}.
    """
    import numpy as np

    axes: list[dict] = []

    if env_name == "TesolloCubePinch":
        if cfg.force_target_sinusoid:
            # The sinusoid sweeps force_target_range within the episode; the
            # per-episode random quantity is the phase.
            lo, hi = (float(v) for v in cfg.force_target_phase)
            if lo != hi:
                full_circle = abs((hi - lo) - 2.0 * np.pi) < 1e-9
                axes.append({
                    "name": "force_phase", "kind": "episode",
                    "values": _linspace(lo, hi, n_episode, endpoint=not full_circle),
                    "source": f"config.force_target_phase [{lo}, {hi}]",
                })
        else:
            lo, hi = (float(v) for v in cfg.force_target_range)
            if lo != hi:
                axes.append({
                    "name": "force_target", "kind": "episode",
                    "values": _linspace(lo, hi, n_episode),
                    "source": f"config.force_target_range [{lo}, {hi}]",
                })
        # Model-DR axes: every non-degenerate range in config.domain_rand.
        for key in ("cube_size", "cube_pos", "cube_mass", "actuator_kp"):
            lo, hi = (float(v) for v in cfg.domain_rand[key])
            if lo != hi:
                axes.append({
                    "name": key, "kind": "model",
                    "values": _linspace(lo, hi, n_model),
                    "source": f"config.domain_rand.{key} [{lo}, {hi}]",
                })

    elif env_name == "TesolloDownwardsRotateZ":
        lo = float(cfg.min_target_angle)
        hi = float(cfg.max_target_angle)
        vals = _linspace(lo, hi, n_episode)
        if bool(cfg.symmetric_target):
            # Training samples U(lo, hi) with a random sign; cover both signs.
            vals = np.unique(np.concatenate([-vals[::-1], vals]))
        axes.append({
            "name": "target_angle", "kind": "episode",
            "values": vals,
            "source": (
                f"config.min/max_target_angle [{lo}, {hi}]"
                + (" (symmetric)" if cfg.symmetric_target else "")
            ),
        })
        # Cube-size DR for this env lives in a module constant, not config, so
        # config.json cannot tell us whether the run trained with
        # --domain_randomization. Include the full range; use --axes to drop it
        # for runs known to have trained without DR.
        from mujoco_playground._src.manipulation.tesollo_hand import (
            downwards_rotate_z as _dz,
        )

        dr_lo, dr_hi = _dz._CUBE_SIZE_RANGE
        axes.append({
            "name": "cube_size", "kind": "model",
            "values": _linspace(float(dr_lo), float(dr_hi), n_model),
            "source": f"downwards_rotate_z._CUBE_SIZE_RANGE [{dr_lo}, {dr_hi}]",
        })

    else:
        raise ValueError(
            f"grid_collect does not support env {env_name!r} "
            f"(supported: {', '.join(SUPPORTED_ENVS)})"
        )

    return axes


def _episode_overrides(env_name: str, axis_name: str, value: float) -> dict:
    """Config keys that pin an episode-level axis to `value` (lo == hi ranges)."""
    v = float(value)
    if env_name == "TesolloCubePinch":
        if axis_name == "force_target":
            return {"force_target_range": [v, v]}
        if axis_name == "force_phase":
            return {"force_target_phase": [v, v]}
    if env_name == "TesolloDownwardsRotateZ" and axis_name == "target_angle":
        # symmetric_target off so the signed value is taken literally;
        # curriculum off so reset() uses the min/max band directly.
        return {
            "min_target_angle": v,
            "max_target_angle": v,
            "symmetric_target": False,
            "curriculum.enable": False,
        }
    raise KeyError(f"no episode override mapping for {env_name}/{axis_name}")


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


def run_grid_eval(
    handles: dict,
    axes: list[dict],
    n_rollouts: int,
    deterministic: bool,
) -> dict:
    """Evaluate every grid cell; returns stacked channels + grid metadata.

    Per cell: rebuild the env with the cell's episode parameters pinned via
    config, pin its model-DR parameters, and reuse collect.run_eval_rollouts
    (same fixed seed stream in every cell, so cells are seed-paired).
    """
    import copy
    import itertools
    import time

    import numpy as np

    from mujoco_playground import registry
    from policy_analyzer import collect

    env_name = handles["env_name"]
    base_cfg = handles["env_cfg"]
    shape = tuple(len(ax["values"]) for ax in axes)
    n_cells = int(np.prod(shape))

    cell_values = np.array(
        [
            [float(axes[i]["values"][idx[i]]) for i in range(len(axes))]
            for idx in itertools.product(*(range(s) for s in shape))
        ]
    )  # [n_cells, n_axes]

    per_cell: list[dict] = []
    for c, idx in enumerate(itertools.product(*(range(s) for s in shape))):
        t0 = time.monotonic()
        cfg = copy.deepcopy(base_cfg)
        overrides: dict = {}
        for i, ax in enumerate(axes):
            if ax["kind"] == "episode":
                overrides.update(
                    _episode_overrides(env_name, ax["name"], ax["values"][idx[i]])
                )
        _apply_overrides_strict(cfg, overrides)
        env = registry.load(env_name, config=cfg)
        for i, ax in enumerate(axes):
            if ax["kind"] == "model":
                _pin_model_axis(env, env_name, ax["name"], ax["values"][idx[i]])

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
    ap.add_argument("--points-episode", type=int, default=DEFAULT_EPISODE_POINTS)
    ap.add_argument("--points-model", type=int, default=DEFAULT_MODEL_POINTS)
    ap.add_argument("--axes", default=None,
                    help="comma-separated axis names to keep (default: all derived)")
    ap.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
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
        axes = build_axes(
            handles["env_name"], handles["env_cfg"],
            n_episode=args.points_episode, n_model=args.points_model,
        )
        if args.axes:
            keep = {a.strip() for a in args.axes.split(",")}
            unknown = keep - {ax["name"] for ax in axes}
            if unknown:
                raise ValueError(
                    f"--axes names {sorted(unknown)} not among derived axes "
                    f"{[ax['name'] for ax in axes]}"
                )
            axes = [ax for ax in axes if ax["name"] in keep]
        if not axes:
            raise ValueError(
                "no grid axes: every randomization range in this run's config "
                "is degenerate (or was filtered out by --axes)"
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
                "reduce --points-episode/--points-model or restrict --axes"
            )
        print(f"[grid] {n_cells} cells x {args.n_rollouts} rollouts ({args.mode})")

        result = run_grid_eval(
            handles, axes, n_rollouts=args.n_rollouts, deterministic=deterministic
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

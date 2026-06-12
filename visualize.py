"""Policy analyzer visualizations, built from rollout.npz.

visualize_input_distributions  — grid of per-input histograms
visualize_dof_evolution         — per-DOF candle/area plot over the rollout
visualize_rollout_video         — env render + DOF evolution with scrubbing red line

Schema is always rebuilt live from the env — never persisted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def rebuild_schema(npz) -> dict:
    """Rebuild the I/O schema from the env identified by id_checkpoint in the npz.

    env_name and sensor_bundle are derived from the checkpoint path + config.json
    (the authoritative source in logs/<run>/checkpoints/) rather than being read
    from the npz, keeping the npz minimal.
    """
    from policy_analyzer.collect import load_env_from_checkpoint
    from policy_analyzer import io_schema

    env_name, cfg, env = load_env_from_checkpoint(str(npz["id_checkpoint"]))
    return io_schema.build_io_schema(
        env, env_name=env_name, sensor_bundle=cfg.sensor_bundle
    )


def _ckpt_step(npz) -> str:
    return Path(str(npz["id_checkpoint"])).name if "id_checkpoint" in npz.files else ""


def _suptitle(npz, schema: dict, extra: str = "") -> str:
    env_name = schema["env_name"]
    sb = schema["sensor_bundle"]
    det = bool(npz["id_deterministic"]) if "id_deterministic" in npz.files else True
    mode = "det" if det else "sto"
    parts = [env_name, sb, f"ckpt {_ckpt_step(npz)}", f"T={npz['obs'].shape[0]}", mode]
    if extra:
        parts.append(extra)
    return " / ".join(parts)


def _dof_layout(schema: dict):
    """Return list of (group_key, elements) in output order."""
    return [(g["key"], g["elements"]) for g in schema["output_groups"]]


# ── input distributions ───────────────────────────────────────────────────────

def visualize_input_distributions(
    run_dir: Path, schema: dict | None = None, ncols: int = 8
) -> Path:
    """One combined grid figure of per-input histograms, colored by group."""
    npz = np.load(run_dir / "rollout.npz", allow_pickle=False)
    if schema is None:
        schema = rebuild_schema(npz)

    obs = np.asarray(npz["obs"])
    T, obs_dim = obs.shape

    groups = schema["input_groups"]
    cmap = plt.get_cmap("tab10")
    group_color = {g["key"]: cmap(i % 10) for i, g in enumerate(groups)}

    nrows = math.ceil(obs_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 1.6), squeeze=False)

    cell = 0
    for g in groups:
        for e in g["elements"]:
            ax = axes[cell // ncols][cell % ncols]
            ax.hist(obs[:, e["index"]], bins=20, color=group_color[g["key"]])
            ax.set_title(e["label"], fontsize=6)
            ax.tick_params(labelsize=5)
            cell += 1

    for j in range(cell, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    legend = [Patch(facecolor=group_color[g["key"]], label=g["key"]) for g in groups]
    fig.legend(handles=legend, loc="lower center", ncol=min(len(groups), 7),
               fontsize=7, frameon=False)
    fig.suptitle(_suptitle(npz, schema, "inputs"), fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))

    path = (run_dir / "plots" / "input_distributions.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}  ({obs_dim} input dims)")
    return path


# ── DOF evolution ─────────────────────────────────────────────────────────────

# Threshold where |tanh(x)| > 0.96 — policy output is effectively saturated.
_TANH_SAT = 2.0


def _draw_dof_evolution(
    axes: np.ndarray,
    layout: list,
    pre_squash: np.ndarray,    # [T, action_dim]
    action_scale: np.ndarray,  # [T, action_dim]
    command: np.ndarray,       # [T, action_dim]  motor targets in radians
    dt: float,
    deterministic: bool,
    vline_t: Optional[float] = None,
    ncols_max: int = 6,
    # axes layout: each group occupies two consecutive rows —
    #   row 2k:   pre_squash (loc ± scale, unbounded)
    #   row 2k+1: command    (motor target in radians)
) -> None:
    """Draw DOF evolution into a 2×ngroups × ncols_max axes grid.

    Top row per group: pre_squash space with tanh-saturation dashed lines at ±2.
    Bottom row per group: command (motor target in radians) with per-DOF ctrl_range rails.
    """
    T = pre_squash.shape[0]
    t = np.arange(T) * dt

    SAT_KW = dict(color="tab:orange", linewidth=0.7, linestyle="--", alpha=0.7)
    CTL_KW = dict(color="tab:red",    linewidth=0.7, linestyle="--", alpha=0.7)

    group_row = 0
    for g_key, elements in layout:
        ps_row  = group_row * 2
        cmd_row = group_row * 2 + 1

        for col, e in enumerate(elements):
            if col >= ncols_max:
                break
            i = e["index"]

            # ── pre_squash panel (loc ± scale) ──────────────────────────────
            ax = axes[ps_row, col]
            lo, hi = pre_squash[:, i] - action_scale[:, i], pre_squash[:, i] + action_scale[:, i]
            ax.fill_between(t, lo, hi, alpha=0.3, linewidth=0)
            ax.plot(t, pre_squash[:, i], linewidth=0.8)
            ax.axhline( _TANH_SAT, **SAT_KW)
            ax.axhline(-_TANH_SAT, **SAT_KW)
            ax.set_title(e["label"], fontsize=5)
            ax.tick_params(labelsize=4)
            if vline_t is not None:
                ax.axvline(vline_t, color="red", linewidth=1.2, zorder=10)

            # ── command panel (motor target in radians) ──────────────────────
            ax2 = axes[cmd_row, col]
            ax2.plot(t, command[:, i], linewidth=0.8, color="tab:green")
            if e.get("ctrl_range"):
                lo_r, hi_r = e["ctrl_range"]
                ax2.axhline(lo_r, **CTL_KW)
                ax2.axhline(hi_r, **CTL_KW)
            ax2.tick_params(labelsize=4)
            if vline_t is not None:
                ax2.axvline(vline_t, color="red", linewidth=1.2, zorder=10)

        # blank unused cells in both rows
        for col in range(len(elements), ncols_max):
            axes[ps_row,  col].axis("off")
            axes[cmd_row, col].axis("off")

        group_row += 1

    # row labels on leftmost column
    for group_row, (g_key, _) in enumerate(layout):
        axes[group_row * 2, 0].set_ylabel(
            f"{g_key}\nloc±σ", fontsize=5, rotation=0, labelpad=32, va="center"
        )
        axes[group_row * 2 + 1, 0].set_ylabel(
            "cmd\n(rad)", fontsize=5, rotation=0, labelpad=32, va="center"
        )


def visualize_dof_evolution(run_dir: Path, schema: dict | None = None) -> Path:
    """Two-row-per-group DOF evolution: pre_squash (loc±σ) + command (radians)."""
    npz = np.load(run_dir / "rollout.npz", allow_pickle=False)
    if schema is None:
        schema = rebuild_schema(npz)

    pre_squash   = np.asarray(npz["pre_squash"])
    action_scale = np.asarray(npz["action_scale"])
    command      = np.asarray(npz["command"])
    dt           = float(npz["id_dt"]) if "id_dt" in npz.files else 1.0
    deterministic = bool(npz["id_deterministic"]) if "id_deterministic" in npz.files else True

    layout    = _dof_layout(schema)
    ngroups   = len(layout)
    ncols_max = max(len(els) for _, els in layout)
    nrows     = ngroups * 2  # two rows per group

    fig, axes = plt.subplots(
        nrows, ncols_max,
        figsize=(ncols_max * 2.0, nrows * 1.2),
        squeeze=False,
    )

    _draw_dof_evolution(
        axes, layout, pre_squash, action_scale, command, dt, deterministic,
        ncols_max=ncols_max,
    )

    fig.suptitle(_suptitle(npz, schema, "DOF evolution"), fontsize=10)
    fig.tight_layout()

    path = run_dir / "plots" / "dof_evolution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


# ── synchronized video ────────────────────────────────────────────────────────

class _RenderState:
    """Minimal state-like object accepted by mjx_env.render_array."""
    class _Data:
        __slots__ = ("qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied")
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def __init__(self, **data_kw):
        self.data = self._Data(**data_kw)


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # tostring_argb is the stable Agg API; convert ARGB → RGB
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, 1:]  # drop alpha channel, keep R G B


def visualize_rollout_video(
    run_dir: Path,
    schema: dict | None = None,
    height: int = 360,
    width: int = 480,
    render_every: int = 2,
) -> Path:
    """Composite video: env render (left) + DOF evolution with scrubbing red line (right)."""
    import mujoco
    import mediapy
    from mujoco_playground import registry
    from mujoco_playground._src.mjx_env import render_array

    npz = np.load(run_dir / "rollout.npz", allow_pickle=False)
    if schema is None:
        schema = rebuild_schema(npz)

    env_name      = str(npz["id_env_name"])
    sensor_bundle = str(npz["id_sensor_bundle"])
    pre_squash    = np.asarray(npz["pre_squash"])
    action_scale  = np.asarray(npz["action_scale"])
    command       = np.asarray(npz["command"])
    dt            = float(npz["id_dt"]) if "id_dt" in npz.files else 1.0
    deterministic = bool(npz["id_deterministic"]) if "id_deterministic" in npz.files else True
    T             = pre_squash.shape[0]

    # Build a render-able env (just for its mj_model; no JAX needed)
    cfg = registry.get_default_config(env_name)
    cfg.sensor_bundle = sensor_bundle
    env = registry.load(env_name, config=cfg)

    # Reconstruct per-step render states from stored traj fields
    states = [
        _RenderState(
            qpos        = npz["traj_qpos"][t],
            qvel        = npz["traj_qvel"][t],
            mocap_pos   = npz["traj_mocap_pos"][t],
            mocap_quat  = npz["traj_mocap_quat"][t],
            xfrc_applied= npz["traj_xfrc_applied"][t],
        )
        for t in range(T)
    ]

    import mujoco
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    env_frames = render_array(
        env.mj_model, states[::render_every],
        height=height, width=width,
        scene_option=scene_option,
    )

    layout    = _dof_layout(schema)
    nrows     = len(layout) * 2   # two rows per group: pre_squash + command
    ncols_max = max(len(els) for _, els in layout)
    plot_w    = int(width * 1.2)   # slightly wider for labels
    plot_h    = height

    combined_frames = []
    frame_ts = list(range(0, T, render_every))

    for frame_idx, (env_frame, t_idx) in enumerate(zip(env_frames, frame_ts)):
        vline_t = t_idx * dt

        fig, axes = plt.subplots(
            nrows, ncols_max,
            figsize=(plot_w / 100, plot_h / 100),
            dpi=100,
            squeeze=False,
        )
        _draw_dof_evolution(
            axes, layout, pre_squash, action_scale, command, dt,
            deterministic, vline_t=vline_t, ncols_max=ncols_max,
        )
        fig.suptitle(f"t={vline_t:.2f}s", fontsize=7)
        fig.tight_layout(pad=0.3)

        plot_frame = _fig_to_rgb(fig)
        plt.close(fig)

        # Resize plot frame to exactly (height, plot_w) for compositing
        if plot_frame.shape[:2] != (height, plot_w):
            import PIL.Image
            plot_frame = np.array(
                PIL.Image.fromarray(plot_frame).resize((plot_w, height), PIL.Image.LANCZOS)
            )

        combined = np.concatenate([np.asarray(env_frame), plot_frame], axis=1)
        combined_frames.append(combined)

    fps = 1.0 / (dt * render_every)
    path = run_dir / "plots" / "rollout_video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), combined_frames, fps=fps)
    print(f"Wrote {path}  ({len(combined_frames)} frames @ {fps:.1f} fps)")
    return path

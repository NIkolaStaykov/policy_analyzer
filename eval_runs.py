"""Evaluate one or more training checkpoints and save videos named by run suffix."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import re
from datetime import datetime

from policy_analyzer.paths import MJPG_ROOT

sys.path.insert(0, str(MJPG_ROOT / "learning"))
import brax_compat  # noqa: F401  -- must precede brax imports

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco_playground
from mujoco_playground import registry, wrapper
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)

PROJECT_ROOT = MJPG_ROOT


def _rollout_step(empty_traj, jit_inference_fn, env, carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act = jit_inference_fn(state.obs, act_key)[0]
    state = env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng), (traj_data, state.metrics)


def _do_rollout(step_fn, episode_length, rng, state):
    _, (traj, metrics) = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)
    return traj, metrics


def _make_empty_traj(sample_state):
    empty_data = sample_state.data.__class__(
        **{k: None for k in sample_state.data.__annotations__}
    )
    empty_traj = sample_state.__class__(**{k: None for k in sample_state.__annotations__})
    return empty_traj.replace(data=empty_data)


def find_runs_with_checkpoints(logs_dir: Path) -> list[Path]:
    runs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name != "_queue"],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [
        r for r in runs
        if (r / "checkpoints").exists()
        and any(p.is_dir() for p in (r / "checkpoints").iterdir())
    ]


def extract_suffix(run_name: str) -> str:
    m = re.search(r"\d{8}-\d{6}-(.+)$", run_name)
    return m.group(1) if m else run_name


def extract_env_name(run_name: str) -> str:
    return run_name.split("-")[0]


def _get_ppo_params(env_name: str, impl: str):
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name, impl)
    raise ValueError(f"Unknown env: {env_name}")


def eval_run(log_dir: Path, output_path: Path, num_videos: int = 4) -> bool:
    env_name = extract_env_name(log_dir.name)
    ckpt_dir = log_dir / "checkpoints"

    env_cfg = registry.get_default_config(env_name)
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                try:
                    env_cfg[k] = v
                except Exception:
                    pass  # ConfigDict rejects keys that don't exist in the schema

    ckpt_subdirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )
    if not ckpt_subdirs:
        print(f"No checkpoint dirs in {ckpt_dir}")
        return False
    restore_path = ckpt_subdirs[-1]
    print(f"Restoring from: {restore_path}")

    ppo_params = _get_ppo_params(env_name, env_cfg.get("impl", "warp"))
    ppo_params.num_timesteps = 0

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    training_params = {
        k: v for k, v in dict(ppo_params).items()
        if k not in ("network_factory", "num_eval_envs")
    }

    env = registry.load(env_name, config=env_cfg)
    eval_env = registry.load(env_name, config=env_cfg)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        **training_params,
        network_factory=network_factory,
        seed=1,
        restore_checkpoint_path=restore_path,
        save_checkpoint_path=None,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.get("num_eval_envs", 128),
        progress_fn=lambda *_: None,
        policy_params_fn=lambda *_: None,
        eval_env=eval_env,
    )

    episode_length = ppo_params.episode_length

    rng = jax.random.PRNGKey(1)
    sample_state = jax.jit(eval_env.reset)(rng)
    empty_traj = _make_empty_traj(sample_state)

    det_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    sto_inference_fn = jax.jit(make_inference_fn(params, deterministic=False))

    n_det = (num_videos + 1) // 2
    n_sto = num_videos // 2

    det_step_fn = functools.partial(_rollout_step, empty_traj, det_inference_fn, eval_env)
    sto_step_fn = functools.partial(_rollout_step, empty_traj, sto_inference_fn, eval_env)
    do_det_rollout = jax.jit(functools.partial(_do_rollout, det_step_fn, episode_length))
    do_sto_rollout = jax.jit(functools.partial(_do_rollout, sto_step_fn, episode_length))

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rngs = jax.random.split(rng, num_videos)
    force_keys = {"effective_force", "f_thumb", "f_index"}

    rollout_specs = (
        [("det", i, do_det_rollout) for i in range(n_det)]
        + [("sto", i, do_sto_rollout) for i in range(n_sto)]
    )

    for tag, i, do_rollout in rollout_specs:
        rollout_rng = rngs[i]
        reset_state = jax.jit(eval_env.reset)(rollout_rng)
        traj_stacked, metrics = do_rollout(rollout_rng, reset_state)
        rollout = [jax.tree.map(lambda x, j=j: x[j], traj_stacked) for j in range(episode_length)]

        video_path = output_path.with_name(f"{output_path.stem}_{tag}_{i}.mp4")
        frames = eval_env.render(rollout[::render_every], height=480, width=640, scene_option=scene_option)
        media.write_video(video_path, frames, fps=fps)
        print(f"Video saved: {video_path}")

        if force_keys.issubset(metrics.keys()):
            _save_force_plot(
                path=video_path.with_suffix(".png"),
                metrics=metrics,
                ctrl_dt=env_cfg.ctrl_dt,
                force_target=env_cfg.get("force_target", None),
                force_tolerance=env_cfg.get("force_tolerance", None),
                title=log_dir.name,
            )

    return True


def _save_force_plot(path, metrics, ctrl_dt, force_target, force_tolerance, title):
    eff_force = np.array(metrics["effective_force"])
    f_thumb = np.array(metrics["f_thumb"])
    f_index = np.array(metrics["f_index"])
    timesteps = np.arange(len(eff_force)) * ctrl_dt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, eff_force, label="effective force", linewidth=1.5)
    ax.plot(timesteps, f_thumb, label="f_thumb", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(timesteps, f_index, label="f_index", linewidth=1.0, linestyle=":", alpha=0.7)
    if force_target is not None:
        ax.axhline(force_target, color="red", linestyle="-.", linewidth=1.2,
                   label=f"target ({force_target} N)")
        if force_tolerance is not None:
            ax.axhspan(force_target - force_tolerance, force_target + force_tolerance,
                       alpha=0.1, color="red", label=f"±{force_tolerance} N tolerance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title(f"Pinch Force — {title}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Force plot saved: {path}")


def resolve_runs(logs_dir: Path, args: list[str]) -> list[Path]:
    all_runs = find_runs_with_checkpoints(logs_dir)

    if not args:
        return all_runs[:1]

    if len(args) == 1 and args[0].isdigit():
        return all_runs[: int(args[0])]

    matched = []
    for token in args:
        for run in all_runs:
            if token in run.name and run not in matched:
                matched.append(run)
    return matched


def main():
    args = sys.argv[1:]

    num_videos = 4
    remaining_args = []
    for arg in args:
        if arg.startswith("--num_videos="):
            num_videos = int(arg.split("=", 1)[1])
        else:
            remaining_args.append(arg)

    logs_dir = PROJECT_ROOT / "logs"
    runs = resolve_runs(logs_dir, remaining_args)

    if not runs:
        print("No matching runs with checkpoints found.")
        sys.exit(1)

    n_det = (num_videos + 1) // 2
    n_sto = num_videos // 2

    videos_dir = PROJECT_ROOT / "videos" / datetime.now().strftime("%Y%m%d-%H%M%S")
    videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {videos_dir}")

    results = []
    for run in runs:
        suffix = extract_suffix(run.name)
        output = videos_dir / f"{suffix}.mp4"
        det_names = ", ".join(f"{suffix}_det_{i}.mp4" for i in range(n_det))
        sto_names = ", ".join(f"{suffix}_sto_{i}.mp4" for i in range(n_sto))
        print(f"\n=== {run.name} -> {det_names} | {sto_names} ===", flush=True)
        ok = eval_run(run, output, num_videos=num_videos)
        results.append((run.name, suffix, ok))

    print("\n=== Summary ===")
    print(f"{'Run':<55} {'det':<4} {'sto':<4} {'Status'}")
    print("-" * 75)
    for run_name, suffix, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"{run_name:<55} {n_det:<4} {n_sto:<4} {status}")


if __name__ == "__main__":
    main()

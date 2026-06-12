"""Capture a single policy rollout from a checkpoint.

Reuses the checkpoint-restore + scan machinery from eval_runs.py,
but records the policy's observation inputs and action outputs each step (which
eval_runs does not — it only keeps data.* for video).

Output artifact (under analysis/<timestamp>/):
    rollout.npz   obs [T, obs_dim], action [T, action_dim], reward_terms [T, K]
    meta.json     env_name, sensor_bundle, checkpoint, episode_length, dt, ...
    schema.json   io_schema.build_io_schema(env)
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json

import jax
import jax.numpy as jp
import numpy as np

from policy_analyzer import eval_runs  # noqa: E402  -- pulls in brax_compat + env vars

from mujoco_playground import registry, wrapper  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from policy_analyzer import io_schema  # noqa: E402


def load_env_from_checkpoint(ckpt_str: str):
    """Derive env_name + config + env from an id_checkpoint string in rollout.npz.

    id_checkpoint layout:  .../logs/<run>/checkpoints/<step>
                                          ^^^^^^^^^^^   ← ckpt_dir
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ← log_dir

    env_name comes from the log_dir name (same as restore_policy uses).
    config comes from logs/<run>/checkpoints/config.json (authoritative source —
    not re-stored in the npz per prefer-minimal-derived-artifacts principle).
    """
    ckpt_path = Path(ckpt_str)
    log_dir = ckpt_path.parent.parent   # <step> → checkpoints → log_dir

    env_name = eval_runs.extract_env_name(log_dir.name)
    cfg = registry.get_default_config(env_name)
    config_path = log_dir / "checkpoints" / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                try:
                    cfg[k] = v
                except Exception:
                    pass
    env = registry.load(env_name, config=cfg)
    return env_name, cfg, env


def list_checkpoints(log_dir: Path) -> list[str]:
    """Return checkpoint step strings for a run, sorted ascending by step number."""
    ckpt_dir = log_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(
        [d.name for d in ckpt_dir.iterdir() if d.is_dir()],
        key=lambda s: int(s),
    )


def restore_policy(log_dir: Path, checkpoint_step: str | None = None):
    """Restore env + inference fn from a logs/<run> checkpoint dir.

    Mirrors eval_runs.eval_run's restore path (lines ~100-150) but returns the
    handles instead of rendering video.
    """
    env_name = eval_runs.extract_env_name(log_dir.name)
    ckpt_dir = log_dir / "checkpoints"

    env_cfg = registry.get_default_config(env_name)
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                try:
                    env_cfg[k] = v
                except Exception:
                    pass  # ConfigDict rejects keys absent from the schema

    ckpt_subdirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name)
    )
    if not ckpt_subdirs:
        raise FileNotFoundError(f"No checkpoint dirs in {ckpt_dir}")

    if checkpoint_step and checkpoint_step != "latest":
        restore_path = ckpt_dir / checkpoint_step
        if not restore_path.exists():
            raise FileNotFoundError(
                f"Checkpoint step {checkpoint_step} not found in {ckpt_dir}"
            )
    else:
        restore_path = ckpt_subdirs[-1]
    print(f"Restoring from: {restore_path}")

    ppo_params = eval_runs._get_ppo_params(env_name, env_cfg.get("impl", "warp"))
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

    # Rebuild the same PPONetworks to read the policy's raw pre-tanh output
    # (the Normal mean `loc`). train.py builds it identically from obs_shape +
    # action_size; the restored policy params apply cleanly. See
    # brax/training/agents/ppo/{train,networks}.py.
    sample_obs = jax.jit(eval_env.reset)(jax.random.PRNGKey(0)).obs
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, sample_obs)
    normalize = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", False)
        else (lambda x, y: x)
    )
    ppo_network = network_factory(
        obs_shape, eval_env.action_size, preprocess_observations_fn=normalize
    )

    return {
        "env_name": env_name,
        "env_cfg": env_cfg,
        "eval_env": eval_env,
        "make_inference_fn": make_inference_fn,
        "params": params,
        "ppo_network": ppo_network,
        "ppo_params": ppo_params,
        "restore_path": restore_path,
    }


def _rollout_step(inference_fn, loc_scale_fn, env, carry, _):
    """One step. Records, per timestep:

    obs           — the observation the policy acted on
    action        — post-tanh policy output in [-1, 1]
    pre_squash    — Normal mean (loc), unbounded; saturation shows here
    action_scale  — Normal std (scale = softplus(raw)+0.001), pre-tanh space
    command       — raw action delta = action * action_scale (config scalar)
    traj_*        — MuJoCo data fields needed for rendering
    metrics       — env reward terms / counters
    """
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act = inference_fn(state.obs, act_key)[0]
    loc, scale = loc_scale_fn(state.obs)
    next_state = env.step(state, act)
    out = {
        "obs": state.obs["state"],
        "action": act,
        "pre_squash": loc,
        "action_scale": scale,
        # raw per-actuator delta the env applies to data.ctrl before clipping +
        # EMA (env: delta = action * action_scale). Derived from the config scalar
        # rather than read from info, since the env doesn't expose it there.
        "command": act * env._config.action_scale,
        # absolute EMA-smoothed motor target (always present in info, regardless
        # of whether the obs bundle exposes a motor_targets group)
        "motor_targets": next_state.info["motor_targets"],
        # render fields from the state BEFORE the step (consistent with obs)
        "traj_qpos": state.data.qpos,
        "traj_qvel": state.data.qvel,
        "traj_mocap_pos": state.data.mocap_pos,
        "traj_mocap_quat": state.data.mocap_quat,
        "traj_xfrc_applied": state.data.xfrc_applied,
        "metrics": next_state.metrics,
    }
    return (next_state, rng), out


def _make_loc_scale_fn(ppo_network, params):
    """obs -> (loc, scale) for NormalTanh: scale = softplus(raw) + min_std.

    Matches NormalTanhDistribution.create_dist in brax/training/distribution.py.
    Verified: tanh(loc) reproduces the deterministic action to ~1e-7.
    """
    policy_network = ppo_network.policy_network
    min_std = 0.001

    def loc_scale_fn(obs):
        logits = policy_network.apply(params[0], params[1], obs)
        loc, scale_raw = jp.split(logits, 2, axis=-1)
        scale = (jax.nn.softplus(scale_raw) + min_std)
        return loc, scale

    return loc_scale_fn


def run_single_rollout(handles: dict, seed: int = 1, deterministic: bool = True) -> dict:
    """Run one rollout using pre-restored policy handles.

    Faster than collect_rollout for multiple rollouts from the same checkpoint
    because it skips the expensive restore step.
    """
    eval_env = handles["eval_env"]
    episode_length = handles["ppo_params"].episode_length

    inference_fn = jax.jit(
        handles["make_inference_fn"](handles["params"], deterministic=deterministic)
    )
    loc_scale_fn = jax.jit(_make_loc_scale_fn(handles["ppo_network"], handles["params"]))
    step_fn = functools.partial(_rollout_step, inference_fn, loc_scale_fn, eval_env)

    rng = jax.random.PRNGKey(seed)
    state = jax.jit(eval_env.reset)(rng)
    _, traj = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)

    obs          = np.asarray(traj["obs"])           # [T, obs_dim]
    action       = np.asarray(traj["action"])         # [T, action_dim]
    pre_squash   = np.asarray(traj["pre_squash"])     # [T, action_dim]
    action_scale = np.asarray(traj["action_scale"])   # [T, action_dim]
    command      = np.asarray(traj["command"])         # [T, action_dim]
    motor_targets = np.asarray(traj["motor_targets"])   # [T, action_dim]
    traj_fields  = {k: np.asarray(traj[k]) for k in (
        "traj_qpos", "traj_qvel", "traj_mocap_pos", "traj_mocap_quat", "traj_xfrc_applied"
    )}
    metrics = {k: np.asarray(v) for k, v in traj["metrics"].items()}

    assert obs.shape[1] == eval_env.obs_size, (
        f"obs dim {obs.shape[1]} != env.obs_size {eval_env.obs_size}"
    )
    for name, arr in (("action", action), ("pre_squash", pre_squash),
                      ("action_scale", action_scale), ("command", command),
                      ("motor_targets", motor_targets)):
        assert arr.shape[1] == eval_env.action_size, (
            f"{name} dim {arr.shape[1]} != env.action_size {eval_env.action_size}"
        )

    schema = io_schema.build_io_schema(
        eval_env,
        env_name=handles["env_name"],
        sensor_bundle=handles["env_cfg"].sensor_bundle,
        policy_hidden_layer_sizes=handles["ppo_params"].network_factory.get(
            "policy_hidden_layer_sizes", None
        ),
    )
    identity = {
        "checkpoint": str(handles["restore_path"]),
        "dt": float(eval_env.dt),
        "deterministic": bool(deterministic),
        "seed": int(seed),
    }
    return {
        "obs": obs,
        "action": action,
        "pre_squash": pre_squash,
        "action_scale": action_scale,
        "command": command,
        "motor_targets": motor_targets,
        "metrics": metrics,
        "schema": schema,
        "identity": identity,
        **traj_fields,
    }


def run_batched_rollout(
    handles: dict, seeds, deterministic: bool = True
) -> list[dict]:
    """Run one rollout per seed in a single vmapped pass on one device.

    All seeds share the JIT compile and run as a batched MJX simulation — far
    cheaper than launching a separate process per seed. Returns one dict per
    seed, each shaped exactly like run_single_rollout's output so the existing
    write_artifacts / export_frontend / visualize pipeline works unchanged.
    """
    seeds = [int(s) for s in seeds]
    eval_env = handles["eval_env"]
    episode_length = handles["ppo_params"].episode_length

    inference_fn = handles["make_inference_fn"](handles["params"], deterministic=deterministic)
    loc_scale_fn = _make_loc_scale_fn(handles["ppo_network"], handles["params"])
    step_fn = functools.partial(_rollout_step, inference_fn, loc_scale_fn, eval_env)

    def single(rng):
        state = eval_env.reset(rng)
        _, traj = jax.lax.scan(step_fn, (state, rng), None, length=episode_length)
        return traj

    rngs = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds))   # [N, 2]
    batched = jax.jit(jax.vmap(single))(rngs)                # each leaf [N, T, ...]

    schema = io_schema.build_io_schema(
        eval_env,
        env_name=handles["env_name"],
        sensor_bundle=handles["env_cfg"].sensor_bundle,
        policy_hidden_layer_sizes=handles["ppo_params"].network_factory.get(
            "policy_hidden_layer_sizes", None
        ),
    )

    results = []
    for i, seed in enumerate(seeds):
        sl = jax.tree_util.tree_map(lambda x: np.asarray(x[i]), batched)  # noqa: B023
        obs = sl["obs"]
        assert obs.shape[1] == eval_env.obs_size, (
            f"obs dim {obs.shape[1]} != env.obs_size {eval_env.obs_size}"
        )
        for name in ("action", "pre_squash", "action_scale", "command", "motor_targets"):
            assert sl[name].shape[1] == eval_env.action_size, (
                f"{name} dim {sl[name].shape[1]} != env.action_size {eval_env.action_size}"
            )
        results.append({
            "obs": obs,
            "action": sl["action"],
            "pre_squash": sl["pre_squash"],
            "action_scale": sl["action_scale"],
            "command": sl["command"],
            "motor_targets": sl["motor_targets"],
            "metrics": sl["metrics"],
            "schema": schema,
            "identity": {
                "checkpoint": str(handles["restore_path"]),
                "dt": float(eval_env.dt),
                "deterministic": bool(deterministic),
                "seed": int(seed),
            },
            "traj_qpos": sl["traj_qpos"],
            "traj_qvel": sl["traj_qvel"],
            "traj_mocap_pos": sl["traj_mocap_pos"],
            "traj_mocap_quat": sl["traj_mocap_quat"],
            "traj_xfrc_applied": sl["traj_xfrc_applied"],
        })
    return results


def collect_rollout(log_dir: Path, seed: int = 1, deterministic: bool = True) -> dict:
    """Restore the policy and run one rollout; return numpy arrays + meta + schema."""
    handles = restore_policy(log_dir)
    return run_single_rollout(handles, seed=seed, deterministic=deterministic)


def write_artifacts(out_dir: Path, rollout: dict) -> Path:
    """Write the single durable artifact, rollout.npz (arrays + identity scalars)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = rollout["metrics"]
    metric_keys = sorted(metrics.keys())
    reward_terms = (
        np.stack([metrics[k] for k in metric_keys], axis=1)
        if metric_keys else np.zeros((rollout["obs"].shape[0], 0))
    )
    npz_path = out_dir / "rollout.npz"
    np.savez(
        npz_path,
        obs=rollout["obs"],
        action=rollout["action"],
        pre_squash=rollout["pre_squash"],
        action_scale=rollout["action_scale"],
        command=rollout["command"],
        motor_targets=rollout["motor_targets"],
        reward_terms=reward_terms,
        reward_term_keys=np.array(metric_keys),
        traj_qpos=rollout["traj_qpos"],
        traj_qvel=rollout["traj_qvel"],
        traj_mocap_pos=rollout["traj_mocap_pos"],
        traj_mocap_quat=rollout["traj_mocap_quat"],
        traj_xfrc_applied=rollout["traj_xfrc_applied"],
        # self-describing identity (0-d arrays); everything else is re-derivable.
        **{f"id_{k}": np.array(v) for k, v in rollout["identity"].items()},
    )
    print(f"Wrote {npz_path}")
    return npz_path

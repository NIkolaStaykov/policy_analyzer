"""Core of the frozen-policy force-probe study (isolated bubble).

Given restored policy handles (from policy_analyzer.collect.restore_policy) this:
  1. rolls out the deterministic policy and captures, per step, the policy input
     obs (`state`) paired with the ground-truth fingertip force magnitudes read
     from the SAME pre-step data (so obs_t and force_t are the same instant);
  2. re-derives the frozen policy's hidden activations from the checkpoint params
     (manual swish-MLP forward, validated against the network's own output);
  3. fits a ridge linear probe (closed form) from each representation to the
     force targets and scores it on a held-out split.

Nothing here mutates the env or training code — it only reads sensors/params.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jp
from brax.training import networks as bnet
from brax.training.acme import running_statistics


# ----------------------------------------------------------------------------
# 1. rollout capture: policy input obs + ground-truth fingertip forces
# ----------------------------------------------------------------------------
def _label_fn(env, task: str):
    """Returns f(data, info) -> [k] ground-truth per-finger/per-tip force (N)."""
    if task in ("pinch", "pinch_sinusoid"):
        s0, s1 = env._FINGER_FORCE_SENSORS[0], env._FINGER_FORCE_SENSORS[1]

        def f(data, info):  # 2: thumb, index per-finger contact force (N)
            return jp.stack([
                env._finger_contact_force(data, s0),
                env._finger_contact_force(data, s1),
            ])
        return f

    # downwards_rotate: 5 per-tip contact force magnitudes. _obs_fingertip_forces
    # returns the clean sensor read divided by _TIP_FORCE_SCALE; multiply back to
    # get raw Newtons (scale is irrelevant to R^2 but keeps units interpretable).
    scale = env._TIP_FORCE_SCALE

    def f(data, info):
        return env._obs_fingertip_forces(data, info) * scale
    return f


def capture_rollouts(handles: dict, task: str, n_rollouts: int = 64, seed0: int = 0):
    """Roll out the deterministic policy; return (obs_state[N*T,D], forces[N*T,k])."""
    env = handles["eval_env"]
    T = int(handles["ppo_params"].episode_length)
    inference_fn = handles["make_inference_fn"](handles["params"], deterministic=True)
    label_fn = _label_fn(env, task)

    def one(rng):
        state = env.reset(rng)

        def step_fn(carry, _):
            state, rng = carry
            rng, ak = jax.random.split(rng)
            out = {
                "obs": state.obs["state"],
                "force": label_fn(state.data, state.info),
            }
            act = inference_fn(state.obs, ak)[0]
            return (env.step(state, act), rng), out

        _, traj = jax.lax.scan(step_fn, (state, rng), None, length=T)
        return traj

    rngs = jax.random.split(jax.random.PRNGKey(seed0), n_rollouts)
    traj = jax.jit(jax.vmap(one))(rngs)
    obs = np.asarray(traj["obs"]).reshape(-1, np.asarray(traj["obs"]).shape[-1])
    force = np.asarray(traj["force"]).reshape(-1, np.asarray(traj["force"]).shape[-1])
    return obs, force


# ----------------------------------------------------------------------------
# 2. frozen-policy activations (manual forward, validated)
# ----------------------------------------------------------------------------
def extract_activations(handles: dict, obs_state: np.ndarray):
    """Return {layer_name: activations[N, width]} for the normalized input and
    every hidden layer, plus the max abs error vs the network's own `loc`."""
    proc = handles["params"][0]
    pol = handles["params"][1]["params"]
    sizes = handles["ppo_params"].network_factory.get("policy_hidden_layer_sizes")
    n_hidden = len(sizes)

    x = running_statistics.normalize(jp.asarray(obs_state), bnet.normalizer_select(proc, "state"))
    acts = {"input": np.asarray(x)}
    h = x
    for i in range(n_hidden):
        W, b = pol[f"hidden_{i}"]["kernel"], pol[f"hidden_{i}"]["bias"]
        h = jax.nn.swish(h @ W + b)
        acts[f"hidden_{i}"] = np.asarray(h)

    # validate: manual output head == network loc
    W, b = pol[f"hidden_{n_hidden}"]["kernel"], pol[f"hidden_{n_hidden}"]["bias"]
    A = int(handles["eval_env"].action_size)
    loc_manual = (h @ W + b)[..., :A]
    logits = handles["ppo_network"].policy_network.apply(
        handles["params"][0], handles["params"][1], {"state": jp.asarray(obs_state)}
    )
    loc_net = jp.split(logits, 2, axis=-1)[0]
    val_err = float(jp.max(jp.abs(loc_manual - loc_net)))
    return acts, val_err


# ----------------------------------------------------------------------------
# 3. ridge linear probe (closed form) with a held-out split
# ----------------------------------------------------------------------------
def train_probe(X: np.ndarray, Y: np.ndarray, alpha: float = 10.0,
                val_frac: float = 0.2, seed: int = 0):
    """Fit force = W @ standardized(X) + b via ridge; score on held-out split.

    Returns dict with per-channel and mean val R^2, val RMSE (N), n_train/n_val.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, Xva, Ytr, Yva = X[tr], X[va], Y[tr], Y[va]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu

    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr_c)          # [d, k]
    pred_va = Xva @ W + ymu

    resid = Yva - pred_va
    ss_res = (resid ** 2).sum(0)
    ss_tot = ((Yva - Yva.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt((resid ** 2).mean(0))
    return {
        "r2_per_channel": [round(float(v), 4) for v in r2],
        "r2_mean": round(float(r2.mean()), 4),
        "rmse_per_channel_N": [round(float(v), 4) for v in rmse],
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
    }

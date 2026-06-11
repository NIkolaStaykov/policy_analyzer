"""I/O schema for Tesollo hand policies.

Produces a JSON-serializable description of a policy's observation inputs and
action outputs: ordered named groups, each expanded into named elements with
their offset into the flat obs/action vector. This is the contract shared by the
rollout collector, the data analyzer, and the dashboard frontend.

The schema is derived from the obs registry (obs.py) + the compiled MuJoCo
model, so Python and the env can never disagree about layout.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from mujoco_playground._src.manipulation.tesollo_hand import obs as obs_module


# Actuator-name → action-group mapping. Wrist actuators share one group; each
# finger (dg_1..dg_5) is its own group, named anatomically.
_FINGER_NAMES = {
    "1": "thumb",
    "2": "index",
    "3": "middle",
    "4": "ring",
    "5": "pinky",
}


def _action_group_for(actuator_name: str) -> str:
    """Group key for an actuator, derived from its name."""
    if "wrist" in actuator_name:
        return "wrist"
    m = re.search(r"dg_(\d)_", actuator_name)
    if m and m.group(1) in _FINGER_NAMES:
        return _FINGER_NAMES[m.group(1)]
    return "other"


def _actuator_names(env) -> list[str]:
    """Ordered actuator names straight from the compiled model."""
    model = env.mj_model
    return [model.actuator(i).name for i in range(model.nu)]


# Channels in which an action can be inspected, with their natural bounds.
# - pre_squash: raw network output (the Normal mean `loc`); unbounded.
# - action:     post-tanh policy output, fed to env.step; bounded [-1, 1].
# - command:    clipped motor target in actuator units (radians); bounded by
#               each actuator's ctrlrange (per-element, see output elements).
OUTPUT_CHANNELS = ("pre_squash", "action", "command")


def build_io_schema(
    env,
    *,
    env_name: Optional[str] = None,
    sensor_bundle: Optional[str] = None,
    policy_hidden_layer_sizes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Build the JSON-serializable I/O schema for a Tesollo hand env.

    Args:
      env: an instantiated Tesollo hand env (e.g. PickAndPlace).
      env_name: registry name, for display; defaults to the class name.
      sensor_bundle: obs bundle; defaults to env config's sensor_bundle.
      policy_hidden_layer_sizes: hidden layer widths, recorded so the dashboard
        can draw the decorative MLP. Passed in by the caller (which has the PPO
        config) to keep this module free of training-config imports.

    Returns:
      dict with input_groups / output_groups, each group carrying its start
      offset, size, and per-element labels. Asserts that the input groups tile
      [0, env.obs_size) exactly and the output groups tile [0, env.action_size).
    """
    sensor_bundle = sensor_bundle or env._config.sensor_bundle
    obs_keys = obs_module.resolve_bundle(sensor_bundle) + env._task_obs_keys()

    input_groups = []
    start = 0
    for key in obs_keys:
        comp = obs_module.get(key)
        labels = obs_module.element_labels(key)
        input_groups.append({
            "key": key,
            "description": comp.description,
            "start": start,
            "size": comp.size,
            "elements": [
                {"index": start + i, "label": labels[i]} for i in range(comp.size)
            ],
        })
        start += comp.size

    if start != env.obs_size:
        raise ValueError(
            f"Input groups sum to {start} but env.obs_size is {env.obs_size}."
        )

    # Output groups: walk actuators in order, opening a new contiguous group each
    # time the group key changes. Actuators of one finger/wrist are contiguous in
    # the model, so this yields tidy [start, start+size) slices.
    actuator_names = _actuator_names(env)
    ctrl_range = env.mj_model.actuator_ctrlrange  # [nu, 2]
    output_groups: list[dict[str, Any]] = []
    for idx, name in enumerate(actuator_names):
        gkey = _action_group_for(name)
        if not output_groups or output_groups[-1]["key"] != gkey:
            output_groups.append({
                "key": gkey,
                "description": f"{gkey} actuators",
                "start": idx,
                "size": 0,
                "elements": [],
            })
        g = output_groups[-1]
        g["elements"].append({
            "index": idx,
            "label": name,
            "ctrl_range": [float(ctrl_range[idx, 0]), float(ctrl_range[idx, 1])],
        })
        g["size"] += 1

    total_out = sum(g["size"] for g in output_groups)
    if total_out != env.action_size:
        raise ValueError(
            f"Output groups sum to {total_out} but env.action_size is "
            f"{env.action_size}."
        )

    return {
        "env_name": env_name or type(env).__name__,
        "sensor_bundle": sensor_bundle,
        "obs_dim": int(env.obs_size),
        "action_dim": int(env.action_size),
        "policy_hidden_layer_sizes": (
            list(policy_hidden_layer_sizes)
            if policy_hidden_layer_sizes is not None
            else None
        ),
        "output_channels": list(OUTPUT_CHANNELS),
        "input_groups": input_groups,
        "output_groups": output_groups,
    }

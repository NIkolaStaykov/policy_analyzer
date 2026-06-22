"""Success-rate-vs-threshold computation for the Compare tab (numpy only, no JAX).

Port of experimentation/plot_success_vs_threshold.py onto the analyzer's own
lightweight eval caches (see compare_collect.py / collect.run_eval_rollouts).

For each policy we sweep a threshold on the x-axis and plot, on y, the fraction
of rollouts that count as successful. A rollout succeeds when its chosen error
channel stays below the threshold for `hold_steps` consecutive steps. Eval seeds
(the N rollouts) are pooled into the rate; training-seed replicates are pooled
into a mean line with a min-max band.

A per-env registry (SUCCESS_CONFIGS, mirroring the reference's CONFIGS) supplies
sensible defaults; every field is overridable from the UI.
"""

from __future__ import annotations

import numpy as np

_DEG_PER_RAD = 180.0 / np.pi


# Per-env defaults. `unit_scale` maps a raw channel value to display x-units
# (e.g. radians→degrees); sweep/ref/xlim are all expressed in DISPLAY units.
SUCCESS_CONFIGS: dict[str, dict] = {
    "TesolloDownwardsRotateZ": {
        "channel": "info.ori_error",
        "abs": False,
        "unit_scale": _DEG_PER_RAD,
        "sweep_min": 0.0, "sweep_max": 18.0, "n_points": 151,
        "xlabel": "orientation-error threshold (deg)", "xlim": 10.0,
        "hold_steps": 10,
        "ref_value": 3.0, "ref_label": "train tol (3°)",
    },
    "TesolloCubePinch": {
        "channel": "info.force_error",
        "abs": True,
        "unit_scale": 1.0,
        "sweep_min": 0.0, "sweep_max": 3.0, "n_points": 151,
        "xlabel": "force-error threshold (N)", "xlim": 3.0,
        "hold_steps": 10,
        "ref_value": 0.75, "ref_label": "force tol (0.75 N)",
    },
}

# Used when the env has no registry entry; the UI is expected to pick a channel.
DEFAULT_CONFIG: dict = {
    "channel": "",
    "abs": False,
    "unit_scale": 1.0,
    "sweep_min": 0.0, "sweep_max": 1.0, "n_points": 101,
    "xlabel": "threshold", "xlim": 1.0,
    "hold_steps": 10,
    "ref_value": None, "ref_label": "",
}


def config_for_env(env_name: str, available_channels: list[str] | None = None) -> dict:
    """Default criterion for an env, merged with a default channel guess."""
    cfg = dict(SUCCESS_CONFIGS.get(env_name, DEFAULT_CONFIG))
    if available_channels:
        cfg["available_channels"] = list(available_channels)
        if cfg["channel"] not in available_channels:
            # Fall back to the first error-looking channel, else the first one.
            guess = next((c for c in available_channels if "error" in c), available_channels[0])
            cfg["channel"] = guess
    return cfg


def max_consecutive_below(arr: np.ndarray, thresh: float) -> np.ndarray:
    """Per-row longest run of consecutive steps with arr < thresh.

    arr is [N, T]; returns [N] (vectorised port of the reference's scalar fn).
    """
    below = arr < thresh
    best = np.zeros(arr.shape[0], dtype=int)
    run = np.zeros(arr.shape[0], dtype=int)
    for t in range(arr.shape[1]):
        run = np.where(below[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def _seed_curve(arr: np.ndarray, sweep_raw: np.ndarray, hold_steps: int, use_abs: bool) -> np.ndarray:
    """Success-rate curve (%) over the sweep for one seed run's [N, T] channel."""
    if use_abs:
        arr = np.abs(arr)
    n = arr.shape[0]
    rates = np.empty(sweep_raw.shape[0], dtype=float)
    for i, thr in enumerate(sweep_raw):
        rates[i] = np.count_nonzero(max_consecutive_below(arr, thr) >= hold_steps) / n
    return rates * 100.0


def compute_curves(policies: list[dict], criterion: dict) -> dict:
    """Compute mean ± min/max success curves for each policy.

    policies: [{ "label", "sensor_bundle", "seeds": [ {channels: {name->[N,T]}}... ] }]
              where each seed dict is a loaded cache (channels keyed by name).
    criterion: merged config dict (channel, abs, unit_scale, sweep_*, hold_steps, ...).

    Returns a Plotly-ready dict.
    """
    channel = criterion["channel"]
    use_abs = bool(criterion.get("abs", False))
    unit_scale = float(criterion.get("unit_scale", 1.0)) or 1.0
    hold_steps = int(criterion.get("hold_steps", 10))
    x = np.linspace(
        float(criterion["sweep_min"]),
        float(criterion["sweep_max"]),
        int(criterion["n_points"]),
    )
    sweep_raw = x / unit_scale  # display units → raw channel units for comparison

    out_policies = []
    for pol in policies:
        curves = []
        n_rollouts = 0
        for seed in pol["seeds"]:
            ch = seed["channels"].get(channel)
            if ch is None:
                continue
            arr = np.asarray(ch, dtype=float)
            n_rollouts = max(n_rollouts, arr.shape[0])
            curves.append(_seed_curve(arr, sweep_raw, hold_steps, use_abs))
        if not curves:
            continue
        stack = np.vstack(curves)  # [n_seeds, n_points]
        out_policies.append({
            "label": pol["label"],
            "sensor_bundle": pol.get("sensor_bundle", ""),
            "n_seeds": stack.shape[0],
            "n_rollouts": n_rollouts,
            "mean": stack.mean(axis=0).tolist(),
            "min": stack.min(axis=0).tolist(),
            "max": stack.max(axis=0).tolist(),
        })

    ref = None
    if criterion.get("ref_value") is not None:
        ref = {"value": float(criterion["ref_value"]), "label": criterion.get("ref_label", "")}

    return {
        "x": x.tolist(),
        "xlabel": criterion.get("xlabel", "threshold"),
        "ylabel": "success rate (%)",
        "xlim": criterion.get("xlim"),
        "channel": channel,
        "hold_steps": hold_steps,
        "ref": ref,
        "policies": out_policies,
    }

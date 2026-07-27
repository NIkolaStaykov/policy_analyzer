"""Success-rate-vs-threshold computation for the Compare tab (numpy only, no JAX).

Port of experimentation/plot_success_vs_threshold.py onto the analyzer's own
lightweight eval caches (see compare_collect.py / collect.run_eval_rollouts).

For each policy we sweep a threshold on the x-axis and plot, on y, the fraction
of rollouts that count as successful. Two success criteria are supported
(criterion["success_mode"]):
  - "hold":    the error channel stays below the threshold for `hold_steps`
               consecutive steps.
  - "average": the rollout-averaged error (mean over the steps after the first
               `skip_first` seconds — a settling period) is below the threshold.
  - "average_above": same rollout mean, but success when it is *above* the
               threshold (a success metric where higher is better).
Eval seeds (the N rollouts) are pooled into the rate; training-seed replicates
are pooled into a mean line with a min-max band.

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
        "channel": "reward/success_per_step",
        "abs": False,
        "unit_scale": 1.0,
        "sweep_min": 0.0, "sweep_max": 1.0, "n_points": 151,
        "xlabel": "min success fraction", "xlim": 1.0,
        "success_mode": "average_above",
        "hold_steps": 10,
        "ref_value": None, "ref_label": "",
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
# success_mode selects how a single rollout is judged:
#   "hold"    — error stays below the threshold for `hold_steps` consecutive steps
#   "average" — the rollout-averaged error (over steps after `skip_first` seconds)
#               is below the threshold
DEFAULT_CONFIG: dict = {
    "channel": "",
    "abs": False,
    "unit_scale": 1.0,
    "sweep_min": 0.0, "sweep_max": 1.0, "n_points": 101,
    "xlabel": "threshold", "xlim": 1.0,
    "success_mode": "hold",
    "hold_steps": 10,
    "skip_first": 0.0,   # seconds at the start excluded from the average (settling)
    "ref_value": None, "ref_label": "",
}


def config_for_env(env_name: str, available_channels: list[str] | None = None) -> dict:
    """Default criterion for an env, merged with a default channel guess.

    Layered over DEFAULT_CONFIG so every key (incl. success_mode / skip_first) is
    always present even for envs whose registry entry predates them.
    """
    cfg = {**DEFAULT_CONFIG, **SUCCESS_CONFIGS.get(env_name, {})}
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


def _seed_curve_hold(arr: np.ndarray, sweep_raw: np.ndarray, hold_steps: int, use_abs: bool) -> np.ndarray:
    """Hold-mode success-rate curve (%): error below thr for hold_steps in a row.

    arr is one seed run's [N, T] channel.
    """
    if use_abs:
        arr = np.abs(arr)
    n = arr.shape[0]
    rates = np.empty(sweep_raw.shape[0], dtype=float)
    for i, thr in enumerate(sweep_raw):
        rates[i] = np.count_nonzero(max_consecutive_below(arr, thr) >= hold_steps) / n
    return rates * 100.0


def _seed_curve_average(
    arr: np.ndarray, sweep_raw: np.ndarray, use_abs: bool, skip_steps: int, above: bool = False
) -> np.ndarray:
    """Average-mode success-rate curve (%) on the rollout-averaged channel.

    The per-rollout statistic is the mean of the (optionally abs) channel over
    the steps after the first `skip_steps` (a settling period). A rollout
    succeeds when that mean is below the threshold (error metric, `above=False`)
    or above it (success metric, `above=True`).
    """
    if use_abs:
        arr = np.abs(arr)
    # Keep at least the final step if skip would consume the whole rollout.
    skip = max(0, min(int(skip_steps), arr.shape[1] - 1)) if arr.shape[1] else 0
    means = arr[:, skip:].mean(axis=1)          # [N]
    n = arr.shape[0]
    # success rate at each threshold = fraction of rollouts on the success side
    cmp = (means[None, :] >= sweep_raw[:, None]) if above else (means[None, :] < sweep_raw[:, None])
    rates = cmp.sum(axis=1) / n
    return rates * 100.0


# Display transforms per grid-axis name (grid_collect axes are in raw units).
_GRID_AXIS_DISPLAY = {
    "target_angle": (_DEG_PER_RAD, "target angle (deg)"),
    "force_target": (1.0, "target force (N)"),
    "force_phase": (1.0, "force phase (rad)"),
    "cube_size": (1.0, "cube size (scale)"),
    "cube_pos": (1.0, "cube pos offset (m)"),
    "cube_mass": (1.0, "cube mass (scale)"),
    "actuator_kp": (1.0, "actuator kp (scale)"),
}


def compute_grid_curves(policies: list[dict], criterion: dict) -> dict:
    """Success rate vs the primary grid axis, at a single threshold.

    policies: like compute_curves, but each seed is a grid_collect cache:
        channels {name -> [n_cells, N, T]}, axes (list of {name, kind, values}),
        cell_values [n_cells, n_axes], dt.
    The primary axis is the first one (grid_collect orders episode axes first);
    cells that share a primary-axis value (secondary axes) are averaged. The
    threshold is criterion.ref_value when set, else the sweep-range midpoint —
    plotted rate = fraction of rollouts succeeding at that single threshold.

    Returns the same Plotly-ready shape as compute_curves (x is the axis value,
    not a threshold), plus "title" and "threshold".
    """
    channel = criterion["channel"]
    use_abs = bool(criterion.get("abs", False))
    unit_scale = float(criterion.get("unit_scale", 1.0)) or 1.0
    success_mode = criterion.get("success_mode", "hold")
    hold_steps = int(criterion.get("hold_steps", 10))
    skip_first = float(criterion.get("skip_first", 0.0))  # seconds

    if criterion.get("ref_value") is not None:
        thr_display = float(criterion["ref_value"])
    else:
        thr_display = 0.5 * (
            float(criterion["sweep_min"]) + float(criterion["sweep_max"])
        )
    sweep_raw = np.array([thr_display / unit_scale])

    # Primary axis from the first available seed; all seeds of a policy group
    # come from the same queue config, so their grids agree.
    first_seed = next(
        (s for pol in policies for s in pol["seeds"] if s.get("axes")), None
    )
    if first_seed is None:
        return {"x": [], "policies": [], "xlabel": "", "ylabel": "success rate (%)"}
    axes = first_seed["axes"]
    ax0 = axes[0]
    ax0_vals = np.asarray(ax0["values"], dtype=float)
    xscale, xlabel = _GRID_AXIS_DISPLAY.get(ax0["name"], (1.0, ax0["name"]))

    skip_steps_used = 0
    out_policies = []
    for pol in policies:
        curves = []
        n_rollouts = 0
        for seed in pol["seeds"]:
            ch = seed["channels"].get(channel)
            if ch is None or seed.get("cell_values") is None:
                continue
            arr = np.asarray(ch, dtype=float)          # [n_cells, N, T]
            cell_primary = np.asarray(seed["cell_values"], dtype=float)[:, 0]
            if arr.shape[0] != cell_primary.shape[0]:
                continue
            n_rollouts = max(n_rollouts, arr.shape[1])
            if success_mode in ("average", "average_above"):
                dt = seed.get("dt")
                skip_steps = int(round(skip_first / dt)) if (dt and skip_first > 0) else 0
                skip_steps_used = skip_steps
                above = success_mode == "average_above"
                rates = np.array([
                    _seed_curve_average(arr[c], sweep_raw, use_abs, skip_steps, above)[0]
                    for c in range(arr.shape[0])
                ])
            else:
                rates = np.array([
                    _seed_curve_hold(arr[c], sweep_raw, hold_steps, use_abs)[0]
                    for c in range(arr.shape[0])
                ])
            # Marginalize secondary axes: mean over cells at each primary value.
            curve = np.array([
                rates[np.isclose(cell_primary, v)].mean() for v in ax0_vals
            ])
            curves.append(curve)
        if not curves:
            continue
        stack = np.vstack(curves)  # [n_seeds, n_primary_values]
        out_policies.append({
            "label": pol["label"],
            "sensor_bundle": pol.get("sensor_bundle", ""),
            "n_seeds": stack.shape[0],
            "n_rollouts": n_rollouts,
            "mean": stack.mean(axis=0).tolist(),
            "min": stack.min(axis=0).tolist(),
            "max": stack.max(axis=0).tolist(),
        })

    title = f"Success rate vs {ax0['name']} · thr {thr_display:g}"
    if len(axes) > 1:
        title += " · avg over " + ", ".join(a["name"] for a in axes[1:])

    return {
        "x": (ax0_vals * xscale).tolist(),
        "xlabel": xlabel,
        "ylabel": "success rate (%)",
        "channel": channel,
        "success_mode": success_mode,
        "hold_steps": hold_steps,
        "skip_first": skip_first,
        "skip_steps": skip_steps_used,
        "threshold": thr_display,
        "title": title,
        "ref": None,
        "policies": out_policies,
    }


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
    success_mode = criterion.get("success_mode", "hold")
    hold_steps = int(criterion.get("hold_steps", 10))
    skip_first = float(criterion.get("skip_first", 0.0))  # seconds
    x = np.linspace(
        float(criterion["sweep_min"]),
        float(criterion["sweep_max"]),
        int(criterion["n_points"]),
    )
    sweep_raw = x / unit_scale  # display units → raw channel units for comparison

    skip_steps_used = 0  # for display: skip_first converted via each seed's dt
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
            if success_mode in ("average", "average_above"):
                dt = seed.get("dt")
                skip_steps = int(round(skip_first / dt)) if (dt and skip_first > 0) else 0
                skip_steps_used = skip_steps
                above = success_mode == "average_above"
                curves.append(_seed_curve_average(arr, sweep_raw, use_abs, skip_steps, above))
            else:
                curves.append(_seed_curve_hold(arr, sweep_raw, hold_steps, use_abs))
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
        "success_mode": success_mode,
        "hold_steps": hold_steps,
        "skip_first": skip_first,
        "skip_steps": skip_steps_used,
        "ref": ref,
        "policies": out_policies,
    }

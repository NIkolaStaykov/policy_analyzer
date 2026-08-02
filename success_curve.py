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

import itertools
import warnings

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
    "force_frequency": (1.0, "target frequency (Hz)"),
    "force_amplitude_scale": (1.0, "amplitude scale"),
    "cube_size": (1.0, "cube size (scale)"),
    "cube_pos": (1.0, "cube pos offset (m)"),
    "cube_mass": (1.0, "cube mass (scale)"),
    "actuator_kp": (1.0, "actuator kp (scale)"),
}


def _cell_axis_indices(cell_values: np.ndarray, axes: list[dict]) -> np.ndarray:
    """Map each grid cell's raw coordinates to integer indices into each axis.

    grid_collect stores every cell's raw per-axis value (cell_values [n_cells,
    n_axes]); the heatmap pivots by axis index, so match each coordinate to the
    nearest of that axis's `values` (floats written and re-read, so exact
    equality is unsafe). Returns int [n_cells, n_axes].
    """
    idx = np.empty(cell_values.shape, dtype=int)
    for i, ax in enumerate(axes):
        vals = np.asarray(ax["values"], dtype=float)
        col = cell_values[:, i]
        idx[:, i] = np.argmin(np.abs(col[:, None] - vals[None, :]), axis=1)
    return idx


def _reduce_group(
    arr: np.ndarray, sweep_raw: np.ndarray, success_mode: str,
    hold_steps: int, skip_steps: int, use_abs: bool,
) -> tuple[float, float]:
    """One cell's (success%, mean) from its rollouts [n_i, T]; NaN if empty."""
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    if success_mode in ("average", "average_above"):
        succ = _seed_curve_average(
            arr, sweep_raw, use_abs, skip_steps, success_mode == "average_above"
        )[0]
    else:
        succ = _seed_curve_hold(arr, sweep_raw, hold_steps, use_abs)[0]
    mean = float((np.abs(arr) if use_abs else arr).mean())
    return float(succ), mean


def _nan_to_none(arr: np.ndarray) -> list:
    """List with JSON null for NaN (json.dumps would otherwise emit invalid NaN)."""
    return [None if not np.isfinite(v) else float(v) for v in arr]


def compute_grid_heatmap(policies: list[dict], criterion: dict) -> dict:
    """Per-cell success%/mean for a 2-D heatmap, from EITHER dataset type.

    Two ways a policy's seeds map to cells, chosen per the cache shape:
      - grid cache (channels [n_cells, N, T] + cell_values + axes): each stored
        cell is a group of N pinned rollouts — the pinned sweep.
      - sampled cache (channels [N, T], no cell_values): rollouts are binned into
        a grid by `criterion["heatmap_axes"]` — each axis carries centers plus the
        recorded per-episode value channel, and every rollout drops into the
        nearest-center bin (empty bins → null). Bin edges come from the config
        ranges, so a sampled heatmap lines up cell-for-cell with the grid one.

    Either way each cell reduces to two scalars (success at a single threshold;
    threshold-free mean) laid out over the full axis product, and the browser
    marginalises the unshown axes. Training-seed replicates are averaged per cell.
    """
    channel = criterion["channel"]
    use_abs = bool(criterion.get("abs", False))
    unit_scale = float(criterion.get("unit_scale", 1.0)) or 1.0
    success_mode = criterion.get("success_mode", "hold")
    hold_steps = int(criterion.get("hold_steps", 10))
    skip_first = float(criterion.get("skip_first", 0.0))  # seconds

    # Single judging threshold: explicit UI threshold, else env ref, else midpoint.
    if criterion.get("threshold") is not None:
        thr_display = float(criterion["threshold"])
    elif criterion.get("ref_value") is not None:
        thr_display = float(criterion["ref_value"])
    else:
        thr_display = 0.5 * (
            float(criterion["sweep_min"]) + float(criterion["sweep_max"])
        )
    sweep_raw = np.array([thr_display / unit_scale])
    heatmap_axes = criterion.get("heatmap_axes")  # sampled binning spec, or None

    first_seed = next(
        (s for pol in policies for s in pol["seeds"]
         if s["channels"].get(channel) is not None), None
    )
    if first_seed is None:
        return {"kind": "grid_heatmap", "axes": [], "policies": []}
    grid_mode = first_seed.get("cell_values") is not None and bool(first_seed.get("axes"))

    # Axis definitions (name + raw-unit centers); grid reads them from the cache,
    # sampled from the heatmap_axes spec (which also names the binning channel).
    if grid_mode:
        axis_defs = [
            {"name": ax["name"], "values": np.asarray(ax["values"], float)}
            for ax in first_seed["axes"]
        ]
    else:
        if not heatmap_axes:
            return {"kind": "grid_heatmap", "axes": [], "policies": []}
        axis_defs = [
            {"name": a["name"], "values": np.asarray(a["values"], float),
             "channel": a["channel"], "scale": float(a.get("channel_scale", 1.0))}
            for a in heatmap_axes
        ]

    shape = [len(ax["values"]) for ax in axis_defs]
    n_cells = int(np.prod(shape)) if shape else 0
    if n_cells == 0:
        return {"kind": "grid_heatmap", "axes": [], "policies": []}
    cell_index = np.array(list(itertools.product(*[range(s) for s in shape])))

    axes_out = []
    for ax in axis_defs:
        scale, label = _GRID_AXIS_DISPLAY.get(ax["name"], (1.0, ax["name"]))
        axes_out.append({
            "name": ax["name"], "label": label, "kind": "",
            "values": [float(v) * scale for v in ax["values"]],
        })

    skip_steps_used = 0
    out_policies = []
    for pol in policies:
        succ_seeds, mean_seeds = [], []
        n_rollouts = 0
        for seed in pol["seeds"]:
            ch = seed["channels"].get(channel)
            if ch is None:
                continue
            arr = np.asarray(ch, dtype=float)
            skip_steps = 0
            if success_mode in ("average", "average_above"):
                dt = seed.get("dt")
                skip_steps = int(round(skip_first / dt)) if (dt and skip_first > 0) else 0
                skip_steps_used = skip_steps

            if grid_mode:
                if arr.ndim != 3:
                    continue
                cv = np.asarray(seed["cell_values"], dtype=float)
                if arr.shape[0] != cv.shape[0]:
                    continue
                flat = np.ravel_multi_index(
                    _cell_axis_indices(cv, seed["axes"]).T, shape
                )
                groups = {int(flat[c]): arr[c] for c in range(arr.shape[0])}
                n_rollouts = max(n_rollouts, arr.shape[1])
            else:
                if arr.ndim != 2:
                    continue
                coords = np.empty((arr.shape[0], len(axis_defs)), int)
                ok = True
                for i, ax in enumerate(axis_defs):
                    cc = seed["channels"].get(ax["channel"])
                    if cc is None:
                        ok = False
                        break
                    v = np.asarray(cc, dtype=float)[:, 0] * ax["scale"]
                    coords[:, i] = np.argmin(
                        np.abs(v[:, None] - ax["values"][None, :]), axis=1
                    )
                if not ok:
                    continue
                flat = np.ravel_multi_index(coords.T, shape)
                groups = {c: arr[flat == c] for c in range(n_cells)}
                n_rollouts = max(n_rollouts, arr.shape[0])

            succ = np.full(n_cells, np.nan)
            mean = np.full(n_cells, np.nan)
            for cid, g in groups.items():
                s, m = _reduce_group(g, sweep_raw, success_mode, hold_steps, skip_steps, use_abs)
                succ[cid], mean[cid] = s, m
            succ_seeds.append(succ)
            mean_seeds.append(mean)
        if not succ_seeds:
            continue
        # Average seed replicates per cell, ignoring cells a seed never populated.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-empty cell → NaN
            succ_avg = np.nanmean(np.vstack(succ_seeds), axis=0)
            mean_avg = np.nanmean(np.vstack(mean_seeds), axis=0)
        out_policies.append({
            "label": pol["label"],
            "sensor_bundle": pol.get("sensor_bundle", ""),
            "n_seeds": len(succ_seeds),
            "n_rollouts": n_rollouts,
            "cell_index": cell_index.tolist(),
            "success": _nan_to_none(succ_avg),
            "mean": _nan_to_none(mean_avg),
        })

    mean_label = f"mean |{channel}|" if use_abs else f"mean {channel}"
    return {
        "kind": "grid_heatmap",
        "axes": axes_out,
        "channel": channel,
        "success_mode": success_mode,
        "hold_steps": hold_steps,
        "skip_first": skip_first,
        "skip_steps": skip_steps_used,
        "threshold": thr_display,
        "binned": not grid_mode,
        "metrics": [
            {"key": "success", "label": "success rate (%)", "zmin": 0.0, "zmax": 100.0},
            {"key": "mean", "label": mean_label, "zmin": None, "zmax": None},
        ],
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
            # A grid cache stores [n_cells, N, T]; pool every cell's rollouts into
            # one flat [·, T] distribution so the curve reflects the whole grid.
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])
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

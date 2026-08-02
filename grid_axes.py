"""Manifest of grid-sweepable randomization axes per env (stdlib-only).

Single source of truth shared by two consumers:

  - the analyzer server (``__main__``), which is import-light (NO numpy/JAX) and
    reads a run's ``checkpoints/config.json`` to offer candidate axes as
    checkboxes in the Compare tab;
  - ``grid_collect`` (JAX side), which sweeps the chosen axes and PINS each to a
    single value per grid cell.

Keeping the manifest here — not in the env config — means it works on
already-trained runs with no env-config change: candidacy and ranges are read
from each run's own config.json, while the "which ranges are sweepable and how
to pin each" knowledge lives in one importable place.

Each axis entry:

  name         axis id (matches the npz ``_grid_axes`` name and the
               success_curve ``_GRID_AXIS_DISPLAY`` map used for heatmap labels)
  label        human label for the checkbox selector
  kind         "episode" (pinned via config overrides) | "model" (mjx transform)
  space        "linear" | "log2" (the config range is in log2 units; sweep and
               display are the exp2 of it, i.e. geometric spacing)
  reads        dotted config key holding ``[lo, hi]``            (range source)
  reads_pair   ``(lo_key, hi_key)`` two scalar keys              (alt source)
  range_const  literal ``[lo, hi]`` (range lives in a module constant, not the
               config — e.g. downwards cube_size)               (alt source)
  requires     dotted bool config key gating candidacy; ``"!key"`` negates
  symmetric    dotted bool config key: when true, the swept values are mirrored
               about 0 (training draws U(lo,hi) with a random sign)
  wrap         range spans a full 2*pi circle -> drop the duplicate endpoint
  pin          how to hold the axis at value ``v`` for one cell:
                 episode: {"range": key}         -> set key = [v, v]
                          {"range_pair": (lo, hi)} -> set both scalar keys = v
                          {"scalar": key, "degenerate": range_key}
                              -> set scalar key = v (in real/display units) and
                                 degenerate the range so reset() takes the scalar
                                 branch (needed for frequency/amplitude, whose
                                 reset condition is ``hi > lo``)
                          {"set": {key: value, ...}} merged into every pin
                                 (e.g. symmetric_target False, curriculum off)
                 model:   {"model": transform}   -> grid_collect._pin_model_axis
"""

from __future__ import annotations

# Default per-axis resolution; the UI/CLI can override per axis.
DEFAULT_POINTS = 8

_TWO_PI = 6.283185307179586

MANIFEST: dict[str, list[dict]] = {
    "TesolloCubePinch": [
        # Episode axes — reset() samples these from config ranges.
        {
            "name": "force_target", "label": "target force (N)",
            "kind": "episode", "space": "linear",
            "reads": "force_target_range", "requires": "!force_target_sinusoid",
            "pin": {"range": "force_target_range"},
            "channel": "info.force_target",
        },
        {
            "name": "force_phase", "label": "force phase (rad)",
            "kind": "episode", "space": "linear",
            "reads": "force_target_phase", "requires": "force_target_sinusoid",
            "wrap": True,
            "pin": {"range": "force_target_phase"},
            "channel": "info.force_phase",
        },
        {
            "name": "force_frequency", "label": "target frequency (Hz)",
            "kind": "episode", "space": "log2",
            "reads": "force_target_frequency_log2_range",
            "requires": "force_target_sinusoid",
            "pin": {"scalar": "force_target_frequency",
                    "degenerate": "force_target_frequency_log2_range"},
            "channel": "info.force_frequency",
        },
        {
            "name": "force_amplitude_scale", "label": "amplitude scale",
            "kind": "episode", "space": "linear",
            "reads": "force_target_amplitude_scale_range",
            "requires": "force_target_sinusoid",
            "pin": {"scalar": "force_target_amplitude_scale",
                    "degenerate": "force_target_amplitude_scale_range"},
            "channel": "info.force_amplitude_scale",
        },
        # Model-DR axes — mirror pinch._domain_randomize_impl; ranges in config.
        {
            "name": "cube_size", "label": "cube size (scale)",
            "kind": "model", "space": "linear",
            "reads": "domain_rand.cube_size", "pin": {"model": "cube_size"},
        },
        {
            "name": "cube_pos", "label": "cube pos offset (m)",
            "kind": "model", "space": "linear",
            "reads": "domain_rand.cube_pos", "pin": {"model": "cube_pos"},
        },
        {
            "name": "cube_mass", "label": "cube mass (scale)",
            "kind": "model", "space": "linear",
            "reads": "domain_rand.cube_mass", "pin": {"model": "cube_mass"},
        },
        {
            "name": "actuator_kp", "label": "actuator kp (scale)",
            "kind": "model", "space": "linear",
            "reads": "domain_rand.actuator_kp", "pin": {"model": "actuator_kp"},
        },
    ],
    "TesolloDownwardsRotateZ": [
        {
            "name": "target_angle", "label": "target angle (rad)",
            "kind": "episode", "space": "linear",
            "reads_pair": ("min_target_angle", "max_target_angle"),
            "symmetric": "symmetric_target",
            "pin": {"range_pair": ("min_target_angle", "max_target_angle"),
                    "set": {"symmetric_target": False, "curriculum.enable": False}},
            # Recorded per-rollout goal is logged in degrees; bin back to radians.
            "channel": "curriculum/goal_angle_per_step", "channel_scale": 0.017453292519943295,
        },
        {
            # Range lives in a module constant, not config, so config.json cannot
            # reveal whether the run trained with --domain_randomization. Offer it
            # always (checkbox default-on); uncheck for runs known to be no-DR.
            "name": "cube_size", "label": "cube size (scale)",
            "kind": "model", "space": "linear",
            "range_const": [0.85, 1.15], "pin": {"model": "cube_size"},
        },
    ],
}


def _get(cfg: dict, dotted: str):
    """Walk a dotted key into a plain dict (config.json or ConfigDict.to_dict())."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _requires_ok(entry: dict, cfg: dict) -> bool:
    req = entry.get("requires")
    if not req:
        return True
    negate = req.startswith("!")
    val = bool(_get(cfg, req[1:] if negate else req))
    return (not val) if negate else val


def _read_range(entry: dict, cfg: dict):
    """Return (lo, hi) in DISPLAY/generation units, or None if unreadable.

    For space "log2" the config stores log2 units; we return exp2 of them so both
    the UI hint and grid_collect's geometric sweep operate in real units (Hz).
    """
    if "range_const" in entry:
        lo, hi = entry["range_const"]
    elif "reads_pair" in entry:
        lo_k, hi_k = entry["reads_pair"]
        lo, hi = _get(cfg, lo_k), _get(cfg, hi_k)
    else:
        rng = _get(cfg, entry["reads"])
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            return None
        lo, hi = rng
    if lo is None or hi is None:
        return None
    lo, hi = float(lo), float(hi)
    if entry.get("space") == "log2":
        lo, hi = 2.0 ** lo, 2.0 ** hi
    return lo, hi


def _source(entry: dict) -> str:
    if "range_const" in entry:
        return f"const {entry['range_const']}"
    if "reads_pair" in entry:
        return "config." + "/".join(entry["reads_pair"])
    return f"config.{entry['reads']}"


def candidates(env_name: str, cfg: dict) -> list[dict]:
    """Sweepable axes for a run, resolved against its config.

    Drops axes whose ``requires`` is unmet or whose range is degenerate
    (``lo >= hi``) — a run that trained a parameter at a fixed value offers no
    axis for it. Returned dicts carry the sweep/pin metadata grid_collect needs;
    the server projects them to UI-safe fields before sending to the browser.
    """
    out: list[dict] = []
    for entry in MANIFEST.get(env_name, []):
        if not _requires_ok(entry, cfg):
            continue
        rng = _read_range(entry, cfg)
        if rng is None:
            continue
        lo, hi = rng
        if lo >= hi:
            continue
        out.append({
            "name": entry["name"],
            "label": entry["label"],
            "kind": entry["kind"],
            "space": entry.get("space", "linear"),
            "lo": lo,
            "hi": hi,
            "default_points": DEFAULT_POINTS,
            "wrap": bool(entry.get("wrap", False)),
            "symmetric": bool(entry.get("symmetric") and _get(cfg, entry["symmetric"])),
            "pin": entry["pin"],
            "source": _source(entry),
            # Per-rollout recorded value channel (for binning sampled rollouts
            # into a heatmap); None for model axes, which don't vary in nominal
            # sampled collection (DR off) and so aren't bin-able.
            "channel": entry.get("channel"),
            "channel_scale": float(entry.get("channel_scale", 1.0)),
        })
    return out


# Fields safe to expose to the browser (drops the internal pin recipe). Includes
# wrap/symmetric/channel_scale so the client can generate sampled-heatmap bin
# centers identically to grid_collect._gen_values.
_UI_FIELDS = ("name", "label", "kind", "space", "lo", "hi", "default_points",
              "channel", "channel_scale", "wrap", "symmetric")


def candidates_ui(env_name: str, cfg: dict) -> list[dict]:
    """Candidate axes projected to the fields the Compare-tab selector needs."""
    return [{k: c[k] for k in _UI_FIELDS} for c in candidates(env_name, cfg)]

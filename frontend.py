"""Export interactive frontend artifacts for a rollout.

Artifacts land directly in run_dir (analysis/<suffix>/):
  index.html     interactive viewer (copied from frontend_template.html)
  data.json      schema output_groups + DOF arrays + meta
  frames/        frame_0000.png … frame_NNNN.png (one per timestep)

analysis/index.html is regenerated as a landing page listing all available runs.

Serve the whole analysis/ directory once:
  python -m policy_analyzer --serve 8000
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

_TANH_SAT = 2.0
_TEMPLATE = Path(__file__).parent / "frontend_template.html"


def _render_frames(npz, mj_model, height: int = 360, width: int = 480) -> list:
    """Render all T frames from stored traj fields in the npz."""
    import mujoco
    from mujoco_playground._src.mjx_env import render_array

    T = int(npz["traj_qpos"].shape[0])

    class _RS:
        class _D:
            __slots__ = ("qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied")
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        def __init__(self, **kw):
            self.data = self._D(**kw)

    states = [
        _RS(
            qpos=npz["traj_qpos"][t],
            qvel=npz["traj_qvel"][t],
            mocap_pos=npz["traj_mocap_pos"][t],
            mocap_quat=npz["traj_mocap_quat"][t],
            xfrc_applied=npz["traj_xfrc_applied"][t],
        )
        for t in range(T)
    ]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    return render_array(mj_model, states, height=height, width=width, scene_option=scene_option)


def _load_run_meta(run_dir: Path) -> dict:
    """Best-effort meta dict for a run dir, tried in priority order:
    1. data.json (written by export_frontend — most complete)
    2. rollout.npz id_* scalars (written by collect — always present)
    Falls back gracefully; never raises.
    """
    data_json = run_dir / "data.json"
    if data_json.exists():
        try:
            with open(data_json) as f:
                return json.load(f).get("meta", {})
        except Exception:
            pass

    npz_path = run_dir / "rollout.npz"
    if not npz_path.exists():
        return {}
    try:
        npz = np.load(npz_path, allow_pickle=False)
        ckpt = str(npz["id_checkpoint"]) if "id_checkpoint" in npz.files else ""
        meta: dict = {}
        if ckpt:
            meta["checkpoint"] = ckpt
            # Try to read sensor_bundle from the run's config.json
            config_path = Path(ckpt).parent.parent / "checkpoints" / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    if "sensor_bundle" in cfg:
                        meta["sensor_bundle"] = cfg["sensor_bundle"]
                except Exception:
                    pass
        if "id_deterministic" in npz.files:
            meta["deterministic"] = bool(npz["id_deterministic"])
        if "obs" in npz.files:
            meta["T"] = int(npz["obs"].shape[0])
        elif "pre_squash" in npz.files:
            meta["T"] = int(npz["pre_squash"].shape[0])
        return meta
    except Exception:
        return {}


def _run_tags(meta: dict) -> str:
    """Build the grey detail string from a meta dict."""
    ckpt = (meta.get("checkpoint") or "").replace("\\", "/").split("/")[-1]
    parts = [
        meta.get("sensor_bundle", ""),
        f"ckpt {ckpt}" if ckpt else "",
        f"T={meta['T']}" if "T" in meta else "",
        "det" if meta.get("deterministic") else ("sto" if "deterministic" in meta else ""),
    ]
    return " / ".join(p for p in parts if p)


def update_root_index(analysis_dir: Path) -> None:
    """Regenerate analysis/index.html listing all subdirectories."""
    runs = []
    for d in sorted(analysis_dir.iterdir()):
        if not d.is_dir():
            continue
        meta = _load_run_meta(d)
        npz_path = d / "rollout.npz"
        data_json = d / "data.json"
        mtime = (
            data_json.stat().st_mtime if data_json.exists()
            else npz_path.stat().st_mtime if npz_path.exists()
            else d.stat().st_mtime
        )
        runs.append((d.name, meta, mtime, (d / "index.html").exists()))

    runs.sort(key=lambda r: r[2], reverse=True)

    rows = ""
    for name, meta, _, has_frontend in runs:
        detail = _run_tags(meta)
        rows += f'<li><a href="{name}/">{name}</a>'
        if not has_frontend:
            rows += ' <span class="no-fe">(no frontend)</span>'
        if detail:
            rows += f' <span class="d">{detail}</span>'
        rows += "</li>\n"

    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <title>Policy Analyzer</title>\n"
        "  <style>\n"
        "    body{background:#111827;color:#f3f4f6;font:13px ui-monospace,monospace;padding:2rem}\n"
        "    h1{font-size:13px;color:#9ca3af;margin-bottom:1.2rem}\n"
        "    ul{list-style:none;padding:0}\n"
        "    li{margin:5px 0}\n"
        "    a{color:#60a5fa;text-decoration:none}\n"
        "    a:hover{text-decoration:underline}\n"
        "    .d{color:#6b7280;margin-left:10px;font-size:11px}\n"
        "    .no-fe{color:#9ca3af}\n"
        "  </style>\n"
        "</head>\n<body>\n"
        f"  <h1>Policy Analyzer — {len(runs)} run{'s' if len(runs) != 1 else ''}</h1>\n"
        f"  <ul>\n{rows}  </ul>\n"
        "</body>\n</html>\n"
    )
    (analysis_dir / "index.html").write_text(html)


def export_frontend(
    run_dir: Path,
    schema: Optional[dict] = None,
    render_height: int = 360,
    render_width: int = 480,
    update_index: bool = True,
    env=None,
    env_name: Optional[str] = None,
    cfg=None,
) -> Path:
    """Generate frontend artifacts in run_dir from rollout.npz.

    Also regenerates analysis/index.html (run_dir.parent) as the landing page.

    Pass a preloaded (env, env_name, cfg) to skip rebuilding the env — useful
    when exporting many seeds from the same checkpoint in one process.
    """
    import PIL.Image
    from policy_analyzer.collect import load_env_from_checkpoint
    from policy_analyzer import io_schema as _io

    npz = np.load(run_dir / "rollout.npz", allow_pickle=False)

    if env is None or env_name is None or cfg is None:
        env_name, cfg, env = load_env_from_checkpoint(str(npz["id_checkpoint"]))

    if schema is None:
        schema = _io.build_io_schema(env, env_name=env_name, sensor_bundle=cfg.sensor_bundle)

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    T = int(npz["pre_squash"].shape[0])
    print(f"Rendering {T} frames …")
    frames = _render_frames(npz, env.mj_model, height=render_height, width=render_width)
    for i, frame in enumerate(frames):
        PIL.Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
    print(f"Wrote {T} frames to {frames_dir}")

    obs_arr = np.asarray(npz["obs"])
    # Prefer the directly-captured absolute motor target (always available);
    # fall back to the obs-derived group for older npz files that predate it.
    if "motor_targets" in npz.files:
        motor_targets = np.asarray(npz["motor_targets"]).tolist()
    else:
        mt_group = next((g for g in schema["input_groups"] if g["key"] == "motor_targets"), None)
        motor_targets = (
            obs_arr[:, mt_group["start"]:mt_group["start"] + mt_group["size"]].tolist()
            if mt_group is not None else None
        )

    reward_term_keys = (
        [str(k) for k in npz["reward_term_keys"]]
        if "reward_term_keys" in npz.files else []
    )
    reward_terms_arr = (
        np.asarray(npz["reward_terms"])
        if "reward_terms" in npz.files else np.zeros((T, 0))
    )

    # The env logs RAW reward components in metrics; scale them by their config
    # weights so the plotted magnitudes match each term's actual contribution to
    # the reward. Non-reward diagnostics (e.g. floor_support_fraction) are dropped.
    rc = getattr(cfg, "reward_config", None)
    scales = dict(rc.scales) if rc is not None and hasattr(rc, "scales") else {}
    success_reward = float(getattr(rc, "success_reward", 1.0)) if rc is not None else 1.0

    def _term_scale(key: str):
        if not key.startswith("reward/"):
            return None  # diagnostic metric, not a reward term → exclude
        name = key[len("reward/"):]
        if name.endswith("_per_step"):
            name = name[: -len("_per_step")]
        if name == "success":
            return success_reward
        return float(scales.get(name, 1.0))

    reward_idx, reward_scales, reward_keys = [], [], []
    for i, k in enumerate(reward_term_keys):
        s = _term_scale(k)
        if s is not None:
            reward_idx.append(i)
            reward_scales.append(s)
            reward_keys.append(k)
    if reward_idx:
        scaled_terms = reward_terms_arr[:, reward_idx] * np.asarray(reward_scales)
    else:
        scaled_terms = np.zeros((T, 0))

    data = {
        "meta": {
            "env_name": env_name,
            "sensor_bundle": cfg.sensor_bundle,
            "checkpoint": str(npz["id_checkpoint"]),
            "T": T,
            "dt": float(npz["id_dt"]),
            "deterministic": bool(npz["id_deterministic"]),
            "seed": int(npz["id_seed"]),
            "tanh_sat": _TANH_SAT,
        },
        "input_groups": schema["input_groups"],
        "output_groups": schema["output_groups"],
        "obs": obs_arr.tolist(),
        "dof": {
            "pre_squash":    np.asarray(npz["pre_squash"]).tolist(),
            "action_scale":  np.asarray(npz["action_scale"]).tolist(),
            "command":       np.asarray(npz["command"]).tolist(),
            "motor_targets": motor_targets,
        },
        "rewards": {
            # scaled reward terms only (diagnostics excluded); total = their sum,
            # which is the actual per-step reward.
            "term_keys": reward_keys,
            "total": scaled_terms.sum(axis=1).tolist(),
            "terms": scaled_terms.tolist(),   # [T, R] scaled per-term time series
        },
    }
    data_path = run_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote {data_path}")

    shutil.copy(_TEMPLATE, run_dir / "index.html")

    if update_index:
        update_root_index(run_dir.parent)
    return run_dir

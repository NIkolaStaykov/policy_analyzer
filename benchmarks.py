from __future__ import annotations

from pathlib import Path

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"

DEFAULT_BENCHMARK = "default"


def load_benchmarks() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not BENCHMARKS_DIR.is_dir():
        return out
    for p in sorted(BENCHMARKS_DIR.glob("*.yaml")):
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue
        name = str(data.get("name") or p.stem)
        out[name] = {
            "name": name,
            "label": str(data.get("label", name)),
            "description": str(data.get("description", "")),
            "envs": list(data.get("envs", ["*"])),
            "env_overrides": dict(data.get("env_overrides", {})),
        }
    return out


def get_benchmark(name: str) -> dict | None:
    return load_benchmarks().get(name)


def benchmarks_for_env(env_name: str) -> list[dict]:
    items = [
        b for b in load_benchmarks().values()
        if "*" in b["envs"] or env_name in b["envs"]
    ]
    items.sort(key=lambda b: (b["name"] != DEFAULT_BENCHMARK, b["label"].lower()))
    return items


def overrides_for(name: str) -> dict:
    b = get_benchmark(name)
    return dict(b["env_overrides"]) if b else {}


def apply_overrides(cfg, overrides: dict) -> None:
    for key, val in overrides.items():
        parts = str(key).split(".")
        node = cfg
        try:
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = val
        except Exception:
            pass

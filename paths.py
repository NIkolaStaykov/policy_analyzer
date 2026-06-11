"""Filesystem locations the analyzer depends on.

The analyzer is decoupled from the mujoco_playground repo and lives beside it:

    workspace/
      mujoco_playground/      ← the training repo (logs, eval_runs, env code)
      policy_analyzer/         ← this package

By default we assume that sibling layout. Override with the
MUJOCO_PLAYGROUND_ROOT environment variable to point at the repo elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent

MJPG_ROOT = Path(
    os.environ.get("MUJOCO_PLAYGROUND_ROOT", PKG_DIR.parent / "mujoco_playground")
).resolve()

# Training run outputs live in the mujoco_playground repo.
LOGS_DIR = MJPG_ROOT / "logs"

# Analyzer-owned output (sessions, served static files) lives next to this package.
ANALYSIS_DIR = PKG_DIR / "analysis"

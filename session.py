"""Multi-rollout session model with thread-safe status tracking."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RolloutStatus = Literal["pending", "running", "done", "error"]


@dataclass
class RolloutInfo:
    name: str
    deterministic: bool
    seed: int
    status: RolloutStatus = "pending"
    error: str | None = None
    detail: str | None = None  # transient status text (e.g. "waiting for VRAM")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "detail": self.detail,
        }


@dataclass
class Session:
    session_id: str
    session_dir: Path
    run: str
    checkpoint_step: str  # "latest" or numeric step string
    rollouts: list[RolloutInfo] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "session_id": self.session_id,
                "run": self.run,
                "checkpoint_step": self.checkpoint_step,
                "rollouts": [r.to_dict() for r in self.rollouts],
                "n_done": sum(r.status == "done" for r in self.rollouts),
                "n_total": len(self.rollouts),
            }

    def update_rollout(
        self, name: str, status: RolloutStatus, error: str | None = None
    ) -> None:
        with self._lock:
            for r in self.rollouts:
                if r.name == name:
                    r.status = status
                    r.error = error
                    r.detail = None  # clear transient detail on terminal status
                    break
        self._save()

    def update_rollout_detail(self, name: str, detail: str) -> None:
        """Update transient status text without changing status or saving to disk."""
        with self._lock:
            for r in self.rollouts:
                if r.name == name:
                    r.detail = detail
                    break

    def _save(self) -> None:
        data = self.to_dict()
        (self.session_dir / "session.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

"""GPU worker process for parallel rollout execution.

Each worker is a dedicated subprocess assigned to one GPU (via CUDA_VISIBLE_DEVICES,
set before any CUDA/JAX import). Workers pull tasks from a shared multiprocessing
Queue, cache model handles across rollouts for the same checkpoint, and retry on
VRAM exhaustion with live status reports back to the main process.

Message protocol on result_q:
  (sid, rollout_name, 'done',   None)        — completed successfully
  (sid, rollout_name, 'error',  error_str)   — failed permanently
  (sid, rollout_name, 'detail', detail_str)  — transient status update (still running)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

MIN_FREE_VRAM_GB = 4.0    # minimum VRAM required before attempting a rollout
VRAM_POLL_SECS = 20       # seconds between VRAM checks while waiting
MAX_OOM_RETRIES = 10      # OOM attempts before giving up


def _free_vram_gb(physical_gpu_id: int) -> float:
    """Query free VRAM for a physical GPU index via nvidia-smi (MiB → GiB)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip()) / 1024
    except Exception:
        return 0.0  # conservative: assume no VRAM available


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "out of memory",
            "resource_exhausted",
            "allocation failed",
            "cannot allocate",
            "oom",
            # Warp CUDA module-load failure caused by OOM (cudaErrorMemoryAllocation)
            "failed to load cuda module",
            "cuda error 2",
        )
    )


def _run_with_vram_retry(fn, physical_gpu_id: int, result_q, sid: str, name: str):
    """Call fn(), retrying on OOM after polling until VRAM is sufficient."""
    for attempt in range(MAX_OOM_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_oom(exc) or attempt >= MAX_OOM_RETRIES:
                raise

            free = _free_vram_gb(physical_gpu_id)
            print(
                f"[GPU {physical_gpu_id}] {name}: OOM (attempt {attempt + 1}), "
                f"{free:.1f} GB free — polling until {MIN_FREE_VRAM_GB} GB available",
                flush=True,
            )

            while True:
                free = _free_vram_gb(physical_gpu_id)
                detail = f"waiting for VRAM — {free:.1f} / {MIN_FREE_VRAM_GB} GB"
                result_q.put((sid, name, "detail", detail))
                if free >= MIN_FREE_VRAM_GB:
                    print(
                        f"[GPU {physical_gpu_id}] {name}: VRAM OK ({free:.1f} GB), retrying",
                        flush=True,
                    )
                    break
                time.sleep(VRAM_POLL_SECS)


def main_loop(gpu_id: int, task_q, result_q) -> None:
    """Entry point for a spawned GPU worker process.

    gpu_id is the physical GPU index (0, 1, ...). We do NOT restrict
    CUDA_VISIBLE_DEVICES (the JAX CUDA plugin fails when device 0 is hidden);
    instead we use jax.default_device to pin all JAX ops to the target GPU.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # ── JAX is already imported (fork inherits parent's modules); select device

    import jax
    cuda_devs = jax.devices("cuda")
    if not cuda_devs:
        print(f"[GPU {gpu_id}] no CUDA devices found, exiting", flush=True)
        return
    device = cuda_devs[min(gpu_id, len(cuda_devs) - 1)]

    from policy_analyzer import collect, frontend, visualize

    print(f"[GPU {gpu_id}] worker ready — device {device}", flush=True)

    handles_cache: dict[tuple, dict] = {}

    # Pin all JAX operations in this worker to `device` for its entire lifetime
    with jax.default_device(device):
        while True:
            task = task_q.get()
            if task is None:  # poison pill — shut down
                break

            sid, name, log_dir_str, checkpoint_step, seed, deterministic, rollout_dir_str = task
            rollout_dir = Path(rollout_dir_str)

            # Capture loop variables for the closure
            _sid, _name, _log, _ckpt, _seed, _det, _rdir = (
                sid, name, log_dir_str, checkpoint_step, seed, deterministic, rollout_dir
            )

            def _do():
                key = (_log, _ckpt)
                if key not in handles_cache:
                    ckpt = None if _ckpt == "latest" else _ckpt
                    handles_cache[key] = collect.restore_policy(
                        Path(_log), checkpoint_step=ckpt
                    )
                handles = handles_cache[key]

                rollout = collect.run_single_rollout(
                    handles, seed=_seed, deterministic=_det
                )
                collect.write_artifacts(_rdir, rollout)
                visualize.visualize_input_distributions(_rdir, schema=rollout["schema"])
                visualize.visualize_dof_evolution(_rdir, schema=rollout["schema"])
                frontend.export_frontend(
                    _rdir, schema=rollout["schema"], update_index=False
                )

            try:
                _run_with_vram_retry(_do, gpu_id, result_q, _sid, _name)
                result_q.put((_sid, _name, "done", None))
                print(f"[GPU {gpu_id}] {_name} done", flush=True)
            except Exception as exc:
                traceback.print_exc()
                result_q.put((_sid, _name, "error", str(exc)))

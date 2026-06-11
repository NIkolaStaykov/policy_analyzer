# Bugs to fix in the analyzer frontend
- [x] Rollout evals not running in parallel, but are sequential
      → subprocess-per-rollout scheduler, one per GPU (CUDA_VISIBLE_DEVICES set
        before launch). Verified: 2 rollouts ran concurrently on GPU 0 + GPU 1.
- [x] After a rollout is done it is not immediately updated on the website. A reload of the page shows it correctly
      → API responses now send `Cache-Control: no-store` + cache-buster query param;
        poll re-renders list/aggregate/video grid on status change.
- [x] Individual rollout evals should not show in a separate tab
      → clicking a rollout now opens its analysis inline (iframe) in the session
        main pane via a view switcher, instead of `target="_blank"`.
- [x] There should be a video pane with all the rollouts in a grid in the session view. It should get filled as individual rollout analyses get ready
      → "Videos" view: grid of looping frame animations, one per completed
        rollout, filled in as each finishes (polling adds tiles).
- [x] The 2x2 plot grid for the individual rollout view output analysis is gone. It should include delta to last action & motor target in time on line 1 and error (target - state) and current state on line 2. All for across the whole rollout span and for each of the groups
      → the embedded per-rollout page (frontend_template.html) already builds this
        2×2 grid; it's reachable again now that the view is shown inline (bug 3).
        Confirmed data.json carries motor_targets + joint_pos for the active env.

## Notes
- Launch the server with the uv venv python so subprocess rollouts inherit jax:
  `.venv/bin/python -m experimentation.policy_analyzer --port 8000` (or `uv run`).
  Subprocesses use `sys.executable`, so the launching interpreter must have jax.
- New file: collect_one.py — standalone single-rollout runner (exit 75 = OOM →
  scheduler requeues after VRAM frees; other nonzero = error with stderr tail).

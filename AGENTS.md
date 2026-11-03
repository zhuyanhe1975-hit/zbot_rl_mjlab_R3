# Project memory: read at session start

- Before working in this repository, read `docs/project-memory.md`. It contains the current decisions, user preferences, experiment identities and validation limits.
- Treat live task cards, saved run configs and the user's latest instructions as authoritative; do not restore old values from historical memory.
- Update the memory after meaningful changes, separating verified facts from assumptions and unverified learning claims. Preserve earlier history.
- If the user asks only to read memory and stand by, read it, briefly confirm readiness, and wait. Do not start training, tuning, archiving or other work. If a concrete task is supplied, proceed after reading.

# Development environment policy

Read `docs/development-environment.md` before installing or upgrading dependencies.

- Reuse `~/dev-envs/robotics-mjlab-py313` for compatible robotics / MJLab work. Run this project's Python via `./scripts/python.sh`.
- Do not automatically create `.venv`, Conda environments, or download another PyTorch/CUDA stack. Inspect the shared environment and `~/.cache/uv` first.
- NVIDIA drivers are system-wide. Do not reinstall a working driver or install a system CUDA toolkit unless a concrete workload requires it.
- Use uv's persistent shared cache and hardlinks; do not clear dependency or Warp caches as routine cleanup.
- Before modifying shared dependencies, inspect all environment consumers, run a dry-run, preserve the frozen inventory, and verify compatibility. Create a separate named environment only for demonstrated conflicts; reuse cached packages.
- Never run `uv sync` at this repository root to prepare training: its `pyproject.toml` / `uv.lock` omit the external MJLab runtime.
- Keep experiment selection, hyperparameters and archived checkpoints unchanged during environment maintenance.

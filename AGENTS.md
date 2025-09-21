# Repository Guidelines

## Project Structure & Module Organization
- `apps/` — runnable app, services, UI and utilities; integration tests live in `apps/tests/` and some `apps/test_*.py` scripts.
- `src/` — training/prototyping code (e.g., `src/main.py`, notebooks).
- `configs/` — YOLO/dataset configs (e.g., `configs/**/dataset.yaml`).
- `models/`, `data/` — model artifacts and datasets (via submodules/LFS).
- `docs/` — design notes and guides. `scripts/` — helper scripts.

## Build, Test, and Development Commands
- Install/activate env: `pixi install` then `pixi shell` (or prefix with `pixi run`).
- Run the demo app: `pixi run python apps/main.py`.
- Unit tests (repo default): `pixi run test` (runs `pytest tests/ -v`).
- App/integration tests: `pixi run pytest apps/tests -v` or `pixi run pytest apps -v`.
- Lint/format: `pixi run fmt` (ruff format), `pixi run ruff-check` (ruff fix), `pixi run lint` (pylint), or all: `pixi run style`; full quality gate: `pixi run quality`.

## Coding Style & Naming Conventions
- Python 3.12, PEP 8 aligned. Max line length 88, 4‑space indents, single quotes (see `ruff.toml`, `.pylintrc`, `.editorconfig`).
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `CONSTANT_CASE` for constants. Keep modules small and focused.
- Type hints for public functions/classes. Prefer pure functions and small, testable units.

## Testing Guidelines
- Framework: `pytest`. Place new unit tests in `tests/`; larger integration/vision flows in `apps/tests/`.
- Name tests `test_*.py`; use fixtures for images/paths. Save visual debug outputs under a test‑scoped temp/outputs dir (see `apps/tests/outputs/` pattern).
- Run fast tests locally (`pixi run test`) before pushing; add at least one test per new feature/bugfix.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`, `docs:`, `style:`) as seen in git history.
- PRs must include: clear description, linked issues, test results, and screenshots/sample images when changing CV behavior.
- Keep changes scoped; pass `pixi run quality` before requesting review.

## Environment & Assets
- Large models/datasets use Git LFS and submodules. After clone: `git lfs install && git submodule update --init`.
- GPU backends vary by OS (CUDA on Linux/Windows; MPS/CoreML on macOS); follow `pixi.toml` for platform extras.

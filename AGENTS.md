# Balatro Game Playing AI - Development Guidelines

## Project Overview
Building a Balatro game playing AI using modern Python tooling and best practices. This project leverages computer vision (YOLO object detection) to understand game state and implements AI decision-making for optimal gameplay.

### Why This Tech Stack?
- **Pixi**: Modern package manager that combines conda/mamba with uv, providing better dependency resolution and faster installs
- **YOLO**: State-of-the-art real-time object detection, perfect for identifying game entities
- **ONNX**: Cross-platform inference format for deploying models efficiently across different hardware
- **Git LFS + Submodules**: Efficient handling of large model files while maintaining version control

## Project Structure & Module Organization
- `src/agent/` — main agent implementation (AI, OCR, services, UI, utils)
- `src/clis/` — command-line utilities and tools
- `src/train/` — training/prototyping code
- `configs/` — YOLO/dataset configs (e.g., `configs/**/dataset.yaml`)
- `models/`, `data/` — model artifacts and datasets (via submodules/LFS)
- `notebooks/` — Jupyter notebooks for experiments and benchmarks
- `docs/` — design notes and guides
- `test/` — test data and fixtures
- `runs/` — training run outputs and metrics

## Build, Test, and Development Commands
- Install/activate env: `pixi install` then `pixi shell` (or prefix with `pixi run`)
- Run the demo app: `pixi run python src/agent/main.py`
- Unit tests (repo default): `pixi run test` (runs `pytest tests/ -v`)
- App/integration tests: `pixi run pytest src/agent/tests -v`
- Lint/format: `pixi run fmt` (ruff format), `pixi run ruff-check` (ruff fix), `pixi run lint` (pylint), or all: `pixi run style`; full quality gate: `pixi run quality`

## Package Management (Pixi)
This project uses **pixi** (conda/mamba alternative written in Rust with uv integration) for dependency management.

### Python Execution
```bash
pixi run python <script_name.py|.sh>
```

### Adding Dependencies
- **Pip packages**: `pixi add --pypi <package_name>`
  - Use for Python-only packages from PyPI
- **C++/CUDA packages**: `pixi add <package_name>`
  - Use for system-level dependencies and compiled libraries
- **Platform-specific CUDA** (for Linux/Windows, since macOS uses MPS/CoreML):
  ```bash
  pixi add cuda --platform win-64 --platform linux-64
  ```
  - **Why platform-specific?** macOS doesn't support CUDA, so we exclude it to avoid conflicts
  - **Alternative on macOS**: Uses Metal Performance Shaders (MPS) and CoreML for GPU acceleration

## Development Standards

### Model Implementation
- Use **transformers** and **onnxruntime** for inference
  - **PyTorch models**: Best for CUDA/MPS workflows (high-performance GPU inference)
  - **ONNX models**: Best for WebGPU, AMD GPU, Intel GPU, Intel CPU, broad hardware compatibility
  - **Focus**: Prioritize CUDA/MPS workflow for optimal performance
- **Device Management**: Always implement proper device detection and setup
  - **Universal utility**: Create device detection function with assertions
  - **CUDA support**: Automatic CUDA detection with fallback
  - **MPS support**: Apple Silicon optimization (Metal Performance Shaders)
  - **CoreML**: Use Apple's CoreML libraries for FastVLM models (Apple-optimized)
- Create dedicated test sub-modules with **pytest** capability
- Implement minimal approaches first, then expand
- Wrap implementations in well-defined classes and abstract classes

### Code Structure
- Maintain clean project architecture
- Use abstract base classes for interface definitions
- Create modular, testable components
- No rushing - prioritize engineering best practices

### Platform Considerations & Device Management
- **macOS**: Use MPS (Metal Performance Shaders) for PyTorch, CoreML for FastVLM
- **Linux/Windows**: CUDA support for high-performance GPU inference
- **Device Detection Strategy**:
  - Implement universal device utility function with proper assertions
  - Auto-detect: CUDA → MPS → CPU (in order of preference)
  - Explicit device specification with validation
- **Maximum Compatibility**: Support all device types with appropriate fallbacks

## Coding Style & Naming Conventions
- Python 3.12, PEP 8 aligned. Max line length 88, 4‑space indents, single quotes (see `ruff.toml`, `.pylintrc`, `.editorconfig`)
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `CONSTANT_CASE` for constants. Keep modules small and focused
- Type hints for public functions/classes. Prefer pure functions and small, testable units

## Testing Guidelines
- Framework: `pytest`. Place new unit tests in `tests/`; larger integration/vision flows in `src/agent/tests/`
- Name tests `test_*.py`; use fixtures for images/paths. Save visual debug outputs under a test‑scoped temp/outputs dir
- Run fast tests locally (`pixi run test`) before pushing; add at least one test per new feature/bugfix
- All model implementations must include pytest-capable test scripts
- Test minimal approaches before expanding functionality
- Ensure reproducible testing environments

### Computer Vision & Debugging
- **Image preservation**: Always save/record intermediate images for debugging
  - **Why critical**: CV bugs are often visual - seeing processed images reveals issues instantly
  - **What to save**: Original images, preprocessed images, detection overlays, cropped regions
- **Visual debugging**: Preserve bounding boxes and processing steps
  - **Best practice**: Save images with drawn bounding boxes, confidence scores, class labels
- **Notebook development**: Use `.ipynb` for CV work - better image visualization than terminals
  - **Advantage**: Inline image display, interactive exploration, step-by-step debugging
  - **When to use**: Prototyping, data exploration, model validation, debugging complex pipelines
- **Progressive development**: Notebooks ideal for REPL-style CV development
  - **Workflow**: Develop in notebooks → Extract to modules → Add tests → Integrate

## Environment & Assets

### HuggingFace Integration
- **Models**: Published to HuggingFace under `proj-airi/games-balatro-2024-yolo-entities-detection`
- **Datasets**: Published to HuggingFace under `proj-airi/games-balatro-2024-entities-detection`
- **Git LFS**: HuggingFace repos use Git LFS, integrated as submodules

### Available Models
- **PyTorch**: `models/games-balatro-2024-yolo-entities-detection/model.pt`
- **ONNX**: `models/games-balatro-2024-yolo-entities-detection/onnx/model.onnx`

### Test Data
- **Training images**: `data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images/`
- **Test fixtures**: `test/testdata/` (image-1.png, image-2.png, etc.)
- **Example**: `out_00001.jpg` (many more available)

### Development Setup
```bash
git clone git@github.com:proj-airi/game-playing-ai-balatro.git
git lfs install                # Required for large model files from HuggingFace
git submodule init            # Initialize submodule tracking
git submodule update          # Download HuggingFace model/dataset repos
pixi install                 # Install all dependencies via pixi
```

**Why this workflow?**
- **Git LFS**: HuggingFace repositories contain large model files (hundreds of MB) that need LFS
- **Submodules**: Models and datasets are separate HuggingFace repos, integrated as submodules for version control
- **Pixi install**: Ensures reproducible environment across all team members

Large models/datasets use Git LFS and submodules. After clone: `git lfs install && git submodule update --init`
GPU backends vary by OS (CUDA on Linux/Windows; MPS/CoreML on macOS); follow `pixi.toml` for platform extras

## Pixi Environment Management

### Shell Access
- **Interactive shell**: `pixi shell` (required for direct `python <script>` calls or pip-installed CLIs)
  - **When to use**: Interactive development, using CLI tools like `huggingface-cli`, `yolo`
  - **Why needed**: Activates the pixi environment so tools are in PATH
- **Direct execution**: `pixi run python <script>` (works without shell)
  - **When to use**: Running specific scripts, automated workflows
  - **Advantage**: No need to activate environment, cleaner for CI/CD

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`, `docs:`, `style:`) as seen in git history
- PRs must include: clear description, linked issues, test results, and screenshots/sample images when changing CV behavior
- Keep changes scoped; pass `pixi run quality` before requesting review

## Documentation Standards

### Design Documents & Plans
- **Location**: `docs/ai/designs/`
- **Naming Convention**: `YYYY-MM-DD-kebab-case-description.md`
- **Frontmatter Required**: All design docs must include YAML frontmatter with metadata
- **Metadata Structure**:
  ```yaml
  ---
  title: "Document Title"
  date: "YYYY-MM-DD"
  coding_agents:
    authors: ["Author 1", "Author 2", "Claude Code"]
    project: "proj-airi/game-playing-ai-balatro"
    context: "Brief description of context"
    technologies: ["tech1", "tech2"]
  tags: ["tag1", "tag2"]
  ---
  ```
- **EDIT history**: Maintain a changelog at the bottom of each doc for updates when huge changes occur

### Documentation Rules
- **Essential Context**: Include project background, technical constraints, and decision rationale
- **Living Documents**: Update status and content as implementation progresses
- **Collaborative Attribution**: Always include human collaborators and Claude Code in authors
- **Technology Stack**: List all relevant technologies in frontmatter for searchability

## Engineering Principles
1. **Quality over speed** - No rushing, focus on solid engineering
2. **Modular design** - Well-defined interfaces and abstractions
3. **Test-driven** - Every component should be testable
4. **Cross-platform** - Consider different inference backends
5. **Maintainable** - Clean code structure and documentation
6. **Visual debugging** - Preserve intermediate images for CV tasks
7. **Progressive development** - Use notebooks for exploratory CV work
8. **Documentation-driven** - Comprehensive design docs before implementation
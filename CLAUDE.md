# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Getting Oriented

Always start by running `git status` to understand the current state of the repo. If you're not on `main`, run `git diff main` to see what changes have been made on the current branch before making modifications.

## Project Overview

AI Toolkit (by Ostris) is a training suite for fine-tuning diffusion models (image and video) on consumer-grade hardware. It supports CLI, Gradio UI, and a Next.js web UI. Version 0.7.20, Python >=3.11.

## uv (Python package manager)

This project uses `uv` for dependency management. Key conventions:
- **Run Python**: `uv run python ...` (never bare `python` outside the .venv)
- **Install deps**: `uv sync` (not `pip install`)
- **Add a dependency**: `uv add <package>` (updates pyproject.toml + uv.lock)
- **Add a dev dependency**: `uv add --group dev <package>`
- **Remove a dependency**: `uv remove <package>`
- **Lock without installing**: `uv lock`

## Common Commands

### Training (CLI)
```bash
uv run python run.py config/train_lora_flux_24gb.yaml
uv run python run.py config/some_config.yaml --recover --name my_run --log output.log
```

### Gradio UI (captioning + training)
```bash
uv run python flux_train_ui.py
```

### Web UI
```bash
cd ui && npm run build_and_start   # http://localhost:8675
```

### Dependencies (uv)
```bash
uv sync                            # Install all dependencies
uv sync --group dev                # Include dev tools (pytest, black, ruff)
```

### Code Quality
```bash
uv run black --line-length 120 .
uv run ruff check .                # Checks: E, F, W, I rules
```

### Tests
```bash
uv run pytest testing/
```

### Utility Scripts
```bash
uv run python scripts/convert_diffusers_to_comfy.py   # ComfyUI format conversion
uv run python scripts/extract_lora_from_flex.py       # LoRA extraction
uv run python scripts/convert_lora_to_peft_format.py  # PEFT format conversion
```

## Architecture

### Execution Flow
```
run.py → toolkit.job.get_job(config) → Job.run() → Process.process()
```

All behavior is driven by YAML config files. The config system supports environment variable substitution (`${VAR}`) and token replacement (`[name]`, `[trigger]`).

### Job System (`jobs/`)
- **BaseJob** → TrainJob, GenerateJob, ExtractJob, ModJob, ExtensionJob
- Each job contains one or more **processes** executed sequentially
- Process types: BaseTrainProcess, TrainFineTuneProcess, GenerateProcess, ExtractLoconProcess

### Core Library (`toolkit/`)
- `stable_diffusion_model.py` — Central model handling (loading, inference, LoRA/LoKr/IP-Adapter)
- `config_modules.py` — Pydantic-based config validation and schemas
- `dataloader_mixins.py` — Data loading with aspect-ratio bucketing
- `pipelines.py` — Custom diffusers pipelines and schedulers
- `train_tools.py` / `train_pipelines.py` — Training loop utilities
- `saving.py` — Checkpoint and model saving
- `accelerator.py` — Hardware acceleration via HuggingFace Accelerate

### Extension System (`extensions_built_in/`)
Plugin architecture with auto-discovery. Major extensions:
- **sd_trainer/** — Main SD/SDXL/Flux training (SDTrainer, DiffusionTrainer)
- **diffusion_models/** — Model-specific implementations (Flux, WAN, LTX2, Chroma, etc.)
- **ultimate_slider_trainer/** — Concept slider training
- **advanced_generator/** — Reference/Img2Img/PureLora generation
- **dataset_tools/** — SuperTagger, dataset sync utilities

### Model Support
- **Text-to-Image**: Flux.1-dev/schnell, SD 1.5, SDXL, SD3, PixArt
- **Video**: LTX-2, WAN 2.2/2.1
- **Edit/Other**: Qwen Image Edit, OmniGen2, Chroma, HiDream

### Web UI (`ui/`)
Next.js + TypeScript + TailwindCSS frontend for job creation, monitoring, dataset management, and training configuration.

## Key Design Patterns

1. **Config-driven**: All training/generation parameters live in YAML configs (see `config/examples/`)
2. **Process pipeline**: Jobs compose multiple sequential processes
3. **Memory-aware**: Gradient checkpointing, quantization (bitsandbytes, optimum-quanto), low-VRAM modes
4. **Checkpoint recovery**: `--recover` flag resumes from latest saved step
5. **Mixed precision**: bfloat16/float16 per-model configuration

## Adding New Model Support

1. Create a directory under `extensions_built_in/diffusion_models/`
2. Implement model loading, forward pass, and training hooks
3. Register in the extension discovery system
4. Add example config in `config/examples/`

## PyTorch CUDA Configuration

The project uses `pyproject.toml` with `[tool.uv]` index configuration for PyTorch CUDA builds. Currently defaults to cu128 (RTX 50-series Blackwell support). Configurable for cu126, cu124, or CPU variants.

## Code Style

- Black formatter, 120 char line length
- Ruff linting (E, F, W, I)
- Python 3.11+ target

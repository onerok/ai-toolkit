# AI Toolkit commands

# List available commands
default:
    @just --list

# Check environment and warn about common issues
check:
    #!/usr/bin/env bash
    echo "Checking environment..."
    # Check for WSL2 nvidia-smi PATH issue
    if grep -q microsoft /proc/version 2>/dev/null; then
        if ! command -v nvidia-smi &>/dev/null; then
            if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
                echo "⚠️  WSL2 detected: nvidia-smi found at /usr/lib/wsl/lib/ but not in PATH"
                echo "   Add to your shell config: fish_add_path /usr/lib/wsl/lib"
                echo "   (GPU monitoring in UI won't work until fixed)"
            else
                echo "⚠️  WSL2 detected but nvidia-smi not found - GPU monitoring unavailable"
            fi
        else
            echo "✓ nvidia-smi found"
        fi
    else
        if command -v nvidia-smi &>/dev/null; then
            echo "✓ nvidia-smi found"
        else
            echo "⚠️  nvidia-smi not found - GPU monitoring in UI unavailable"
        fi
    fi

# First-time setup: install Python and Node dependencies, initialize database
setup: check
    uv sync
    cd ui && npm install
    cd ui && npx prisma generate
    cd ui && npx prisma db push

# Start the UI and worker (accessible from network)
start:
    cd ui && npx concurrently --restart-tries -1 --restart-after 1000 -n WORKER,UI "node dist/cron/worker.js" "npx next start --port 8675 --hostname 0.0.0.0"

# Development mode
dev:
    cd ui && npm run dev

# Build the project
build:
    cd ui && npm run build

# Build and start
build-and-start: build start

# Clean the UI build artifacts
clean:
    rm -rf ui/.next ui/dist

# Clean and rebuild
clean-build: clean build

# =============================================================================
# Performance Benchmarking (Phase 0 Test Harness)
# =============================================================================

# Download test dataset for benchmarking (naruto-blip-captions subset)
perf-dataset count="20":
    uv run python scripts/download_perf_dataset.py --count {{count}}

# Quick memory-only benchmark (no dataset needed, ~5 sec)
perf-memory:
    uv run python scripts/perf_benchmark.py --memory-only

# Run benchmark with training (requires datasets/perf_test/)
perf-train steps="20" save_at="10" clipscore_threshold="0.20":
    #!/usr/bin/env bash
    steps_val="{{steps}}"; steps_val="${steps_val#*=}"
    save_at_val="{{save_at}}"; save_at_val="${save_at_val#*=}"
    clipscore_threshold_val="{{clipscore_threshold}}"; clipscore_threshold_val="${clipscore_threshold_val#*=}"
    uv run python scripts/perf_benchmark.py --steps "${steps_val}" --save-at "${save_at_val}" --clean-output --validate-generation --clipscore-threshold "${clipscore_threshold_val}"

# Capture baseline with a tag (memory-only)
perf-baseline tag:
    #!/usr/bin/env bash
    tag_val="{{tag}}"; tag_val="${tag_val#*=}"
    uv run python scripts/collect_baselines.py --tag "${tag_val}" --memory-only

# Capture baseline with training (requires datasets/perf_test/)
perf-baseline-train tag steps="20" save_at="10" clipscore_threshold="0.20":
    #!/usr/bin/env bash
    tag_val="{{tag}}"; tag_val="${tag_val#*=}"
    steps_val="{{steps}}"; steps_val="${steps_val#*=}"
    save_at_val="{{save_at}}"; save_at_val="${save_at_val#*=}"
    clipscore_threshold_val="{{clipscore_threshold}}"; clipscore_threshold_val="${clipscore_threshold_val#*=}"
    #uv run python scripts/collect_baselines.py --tag "${tag_val}" --steps "${steps_val}" --save-at "${save_at_val}" --clean-output --validate-generation --clipscore-threshold "${clipscore_threshold_val}"
    uv run python scripts/collect_baselines.py --tag "${tag_val}" --steps "${steps_val}" --save-at "${save_at_val}" --clean-output 

# Compare current state against a saved baseline
perf-compare baseline:
    #!/usr/bin/env bash
    baseline_val="{{baseline}}"; baseline_val="${baseline_val#*=}"
    uv run python scripts/compare_baselines.py --baseline "${baseline_val}" --current --memory-only

# Compare two saved baselines
perf-diff baseline1 baseline2:
    #!/usr/bin/env bash
    baseline1_val="{{baseline1}}"; baseline1_val="${baseline1_val#*=}"
    baseline2_val="{{baseline2}}"; baseline2_val="${baseline2_val#*=}"
    uv run python scripts/compare_baselines.py --baseline "${baseline1_val}" --compare "${baseline2_val}"

# List saved baselines
perf-list:
    @ls -la docs/perf/baselines/*.json 2>/dev/null || echo "No baselines found in docs/perf/baselines/"

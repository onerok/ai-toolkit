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

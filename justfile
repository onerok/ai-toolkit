# AI Toolkit commands

# List available commands
default:
    @just --list

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

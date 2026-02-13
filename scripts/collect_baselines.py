#!/usr/bin/env python3
"""
Baseline Collection Script for AI Toolkit Performance Testing

Captures performance metrics and saves them as a tagged baseline JSON file.
Use this before and after each phase to track improvements.

Usage:
    # Capture baseline with tag
    uv run python scripts/collect_baselines.py --tag "main-baseline"

    # Capture with custom config
    uv run python scripts/collect_baselines.py --tag "phase-1-before" --config config/perf_test.yaml

    # Specify output directory
    uv run python scripts/collect_baselines.py --tag "phase-2-before" --output docs/perf/baselines/

    # Quick memory-only baseline
    uv run python scripts/collect_baselines.py --tag "memory-test" --memory-only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.perf_benchmark import (
    run_training_benchmark,
    run_memory_only_benchmark,
    get_git_commit,
    print_result_summary,
)


def get_git_status() -> dict:
    """Get git status info for reproducibility."""
    info = {
        "commit": get_git_commit(),
        "branch": "",
        "dirty": False,
    }

    try:
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

    except Exception:
        pass

    return info


def main():
    parser = argparse.ArgumentParser(description="Collect performance baseline")

    parser.add_argument("--tag", "-t", type=str, required=True,
                        help="Tag for this baseline (e.g., 'phase-1-before')")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to training config YAML")
    parser.add_argument("--output", "-o", type=str, default="docs/perf/baselines/",
                        help="Output directory for baseline JSON")
    parser.add_argument("--steps", "-s", type=int, default=20,
                        help="Number of training steps (default: 20)")
    parser.add_argument("--resolution", "-r", type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument("--save-at", type=int, default=None,
                        help="Step to trigger checkpoint save (for VRAM leak testing)")
    parser.add_argument("--memory-only", action="store_true",
                        help="Run memory-only test (faster, no dataset needed)")
    parser.add_argument("--notes", type=str, default="",
                        help="Optional notes to include in baseline")

    args = parser.parse_args()

    print(f"Collecting baseline: {args.tag}")
    print(f"Output: {args.output}")

    # Get git info
    git_info = get_git_status()
    print(f"Git: {git_info['branch']} @ {git_info['commit']}" +
          (" (dirty)" if git_info['dirty'] else ""))

    if git_info['dirty']:
        print("\nWARNING: Working directory has uncommitted changes!")
        print("Consider committing before capturing baseline for reproducibility.\n")

    # Run benchmark
    if args.memory_only:
        result = run_memory_only_benchmark()
    else:
        result = run_training_benchmark(
            config_path=args.config,
            steps=args.steps,
            resolution=args.resolution,
            save_at=args.save_at,
        )

    result.tag = args.tag

    # Add git info to result
    result_dict = result.to_dict()
    result_dict["git"] = git_info
    if args.notes:
        result_dict["notes"] = args.notes

    # Print summary
    print_result_summary(result)

    # Save baseline
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename: tag_commit_timestamp.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = args.tag.replace("/", "-").replace(" ", "_")
    filename = f"{safe_tag}.json"
    output_path = output_dir / filename

    # Check if file exists
    if output_path.exists():
        print(f"\nBaseline already exists: {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            # Save with timestamp
            filename = f"{safe_tag}_{timestamp}.json"
            output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nBaseline saved to: {output_path}")

    # Also save a "latest" symlink for convenience
    latest_path = output_dir / "latest.json"
    if latest_path.is_symlink():
        latest_path.unlink()
    try:
        latest_path.symlink_to(filename)
        print(f"Latest symlink: {latest_path} -> {filename}")
    except OSError:
        # Symlinks may not work on all systems
        pass

    return result_dict


if __name__ == "__main__":
    main()

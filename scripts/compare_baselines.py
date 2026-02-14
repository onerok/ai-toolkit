#!/usr/bin/env python3
"""
Baseline Comparison Script for AI Toolkit Performance Testing

Compares two baseline JSON files and reports improvements/regressions.
Use after implementing a phase to verify improvements.

Usage:
    # Compare saved baseline against current run
    uv run python scripts/compare_baselines.py \
        --baseline docs/perf/baselines/phase-1-before.json \
        --current

    # Compare two saved baselines
    uv run python scripts/compare_baselines.py \
        --baseline docs/perf/baselines/phase-1-before.json \
        --compare docs/perf/baselines/phase-1-after.json

    # With custom thresholds
    uv run python scripts/compare_baselines.py \
        --baseline docs/perf/baselines/main.json \
        --current \
        --vram-threshold 10 \
        --time-threshold 10
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Threshold:
    """Regression threshold for a metric."""
    name: str
    direction: str  # "lower_better" or "higher_better"
    warning_pct: float  # Percentage change to warn
    error_pct: float  # Percentage change to fail
    absolute_max: Optional[float] = None  # Absolute value that's always bad


# Default thresholds
DEFAULT_THRESHOLDS = {
    "peak_vram_mb": Threshold("Peak VRAM", "lower_better", 5.0, 10.0),
    "post_save_vram_mb": Threshold("Post-Save VRAM", "lower_better", 5.0, 10.0),
    "leaked_mb": Threshold("VRAM Leaked", "lower_better", 50.0, 100.0, absolute_max=100.0),
    "avg_step_time_s": Threshold("Avg Step Time", "lower_better", 5.0, 10.0),
    "post_save_step_time_s": Threshold("Post-Save Step Time", "lower_better", 5.0, 10.0),
    "slowdown_ratio": Threshold("Slowdown Ratio", "lower_better", 5.0, 10.0, absolute_max=1.10),
    "num_alloc_retries": Threshold("Alloc Retries", "lower_better", 50.0, 100.0),
    "fragmentation_ratio": Threshold("Fragmentation", "lower_better", 20.0, 50.0),
}


def load_baseline(path: str) -> dict:
    """Load baseline JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_change(baseline: float, current: float, direction: str) -> Tuple[float, str]:
    """
    Compute percentage change and improvement direction.

    Returns:
        (pct_change, status) where status is "improved", "regressed", or "same"
    """
    if baseline == 0:
        if current == 0:
            return 0.0, "same"
        return float("inf"), "regressed" if direction == "lower_better" else "improved"

    pct = ((current - baseline) / abs(baseline)) * 100

    if abs(pct) < 0.5:  # Less than 0.5% change
        status = "same"
    elif direction == "lower_better":
        status = "improved" if pct < 0 else "regressed"
    else:
        status = "improved" if pct > 0 else "regressed"

    return pct, status


def format_metric(name: str, baseline: float, current: float, threshold: Threshold) -> dict:
    """Format a metric comparison."""
    pct_change, status = compute_change(baseline, current, threshold.direction)

    # Check absolute threshold
    if threshold.absolute_max is not None and current > threshold.absolute_max:
        status = "regressed"
        level = "FAIL"
    elif status == "regressed":
        if abs(pct_change) >= threshold.error_pct:
            level = "FAIL"
        elif abs(pct_change) >= threshold.warning_pct:
            level = "WARN"
        else:
            level = "OK"
    else:
        level = "OK"

    # Format values nicely
    if "time" in name.lower():
        baseline_str = f"{baseline:.3f}s"
        current_str = f"{current:.3f}s"
    elif "ratio" in name.lower():
        baseline_str = f"{baseline:.2f}x"
        current_str = f"{current:.2f}x"
    elif "mb" in name.lower() or "vram" in name.lower():
        baseline_str = f"{baseline:.0f} MB"
        current_str = f"{current:.0f} MB"
    else:
        baseline_str = f"{baseline:.2f}"
        current_str = f"{current:.2f}"

    return {
        "name": threshold.name,
        "baseline": baseline_str,
        "current": current_str,
        "pct_change": pct_change,
        "status": status,
        "level": level,
    }


def compare_baselines(baseline: dict, current: dict, thresholds: Dict[str, Threshold] = None) -> dict:
    """Compare two baselines and return comparison report."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    base_metrics = baseline.get("metrics", {})
    curr_metrics = current.get("metrics", {})

    comparisons = []
    has_warnings = False
    has_failures = False

    for key, threshold in thresholds.items():
        base_val = base_metrics.get(key, 0)
        curr_val = curr_metrics.get(key, 0)

        comp = format_metric(key, base_val, curr_val, threshold)
        comparisons.append(comp)

        if comp["level"] == "WARN":
            has_warnings = True
        elif comp["level"] == "FAIL":
            has_failures = True

    return {
        "baseline_tag": baseline.get("tag", "unknown"),
        "baseline_commit": baseline.get("commit", baseline.get("git", {}).get("commit", "unknown")),
        "current_tag": current.get("tag", "unknown"),
        "current_commit": current.get("commit", current.get("git", {}).get("commit", "unknown")),
        "comparisons": comparisons,
        "has_warnings": has_warnings,
        "has_failures": has_failures,
        "overall": "FAIL" if has_failures else ("WARN" if has_warnings else "PASS"),
    }


def print_comparison(report: dict, use_color: bool = True):
    """Print comparison report to console."""
    # ANSI colors
    if use_color:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        GREEN = YELLOW = RED = BOLD = RESET = ""

    print("\n" + "=" * 70)
    print(f"{BOLD}BASELINE COMPARISON{RESET}")
    print("=" * 70)

    print(f"\nBaseline: {report['baseline_tag']} @ {report['baseline_commit']}")
    print(f"Current:  {report['current_tag']} @ {report['current_commit']}")

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>10} {'Status':>8}")
    print("-" * 70)

    for comp in report["comparisons"]:
        # Color based on level
        if comp["level"] == "FAIL":
            color = RED
            symbol = "✗"
        elif comp["level"] == "WARN":
            color = YELLOW
            symbol = "!"
        elif comp["status"] == "improved":
            color = GREEN
            symbol = "✓"
        else:
            color = RESET
            symbol = "-"

        # Format change
        if comp["pct_change"] == float("inf"):
            change_str = "N/A"
        else:
            sign = "+" if comp["pct_change"] > 0 else ""
            change_str = f"{sign}{comp['pct_change']:.1f}%"

        print(f"{comp['name']:<25} {comp['baseline']:>12} {comp['current']:>12} "
              f"{color}{change_str:>10}{RESET} {color}{symbol:>8}{RESET}")

    print("-" * 70)

    # Overall result
    if report["overall"] == "PASS":
        result_color = GREEN
        result_text = "PASS - No regressions detected"
    elif report["overall"] == "WARN":
        result_color = YELLOW
        result_text = "WARN - Minor regressions detected"
    else:
        result_color = RED
        result_text = "FAIL - Significant regressions detected"

    print(f"\n{BOLD}Result: {result_color}{result_text}{RESET}")
    print("=" * 70)


def run_current_benchmark(config: str = None, steps: int = 20,
                          resolution: int = 256, save_at: int = None,
                          memory_only: bool = False,
                          clean_output: bool = False,
                          validate_generation: bool = False,
                          clipscore_threshold: Optional[float] = None,
                          clipscore_model: str = "openai/clip-vit-base-patch32") -> dict:
    """Run current benchmark and return results."""
    from scripts.perf_benchmark import (
        run_memory_only_benchmark,
        run_training_benchmark,
    )  # noqa: E402

    if memory_only:
        result = run_memory_only_benchmark()
    else:
        result = run_training_benchmark(
            config_path=config,
            steps=steps,
            resolution=resolution,
            save_at=save_at,
            clean_output=clean_output,
            validate_generation=validate_generation,
            clipscore_threshold=clipscore_threshold,
            clipscore_model=clipscore_model,
        )

    result.tag = "current"
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Compare performance baselines")

    parser.add_argument("--baseline", "-b", type=str, required=True,
                        help="Path to baseline JSON file")
    parser.add_argument("--compare", "-c", type=str, default=None,
                        help="Path to second baseline to compare (if not using --current)")
    parser.add_argument("--current", action="store_true",
                        help="Run current benchmark and compare against baseline")
    parser.add_argument("--config", type=str, default=None,
                        help="Config for current benchmark")
    parser.add_argument("--steps", "-s", type=int, default=20,
                        help="Steps for current benchmark")
    parser.add_argument("--resolution", "-r", type=int, default=256,
                        help="Resolution for current benchmark")
    parser.add_argument("--save-at", type=int, default=None,
                        help="Save step for current benchmark")
    parser.add_argument("--memory-only", action="store_true",
                        help="Run memory-only benchmark")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove prior benchmark output folder before current run")
    parser.add_argument("--validate-generation", action="store_true",
                        help="Run generation smoke validation in current benchmark run")
    parser.add_argument("--clipscore-threshold", type=float, default=0.20,
                        help="Minimum CLIPScore when --validate-generation is enabled")
    parser.add_argument("--clipscore-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model ID for CLIPScore calculation")
    parser.add_argument("--vram-threshold", type=float, default=5.0,
                        help="VRAM regression threshold percentage")
    parser.add_argument("--time-threshold", type=float, default=5.0,
                        help="Timing regression threshold percentage")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error code on any regression")

    args = parser.parse_args()

    if not args.current and not args.compare:
        parser.error("Either --current or --compare must be specified")

    # Load baseline
    print(f"Loading baseline: {args.baseline}")
    baseline = load_baseline(args.baseline)

    # Get comparison target
    if args.current:
        print("Running current benchmark...")
        current = run_current_benchmark(
            config=args.config,
            steps=args.steps,
            resolution=args.resolution,
            save_at=args.save_at,
            memory_only=args.memory_only,
            clean_output=args.clean_output,
            validate_generation=args.validate_generation,
            clipscore_threshold=(args.clipscore_threshold if args.validate_generation else None),
            clipscore_model=args.clipscore_model,
        )
    else:
        print(f"Loading comparison: {args.compare}")
        current = load_baseline(args.compare)

    # Customize thresholds if specified
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.vram_threshold != 5.0:
        for key in ["peak_vram_mb", "post_save_vram_mb"]:
            thresholds[key] = Threshold(
                thresholds[key].name,
                "lower_better",
                args.vram_threshold,
                args.vram_threshold * 2,
            )
    if args.time_threshold != 5.0:
        for key in ["avg_step_time_s", "post_save_step_time_s"]:
            thresholds[key] = Threshold(
                thresholds[key].name,
                "lower_better",
                args.time_threshold,
                args.time_threshold * 2,
            )

    # Compare
    report = compare_baselines(baseline, current, thresholds)

    # Print results
    print_comparison(report, use_color=not args.no_color)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    # Exit code
    if args.strict and report["overall"] != "PASS":
        sys.exit(1)
    elif report["overall"] == "FAIL":
        sys.exit(2)

    return report


if __name__ == "__main__":
    main()

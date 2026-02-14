#!/usr/bin/env python3
"""
Run Phase 4-6 control/ablation training matrix.

Default usage:
    uv run python scripts/run_phase4_ablations.py

Custom config:
    uv run python scripts/run_phase4_ablations.py --config config/examples/train_lora_flux_24gb.yaml

Generate configs only (no training):
    uv run python scripts/run_phase4_ablations.py --generate-only
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "perf_test.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "phase4_ablations"


def _get_process_block(config_data: dict[str, Any]) -> dict[str, Any]:
    try:
        process_list = config_data["config"]["process"]
    except KeyError as exc:
        raise ValueError("Config must contain top-level 'config.process' list.") from exc
    if not isinstance(process_list, list) or not process_list:
        raise ValueError("Config field 'config.process' must be a non-empty list.")

    # Prefer sd_trainer, otherwise use the first process entry.
    for proc in process_list:
        if isinstance(proc, dict) and proc.get("type") == "sd_trainer":
            return proc
    if not isinstance(process_list[0], dict):
        raise ValueError("First process entry in 'config.process' must be a mapping.")
    return process_list[0]


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    return child


def _set_run_name_and_output(config_data: dict[str, Any], case_name: str, run_dir: Path) -> None:
    cfg_root = _ensure_dict(config_data, "config")
    base_name = cfg_root.get("name", "phase4_ablation")
    cfg_root["name"] = f"{base_name}_{case_name}"

    process = _get_process_block(config_data)
    process["training_folder"] = str(run_dir)


def _apply_reproducibility_defaults(
    process: dict[str, Any],
    seed: int,
    force_optimizer: str | None,
) -> None:
    train = _ensure_dict(process, "train")
    sample = _ensure_dict(process, "sample")

    # Fixed training/sample seeds for comparable ablations.
    # `training_seed` is consumed by BaseTrainProcess and controls model/init/training RNG.
    process["training_seed"] = int(seed)
    train["seed"] = int(seed)
    sample["seed"] = int(seed)
    sample["walk_seed"] = False

    if force_optimizer:
        train["optimizer"] = force_optimizer
    else:
        # bitsandbytes AdamW8bit does not support step_parameter; use in-repo AdamW8 path.
        if str(train.get("optimizer", "")).lower() == "adamw8bit":
            train["optimizer"] = "adamw8"


def _apply_case_flags(
    process: dict[str, Any],
    case_name: str,
    stable_loss_path: str | None,
) -> None:
    train = _ensure_dict(process, "train")
    model = _ensure_dict(process, "model")

    # Baseline: all knobs off
    train["fused_back_pass"] = False
    train["stable_loss_enabled"] = False
    model["use_offload_conductor"] = False
    model["layer_offloading"] = False

    if case_name in {"control", "control_repeat"}:
        return
    if case_name == "layer_offloading":
        model["layer_offloading"] = True
        arch = str(model.get("arch", "")).lower()
        if arch.startswith("flux2") and bool(model.get("quantize", False)):
            # Quantized Flux2 layer offloading currently routes through conductor and can be unstable.
            # Force this case onto the plain memory-manager offloading path so it is active and train-valid.
            model["quantize"] = False
            model["quantize_te"] = False
            model["use_offload_conductor"] = False
        return
    if case_name == "offload_conductor":
        model["use_offload_conductor"] = True
        return
    if case_name == "fused_back_pass":
        train["fused_back_pass"] = True
        return
    if case_name == "stable_loss":
        train["stable_loss_enabled"] = True
        if stable_loss_path:
            train["stable_loss_path"] = stable_loss_path
        return
    raise ValueError(f"Unknown case '{case_name}'")


def _warn_if_case_will_auto_disable(process: dict[str, Any], case_name: str) -> None:
    if case_name != "layer_offloading":
        return
    model = _ensure_dict(process, "model")
    arch = str(model.get("arch", "")).lower()
    quantize = bool(model.get("quantize", False))
    use_offload_conductor = bool(model.get("use_offload_conductor", False))
    if arch.startswith("flux2") and not quantize:
        print(
            "Info: layer_offloading case disables Flux2 quantization so layer_offloading remains "
            "active on the plain memory-manager path."
        )
    elif arch.startswith("flux2") and quantize and not use_offload_conductor:
        print(
            "Warning: layer_offloading case uses quantized Flux2 without offload conductor; "
            "runtime model startup will disable layer_offloading for compatibility. "
            "This case will effectively track control unless config/model settings change."
        )


def _run_training(config_path: Path) -> int:
    cmd = ["uv", "run", "python", "run.py", str(config_path)]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def _run_analyzer(
    run_root: Path,
    warn_relative_delta: float | None,
    fail_relative_delta: float | None,
    control_repeat_warn: float | None,
) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/analyze_phase4_ablations.py",
        "--run-root",
        str(run_root),
    ]
    if warn_relative_delta is not None:
        cmd.extend(["--warn-relative-delta", str(warn_relative_delta)])
    if fail_relative_delta is not None:
        cmd.extend(["--fail-relative-delta", str(fail_relative_delta)])
    if control_repeat_warn is not None:
        cmd.extend(["--warn-control-repeat-relative-delta", str(control_repeat_warn)])
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and run Phase 4 ablation configs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Base YAML config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root for generated configs and run folders (default: {DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--stable-loss-path",
        type=str,
        default=None,
        help="Path to 1-2 representative images for the stable-loss ablation run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for all ablation configs (train.seed and sample.seed).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Force optimizer across all ablations (e.g. adamw8, adamw, adafactor).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate ablation configs; do not execute training runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs when a case fails.",
    )
    parser.add_argument(
        "--double-control",
        action="store_true",
        help="Run a second control case (control_repeat) to measure baseline run-to-run drift.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run scripts/analyze_phase4_ablations.py automatically after all runs complete.",
    )
    parser.add_argument(
        "--analyze-warn-relative-delta",
        type=float,
        default=None,
        help="Pass-through threshold for analyzer --warn-relative-delta.",
    )
    parser.add_argument(
        "--analyze-fail-relative-delta",
        type=float,
        default=None,
        help="Pass-through threshold for analyzer --fail-relative-delta.",
    )
    parser.add_argument(
        "--analyze-warn-control-repeat-relative-delta",
        type=float,
        default=None,
        help="Pass-through threshold for analyzer control-repeat sanity warning.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 2

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    if not isinstance(base_config, dict):
        print("Error: config YAML root must be a mapping.", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)) / ts
    configs_dir = run_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        "control",
        "layer_offloading",
        "offload_conductor",
        "fused_back_pass",
        "stable_loss",
    ]
    if args.double_control:
        cases.insert(1, "control_repeat")

    if args.stable_loss_path is None:
        # Stable loss run without a path will be auto-disabled by trainer startup gating.
        print("Note: --stable-loss-path not set; stable_loss case will auto-disable stable loss.")

    case_config_paths: list[tuple[str, Path]] = []
    for case_name in cases:
        cfg = copy.deepcopy(base_config)
        process = _get_process_block(cfg)
        case_run_dir = run_root / "runs" / case_name
        _set_run_name_and_output(cfg, case_name, case_run_dir)
        _apply_reproducibility_defaults(process, seed=args.seed, force_optimizer=args.optimizer)
        _apply_case_flags(process, case_name=case_name, stable_loss_path=args.stable_loss_path)
        _warn_if_case_will_auto_disable(process, case_name=case_name)

        out_cfg = configs_dir / f"{case_name}.yaml"
        with open(out_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        case_config_paths.append((case_name, out_cfg))

    print(f"Generated configs in: {configs_dir}")
    for case_name, path in case_config_paths:
        print(f"  - {case_name}: {path}")

    if args.generate_only:
        return 0

    print("\nRunning ablations in order:")
    for case_name, cfg_path in case_config_paths:
        print(f"\n=== [{case_name}] ===")
        rc = _run_training(cfg_path)
        if rc != 0:
            print(f"Case failed: {case_name} (exit={rc})", file=sys.stderr)
            if not args.continue_on_error:
                return rc

    print("\nAll requested ablation runs completed.")
    if args.analyze:
        print("\nRunning analyzer...")
        rc = _run_analyzer(
            run_root=run_root,
            warn_relative_delta=args.analyze_warn_relative_delta,
            fail_relative_delta=args.analyze_fail_relative_delta,
            control_repeat_warn=args.analyze_warn_control_repeat_relative_delta,
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

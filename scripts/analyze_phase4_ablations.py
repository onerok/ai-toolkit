#!/usr/bin/env python3
"""
Analyze Phase 4 ablation run outputs against control.

Usage:
    uv run python scripts/analyze_phase4_ablations.py
    uv run python scripts/analyze_phase4_ablations.py --run-root output/phase4_ablations/20260214_101704
    uv run python scripts/analyze_phase4_ablations.py --json-out output/phase4_ablations/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "output" / "phase4_ablations"
STEP_RE = re.compile(r"_(\d{9,})\.safetensors$")
DEFAULT_WARN_RELATIVE_DELTA = 0.25
DEFAULT_WARN_CONTROL_REPEAT_RELATIVE_DELTA = 0.05


@dataclass
class CheckpointStats:
    total_l2: float
    mean_abs: float
    max_abs: float
    num_params: int
    lora_a_l2: float
    lora_b_l2: float


def _latest_run_root(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"No ablation output directory found: {runs_root}")
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directories found in: {runs_root}")
    return sorted(candidates)[-1]


def _single_job_dir(case_dir: Path) -> Path:
    children = [p for p in case_dir.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No job directory found under case dir: {case_dir}")
    if len(children) > 1:
        children = sorted(children, key=lambda p: p.stat().st_mtime, reverse=True)
    return children[0]


def _pick_checkpoint(job_dir: Path, step: int | None) -> Path:
    ckpts = sorted(job_dir.glob("*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {job_dir}")

    step_ckpts: list[tuple[int, Path]] = []
    final_ckpts: list[Path] = []
    for ckpt in ckpts:
        m = STEP_RE.search(ckpt.name)
        if m:
            step_ckpts.append((int(m.group(1)), ckpt))
        else:
            final_ckpts.append(ckpt)

    if step is not None:
        for s, p in step_ckpts:
            if s == step:
                return p
        raise FileNotFoundError(f"Requested step {step} not found in {job_dir}")

    if step_ckpts:
        step_ckpts.sort(key=lambda item: item[0])
        return step_ckpts[-1][1]
    return sorted(final_ckpts)[-1]


def _parse_knobs(job_dir: Path) -> dict[str, Any]:
    runtime_path = job_dir / "runtime_knobs.json"
    if runtime_path.exists():
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                runtime = json.load(f) or {}
            effective = runtime.get("effective", {})
            model = effective.get("model", {}) if isinstance(effective, dict) else {}
            train = effective.get("train", {}) if isinstance(effective, dict) else {}
            return {
                "fused_back_pass": bool(train.get("fused_back_pass", False)),
                "use_offload_conductor": bool(model.get("use_offload_conductor", False)),
                "layer_offloading": bool(model.get("layer_offloading", False)),
                "stable_loss_enabled": bool(train.get("stable_loss_enabled", False)),
            }
        except Exception:
            pass

    cfg_path = job_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try:
        process_list = data["config"]["process"]
        process = process_list[0]
        train = process.get("train", {})
        model = process.get("model", {})
        return {
            "fused_back_pass": bool(train.get("fused_back_pass", False)),
            "use_offload_conductor": bool(model.get("use_offload_conductor", False)),
            "layer_offloading": bool(model.get("layer_offloading", False)),
            "stable_loss_enabled": bool(train.get("stable_loss_enabled", False)),
        }
    except Exception:
        return {}


def _parse_runtime_knobs(job_dir: Path) -> dict[str, Any]:
    runtime_path = job_dir / "runtime_knobs.json"
    if not runtime_path.exists():
        return {}
    try:
        with open(runtime_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _extract_thresholds_from_config(config_data: dict[str, Any]) -> dict[str, float]:
    key_aliases = {
        "warn_relative_delta": ["warn_relative_delta", "warn_rel_delta"],
        "warn_control_repeat_relative_delta": [
            "warn_control_repeat_relative_delta",
            "control_repeat_warn_relative_delta",
            "warn_control_repeat",
        ],
        "fail_relative_delta": ["fail_relative_delta"],
    }

    candidates: list[dict[str, Any]] = []
    top = config_data.get("ablation_analyzer")
    if isinstance(top, dict):
        candidates.append(top)

    meta = config_data.get("meta")
    if isinstance(meta, dict):
        v = meta.get("ablation_analyzer")
        if isinstance(v, dict):
            candidates.append(v)

    cfg = config_data.get("config")
    if isinstance(cfg, dict):
        v = cfg.get("ablation_analyzer")
        if isinstance(v, dict):
            candidates.append(v)
        process = cfg.get("process")
        if isinstance(process, list) and process and isinstance(process[0], dict):
            pv = process[0].get("ablation_analyzer")
            if isinstance(pv, dict):
                candidates.append(pv)

    resolved: dict[str, float] = {}
    for candidate in candidates:
        for canonical_key, aliases in key_aliases.items():
            for key in aliases:
                if key not in candidate:
                    continue
                try:
                    resolved[canonical_key] = float(candidate[key])
                except (TypeError, ValueError):
                    pass
                break
    return resolved


def _load_thresholds_from_job_config(job_dir: Path) -> dict[str, float]:
    cfg_path = job_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    return _extract_thresholds_from_config(cfg)


def _tensor_stats(state_dict: dict[str, torch.Tensor]) -> CheckpointStats:
    total_sq = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    numel = 0
    lora_a_sq = 0.0
    lora_b_sq = 0.0

    for key, t in state_dict.items():
        tf = t.float()
        sq = float(torch.sum(tf * tf).item())
        total_sq += sq
        abs_t = torch.abs(tf)
        abs_sum += float(torch.sum(abs_t).item())
        max_abs = max(max_abs, float(torch.max(abs_t).item()))
        numel += tf.numel()
        if ".lora_A." in key:
            lora_a_sq += sq
        elif ".lora_B." in key:
            lora_b_sq += sq

    mean_abs = abs_sum / numel if numel > 0 else 0.0
    return CheckpointStats(
        total_l2=math.sqrt(total_sq),
        mean_abs=mean_abs,
        max_abs=max_abs,
        num_params=numel,
        lora_a_l2=math.sqrt(lora_a_sq),
        lora_b_l2=math.sqrt(lora_b_sq),
    )


def _delta_metrics(
    control: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
) -> dict[str, Any]:
    common = sorted(set(control.keys()) & set(other.keys()))
    missing = sorted(set(control.keys()) - set(other.keys()))
    extra = sorted(set(other.keys()) - set(control.keys()))

    delta_sq = 0.0
    control_sq = 0.0
    other_sq = 0.0
    dot_product = 0.0
    lora_dot_product = 0.0
    lora_control_sq = 0.0
    lora_other_sq = 0.0
    per_key_delta: list[tuple[float, str]] = []
    for k in common:
        c = control[k].float()
        o = other[k].float()
        d = o - c
        d_sq = float(torch.sum(d * d).item())
        delta_sq += d_sq
        c_sq = float(torch.sum(c * c).item())
        o_sq = float(torch.sum(o * o).item())
        control_sq += c_sq
        other_sq += o_sq
        dot_product += float(torch.sum(c * o).item())
        if ".lora_A." in k or ".lora_B." in k:
            lora_dot_product += float(torch.sum(c * o).item())
            lora_control_sq += c_sq
            lora_other_sq += o_sq
        per_key_delta.append((math.sqrt(d_sq), k))

    per_key_delta.sort(reverse=True)
    delta_l2 = math.sqrt(delta_sq)
    control_l2_common = math.sqrt(control_sq)
    rel = delta_l2 / (control_l2_common + 1e-12)
    cosine_similarity = dot_product / ((math.sqrt(control_sq) * math.sqrt(other_sq)) + 1e-12)
    lora_cosine_similarity = lora_dot_product / (
        (math.sqrt(lora_control_sq) * math.sqrt(lora_other_sq)) + 1e-12
    )

    return {
        "shared_keys": len(common),
        "missing_keys": missing,
        "extra_keys": extra,
        "delta_l2": delta_l2,
        "control_l2_common": control_l2_common,
        "relative_delta": rel,
        "cosine_similarity": cosine_similarity,
        "lora_cosine_similarity": lora_cosine_similarity,
        "top_delta_keys": [{"key": k, "delta_l2": v} for v, k in per_key_delta[:10]],
    }


def _samples_info(job_dir: Path) -> dict[str, Any]:
    sample_dir = job_dir / "samples"
    if not sample_dir.exists():
        return {"present": False, "count": 0}
    images = [
        p for p in sample_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    return {"present": True, "count": len(images)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase 4 ablation outputs.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Specific ablation run root (timestamp directory). Default: latest in output/phase4_ablations/",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific checkpoint step suffix to compare (e.g. 10). Defaults to latest step checkpoint.",
    )
    parser.add_argument(
        "--warn-relative-delta",
        type=float,
        default=None,
        help="Warn if relative delta vs control exceeds this threshold.",
    )
    parser.add_argument(
        "--warn-control-repeat-relative-delta",
        type=float,
        default=None,
        help="Warn if control_repeat drift vs control exceeds this threshold.",
    )
    parser.add_argument(
        "--fail-relative-delta",
        type=float,
        default=None,
        help="Exit non-zero if any case exceeds this relative delta threshold.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report as JSON.",
    )
    args = parser.parse_args()

    run_root = args.run_root if args.run_root else _latest_run_root(DEFAULT_RUNS_ROOT)
    if not run_root.is_absolute():
        run_root = (PROJECT_ROOT / run_root).resolve()
    runs_dir = run_root / "runs"
    if not runs_dir.exists():
        print(f"Error: run root has no 'runs' directory: {run_root}", file=sys.stderr)
        return 2

    case_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        print(f"Error: no case directories found in {runs_dir}", file=sys.stderr)
        return 2

    if not any(p.name == "control" for p in case_dirs):
        print("Error: required control case missing.", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "run_root": str(run_root),
        "cases": {},
        "warnings": [],
        "failures": [],
        "runtime_adjustments": {},
    }

    # Load control first.
    control_job = _single_job_dir(runs_dir / "control")
    control_ckpt = _pick_checkpoint(control_job, args.step)
    control_state = load_file(str(control_ckpt))
    control_stats = _tensor_stats(control_state)
    config_thresholds = _load_thresholds_from_job_config(control_job)
    warn_relative_delta = (
        args.warn_relative_delta
        if args.warn_relative_delta is not None
        else config_thresholds.get("warn_relative_delta", DEFAULT_WARN_RELATIVE_DELTA)
    )
    warn_control_repeat_relative_delta = (
        args.warn_control_repeat_relative_delta
        if args.warn_control_repeat_relative_delta is not None
        else config_thresholds.get(
            "warn_control_repeat_relative_delta",
            DEFAULT_WARN_CONTROL_REPEAT_RELATIVE_DELTA,
        )
    )
    fail_relative_delta = (
        args.fail_relative_delta
        if args.fail_relative_delta is not None
        else config_thresholds.get("fail_relative_delta")
    )

    report["cases"]["control"] = {
        "job_dir": str(control_job),
        "checkpoint": str(control_ckpt),
        "knobs": _parse_knobs(control_job),
        "runtime_knobs": _parse_runtime_knobs(control_job),
        "stats": control_stats.__dict__,
        "samples": _samples_info(control_job),
    }
    control_adjustments = report["cases"]["control"]["runtime_knobs"].get("adjustments", [])
    if control_adjustments:
        report["runtime_adjustments"]["control"] = control_adjustments
    report["thresholds"] = {
        "warn_relative_delta": warn_relative_delta,
        "warn_control_repeat_relative_delta": warn_control_repeat_relative_delta,
        "fail_relative_delta": fail_relative_delta,
        "config_overrides": config_thresholds,
    }

    for case_dir in case_dirs:
        case_name = case_dir.name
        if case_name == "control":
            continue

        job_dir = _single_job_dir(case_dir)
        ckpt = _pick_checkpoint(job_dir, args.step)
        state = load_file(str(ckpt))
        stats = _tensor_stats(state)
        delta = _delta_metrics(control_state, state)
        samples = _samples_info(job_dir)

        case_report = {
            "job_dir": str(job_dir),
            "checkpoint": str(ckpt),
            "knobs": _parse_knobs(job_dir),
            "runtime_knobs": _parse_runtime_knobs(job_dir),
            "stats": stats.__dict__,
            "delta_vs_control": delta,
            "samples": samples,
        }
        report["cases"][case_name] = case_report
        adjustments = case_report["runtime_knobs"].get("adjustments", []) if case_report["runtime_knobs"] else []
        if adjustments:
            report["runtime_adjustments"][case_name] = adjustments

        rel = float(delta["relative_delta"])
        if rel > warn_relative_delta:
            report["warnings"].append(
                f"{case_name}: relative_delta={rel:.4f} exceeds warn threshold {warn_relative_delta:.4f}"
            )
        if fail_relative_delta is not None and rel > fail_relative_delta:
            report["failures"].append(
                f"{case_name}: relative_delta={rel:.4f} exceeds fail threshold {fail_relative_delta:.4f}"
            )

    print(f"Run root: {run_root}")
    print(f"Control checkpoint: {control_ckpt.name}")
    print(
        "Thresholds: "
        f"warn_relative_delta={warn_relative_delta:.4f}, "
        f"warn_control_repeat_relative_delta={warn_control_repeat_relative_delta:.4f}, "
        f"fail_relative_delta={'none' if fail_relative_delta is None else f'{fail_relative_delta:.4f}'}"
    )
    print("")
    print(
        "Case                          RelDelta   CosSim   LoraCos  DeltaL2      TotalL2      LoraA_L2     LoraB_L2     SharedKeys"
    )
    print(
        "----------------------------  ---------  -------  -------  -----------  -----------  -----------  -----------  ----------"
    )
    for case_name, case_data in sorted(report["cases"].items()):
        stats = case_data["stats"]
        if case_name == "control":
            print(
                f"{case_name:28}  {'-':>9}  {'-':>7}  {'-':>7}  {'-':>11}  "
                f"{stats['total_l2']:11.4f}  {stats['lora_a_l2']:11.4f}  {stats['lora_b_l2']:11.4f}  {'-':>10}"
            )
            continue
        delta = case_data["delta_vs_control"]
        print(
            f"{case_name:28}  {delta['relative_delta']:9.4f}  {delta['cosine_similarity']:7.4f}  "
            f"{delta['lora_cosine_similarity']:7.4f}  {delta['delta_l2']:11.4f}  {stats['total_l2']:11.4f}  "
            f"{stats['lora_a_l2']:11.4f}  {stats['lora_b_l2']:11.4f}  {delta['shared_keys']:10d}"
        )

    print("")
    if "control_repeat" in report["cases"]:
        repeat_rel = float(report["cases"]["control_repeat"]["delta_vs_control"]["relative_delta"])
        repeat_cos = float(report["cases"]["control_repeat"]["delta_vs_control"]["cosine_similarity"])
        print(
            f"Control-repeat sanity: relative_delta={repeat_rel:.4f}, cosine={repeat_cos:.4f}"
        )
        if repeat_rel > warn_control_repeat_relative_delta:
            report["warnings"].append(
                f"control_repeat drift={repeat_rel:.4f} exceeds threshold {warn_control_repeat_relative_delta:.4f}"
            )
        print("")

    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    else:
        print("Warnings: none")

    if report["runtime_adjustments"]:
        print("")
        print("Runtime Adjustments:")
        for case_name, adjustments in sorted(report["runtime_adjustments"].items()):
            print(f"  - {case_name}:")
            for adj in adjustments:
                feature = str(adj.get("feature", "unknown"))
                requested = adj.get("requested")
                effective = adj.get("effective")
                reason = str(adj.get("reason", ""))
                source = str(adj.get("source", ""))
                src_suffix = f" [{source}]" if source else ""
                print(
                    f"      {feature}: requested={requested!r}, effective={effective!r}, "
                    f"reason={reason}{src_suffix}"
                )

    if report["failures"]:
        print("")
        print("Failures:")
        for f in report["failures"]:
            print(f"  - {f}")

    if args.json_out is not None:
        json_out = args.json_out if args.json_out.is_absolute() else (PROJECT_ROOT / args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report: {json_out}")

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

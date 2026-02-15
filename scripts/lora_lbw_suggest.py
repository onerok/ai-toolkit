#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "safetensors",
# ]
# ///
"""
Analyze a LoRA safetensors file and suggest LoRA Block Weights (LBW).

Usage:
    uv run scripts/lora_lbw_suggest.py path/to/lora.safetensors
    uv run scripts/lora_lbw_suggest.py path/to/lora.safetensors --strategy bins
    uv run scripts/lora_lbw_suggest.py path/to/lora.safetensors --strategy rank_weighted
    uv run scripts/lora_lbw_suggest.py path/to/lora.safetensors --strategy bins --bins 4

Analyzes per-block weight norms and stable rank, then suggests LBW values.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


# ── Block detection ──────────────────────────────────────────────────────────


BLOCK_PATTERNS = [
    # Flux
    (re.compile(r"double_blocks\.(\d+)"), "double", lambda m: int(m.group(1))),
    (re.compile(r"single(?:_transformer)?_blocks\.(\d+)"), "single", lambda m: int(m.group(1))),
    # SDXL / SD 1.5 UNet
    (re.compile(r"down_blocks\.(\d+)"), "down", lambda m: int(m.group(1))),
    (re.compile(r"up_blocks\.(\d+)"), "up", lambda m: int(m.group(1))),
    (re.compile(r"mid_block"), "mid", lambda _: 0),
    # Generic transformer
    (re.compile(r"(?:transformer_)?blocks\.(\d+)"), "block", lambda m: int(m.group(1))),
]


def detect_block(key: str) -> str | None:
    """Map a LoRA weight key to a block name like 'double_0', 'single_12', 'down_2'."""
    for pattern, prefix, idx_fn in BLOCK_PATTERNS:
        m = pattern.search(key)
        if m:
            return f"{prefix}_{idx_fn(m)}"
    return None


def get_layer_id(key: str) -> str:
    """Strip lora_A/lora_B suffix to get the layer pair identifier."""
    return re.sub(r"\.lora_[AB](\.weight)?$", "", key)


# ── Stable rank ──────────────────────────────────────────────────────────────


def stable_rank(A: torch.Tensor, B: torch.Tensor) -> float:
    """Compute stable rank of B@A via efficient r×r SVD (no full product)."""
    U_B, S_B, Vh_B = torch.linalg.svd(B.float(), full_matrices=False)
    U_A, S_A, Vh_A = torch.linalg.svd(A.float(), full_matrices=False)
    M = torch.diag(S_B) @ Vh_B @ U_A @ torch.diag(S_A)
    sv = torch.linalg.svdvals(M)
    sv_sq = sv * sv
    sigma_max_sq = sv_sq[0].item()
    if sigma_max_sq < 1e-12:
        return 1.0
    return sv_sq.sum().item() / sigma_max_sq


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_lora(filepath: Path) -> dict:
    """Analyze a LoRA file and return per-block statistics."""
    f = safe_open(str(filepath), framework="pt")
    keys = list(f.keys())

    # Pair lora_A and lora_B by layer
    layer_pairs: dict[str, dict] = defaultdict(dict)
    for key in keys:
        block = detect_block(key)
        if block is None:
            continue
        tensor = f.get_tensor(key)
        lid = get_layer_id(key)
        if "lora_A" in key or "lora_down" in key:
            layer_pairs[lid]["A"] = tensor
            layer_pairs[lid]["block"] = block
        elif "lora_B" in key or "lora_up" in key:
            layer_pairs[lid]["B"] = tensor
            layer_pairs[lid]["block"] = block

    # Compute per-layer metrics, group by block
    block_layers: dict[str, list] = defaultdict(list)
    rank = None

    for lid, pair in layer_pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        A, B = pair["A"], pair["B"]
        block = pair["block"]
        if rank is None:
            rank = A.shape[0]

        norm_product = B.float().norm().item() * A.float().norm().item()
        sr = stable_rank(A, B)

        block_layers[block].append({"norm": norm_product, "stable_rank": sr})

    # Aggregate per block
    blocks = {}
    for block, layers in block_layers.items():
        n = len(layers)
        blocks[block] = {
            "norm": sum(l["norm"] for l in layers) / n,
            "stable_rank": sum(l["stable_rank"] for l in layers) / n,
            "num_layers": n,
        }

    # Detect architecture
    prefixes = set(b.split("_")[0] for b in blocks)
    if "double" in prefixes and "single" in prefixes:
        arch = "flux"
    elif "down" in prefixes and "up" in prefixes:
        arch = "unet"
    else:
        arch = "transformer"

    return {"blocks": blocks, "rank": rank or 0, "architecture": arch}


# ── LBW calculation ──────────────────────────────────────────────────────────


def natural_sort_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def compute_raw_scores(
    blocks: dict,
    strategy: str = "norm",
) -> dict[str, float]:
    """Compute raw (unnormalized) per-block scores based on strategy."""
    sorted_blocks = sorted(blocks.keys(), key=natural_sort_key)

    if strategy in ("rank_weighted", "rank_weighted_bins"):
        max_sr = max(b["stable_rank"] for b in blocks.values())
        return {
            name: blocks[name]["norm"] * (blocks[name]["stable_rank"] / max_sr)
            for name in sorted_blocks
        }
    else:
        return {name: blocks[name]["norm"] for name in sorted_blocks}


def compute_lbw(
    blocks: dict,
    threshold: float = 0.0,
    strategy: str = "norm",
    num_bins: int = 3,
) -> list[tuple[str, float, dict]]:
    """
    Compute suggested LBW values.

    Returns list of (block_name, lbw_value, stats) sorted by block order.

    Strategies:
      - "norm": proportional to weight norm (continuous 0-1)
      - "rank_weighted": norm * (stable_rank/max_sr) (continuous 0-1)
      - "bins": binned into discrete tiers based on norm
      - "rank_weighted_bins": binned into discrete tiers based on rank-weighted score
    """
    use_bins = strategy in ("bins", "rank_weighted_bins")
    raw = compute_raw_scores(blocks, strategy)
    sorted_blocks = sorted(blocks.keys(), key=natural_sort_key)
    max_val = max(raw.values()) if raw else 1.0

    # Normalize to 0-1
    normalized = {
        name: raw[name] / max_val if max_val > 0 else 0
        for name in sorted_blocks
    }

    if use_bins:
        # Assign each block to a bin using percentile-based thresholds.
        # Bins are evenly spaced tiers from 0.0 to 1.0.
        # The lowest tier zeros out blocks (off).
        vals = sorted(normalized.values())
        n = len(vals)

        # Determine bin edges from the data distribution.
        # With num_bins=3: off / mid / full
        # With num_bins=4: off / low / mid / full
        bin_edges = []
        for i in range(1, num_bins):
            pct_idx = int(n * i / num_bins)
            bin_edges.append(vals[min(pct_idx, n - 1)])

        # Bin values: 0.0, then evenly spaced up to 1.0
        bin_values = [0.0] + [round(i / (num_bins - 1), 2) for i in range(1, num_bins)]

        result = []
        for name in sorted_blocks:
            v = normalized[name]
            # Find which bin this block falls into
            bin_idx = 0
            for edge in bin_edges:
                if v >= edge:
                    bin_idx += 1
            lbw = bin_values[min(bin_idx, len(bin_values) - 1)]
            result.append((name, lbw, blocks[name]))
    else:
        result = []
        for name in sorted_blocks:
            v = normalized[name]
            lbw = round(v, 3) if v >= threshold else 0.0
            result.append((name, lbw, blocks[name]))

    return result


# ── Output formatting ────────────────────────────────────────────────────────


TIER_LABELS = {0.0: "OFF", 0.5: "MID", 1.0: "FULL", 0.33: "LOW", 0.67: "HIGH", 0.25: "LOW", 0.75: "HIGH"}


def tier_label(lbw: float) -> str:
    """Map a binned LBW value to a human-readable tier name."""
    if lbw == 0.0:
        return "OFF "
    if lbw == 1.0:
        return "FULL"
    if lbw <= 0.34:
        return "LOW "
    if lbw <= 0.51:
        return "MID "
    return "HIGH"


def print_table(lbw_results: list, rank: int, arch: str, is_binned: bool = False):
    """Print a human-readable analysis table."""
    max_name = max(len(r[0]) for r in lbw_results)

    print(f"\n{'─' * 78}")
    print(f"  Architecture: {arch}   Rank: {rank}   Blocks: {len(lbw_results)}")
    print(f"{'─' * 78}")

    if is_binned:
        print(
            f"  {'Block':<{max_name}}  {'Tier':>4}  {'LBW':>5}  {'Norm':>8}  "
            f"{'Stable Rank':>11}  {'Rank Use':>8}"
        )
        print(f"  {'─' * (max_name + 48)}")

        for name, lbw, stats in lbw_results:
            sr = stats["stable_rank"]
            rank_use = f"{sr:.1f}/{rank}" if rank > 0 else f"{sr:.1f}"
            label = tier_label(lbw)
            bar_len = int(lbw * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(
                f"  {name:<{max_name}}  {label:>4}  {lbw:>5.2f}  {stats['norm']:>8.3f}  "
                f"{sr:>11.2f}  {rank_use:>8}  {bar}"
            )
    else:
        print(
            f"  {'Block':<{max_name}}  {'LBW':>5}  {'Norm':>8}  "
            f"{'Stable Rank':>11}  {'Rank Use':>8}  {'Layers':>6}"
        )
        print(f"  {'─' * (max_name + 48)}")

        for name, lbw, stats in lbw_results:
            sr = stats["stable_rank"]
            rank_use = f"{sr:.1f}/{rank}" if rank > 0 else f"{sr:.1f}"
            bar_len = int(lbw * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(
                f"  {name:<{max_name}}  {lbw:>5.2f}  {stats['norm']:>8.3f}  "
                f"{sr:>11.2f}  {rank_use:>8}  {stats['num_layers']:>6}  {bar}"
            )

    print()


def print_lbw_string(lbw_results: list, arch: str):
    """Print the comma-separated LBW string for ComfyUI."""
    values = [r[1] for r in lbw_results]
    lbw_str = ",".join(f"{v:.2f}" for v in values)

    print(f"  LBW string ({len(values)} blocks):")
    print(f"  {lbw_str}")
    print()

    # Also print grouped if flux
    if arch == "flux":
        doubles = [r for r in lbw_results if r[0].startswith("double_")]
        singles = [r for r in lbw_results if r[0].startswith("single_")]
        if doubles:
            d_str = ",".join(f"{r[1]:.2f}" for r in doubles)
            print(f"  Double blocks ({len(doubles)}): {d_str}")
        if singles:
            s_str = ",".join(f"{r[1]:.2f}" for r in singles)
            print(f"  Single blocks ({len(singles)}): {s_str}")
        print()


def print_recommendations(lbw_results: list, rank: int):
    """Print actionable recommendations based on the analysis."""
    print(f"{'─' * 72}")
    print("  Recommendations")
    print(f"{'─' * 72}")

    # Find blocks that could be dropped
    zero_blocks = [r for r in lbw_results if r[1] < 0.05]
    low_blocks = [r for r in lbw_results if 0.05 <= r[1] < 0.2]
    low_rank_blocks = [r for r in lbw_results if r[2]["stable_rank"] < 2.0 and rank >= 8]

    if zero_blocks:
        names = ", ".join(r[0] for r in zero_blocks)
        print(f"\n  Candidates to disable (LBW < 0.05):")
        print(f"    {names}")
        print(f"    These blocks learned almost nothing. Safe to set to 0 in LBW")
        print(f"    or exclude via ignore_if_contains in training config.")

    if low_blocks:
        names = ", ".join(r[0] for r in low_blocks)
        print(f"\n  Low-impact blocks (LBW 0.05-0.20):")
        print(f"    {names}")
        print(f"    Consider reducing or zeroing these for a tighter LoRA.")

    if low_rank_blocks:
        names = ", ".join(r[0] for r in low_rank_blocks)
        print(f"\n  Overprovisioned rank (stable_rank < 2.0, trained at rank {rank}):")
        print(f"    {names}")
        print(f"    These blocks converged to ~1-2 effective dimensions out of {rank}.")
        print(f"    If retraining, consider rank {min(4, rank)} for these blocks via block_dims.")

    top = sorted(lbw_results, key=lambda r: r[1], reverse=True)[:3]
    print(f"\n  Highest-impact blocks:")
    for name, lbw, stats in top:
        print(f"    {name}: LBW {lbw:.2f}, stable_rank {stats['stable_rank']:.1f}/{rank}")

    if not zero_blocks and not low_blocks and not low_rank_blocks:
        print("\n  All blocks are contributing meaningfully. No obvious waste detected.")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a LoRA and suggest LBW (LoRA Block Weight) values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/lora_lbw_suggest.py my_lora.safetensors
  uv run scripts/lora_lbw_suggest.py my_lora.safetensors --strategy bins
  uv run scripts/lora_lbw_suggest.py my_lora.safetensors --strategy bins --bins 4
  uv run scripts/lora_lbw_suggest.py my_lora.safetensors --strategy rank_weighted_bins
  uv run scripts/lora_lbw_suggest.py my_lora.safetensors --strategy rank_weighted
        """,
    )
    parser.add_argument("lora_file", type=Path, help="Path to a LoRA safetensors file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Blocks with normalized impact below this get zeroed (default: 0.05)",
    )
    parser.add_argument(
        "--strategy",
        choices=["norm", "rank_weighted", "bins", "rank_weighted_bins"],
        default="norm",
        help="LBW calculation strategy (default: norm)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=3,
        help="Number of tiers for bins/rank_weighted_bins strategy (default: 3 = off/mid/full)",
    )
    args = parser.parse_args()

    if not args.lora_file.exists():
        print(f"Error: {args.lora_file} not found", file=sys.stderr)
        sys.exit(1)

    is_binned = args.strategy in ("bins", "rank_weighted_bins")
    tier_desc = ""
    if is_binned:
        if args.bins == 3:
            tier_desc = " (off / mid / full)"
        elif args.bins == 4:
            tier_desc = " (off / low / high / full)"
        else:
            tier_desc = f" ({args.bins} tiers)"

    print(f"\n  Analyzing: {args.lora_file.name}")
    if is_binned:
        print(f"  Strategy: {args.strategy}, {args.bins} bins{tier_desc}")
    else:
        print(f"  Strategy: {args.strategy}, threshold: {args.threshold}")

    analysis = analyze_lora(args.lora_file)
    lbw_results = compute_lbw(
        analysis["blocks"],
        threshold=args.threshold,
        strategy=args.strategy,
        num_bins=args.bins,
    )

    print_table(lbw_results, analysis["rank"], analysis["architecture"], is_binned=is_binned)
    print_lbw_string(lbw_results, analysis["architecture"])
    print_recommendations(lbw_results, analysis["rank"])


if __name__ == "__main__":
    main()

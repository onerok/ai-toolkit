# Phase 5 Offload Fraction Pareto (2026-02-14)

## Purpose
Provide a user-facing tradeoff view for quantized Flux2 offload conductor fractions.

This is intentionally a Pareto surface, not a single winner. Users can choose by objective:
- speed (`0.25`)
- balanced (`0.50`)
- memory (`0.75`)

## Dataset and Method
- Model/config base: `config/perf_test_quantized_offload_conductor.yaml`
- Device: `NVIDIA GeForce RTX 5090` (31.84 GB)
- Steps: `20`
- Save step: `10`
- Repeats: `2` per case
- Cases per fraction: `control`, `offload_conductor`
- Sweep fractions: `0.25`, `0.50`, `0.75`

Sweep root:
- `output/phase5_offload_fraction_sweep/20260214_175255`

Memory capture root:
- `output/phase5_offload_fraction_sweep/20260214_175255/memory_capture`

## Activation Guardrails
All fractions passed activation + analyzer checks:
- Control: `requested=false`, `effective=false`, `active=false`
- Offload conductor: `requested=true`, `effective=true`, `active=true`
- Analyzer: exit `0`, `Warnings: none`

## Offload Conductor Results (Average of 2 Repeats)
| Fraction | Preset | Peak VRAM (MB) | Post-save VRAM (MB) | Step Time (s) | Steps/s | Alloc Retries |
|---|---|---:|---:|---:|---:|---:|
| 0.25 | speed | 24066 | 24004 | 1.8828 | 0.5311 | 0 |
| 0.50 | balanced | 23024 | 22818 | 3.0636 | 0.3264 | 0 |
| 0.75 | memory | 21984 | 21504 | 4.2517 | 0.2352 | 0 |

## Relative to Control (Same Fraction)
| Fraction | Delta Peak VRAM (MB) | Delta Post-save VRAM (MB) | Offload Time Ratio | Offload Steps/s Ratio |
|---|---:|---:|---:|---:|
| 0.25 | +3860 | +3798 | 2.4868x | 0.4021x |
| 0.50 | +2818 | +2612 | 4.0875x | 0.2446x |
| 0.75 | +1778 | +1298 | 5.6774x | 0.1761x |

## Preset Guidance
- `speed` (`0.25`): best throughput among offload options, highest offload VRAM.
- `balanced` (`0.50`): middle tradeoff between throughput and VRAM.
- `memory` (`0.75`): lowest offload VRAM in this sweep, slowest throughput.

## Source Artifacts
- `output/phase5_offload_fraction_sweep/20260214_175255/summary.tsv`
- `output/phase5_offload_fraction_sweep/20260214_175255/memory_capture/metrics_detailed.tsv`
- `output/phase5_offload_fraction_sweep/20260214_175255/memory_capture/metrics_summary.tsv`
- `output/phase5_offload_fraction_sweep/20260214_175255/memory_capture/metrics_pairwise.tsv`

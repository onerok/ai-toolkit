# Handoff: Phase 4 Drift Status (2026-02-14)

## Status Note
- Historical snapshot from mid-day 2026-02-14.
- Later same-day quantized offload-conductor stabilization and runtime-knob metadata details are in:
  - `.agents/handoff/quantized-offload-conductor-stability-sweep-20260214.md`

## Summary
- `layer_offloading` is no longer the outlier in ablation results for quantized Flux2 path.
- Relative regression is fixed enough to proceed with the full performance plan.
- Baseline determinism remains somewhat noisy (`control_repeat` above warning threshold in multiple runs).

## Key Outcome
Latest run root:
- `output/phase4_ablations/20260214_125702`

Analyzer output:
- `control_repeat`: `0.0930`
- `fused_back_pass`: `0.0856`
- `layer_offloading`: `0.0869`
- `offload_conductor`: `0.0865`
- `stable_loss`: `0.1045`

Interpretation:
- `layer_offloading` now tracks baseline drift rather than diverging.
- Baseline run-to-run drift is still above preferred threshold (`0.05`), but this appears to predate the final drift fix sequence.

## Commits Created
1. `1266a77` — `Fix phase4 drift by gating quantized Flux2 layer offloading`
2. `6233e06` — `Land broader phase4 ablation, optimizer, and metrics work`

## Code Changes Relevant to Drift Fix
- `toolkit/config_modules.py`
  - Removed implicit mutation that changed quantization type (`qfloat8 -> float8`) when `layer_offloading=true`.
- `extensions_built_in/diffusion_models/flux2/flux2_model.py`
  - Added guard to disable layer offloading for quantized Flux2 when offload conductor is not active.
  - Added quantized text-encoder offload skip warning behavior.
- `extensions_built_in/diffusion_models/flux2/flux2_klein_model.py`
  - Added quantized text-encoder offload skip warning behavior for Klein path.

## Current Working Tree
- Clean except local-only/untracked files:
  - `.agents/`
  - `AGENTS.md`

## Recommended Next Steps
1. Proceed with remaining perf phases; do not block on strict determinism right now.
2. Keep running the existing ablation analyzer as a guardrail after major perf changes.
3. Track determinism tightening as a separate follow-up task (goal: reduce `control_repeat` toward/under `0.05`).

## Optional Follow-up Validation
If needed, quantify whether baseline noise predates recent commits by rerunning double-control on earlier branch commits where harness already exists.

## Follow-up (Later 2026-02-14)
- Added offload-conductor instrumentation for layer order and live-gradient offload behavior:
  - Counters: `load_ops`, `offload_ops`, `grad_blocked_offloads`, `deferred_retries`, `deferred_requeues`
  - Trace tail: `recent_events` ring buffer (load/offload/defer/requeue scheduling order)
- Exposed conductor stats via:
  - `Flux2.get_offload_conductor_stats()`
  - `StableDiffusion.get_offload_conductor_stats()`
- Logged conductor diagnostics into per-step `system_metrics` in `BaseSDTrainProcess` when conductor is enabled:
  - `offload_loaded_layers`, `offload_deferred_layers`, `offload_load_ops`, `offload_offload_ops`,
    `offload_grad_blocked`, `offload_deferred_retries`, `offload_deferred_requeues`, `offload_recent_events`

### Bug Fix Landed
- Fixed deferred offload handling bug in `OffloadConductor._process_deferred_offloads`:
  - Previously, deferred entries matching the current layer (`except_layer`) were dropped.
  - Now they are re-queued and retried on subsequent layers.
  - This prevents silent schedule drift in layer residency/offload ordering.

### Test Coverage Added
- `testing/test_offload_conductor_integration.py`
  - Added regression test ensuring deferred current-layer offloads are preserved.
  - Updated setup tests to current Phase 4 behavior (offload conductor no longer requires fused backward manager enablement).

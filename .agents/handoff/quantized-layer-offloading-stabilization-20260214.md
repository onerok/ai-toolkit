# Handoff: Quantized Layer-Offloading Stabilization (2026-02-14)

## Status Note
- Historical snapshot from earlier on 2026-02-14.
- Superseded in part by `.agents/handoff/quantized-offload-conductor-stability-sweep-20260214.md` for later same-day stabilization and validation results.

## Source of Truth
- `scripts/run_phase4_ablations.py`
- `toolkit/memory_management/offload_conductor.py`
- `toolkit/memory_management/manager.py`
- `toolkit/memory_management/manager_modules.py`
- `toolkit/models/base_model.py`
- `extensions_built_in/diffusion_models/flux2/flux2_model.py`
- `docs/perf.md`

## What Was Done In This Session

### 1) Added instrumentation for layer-offloading drift analysis
- `toolkit/memory_management/manager_modules.py`
  - Added env-gated trace infra:
    - `AITK_LAYER_OFFLOAD_TRACE=1`
    - `AITK_LAYER_OFFLOAD_TRACE_MAX_EVENTS`
    - `AITK_LAYER_OFFLOAD_TRACE_PATH`
  - Added trace events for:
    - layer attach
    - linear/conv forward stage
    - linear/conv backward stage
  - Added optional JSONL flush at process exit.
- `toolkit/memory_management/manager.py`
  - Added layer debug naming/indexing during attach.
  - Added trace events for manager decisions:
    - `manager_attach`
    - `manager_skip` (quantized, offload-percent skip, unmanaged class).

### 2) Fixed ablation workflow so `layer_offloading` case is no longer silently fake
Observed:
- Quantized Flux2 with `layer_offloading=true` and `use_offload_conductor=false` is auto-disabled at runtime in:
  - `extensions_built_in/diffusion_models/flux2/flux2_model.py` (warning + disable).

Changes:
- `scripts/run_phase4_ablations.py`
  - Added explicit warning/info for cases that would auto-disable.
  - Updated case generation for `layer_offloading` on Flux2:
    - If base config is quantized Flux2, force this case to:
      - `quantize: false`
      - `quantize_te: false`
      - `use_offload_conductor: false`
      - `layer_offloading: true`
  - Intent: ensure the `layer_offloading` ablation is truly active and train-valid rather than silently matching control.

### 3) Attempted to make quantized+conductor path active; found blockers
Tried enabling quantized Flux2 layer-offloading through conductor path.

Failures observed (in order):
1. `copy_` dispatch/type mismatch on quanto wrappers in offload transfer.
2. Missing `get_offload_conductor_stats` on `Flux2Klein9BModel` causing trainer metrics crash.
3. After transfer fixes, instability in quantized path (`loss is nan`) and then autograd failure (`loss does not require grad`).

Mitigations landed:
- `toolkit/memory_management/offload_conductor.py`
  - Added wrapper-aware direct-move path for quantized/quanto tensors.
  - Avoids allocator-backed `copy_` for wrapper tensors.
  - Synchronous `.to(..., non_blocking=False)` for wrapper moves.
- `toolkit/models/base_model.py`
  - Added default `get_offload_conductor_stats()` returning `None`.

Status after mitigations:
- Transfer-level crashes fixed.
- Quantized+conductor training path still not reliable/correct.

## Key Runs + Results

### A) Full ablation (before forcing active layer_offloading case)
- Run root: `output/phase4_ablations/20260214_134929`
- Analyzer:
  - `control_repeat`: `0.0700`
  - `layer_offloading`: `0.0679`
- Interpretation:
  - Looked healthy, but later confirmed this was because quantized Flux2 runtime disabled layer offloading.

### B) Full ablation (second confirmation of same behavior)
- Run root: `output/phase4_ablations/20260214_135533`
- Analyzer:
  - `control_repeat`: `0.0781`
  - `layer_offloading`: `0.0852`
- Runtime still printed layer-offloading auto-disable warning.

### C) Full ablation after forcing active `layer_offloading` case (unquantized for that case)
- Run root: `output/phase4_ablations/20260214_141148`
- Analyzer:
  - `control_repeat`: `0.0548`
  - `layer_offloading`: `0.1379`
  - `offload_conductor`: `0.0544`
  - `fused_back_pass`: `0.0616`
  - `stable_loss`: `0.0528`
- Interpretation:
  - `layer_offloading` is now truly active and shows meaningful divergence.
  - Comparison is not apples-to-apples because this case is unquantized while control remains quantized.

## Current State / Reality Check
- We now have an ablation that actually exercises `layer_offloading`.
- Quantized Flux2 + active offloading (via conductor) is still unstable/correctness-broken.
- For perf-plan intent (`docs/perf.md` says quantized support is required), this is a real remaining gap.

## Uncommitted Files Changed
- `scripts/run_phase4_ablations.py`
- `toolkit/memory_management/manager.py`
- `toolkit/memory_management/manager_modules.py`
- `toolkit/memory_management/offload_conductor.py`
- `toolkit/models/base_model.py`

## Next Session Priority

### Primary objective
Stabilize **quantized Flux2 layer offloading** so Phase 5/6 work is valid on quantized path.

### Suggested sequence
1. Reproduce quantized+conductor failure on fresh run with a minimal deterministic config.
2. Add targeted diagnostics in conductor path for quantized tensors:
   - tensor class/module
   - device transitions
   - requires_grad / grad_fn propagation around offload/load.
3. Verify whether parameter replacement (`param.data = ...`) is severing graph expectations for quantized wrappers.
4. If yes, implement wrapper-safe state transition strategy (likely avoiding direct data replacement semantics used for dense tensors).
5. Re-run single-case quantized layer_offloading until:
   - no NaN
   - backward works
   - non-zero gradients on expected LoRA params.
6. Re-run full ablation matrix and compare against a matched quantized control.

## Guardrail Follow-up (implemented later the same day)
- Runtime metadata now records `runtime_knobs.json` with:
  - `requested`
  - `effective` (config snapshot at metadata write time)
  - `adjustments` (auto-disables and reasons)
- Analyzer now reads runtime metadata and prints runtime adjustments when present.
- Caveat: this metadata is not full runtime activation telemetry for every feature; it reflects config-effective state plus recorded adjustments.

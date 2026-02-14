# Handoff: Quantized Offload-Conductor Stability Sweep (2026-02-14)

## Scope Completed
- Stabilized quantized Flux2 + offload conductor gradient-path failure (`loss does not require grad`).
- Added ablation analyzer support for per-config thresholds.
- Added runtime guardrail metadata so analyzer can distinguish requested vs config-effective knobs and surface runtime adjustments.
- Ran a longer 3-seed stability sweep on quantized + conductor config.

## Key Code Changes

### 1) Quantized offload-conductor stabilization
- `extensions_built_in/diffusion_models/flux2/src/model.py`
  - Offload conductor no longer forces reentrant checkpointing.
  - Added output grad hook to trigger `offload_conductor.start_backward()`.
- `toolkit/memory_management/offload_conductor.py`
  - Added wrapper-aware tensor move path for quantized/quanto tensors to avoid invalid allocator-backed `copy_` dispatch.
- `toolkit/models/base_model.py`
  - Added default offload stats API and runtime-adjustment tracking helpers.

### 2) Ablation analyzer thresholding
- `scripts/analyze_phase4_ablations.py`
  - Warn/fail thresholds can be read from config (with CLI override precedence).
  - Prints resolved thresholds in report output.
- `scripts/run_phase4_ablations.py`
  - Added `--analyze-warn-relative-delta`.
  - Pass-through warn/fail args only when explicitly provided.
- `config/perf_test.yaml`
  - Added:
    - `ablation_analyzer.warn_relative_delta: 0.25`
    - `ablation_analyzer.warn_control_repeat_relative_delta: 0.08`

### 3) Runtime knob guardrail
- `jobs/process/BaseSDTrainProcess.py`
  - Captures requested knobs before model load.
  - Writes per-run `runtime_knobs.json` with:
    - `requested`
    - `effective` (config snapshot at metadata write time)
    - `adjustments` (auto-disables and reasons)
- `extensions_built_in/diffusion_models/flux2/flux2_model.py`
  - Records runtime adjustments when Flux2 auto-disables layer offloading.
- `scripts/analyze_phase4_ablations.py`
  - Reads `runtime_knobs.json`.
  - Uses effective knobs when available.
  - Prints `Runtime Adjustments` section when adjustments exist.
  - Note: this is not full runtime activation telemetry for all features; it reflects config-effective state plus recorded adjustments.

### 4) Reference implementation alignment note
- `tmp/OneTrainer` still uses reentrant checkpointing for offload-checkpoint wrappers (`use_reentrant=True`) plus a dummy-grad workaround in wrapper output handling.
- This repo intentionally diverges for Flux2 quantized + conductor path by keeping checkpointing non-reentrant and triggering conductor backward transition via output grad hook.

## Validation / Runs

### A) Quantized+conductor ablation run with analyzer
- Command:
  - `uv run python scripts/run_phase4_ablations.py --config config/perf_test_quantized_offload_conductor.yaml --stable-loss-path datasets/stable_loss/ --double-control --analyze`
- Run root:
  - `output/phase4_ablations/20260214_153519`
- Result:
  - Completed successfully, `Warnings: none`.
  - `control_repeat relative_delta=0.0602`, below configured warn threshold `0.08`.

### B) Multi-seed stability sweep (longer run)
- Config basis:
  - `config/perf_test_quantized_offload_conductor.yaml`
- Generated seed configs:
  - `.agents/tmp/stability_sweep/seed_101.yaml`
  - `.agents/tmp/stability_sweep/seed_202.yaml`
  - `.agents/tmp/stability_sweep/seed_303.yaml`
- Sweep status log:
  - `.agents/tmp/stability_sweep/sweep_status.log`
- Outcomes:
  - Seed `101`: rc=0
  - Seed `202`: rc=0
  - Seed `303`: rc=0
  - No `nan` / grad-graph / traceback markers in logs.
  - All runs produced checkpoints + `runtime_knobs.json`.

## Commits Made
- `6b37901` - Stabilize quantized offload conductor and add config-aware ablation thresholds
- `04b0a00` - Record effective runtime knobs for ablation guardrails

## Current State
- Quantized + offload conductor path is currently stable for tested short/medium training windows.
- Guardrails now reduce misinterpretation by recording requested/effective config snapshots plus runtime adjustments.
- Runtime metadata currently does not guarantee post-setup activation state for every feature (for example, conductor activation success/failure is logged but not yet emitted as a dedicated runtime knob field).
- Ablation warnings are now calibrated via config thresholds rather than hardcoded defaults only.

## Next Step
1. Move to Phase 5/6 quantized-path performance tuning:
   - offload fraction policy,
   - transfer overlap behavior,
   - memory/throughput profiling under quantized conductor path.

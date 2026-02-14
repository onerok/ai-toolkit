# Phase 4 Handoff: Layer Offloading Drift (2026-02-14)

## Source of Truth
- `docs/perf.md`
- `docs/perf/Phase-4-architecture-agnostic-training.md`
- `scripts/run_phase4_ablations.py`
- `scripts/analyze_phase4_ablations.py`

## What Was Completed
- Phase 4 capability and startup gating work landed (model capability contract + trainer compatibility checks).
- Offload conductor was decoupled from fused-backward hard requirement.
- Optimizer `step_parameter` support added for:
  - `adafactor`
  - in-repo `adam8bit` / `adamw8`
  - `prodigy8bit`
  - `adamw` wrapper path
- Multi-device VRAM logging added:
  - aggregate `vram_gb`
  - per-device `vram_gpu_{idx}_gb`
- Ablation tooling was automated end-to-end:
  - runner supports fixed seed, `--double-control`, and optional `--analyze`
  - analyzer reports relative delta + cosine signals + control-repeat sanity

## Latest Verified Run
- Run root: `output/phase4_ablations/20260214_110624`
- Command used:
  - `uv run python scripts/run_phase4_ablations.py --config config/perf_test.yaml --stable-loss-path /path/to/stable_loss_images --double-control --analyze`

### Analyzer results
- `control_repeat`: `relative_delta=0.0677`, `cosine=0.9977`
- `fused_back_pass`: `relative_delta=0.0619`, `cosine=0.9981`
- `offload_conductor`: `relative_delta=0.0705`, `cosine=0.9975`
- `stable_loss`: `relative_delta=0.0914`, `cosine=0.9958`
- `layer_offloading`: `relative_delta=0.1528`, `cosine=0.9884` (largest deviation)

## Interpretation
- Determinism baseline is acceptable now (control repeat close to control).
- `layer_offloading` is the current outlier and primary regression surface.
- `fused_back_pass` and `offload_conductor` currently track within noise band.

## Next Session Priority
1. Investigate layer-offloading drift in:
   - `toolkit/memory_management/manager.py`
   - `toolkit/memory_management/manager_modules.py`
2. Add instrumentation around layer offload/load order and gradient-present offload behavior.
3. Re-run same fixed-seed ablation command and confirm `layer_offloading` moves closer to control-repeat band.

## Notes / Gotchas
- `training_seed` is required for deterministic training in this codebase (process-level seed).
- Runner now sets:
  - `process.training_seed`
  - `train.seed`
  - `sample.seed`
  - `sample.walk_seed=false`
- If base config uses `adamw8bit`, runner auto-switches to `adamw8` so fused mode can be tested.

# Phase 4: Architecture-Agnostic Training Features

## Goal
Make `fused_back_pass`, `use_offload_conductor`, and `stable_loss` work safely across model families without relying on Flux2-specific hooks.

## Scope
- In scope: training/runtime integration (`jobs/`, `toolkit/`, model loaders).
- Out of scope: benchmark config generalization (perf tooling may remain model-specific for now).

## Current Gaps
1. `fused_back_pass` depends on `optimizer.step_parameter`; common optimizers in this repo/env do not expose it.
2. `use_offload_conductor` is hard-gated by fused back pass and wired only for Flux2 path.
3. `stable_loss` is globally configurable but not safe for all models (some require `batch` in `predict_noise` paths).
4. system VRAM logging reports a single device only.
5. memory/offload path changes may alter effective LoRA strength; this must be treated as a regression surface.

## Design
### 1) Model Capability Contract
Add a small capability API on base model classes, e.g.:
- `supports_offload_conductor`
- `supports_stable_loss`
- `stable_loss_requires_batch`
- `supports_fused_parameter_step` (optional, optimizer-derived in trainer)

Trainer reads capabilities once and gates features explicitly.

### 2) Decouple Conductor from Fused Back Pass
- Remove hard requirement that conductor needs `train.fused_back_pass`.
- Keep fused mode as an optimization, not a prerequisite.
- Add generic conductor binding via checkpoint wrappers/hooks so non-Flux2 models can opt in without custom forward rewrites.

### 3) Safety Invariants (Fail Fast)
- Preserve strict offload safety: error if offloading layer with live grads in unsupported contexts.
- Allow deferred offload only for explicitly allowed async multi-GPU reduce path.
- Validate incompatible config combinations early (startup-time), not only via runtime warnings.

### 4) Optimizer Capability Strategy
- Add explicit optimizer capability checks/logging for fused parameter stepping.
- Implement optimizer patching for in-repo/common optimizers (`adafactor`, `adam8bit`, `prodigy8bit`, `adamw`) with `step_parameter` to make fused mode practical.
- Mirror OneTrainer-style config-time guard behavior: fail early when fused/offload settings are incompatible.

### 5) Stable Loss Capability Gating
- Gate stable loss by model capability before initialization.
- If a model requires `batch` for noise prediction and no eval-batch provider exists, disable stable loss with explicit reason.

### 6) Multi-GPU Metrics
- Replace single-device `torch.cuda.memory_reserved()` metric with per-device aggregation and per-device keys.

## Implementation Order
1. Capability contract + startup validation.
2. Conductor decoupling from fused back pass.
3. Generic checkpoint-wrapper conductor integration.
4. Optimizer `step_parameter` support + fused/offload compatibility guards.
5. Stable-loss gating and disable reasons.
6. Multi-device VRAM logging.

## Acceptance Criteria
- `use_offload_conductor` can activate on at least one non-Flux2 architecture with gradient checkpointing enabled.
- Training no longer silently claims fused mode when optimizer lacks per-parameter stepping support.
- `stable_loss` never crashes due to missing `batch`; it runs or is cleanly disabled.
- Metrics include total VRAM and per-device VRAM in multi-GPU runs.
- LoRA-strength behavior remains within expected tolerance when optimization knobs are toggled.

## Regression Validation (From Handoff)
Run a fixed-seed control and ablations in this order:
1. Baseline control with all new knobs off:
   - `train.fused_back_pass: false`
   - `model.use_offload_conductor: false`
   - `model.layer_offloading: false`
   - `train.stable_loss_enabled: false`
2. Re-enable one knob at a time (start with `model.layer_offloading`), compare:
   - generated-image strength at same LoRA weight
   - LoRA tensor norm / delta magnitude stats
3. If divergence appears with layer offloading, prioritize debugging:
   - `toolkit/memory_management/manager_modules.py`
   - `toolkit/memory_management/manager.py`

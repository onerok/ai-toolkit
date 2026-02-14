# Phase 3: Fused Backward Pass

**Impact:** Reduces peak memory by freeing gradients immediately after optimizer consumption.
**Risk:** Medium - optimizer/scaler integration and training-loop semantics.
**Effort:** 3-5 days
**Dependencies:** Phase 1 (parallel with Phase 2)

---

## Problem

Current training flow in this repo is effectively:

```text
backward(...) -> grads live on many params -> optimizer.step() -> zero_grad()
```

This keeps a large gradient set resident in VRAM until the step finishes. For large models, this can be multiple GB.

---

## Important Reality in This Repo

Before implementing, align with current architecture:

1. `SDTrainer` can call backward multiple times in one train step (for preservation paths), not just once.
2. Optimizer support is heterogeneous (`torch`, `prodigyopt`, custom optimizers).
3. Several custom optimizers already use `register_post_accumulate_grad_hook` for stochastic grad accumulation.
4. `TrainConfig` currently has no `fused_back_pass` flag.
5. Config blocks use `train:` (not `training:`). Keep this phase's flag under `train`, not `memory_management`.

---

## OneTrainer Reference (What To Reuse)

Reference: `tmp/OneTrainer/modules/trainer/GenericTrainer.py` (`__apply_fused_back_pass`).

Keep these patterns:
- Register per-parameter `register_post_accumulate_grad_hook`.
- Execute per-parameter optimizer logic only on update steps.
- Immediately set `tensor.grad = None` after successful parameter update.

Do not copy unsafe assumptions:
- No generic "manual AdamW fallback" for unsupported optimizers.
- No reliance on private PyTorch/GradScaler internals.

---

## Files To Modify

| File | Change |
|------|--------|
| NEW: `toolkit/fused_backward.py` | Manager + compatibility guardrails |
| `toolkit/config_modules.py` | Add `train.fused_back_pass` boolean (default `False`) |
| `jobs/process/BaseSDTrainProcess.py` | Setup/teardown manager and warnings |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Gate fused execution to correct backward/update points |
| `docs/perf.md` | Move/clarify phase-3 config under `train:` |

---

## Implementation Plan

### Step 1: Add Config Flag

In `TrainConfig`:

```python
self.fused_back_pass = kwargs.get("fused_back_pass", False)
```

Config example:

```yaml
train:
  fused_back_pass: false
```

### Step 2: Build `FusedBackwardManager` With Strict Compatibility

Manager responsibilities:
- Attach/detach post-accumulate hooks.
- Track whether current backward is an update step.
- Optionally disable for intermediate backward calls inside one train step.
- Count events for debugging/metrics.

Compatibility rule:
- Only enable fused optimizer stepping when optimizer exposes `step_parameter`.
- Otherwise, keep standard `optimizer.step()` path.
- Never implement optimizer math in manager fallback.

### Step 3: Mixed Precision Handling

Do not use private scaler fields.
If AMP support is needed for fused stepping, add a dedicated wrapper API (OneTrainer-style) and explicitly test it.

Initial safe rollout:
- Enable fused back pass for non-scaled path first.
- Keep AMP/fused disabled until scaler integration is validated.

### Step 4: Training Loop Integration

In `SDTrainer`:
- Set manager mode before each backward call:
  - non-update/backward fragments: hooks disabled
  - final update backward: hooks enabled when `is_optimizer_step`
- When fused optimizer stepping ran:
  - skip global `optimizer.step()`
  - keep scheduler/EMA semantics identical to baseline update cadence
- Keep existing gradient accumulation behavior unchanged.

### Step 5: Guardrails

- Warn when `fused_back_pass=true` and accumulation > 1 (little/no VRAM gain).
- Warn and auto-disable fused path when optimizer lacks required API.
- Ensure hooks are detached on teardown/exceptions.

---

## Constraints And Warnings

### Gradient Accumulation

With accumulation > 1, memory savings are limited because gradients must survive across microbatches.
`gradient_accumulation` and `gradient_accumulation_steps` both exist in this repo; handle both paths.

### Multi-Backward In A Single Step

Because this trainer can run multiple `backward(...)` calls in one logical step, fused hooks must not step parameters during intermediate losses unless explicitly intended.

### Gradient Clipping Semantics

Global norm clipping is not equivalent to per-parameter clipping.  
If fused path cannot preserve clipping semantics, document and gate it.

---

## Verification

Use the existing perf harness:

```bash
just perf-baseline "phase-3-before"
just perf-train steps=20 save_at=10
just perf-compare docs/perf/baselines/phase-3-before.json
```

Additional checks:
1. Peak VRAM decreases with single-step updates (`gradient_accumulation == 1` and `gradient_accumulation_steps == 1`).
2. Loss curve remains close to baseline over 100+ steps.
3. No NaN/inf regressions.
4. Scheduler and EMA step counts match baseline update cadence.
5. Fallback path (unsupported optimizer) behaves identically to current training.

---

## Rollback Plan

1. Set `train.fused_back_pass: false`.
2. Keep manager creation guarded by config and optimizer compatibility.
3. Training continues on standard backward/optimizer path.

---

## Success Criteria

- [ ] `train.fused_back_pass` added and documented.
- [ ] Fused path only enabled for compatible optimizers.
- [ ] No manual optimizer math fallback exists.
- [ ] Peak VRAM reduced for single-step updates (`gradient_accumulation == 1` and `gradient_accumulation_steps == 1`).
- [ ] Loss/quality parity within expected numerical tolerance.
- [ ] Safe fallback to standard path on incompatibility.

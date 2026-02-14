# Phase 7: Accelerate Bypass for Single-GPU

**Impact:** Eliminates Accelerate dispatch overhead (~1-5% per step)
**Risk:** Low — isolated to single-GPU path
**Effort:** 1-2 days
**Dependencies:** None (can be done anytime)

---

## Problem

Every `self.accelerator.backward(loss)` call goes through HuggingFace Accelerate's dispatch layer. For single-GPU training (the common case), this adds overhead with zero benefit.

### Current Path

```python
self.accelerator.backward(loss)
# → Accelerate checks distributed state
# → Accelerate handles gradient scaling
# → Accelerate calls loss.backward()
# → Accelerate handles gradient sync (no-op for single GPU)
```

### OneTrainer Path

```python
loss.backward()  # Direct call
```

---

## Files to Modify

| File | Change |
|------|--------|
| `jobs/process/BaseSDTrainProcess.py` | Bypass detection, context management, and prepare-path branching |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Backward/grad-clip path selection through `_backward` |
| `toolkit/config_modules.py` | Optional config field for explicit bypass policy |

---

## Correctness Guardrails

1. **Bypass must be process-aware, not `torch.cuda.device_count()` based.**
   - Use accelerator state (`num_processes`, distributed type, deepspeed/fsdp flags), not global visible devices.

2. **This repo’s loop uses Accelerate in more places than `backward()`.**
   - `accelerator.accumulate(...)`, `accelerator.prepare(...)`, `clip_grad_norm_`, and synchronization helpers are in the main path.
   - A partial bypass should be explicit about which calls stay on Accelerate.

3. **Mixed precision and fused backward safety must stay consistent.**
   - Existing fused-backward guards currently disable fused stepping with AMP GradScaler.
   - A bypass path that introduces manual scaling must preserve that behavior (or keep scaler off in bypass mode).

4. **Do not bypass in multi-process or plugin-driven runs.**
   - Always keep Accelerate path for DDP/FSDP/DeepSpeed and any multi-process launch.

### Step 1: Add explicit bypass policy and detection

**Primary file:** `jobs/process/BaseSDTrainProcess.py`

Add a policy like:
- `None`: auto
- `true`: force bypass (single-process only)
- `false`: force Accelerate

Auto mode should enable bypass only when:
- single process (`accelerator.num_processes == 1`)
- no deepspeed/fsdp/distributed plugin active
- AMP/scaler/fused-backward constraints are satisfied.

### Step 2: Route backward and grad clipping through one switch

**Primary file:** `extensions_built_in/sd_trainer/SDTrainer.py`

Current single call site is `_backward()`:
- Accelerate path: `self.accelerator.backward(loss)`
- Bypass path: `loss.backward()`

If bypass is active:
- use native `torch.nn.utils.clip_grad_norm_` where clipping is currently applied
- keep optimizer stepping semantics unchanged
- preserve fused-backward manager state transitions around backward.

### Step 3: Scope the first iteration to low-risk bypass

Start with:
- bypass only `accelerator.backward` and `accelerator.clip_grad_norm_`
- keep `accelerator.prepare` and `accelerator.accumulate` unchanged initially

Then benchmark. If overhead reduction is insufficient, phase a second pass that bypasses prepare/accumulate for single-process runs with full regression coverage.

---

## Configuration

```yaml
train:
  # Proposed new key (not present today):
  # bypass_accelerate: null  # null=auto, true=force single-process bypass, false=always use Accelerate
```

---

## Verification

### Test 1: Overhead measurement

```python
import time

# Warmup
for _ in range(10):
    train_step()

# With Accelerate
times_accelerate = []
for _ in range(100):
    start = time.time()
    train_step_accelerate()
    torch.cuda.synchronize()
    times_accelerate.append(time.time() - start)

# Without Accelerate
times_bypass = []
for _ in range(100):
    start = time.time()
    train_step_bypass()
    torch.cuda.synchronize()
    times_bypass.append(time.time() - start)

avg_accelerate = sum(times_accelerate) / len(times_accelerate)
avg_bypass = sum(times_bypass) / len(times_bypass)

print(f"Accelerate: {avg_accelerate * 1000:.2f} ms/step")
print(f"Bypass: {avg_bypass * 1000:.2f} ms/step")
print(f"Speedup: {avg_accelerate / avg_bypass:.2%}")
```

Expected: 1-5% speedup.

### Test 2: Loss equivalence

```python
# Run same seeds with both paths
torch.manual_seed(42)
losses_accelerate = [train_step_accelerate() for _ in range(100)]

torch.manual_seed(42)
losses_bypass = [train_step_bypass() for _ in range(100)]

# Should be identical
for i, (a, b) in enumerate(zip(losses_accelerate, losses_bypass)):
    assert abs(a - b) < 1e-5, f"Step {i}: {a} vs {b}"

print("Loss equivalence verified")
```

### Test 3: Multi-GPU still uses Accelerate

```python
process = BaseSDTrainProcess(...)
process.setup_accelerate_bypass()

# Should use Accelerate whenever process count > 1
assert process.accelerator.num_processes > 1
assert process.use_accelerate == True
```

---

## Edge Cases

- DeepSpeed/FSDP/DDP: force Accelerate path.
- AMP + fused backward: keep existing disablement/guards intact.
- Gradient accumulation: if `accelerator.accumulate` remains enabled, ensure bypassed backward still obeys update-step gating already used by the fused manager.

---

## Success Criteria

- [ ] 1-5% speedup on single GPU
- [ ] Loss curves identical to Accelerate path
- [ ] Multi-GPU correctly uses Accelerate
- [ ] Gradient scaling works correctly with FP16
- [ ] BF16 works without scaler
- [ ] Gradient accumulation works correctly

# Phase 6: Accelerate Bypass for Single-GPU

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
| `jobs/process/BaseSDTrainProcess.py` | Add bypass detection |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Conditional backward |

---

## Implementation

### Step 1: Add configuration and detection

**File:** `jobs/process/BaseSDTrainProcess.py`

```python
class BaseSDTrainProcess:
    def __init__(self, ...):
        # ... existing init ...
        self.use_accelerate = True  # Default to using Accelerate
        self._grad_scaler = None

    def setup_accelerate_bypass(self):
        """Configure whether to use Accelerate based on hardware and config."""
        # Check config override
        bypass = self.train_config.bypass_accelerate

        if bypass is None:
            # Auto-detect: bypass for single GPU
            self.use_accelerate = torch.cuda.device_count() > 1
        else:
            self.use_accelerate = not bypass

        if not self.use_accelerate:
            print("Accelerate bypass enabled (single-GPU optimization)")

            # Set up manual gradient scaler if needed
            if self.train_config.mixed_precision in ['fp16', 'bf16']:
                # bf16 doesn't need scaling on modern GPUs
                if self.train_config.mixed_precision == 'fp16':
                    self._grad_scaler = torch.cuda.amp.GradScaler()
                else:
                    self._grad_scaler = None
```

### Step 2: Conditional backward pass

**File:** `extensions_built_in/sd_trainer/SDTrainer.py`

```python
def backward_loss(self, loss: torch.Tensor) -> None:
    """Backward pass with optional Accelerate bypass.

    Args:
        loss: The loss tensor to backpropagate.
    """
    if self.use_accelerate:
        # Standard Accelerate path
        self.accelerator.backward(loss)
    else:
        # Direct backward (single-GPU optimization)
        if self._grad_scaler is not None:
            # FP16 needs scaling
            self._grad_scaler.scale(loss).backward()
        else:
            # BF16 or FP32: direct backward
            loss.backward()


def optimizer_step(self) -> None:
    """Optimizer step with optional Accelerate bypass."""
    if self.use_accelerate:
        # Accelerate handles everything
        if self.train_config.max_grad_norm:
            self.accelerator.clip_grad_norm_(
                self.params_to_optimize,
                self.train_config.max_grad_norm
            )
        self.optimizer.step()
    else:
        # Manual handling
        if self._grad_scaler is not None:
            # Unscale before clipping
            self._grad_scaler.unscale_(self.optimizer)

        if self.train_config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.params_to_optimize,
                self.train_config.max_grad_norm
            )

        if self._grad_scaler is not None:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()


# Update training loop
def hook_train_loop(self, batch):
    # ... existing code up to loss computation ...

    # Backward
    self.backward_loss(loss)

    # Optimizer step (if update step)
    if is_update_step:
        self.optimizer_step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    return loss.detach()
```

### Step 3: Model preparation bypass

When bypassing Accelerate, skip `accelerator.prepare()` for models:

```python
def setup_model(self):
    # ... existing model loading ...

    if self.use_accelerate:
        # Standard path: let Accelerate prepare models
        self.sd.unet, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.sd.unet, self.optimizer, self.lr_scheduler
        )
    else:
        # Bypass: move model to device manually
        self.sd.unet = self.sd.unet.to(self.device_torch)

        # Still use Accelerator for some utilities
        # but skip the model wrapping
```

---

## Configuration

```yaml
training:
  bypass_accelerate: null  # null = auto-detect, true = force bypass, false = always use Accelerate
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
# On multi-GPU system
assert torch.cuda.device_count() > 1

process = BaseSDTrainProcess(...)
process.setup_accelerate_bypass()

# Should auto-detect and use Accelerate
assert process.use_accelerate == True
```

---

## Edge Cases

### DeepSpeed / FSDP

If using DeepSpeed or FSDP (Accelerate integrations), bypass should be disabled:

```python
def setup_accelerate_bypass(self):
    if self.train_config.deepspeed or self.train_config.fsdp:
        self.use_accelerate = True
        print("Accelerate required for DeepSpeed/FSDP, bypass disabled")
        return

    # ... rest of detection logic ...
```

### Gradient Accumulation

Bypass works with gradient accumulation, but must handle scaling:

```python
def backward_loss(self, loss, is_last_micro_batch=False):
    # Scale loss for accumulation
    if self.train_config.gradient_accumulation_steps > 1:
        loss = loss / self.train_config.gradient_accumulation_steps

    if self.use_accelerate:
        self.accelerator.backward(loss)
    else:
        if self._grad_scaler:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()
```

---

## Success Criteria

- [ ] 1-5% speedup on single GPU
- [ ] Loss curves identical to Accelerate path
- [ ] Multi-GPU correctly uses Accelerate
- [ ] Gradient scaling works correctly with FP16
- [ ] BF16 works without scaler
- [ ] Gradient accumulation works correctly

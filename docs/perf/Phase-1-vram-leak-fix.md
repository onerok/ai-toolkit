# Phase 1: VRAM Leak Fix

**Impact:** Eliminates the #1 reported performance issue (3.6x-64x slowdowns)
**Risk:** Low — isolated to save path
**Effort:** 1-2 days
**Dependencies:** None

---

## Problem

The `save()` method at `BaseSDTrainProcess.py:495` calls `flush()` (which does `torch.cuda.empty_cache()` + `gc.collect()`), then performs model serialization. Multiple potential leak sources:

1. **Ping-pong buffers not cleared:** `_DEVICE_STATE` maintains per-device buffers (`w_buffers`, `b_buffers`, `w_bwd_buffers`, `w_grad_buffers`, `b_grad_buffers`) that hold references to GPU tensors
2. **State dict cloning:** `copy.deepcopy(self.meta)` and state dict operations may create tensor copies that aren't released
3. **Async operations not synchronized:** Non-blocking `.to()` operations may complete after `flush()` is called

### Related Issues
- Issue #504
- Issue #575
- Issue #387

---

## Files to Modify

| File | Change |
|------|--------|
| `toolkit/memory_management/manager_modules.py` | Add `clear_device_state_buffers()` |
| `jobs/process/BaseSDTrainProcess.py` | Modify module-level `flush()` and `save()` to use buffer-clear + synchronization on checkpoint path |

---

## Implementation

### Step 1: Add buffer cleanup function

**File:** `toolkit/memory_management/manager_modules.py`

```python
def clear_device_state_buffers(device: torch.device = None):
    """Clear all ping-pong buffers to release GPU memory.

    Call this before checkpoint saves to ensure bouncing buffers
    don't hold references to GPU tensors.

    Args:
        device: Specific device to clear, or None for all devices.
    """
    devices = [device] if device else list(_DEVICE_STATE.keys())
    for dev in devices:
        if dev in _DEVICE_STATE:
            state = _DEVICE_STATE[dev]
            for key in ['w_buffers', 'b_buffers', 'w_bwd_buffers',
                        'w_grad_buffers', 'b_grad_buffers']:
                if key in state:
                    state[key] = [None, None]
```

### Step 2: Harden `flush()` in save owner module

**File:** `jobs/process/BaseSDTrainProcess.py`

`save()` resolves `flush()` to the module-level function in this file, so this is the canonical target for save-path memory cleanup.

```python
def flush(clear_bouncing_buffers=False, synchronize=False, sync_device: torch.device = None):
    """Clean up GPU memory.

    Args:
        clear_bouncing_buffers: If True, also clear the ping-pong buffers
            used for CPU/GPU weight bouncing. Required before saves.
        synchronize: If True, wait for pending CUDA ops before cleanup.
        sync_device: Device to synchronize. If None, synchronize each CUDA
            device present in memory-manager state.
    """
    if torch.cuda.is_available() and synchronize:
        from toolkit.memory_management.manager_modules import _DEVICE_STATE

        # Ensure async .to() transfers finish before clearing references.
        # Prefer known manager devices, fallback to provided device, then current device.
        cuda_devices = [
            dev for dev in _DEVICE_STATE.keys()
            if isinstance(dev, torch.device) and dev.type == "cuda"
        ]
        if sync_device is not None:
            cuda_devices = [sync_device]
        if not cuda_devices:
            cuda_devices = [torch.cuda.current_device()]
        for dev in cuda_devices:
            with torch.cuda.device(dev):
                torch.cuda.synchronize()

    if clear_bouncing_buffers:
        from toolkit.memory_management.manager_modules import clear_device_state_buffers
        clear_device_state_buffers()

    # Release Python references before returning memory to CUDA allocator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Step 3: Update `save()` checkpoint cleanup calls

**File:** `jobs/process/BaseSDTrainProcess.py`

```python
def save(self, step=None):
    if not self.accelerator.is_main_process:
        return

    # Pre-save cleanup: finish async ops, then clear bouncing buffers
    flush(
        clear_bouncing_buffers=True,
        synchronize=True,
        sync_device=self.device_torch,
    )

    if self.ema is not None:
        self.ema.eval()

    # ... existing save logic (unchanged) ...

    # Post-save cleanup
    self.post_save_cleanup()

def post_save_cleanup(self):
    """Ensure clean state after checkpoint save."""
    flush(
        clear_bouncing_buffers=True,
        synchronize=True,
        sync_device=self.device_torch,
    )
```

Also keep existing control flow unchanged:
- `self.clean_up_saves()`
- `self.post_save_hook(file_path)`
- `if self.ema is not None: self.ema.train()`

### Step 4: Add diagnostic mode (optional but recommended)

**File:** `jobs/process/BaseSDTrainProcess.py`

```python
import os

def save(self, step=None):
    if not self.accelerator.is_main_process:
        return

    # Memory debugging (enable with AITK_MEMORY_DEBUG=1)
    memory_debug = os.environ.get('AITK_MEMORY_DEBUG')
    debug_enabled = bool(memory_debug and torch.cuda.is_available())
    if debug_enabled:
        torch.cuda.memory._record_memory_history()

    flush(
        clear_bouncing_buffers=True,
        synchronize=True,
        sync_device=self.device_torch,
    )

    try:
        # ... existing save logic ...
        pass
    finally:
        if debug_enabled:
            snapshot_path = f"checkpoint_memory_step{step}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"Memory snapshot saved to {snapshot_path}")

    self.post_save_cleanup()
```

---

## Verification

### Test 0: Reproduce baseline before patch

Run once on current branch and record:
- VRAM before save, peak during save, VRAM 1-2 steps after save
- s/it for 10 steps before and after save

This avoids merging a fix for a non-reproducible issue.

### Test 1: Basic VRAM recovery

Use a config that already saves checkpoints periodically (set in YAML), then run:
```bash
uv run python run.py config/train_lora_flux_24gb.yaml
```

Monitor with:
```bash
watch -n 1 nvidia-smi
```

**Expected:** VRAM usage should return to pre-save levels within 1-2 steps after save.

### Test 2: Measure s/it impact

```python
# Add timing instrumentation
import time

# Before save
pre_save_times = []
for i in range(10):
    start = time.time()
    # run training step
    pre_save_times.append(time.time() - start)

# Trigger save
save(step=50)

# After save
post_save_times = []
for i in range(10):
    start = time.time()
    # run training step
    post_save_times.append(time.time() - start)

print(f"Pre-save avg: {sum(pre_save_times)/10:.3f}s")
print(f"Post-save avg: {sum(post_save_times)/10:.3f}s")
print(f"Ratio: {sum(post_save_times)/sum(pre_save_times):.2f}x")
```

**Expected:** Ratio should be ~1.0x (no slowdown after save).

### Test 3: Memory allocation tracking

```python
# Check memory stats
before = torch.cuda.memory_allocated()
save(step=50)
after = torch.cuda.memory_allocated()

print(f"Before: {before / 1e9:.2f} GB")
print(f"After: {after / 1e9:.2f} GB")
print(f"Leaked: {(after - before) / 1e6:.2f} MB")
```

**Expected:** Leaked should be < 10 MB.

### Test 4: Debug mode snapshot analysis

```bash
AITK_MEMORY_DEBUG=1 uv run python run.py config/train_lora_flux_24gb.yaml
```

Then analyze:
```python
import pickle
import torch.cuda.memory as mem

with open("checkpoint_memory_step50.pickle", "rb") as f:
    snapshot = pickle.load(f)

# Use torch memory visualizer or analyze programmatically
```

---

## Rollback Plan

If issues arise, revert to original `flush()` without parameters:

```python
def flush():
    torch.cuda.empty_cache()
    gc.collect()
```

The changes are isolated to the save path and don't affect training correctness.

---

## Success Criteria

- [ ] VRAM usage returns to pre-save levels after checkpoint
- [ ] No performance degradation after save (post/pre s/it <= 1.05x)
- [ ] No training correctness impact (loss curves identical)
- [ ] `clear_device_state_buffers()` properly resets all buffer types

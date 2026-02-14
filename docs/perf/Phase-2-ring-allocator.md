# Phase 2: Static Ring Buffer Memory Allocator

**Impact:** Eliminates per-transfer allocation overhead, reduces fragmentation
**Risk:** Low-Medium — memory management change
**Effort:** 3-4 days
**Dependencies:** Phase 1 (VRAM leak fix)

---

## Problem

Current approach: Each bouncing forward/backward creates new GPU tensors via `.to(device, non_blocking=True)`. The ping-pong buffers (`w_buffers[idx]`) are reassigned each call, but old buffers linger until GC.

This causes:
- Per-transfer allocation overhead (CUDA malloc is expensive)
- Memory fragmentation over thousands of steps
- Unpredictable VRAM usage patterns

### OneTrainer Reference

`LayerOffloadConductor.py:45-221` implements `StaticLayerAllocator` + `StaticLayerTensorAllocator`:
- Pre-allocate int8 byte tensors with 16-byte alignment
- All layer weights are sliced from this pre-allocated pool
- Zero allocation overhead after initialization

---

## Files to Create/Modify

| File | Change |
|------|--------|
| NEW: `toolkit/memory_management/ring_allocator.py` | Ring buffer allocator class |
| `toolkit/memory_management/manager_modules.py` | Integrate allocator with bouncing |
| `toolkit/memory_management/manager.py` | Add initialization hooks |

---

## Implementation

### Step 1: Create RingBufferAllocator

**File:** `toolkit/memory_management/ring_allocator.py`

```python
"""
Static ring buffer allocator for layer weight transfers.

Adapted from OneTrainer's StaticLayerAllocator pattern.
Pre-allocates memory pools to eliminate per-transfer allocation overhead.
"""

import torch
from torch import nn
from typing import Optional


def _ceil_16(n: int) -> int:
    """Round up to nearest 16-byte boundary."""
    return n + (16 - (n % 16)) % 16


def _floor_16(n: int) -> int:
    """Round down to nearest 16-byte boundary."""
    return n - (n % 16)


class RingBufferAllocator:
    """Pre-allocated ring buffer for layer weight transfers.

    Uses multiple cache tensors to reduce fragmentation and allow
    concurrent allocations. Tensors are lazily allocated on first use.

    Safety: This allocator guards against:
    - Wrap-around overwrites of live allocations
    - Tensors larger than a single cache tensor
    - Out-of-order deallocations leaving holes

    Args:
        device: Target device for allocations (GPU or CPU).
        target_bytes: Total bytes to allocate across all cache tensors.
        num_cache_tensors: Number of cache tensors to split allocation across.
    """

    def __init__(
        self,
        device: torch.device,
        target_bytes: int,
        num_cache_tensors: int = 4
    ):
        self.device = device
        self.is_pinned = device.type == "cpu"

        # Calculate tensor size with overhead for alignment
        # Add max possible tensor size + alignment padding
        max_tensor_bytes = target_bytes // num_cache_tensors
        self.cache_tensor_size = _ceil_16(max_tensor_bytes + 4096)

        # Total capacity across all cache tensors
        self.total_capacity = self.cache_tensor_size * num_cache_tensors

        # Lazy allocation - tensors created on first use
        self.cache_tensors: list[Optional[torch.Tensor]] = [None] * num_cache_tensors

        # Ring buffer tracking (byte positions in virtual linear space)
        self.allocation_start = 0  # First allocated byte (oldest live allocation)
        self.allocation_end = 0    # First unallocated byte

        # Track allocations per layer for ordered deallocation
        self._layer_allocations: dict[int, tuple[int, int]] = {}
        # Track deallocation order for hole detection
        self._pending_deallocations: set[tuple[int, int]] = set()

    def _ensure_tensor(self, index: int) -> None:
        """Lazily allocate cache tensor on first use."""
        if self.cache_tensors[index] is None:
            # Clear any fragmented memory first
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Allocate as int8 byte array
            tensor = torch.zeros(
                self.cache_tensor_size,
                dtype=torch.int8,
                device=self.device
            )

            # Pin CPU tensors for faster H2D transfers
            if self.is_pinned:
                tensor = tensor.pin_memory()

            self.cache_tensors[index] = tensor

    def allocate_like(
        self,
        source_tensor: torch.Tensor,
        layer_index: int = -1
    ) -> Optional[torch.Tensor]:
        """Allocate a view into the ring buffer matching source shape/dtype.

        Args:
            source_tensor: Template tensor for shape and dtype.
            layer_index: Optional layer index for tracking (enables deallocation).

        Returns:
            A tensor view into the ring buffer with matching shape/dtype,
            or None if allocation cannot be satisfied (tensor too large or
            would overwrite live data).

        Note:
            Returns None instead of raising to allow fallback to dynamic allocation.
        """
        num_bytes = _ceil_16(source_tensor.numel() * source_tensor.element_size())

        # Guard: tensor larger than single cache tensor cannot be allocated
        if num_bytes > self.cache_tensor_size:
            return None

        # Calculate current position in ring
        cache_idx = self.allocation_end // self.cache_tensor_size
        offset = _ceil_16(self.allocation_end % self.cache_tensor_size)

        # Check if we need to wrap to next tensor
        if offset + num_bytes > self.cache_tensor_size:
            cache_idx = (cache_idx + 1) % len(self.cache_tensors)
            offset = 0

        # Calculate new position (may wrap around)
        new_position = cache_idx * self.cache_tensor_size + offset
        if new_position >= self.total_capacity:
            # Wrap to beginning of ring
            cache_idx = 0
            offset = 0
            new_position = 0

        new_end = new_position + num_bytes

        # Guard: check for wrap-around overwrite of live allocations
        # Live region is [allocation_start, allocation_end)
        # We're trying to allocate [new_position, new_end)
        if self._would_overwrite_live(new_position, new_end):
            return None

        self._ensure_tensor(cache_idx)

        # Get view with correct dtype/shape
        byte_view = self.cache_tensors[cache_idx][offset:offset + num_bytes]
        tensor_view = byte_view.view(dtype=source_tensor.dtype).view(source_tensor.shape)

        # Track allocation
        self.allocation_end = new_end

        if layer_index >= 0:
            self._layer_allocations[layer_index] = (new_position, new_end)

        return tensor_view

    def _would_overwrite_live(self, new_start: int, new_end: int) -> bool:
        """Check if new allocation would overwrite live data.

        Handles wrap-around case where allocation_end < allocation_start.
        """
        # No live allocations
        if self.allocation_start == self.allocation_end:
            return False

        # Calculate live bytes (handles wrap-around)
        if self.allocation_end >= self.allocation_start:
            # No wrap: live region is [start, end)
            live_bytes = self.allocation_end - self.allocation_start
        else:
            # Wrapped: live region spans [start, capacity) + [0, end)
            live_bytes = (self.total_capacity - self.allocation_start) + self.allocation_end

        # Check if adding this allocation would exceed capacity
        new_bytes = new_end - new_start
        if live_bytes + new_bytes > self.total_capacity:
            return True

        # Check direct overlap with live region
        if self.allocation_end >= self.allocation_start:
            # Simple case: live is [start, end)
            # New allocation overlaps if it intersects this range
            if new_start < self.allocation_end and new_end > self.allocation_start:
                # But we're allocating at allocation_end, so this should be fine
                # unless we wrapped
                if new_start < self.allocation_start:
                    return True
        else:
            # Wrapped case: live spans [start, capacity) and [0, end)
            # New allocation in [0, allocation_end) would overwrite
            if new_start < self.allocation_end:
                return True
            # New allocation overlapping [allocation_start, capacity) would overwrite
            if new_end > self.allocation_start and new_start < self.total_capacity:
                return True

        return False

    def deallocate_layer(self, layer_index: int) -> None:
        """Mark a layer's allocation as reclaimable.

        Handles out-of-order deallocations by tracking pending frees
        and only advancing allocation_start when contiguous space is freed.

        Args:
            layer_index: The layer index passed to allocate_like().
        """
        if layer_index not in self._layer_allocations:
            return

        start, end = self._layer_allocations.pop(layer_index)

        # If this is the oldest allocation, advance start immediately
        if start == self.allocation_start:
            self.allocation_start = end
            # Also consume any pending deallocations that are now contiguous
            self._consume_pending_deallocations()
        else:
            # Out-of-order deallocation - mark as pending
            self._pending_deallocations.add((start, end))

    def _consume_pending_deallocations(self) -> None:
        """Advance allocation_start through any contiguous pending frees."""
        changed = True
        while changed and self._pending_deallocations:
            changed = False
            for start, end in list(self._pending_deallocations):
                if start == self.allocation_start:
                    self.allocation_start = end
                    self._pending_deallocations.discard((start, end))
                    changed = True
                    break

    def clear(self) -> None:
        """Reset all allocations without releasing memory."""
        self.allocation_start = 0
        self.allocation_end = 0
        self._layer_allocations.clear()
        self._pending_deallocations.clear()

    def deallocate_cache(self) -> None:
        """Release all pre-allocated memory."""
        for i, tensor in enumerate(self.cache_tensors):
            if tensor is not None and self.is_pinned:
                # Unpin before releasing
                # Note: PyTorch doesn't have explicit unpin, just release reference
                pass
        self.cache_tensors = [None] * len(self.cache_tensors)
        self.clear()

    def get_stats(self) -> dict:
        """Get allocation statistics for debugging."""
        allocated_tensors = sum(1 for t in self.cache_tensors if t is not None)
        total_bytes = allocated_tensors * self.cache_tensor_size

        # Handle wrap-around for used_bytes calculation
        if self.allocation_end >= self.allocation_start:
            used_bytes = self.allocation_end - self.allocation_start
        else:
            used_bytes = (self.total_capacity - self.allocation_start) + self.allocation_end

        return {
            "device": str(self.device),
            "cache_tensors_allocated": allocated_tensors,
            "cache_tensors_total": len(self.cache_tensors),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "utilization": used_bytes / total_bytes if total_bytes > 0 else 0,
            "active_layers": len(self._layer_allocations),
            "pending_deallocations": len(self._pending_deallocations),
        }


class ActivationAllocator:
    """Allocator specialized for activation tensors.

    Unlike RingBufferAllocator, this grows dynamically during the first
    epoch and consolidates into a single buffer afterward.

    Adapted from OneTrainer's StaticActivationAllocator.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_pinned = device.type == "cpu"

        self._cache_tensors: list[torch.Tensor] = []
        self._current_tensor_idx = 0
        self._current_offset = 0
        self._allocated_bytes = 0
        self._max_allocated_bytes = 0

    def reserve_cache(self, tensors: list[torch.Tensor]) -> None:
        """Reserve space for a list of tensors."""
        num_bytes = sum(
            t.element_size() * t.numel() for t in tensors
        ) + len(tensors) * 16  # Alignment padding

        if num_bytes == 0:
            return

        # On first epoch, use max observed
        if len(self._cache_tensors) == 0:
            num_bytes = max(num_bytes, self._max_allocated_bytes)

        # Find space in existing tensors
        while self._current_tensor_idx < len(self._cache_tensors):
            available = (
                self._cache_tensors[self._current_tensor_idx].shape[0]
                - self._current_offset
            )
            if available >= num_bytes:
                return
            self._current_tensor_idx += 1
            self._current_offset = 0

        # Need to allocate new tensor
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        cache_tensor = torch.zeros(num_bytes, dtype=torch.int8, device=self.device)
        if self.is_pinned:
            cache_tensor = cache_tensor.pin_memory()

        self._cache_tensors.append(cache_tensor)
        self._allocated_bytes += num_bytes
        self._max_allocated_bytes = max(self._max_allocated_bytes, self._allocated_bytes)

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        """Allocate space for a tensor."""
        num_bytes = source_tensor.element_size() * source_tensor.numel()

        cache_tensor = self._cache_tensors[self._current_tensor_idx]
        allocated = cache_tensor[self._current_offset:self._current_offset + num_bytes]
        self._current_offset += _ceil_16(num_bytes)

        return allocated.view(dtype=source_tensor.dtype).view(source_tensor.shape)

    def deallocate(self) -> None:
        """Reset allocations and optionally consolidate."""
        # Consolidate multiple tensors into one for efficiency
        if len(self._cache_tensors) > 1:
            for tensor in self._cache_tensors:
                if self.is_pinned:
                    pass  # Release pinned memory

            self._cache_tensors = []

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Allocate single consolidated tensor
            num_bytes = self._allocated_bytes + 4096  # Alignment overhead
            cache_tensor = torch.zeros(num_bytes, dtype=torch.int8, device=self.device)
            if self.is_pinned:
                cache_tensor = cache_tensor.pin_memory()

            self._cache_tensors = [cache_tensor]

        self._current_tensor_idx = 0
        self._current_offset = 0
        self._allocated_bytes = sum(t.shape[0] for t in self._cache_tensors)

    def deallocate_cache(self) -> None:
        """Release all memory."""
        self._cache_tensors = []
        self._current_tensor_idx = 0
        self._current_offset = 0
        self._allocated_bytes = 0
```

### Step 2: Integrate with bouncing modules

**File:** `toolkit/memory_management/manager_modules.py`

Add to `_DEVICE_STATE` initialization:

```python
def _get_device_state(device: torch.device) -> dict:
    if device not in _DEVICE_STATE:
        _DEVICE_STATE[device] = {
            # ... existing fields ...
            "ring_allocator_gpu": None,  # RingBufferAllocator instance
            "use_ring_allocator": False,  # Enable/disable flag
        }
    return _DEVICE_STATE[device]
```

Modify `_BouncingLinearFn.forward` (note: actual signature is `forward(ctx, x, weight_cpu, bias_cpu, device)`):

```python
@staticmethod
def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
    # choose compute dtype to match activations
    target_dtype = (
        x.dtype
        if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        else torch.bfloat16
    )

    state = _get_device_state(device)
    allocator = state.get("ring_allocator_gpu") if state.get("use_ring_allocator") else None

    # GPU-side dequant/cast for quantized; float path unchanged
    # NOTE: Ring allocator is used for the OUTPUT buffer, not the source tensor
    def _materialize_linear_weight(cpu_w, dev, alloc=None, layer_idx=-1):
        if _is_quantized_tensor(cpu_w):
            # Quantized path: move to GPU -> dequantize -> cast
            # Cannot use ring allocator for quantized (size changes after dequant)
            w_q_gpu = cpu_w.to(dev, non_blocking=True)
            try:
                w_fp_gpu = w_q_gpu.dequantize()
            except Exception:
                w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
            if w_fp_gpu.dtype != target_dtype:
                w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
            return w_fp_gpu

        # Float path: use ring allocator if available
        if alloc is not None:
            buf = alloc.allocate_like(cpu_w, layer_index=layer_idx)
            if buf is not None:
                buf.copy_(cpu_w, non_blocking=True)
                return buf
            # Fallback if allocator returns None (too large or would overwrite)

        # Original dynamic allocation path
        return cpu_w.to(dev, non_blocking=True)

    # ... CPU fallback path unchanged ...

    ts = state["transfer_stream"]
    w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
    ev_tx_f = state["transfer_forward_finished_event"]
    ev_cu_s = state["compute_forward_start_event"]
    idx = state["forward_clk"]

    with torch.cuda.stream(ts):
        ts.wait_event(ev_cu_s)
        # Deallocate previous allocation in this ping-pong slot before reusing it.
        # With bouncing, idx alternates 0/1 so this tracks the two in-flight buffers.
        if allocator is not None:
            allocator.deallocate_layer(idx)
        # Pass allocator and layer index for ring buffer allocation
        w_bufs[idx] = _materialize_linear_weight(
            weight_cpu, device, alloc=allocator, layer_idx=idx
        )
        b_bufs[idx] = (
            bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
        )
        state["forward_clk"] ^= 1
        ev_tx_f.record()

    # ... rest of forward unchanged ...
```

**Key changes from original:**
1. Correct function signature: `(ctx, x, weight_cpu, bias_cpu, device)`
2. Ring allocator only used for float tensors (quantized tensors change size after dequant)
3. Graceful fallback to dynamic allocation if `allocate_like` returns `None`
4. Use ping-pong slot index (`idx`) for allocation/deallocation tracking in Phase 2
5. Call `deallocate_layer(idx)` before slot reuse to prevent stale live ranges

### Step 3: Add initialization to MemoryManager

**File:** `toolkit/memory_management/manager.py`

```python
from .ring_allocator import RingBufferAllocator

class MemoryManager:
    # ... existing code ...

    def initialize_ring_allocators(
        self,
        model_size_bytes: int,
        device: torch.device,
        gpu_fraction: float = 0.25,
    ) -> None:
        """Initialize pre-allocated GPU buffer based on model size.

        The ring allocator pre-allocates a GPU-side buffer for weight staging
        during bouncing transfers. This eliminates per-transfer CUDA malloc
        overhead after the first epoch.

        Args:
            model_size_bytes: Total size of model parameters in bytes.
            device: GPU device to allocate on.
            gpu_fraction: Fraction of model size for GPU staging buffer.
                         Default 0.25 assumes ~4 layers active concurrently.

        Note:
            CPU-side allocator is NOT implemented here. CPU pinned memory for
            offloaded weights is handled separately in Phase 5 (Offload Conductor).
        """
        from .manager_modules import _get_device_state

        state = _get_device_state(device)

        # GPU-side buffer for weight staging during compute
        target_bytes = int(model_size_bytes * gpu_fraction)
        state["ring_allocator_gpu"] = RingBufferAllocator(
            device,
            target_bytes=target_bytes
        )

        state["use_ring_allocator"] = True

        print(f"Ring allocator initialized:")
        print(f"  GPU buffer: {target_bytes / 1e9:.2f} GB on {device}")
        print(f"  Cache tensors: {state['ring_allocator_gpu'].cache_tensors.__len__()}")

    def disable_ring_allocators(self, device: torch.device) -> None:
        """Disable and release ring allocators."""
        from .manager_modules import _get_device_state

        state = _get_device_state(device)

        if state.get("ring_allocator_gpu"):
            state["ring_allocator_gpu"].deallocate_cache()
            state["ring_allocator_gpu"] = None

        state["use_ring_allocator"] = False
```

---

## Verification

### Test 1: CUDA allocation retry reduction

Use `torch.cuda.memory_stats()` to measure allocator pressure. The `num_alloc_retries`
metric indicates how often the CUDA caching allocator had to retry due to fragmentation.

```python
import torch

device = torch.device("cuda:0")

# Reset stats before test
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.reset_accumulated_memory_stats(device)

# Run 100 training steps
# ...

# Check allocation metrics
stats = torch.cuda.memory_stats(device)

print(f"Allocation retries: {stats.get('num_alloc_retries', 0)}")
print(f"Allocations: {stats.get('allocation.all.allocated', 0)}")
print(f"OOM retries: {stats.get('num_ooms', 0)}")

# With ring allocator: retries should be near-zero after warmup
# Without: retries accumulate over training due to fragmentation
```

### Test 2: Memory fragmentation

Compare reserved vs allocated memory. High fragmentation = large gap between these.

```python
import torch

device = torch.device("cuda:0")

# Before training
torch.cuda.reset_peak_memory_stats(device)

# Run training
# ...

# After training
stats = torch.cuda.memory_stats(device)
reserved = stats['reserved_bytes.all.current']
allocated = stats['allocated_bytes.all.current']
peak_reserved = stats['reserved_bytes.all.peak']
peak_allocated = stats['allocated_bytes.all.peak']

fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0
peak_fragmentation = (peak_reserved - peak_allocated) / peak_reserved if peak_reserved > 0 else 0

print(f"Current: {allocated / 1e9:.2f} GB allocated, {reserved / 1e9:.2f} GB reserved")
print(f"Current fragmentation: {fragmentation:.1%}")
print(f"Peak: {peak_allocated / 1e9:.2f} GB allocated, {peak_reserved / 1e9:.2f} GB reserved")
print(f"Peak fragmentation: {peak_fragmentation:.1%}")

# Target: fragmentation < 10% with ring allocator
```

### Test 3: Ring allocator stats

```python
from toolkit.memory_management.manager_modules import _get_device_state

state = _get_device_state(torch.device("cuda:0"))
if state.get("ring_allocator_gpu"):
    stats = state["ring_allocator_gpu"].get_stats()
    print(f"Ring allocator stats:")
    print(f"  Device: {stats['device']}")
    print(f"  Cache tensors: {stats['cache_tensors_allocated']}/{stats['cache_tensors_total']}")
    print(f"  Total bytes: {stats['total_bytes'] / 1e9:.2f} GB")
    print(f"  Used bytes: {stats['used_bytes'] / 1e9:.2f} GB")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Active layers: {stats['active_layers']}")
```

### Test 4: Fallback behavior

Verify graceful fallback when allocator cannot satisfy request:

```python
from toolkit.memory_management.ring_allocator import RingBufferAllocator
import torch

# Create small allocator
alloc = RingBufferAllocator(
    torch.device("cuda:0"),
    target_bytes=1024 * 1024,  # 1 MB total
    num_cache_tensors=2
)

# Try to allocate tensor larger than cache tensor
large_tensor = torch.zeros(1024, 1024, dtype=torch.float32)  # 4 MB
result = alloc.allocate_like(large_tensor)
assert result is None, "Should return None for oversized tensor"

# Verify small allocations work
small_tensor = torch.zeros(128, 128, dtype=torch.float32)  # 64 KB
result = alloc.allocate_like(small_tensor)
assert result is not None, "Should succeed for small tensor"
assert result.shape == small_tensor.shape
```

---

## Configuration

Add to training config:

```yaml
memory_management:
  use_ring_allocator: true  # default: true
  ring_allocator_gpu_fraction: 0.25  # Fraction of model size for GPU staging buffer
```

**Note:** `ring_allocator_cpu_fraction` is NOT implemented in Phase 2. CPU-side pinned
memory pooling is deferred to Phase 5 (Offload Conductor) where it can be coordinated
with the 3-stream transfer architecture.

---

## Success Criteria

- [ ] CUDA allocation calls drop to near-zero after first epoch
- [ ] Peak VRAM usage is more predictable
- [ ] No training correctness impact
- [ ] Allocator stats show healthy utilization (50-80%)

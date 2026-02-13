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

        # Lazy allocation - tensors created on first use
        self.cache_tensors: list[Optional[torch.Tensor]] = [None] * num_cache_tensors

        # Ring buffer tracking
        self.allocation_start = 0  # First allocated byte
        self.allocation_end = 0    # First unallocated byte

        # Track allocations per layer for deallocation
        self._layer_allocations: dict[int, tuple[int, int]] = {}

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
    ) -> torch.Tensor:
        """Allocate a view into the ring buffer matching source shape/dtype.

        Args:
            source_tensor: Template tensor for shape and dtype.
            layer_index: Optional layer index for tracking (enables deallocation).

        Returns:
            A tensor view into the ring buffer with matching shape/dtype.
            The returned tensor shares memory with the ring buffer.
        """
        num_bytes = source_tensor.numel() * source_tensor.element_size()
        total_cache_bytes = self.cache_tensor_size * len(self.cache_tensors)

        # Find slot in ring buffer
        cache_idx = self.allocation_end // self.cache_tensor_size
        offset = _ceil_16(self.allocation_end % self.cache_tensor_size)

        # Check if we need to wrap to next tensor
        if offset + num_bytes > self.cache_tensor_size:
            cache_idx = (cache_idx + 1) % len(self.cache_tensors)
            offset = 0

        # Check for ring buffer overflow (wrap around)
        new_position = cache_idx * self.cache_tensor_size + offset
        if new_position + num_bytes > total_cache_bytes:
            # Wrap to beginning
            cache_idx = 0
            offset = 0
            new_position = 0

        self._ensure_tensor(cache_idx)

        # Get view with correct dtype/shape
        byte_view = self.cache_tensors[cache_idx][offset:offset + num_bytes]
        tensor_view = byte_view.view(dtype=source_tensor.dtype).view(source_tensor.shape)

        # Track allocation
        old_end = self.allocation_end
        self.allocation_end = new_position + num_bytes

        if layer_index >= 0:
            self._layer_allocations[layer_index] = (old_end, self.allocation_end)

        return tensor_view

    def deallocate_layer(self, layer_index: int) -> None:
        """Mark a layer's allocation as reclaimable.

        Args:
            layer_index: The layer index passed to allocate_like().
        """
        if layer_index in self._layer_allocations:
            start, end = self._layer_allocations.pop(layer_index)
            # Move allocation_start forward if this was the oldest allocation
            if start == self.allocation_start:
                self.allocation_start = end

    def clear(self) -> None:
        """Reset all allocations without releasing memory."""
        self.allocation_start = 0
        self.allocation_end = 0
        self._layer_allocations.clear()

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
        used_bytes = self.allocation_end - self.allocation_start

        return {
            "device": str(self.device),
            "cache_tensors_allocated": allocated_tensors,
            "cache_tensors_total": len(self.cache_tensors),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "utilization": used_bytes / total_bytes if total_bytes > 0 else 0,
            "active_layers": len(self._layer_allocations),
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
            "ring_allocator_gpu": None,
            "ring_allocator_cpu": None,
            "use_ring_allocator": False,
        }
    return _DEVICE_STATE[device]
```

Modify `_BouncingLinearFn.forward`:

```python
@staticmethod
def forward(ctx, input, module, device, bias_device):
    # ... existing setup code ...

    state = _get_device_state(device)
    allocator = state.get("ring_allocator_gpu") if state.get("use_ring_allocator") else None

    # Transfer weight to GPU
    if allocator:
        # Use pre-allocated buffer
        w_bufs[idx] = allocator.allocate_like(weight_cpu)
        w_bufs[idx].copy_(weight_cpu, non_blocking=True)
    else:
        # Original path
        w_bufs[idx] = weight_cpu.to(device, non_blocking=True)

    # ... rest of forward ...
```

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
        cpu_fraction: float = 0.50
    ) -> None:
        """Initialize pre-allocated buffers based on model size.

        Args:
            model_size_bytes: Total size of model parameters in bytes.
            device: GPU device to allocate on.
            gpu_fraction: Fraction of model size for GPU staging buffer.
            cpu_fraction: Fraction of model size for CPU offload buffer.
        """
        from .manager_modules import _get_device_state

        state = _get_device_state(device)

        # GPU-side buffer for weight staging during compute
        state["ring_allocator_gpu"] = RingBufferAllocator(
            device,
            target_bytes=int(model_size_bytes * gpu_fraction)
        )

        # CPU-side pinned buffer for offloaded weights
        state["ring_allocator_cpu"] = RingBufferAllocator(
            torch.device("cpu"),
            target_bytes=int(model_size_bytes * cpu_fraction)
        )

        state["use_ring_allocator"] = True

        print(f"Ring allocators initialized:")
        print(f"  GPU buffer: {int(model_size_bytes * gpu_fraction) / 1e9:.2f} GB")
        print(f"  CPU buffer: {int(model_size_bytes * cpu_fraction) / 1e9:.2f} GB")

    def disable_ring_allocators(self, device: torch.device) -> None:
        """Disable and release ring allocators."""
        from .manager_modules import _get_device_state

        state = _get_device_state(device)

        if state.get("ring_allocator_gpu"):
            state["ring_allocator_gpu"].deallocate_cache()
            state["ring_allocator_gpu"] = None

        if state.get("ring_allocator_cpu"):
            state["ring_allocator_cpu"].deallocate_cache()
            state["ring_allocator_cpu"] = None

        state["use_ring_allocator"] = False
```

---

## Verification

### Test 1: Allocation count reduction

```python
import torch

# Monkey-patch to count allocations
original_empty = torch.empty
allocation_count = 0

def counting_empty(*args, **kwargs):
    global allocation_count
    allocation_count += 1
    return original_empty(*args, **kwargs)

torch.empty = counting_empty

# Run 100 training steps
# ...

print(f"Total allocations: {allocation_count}")
# Should be near-zero after first epoch with ring allocator
```

### Test 2: Memory fragmentation

```python
# Before training
stats_before = torch.cuda.memory_stats()
peak_before = stats_before['active_bytes.all.peak']

# Run training
# ...

# After training
stats_after = torch.cuda.memory_stats()
peak_after = stats_after['active_bytes.all.peak']

print(f"Peak memory before: {peak_before / 1e9:.2f} GB")
print(f"Peak memory after: {peak_after / 1e9:.2f} GB")
```

### Test 3: Allocator stats

```python
from toolkit.memory_management.manager_modules import _get_device_state

state = _get_device_state(torch.device("cuda:0"))
if state.get("ring_allocator_gpu"):
    print(state["ring_allocator_gpu"].get_stats())
```

---

## Configuration

Add to training config:

```yaml
memory_management:
  use_ring_allocator: true  # default: true
  ring_allocator_gpu_fraction: 0.25
  ring_allocator_cpu_fraction: 0.50
```

---

## Success Criteria

- [ ] CUDA allocation calls drop to near-zero after first epoch
- [ ] Peak VRAM usage is more predictable
- [ ] No training correctness impact
- [ ] Allocator stats show healthy utilization (50-80%)

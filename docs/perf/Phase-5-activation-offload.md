# Phase 5: Activation Offloading to CPU

**Impact:** Reduces VRAM by offloading activations instead of recomputing
**Risk:** Medium — requires checkpoint integration
**Effort:** 1-2 weeks
**Dependencies:** Phase 2 (ring allocator), Phase 4 (conductor)

---

## Problem

Current approach: Gradient checkpointing recomputes activations during backward.
- Activations are discarded after forward
- Backward recomputes each layer's activations before computing gradients
- **Trade-off: compute for memory**

### When Recompute Hurts

- Complex attention mechanisms (slow to recompute)
- High PCIe bandwidth relative to compute (newer GPUs)
- CPU RAM plentiful but VRAM constrained

### OneTrainer Approach

`LayerOffloadConductor.py:770-784` + `StaticActivationAllocator`:
- Activations saved to pre-allocated CPU pinned memory during forward
- Transferred back to GPU during backward on dedicated stream
- **Trade-off: PCIe bandwidth for compute**

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `toolkit/memory_management/offload_conductor.py` | Add ActivationOffloader |
| NEW: `toolkit/checkpoint_offload.py` | Custom checkpoint function |
| Model-specific integrations | Replace torch.checkpoint |

---

## Implementation

### Step 1: ActivationOffloader class

Add to `toolkit/memory_management/offload_conductor.py`:

```python
from .ring_allocator import ActivationAllocator


class ActivationOffloader:
    """Offloads activations to CPU instead of discarding for recompute.

    During forward pass: saves activations to pinned CPU memory
    During backward pass: restores activations from CPU

    Args:
        conductor: The OffloadConductor managing streams.
    """

    def __init__(self, conductor: OffloadConductor):
        self.conductor = conductor

        # Storage for activations per call_index
        self.activations_map: dict[int, Any] = {}
        self.transfer_events: dict[int, SyncEvent] = {}

        # Use conductor's activation stream
        self.transfer_stream = conductor.activations_transfer_stream

        # CPU allocator for activations
        self.cpu_allocator = ActivationAllocator(torch.device("cpu"))

        # Tensor indices to offload (model-specific)
        self.tensor_indices: dict[int, List[int]] = {}

    def configure_layer(self, layer_index: int, tensor_indices: List[int]) -> None:
        """Configure which tensors to offload for a layer.

        Args:
            layer_index: The layer index.
            tensor_indices: Indices of tensors in activations tuple to offload.
        """
        self.tensor_indices[layer_index] = tensor_indices

    def save_activations(
        self,
        call_index: int,
        layer_index: int,
        activations: Any
    ) -> None:
        """Save activations to CPU during forward pass.

        Args:
            call_index: Unique identifier for this forward call.
            layer_index: The layer that produced these activations.
            activations: The activation tensors to save.
        """
        self.activations_map[call_index] = activations
        tensor_indices = self.tensor_indices.get(layer_index, [])

        if self.conductor.async_transfer:
            with torch.cuda.stream(self.transfer_stream):
                # Wait for compute to finish
                self.transfer_stream.wait_stream(self.conductor.train_stream)

                # Get tensors to offload
                tensors = self._extract_tensors(activations, tensor_indices)

                # Reserve space
                self.cpu_allocator.reserve_cache(tensors)

                # Copy to CPU
                for i, tensor in enumerate(tensors):
                    cpu_tensor = self.cpu_allocator.allocate_like(tensor)
                    cpu_tensor.copy_(tensor, non_blocking=True)
                    self._replace_tensor(activations, tensor_indices[i], cpu_tensor)

                # Record completion
                event = SyncEvent(torch.cuda.Event(), f"activation_save_{call_index}")
                event.record(self.transfer_stream)
                self.transfer_events[call_index] = event
        else:
            tensors = self._extract_tensors(activations, tensor_indices)
            self.cpu_allocator.reserve_cache(tensors)
            for i, tensor in enumerate(tensors):
                cpu_tensor = self.cpu_allocator.allocate_like(tensor)
                cpu_tensor.copy_(tensor)
                self._replace_tensor(activations, tensor_indices[i], cpu_tensor)

    def restore_activations(self, call_index: int) -> Any:
        """Restore activations from CPU during backward pass.

        Args:
            call_index: The call_index used in save_activations.

        Returns:
            The restored activations on GPU.
        """
        # Wait for offload to complete
        if call_index in self.transfer_events:
            if self.conductor.async_transfer:
                self.transfer_events[call_index].wait(self.conductor.train_stream)
            del self.transfer_events[call_index]

        activations = self.activations_map.pop(call_index, None)
        if activations is None:
            return None

        # Transfer back to GPU
        layer_index = self._get_layer_for_call(call_index)
        tensor_indices = self.tensor_indices.get(layer_index, [])

        tensors = self._extract_tensors(activations, tensor_indices)
        for i, tensor in enumerate(tensors):
            if tensor.device.type == "cpu":
                gpu_tensor = tensor.to(self.conductor.train_device, non_blocking=True)
                self._replace_tensor(activations, tensor_indices[i], gpu_tensor)

        return activations

    def prefetch_activations(self, call_index: int) -> None:
        """Start prefetching activations before they're needed.

        Call this during backward of layer N+1 to prefetch layer N's activations.
        """
        if call_index not in self.activations_map:
            return

        if self.conductor.async_transfer:
            with torch.cuda.stream(self.transfer_stream):
                activations = self.activations_map[call_index]
                layer_index = self._get_layer_for_call(call_index)
                tensor_indices = self.tensor_indices.get(layer_index, [])

                tensors = self._extract_tensors(activations, tensor_indices)
                for i, tensor in enumerate(tensors):
                    if tensor.device.type == "cpu":
                        gpu_tensor = tensor.to(
                            self.conductor.train_device,
                            non_blocking=True
                        )
                        self._replace_tensor(activations, tensor_indices[i], gpu_tensor)

    def clear(self) -> None:
        """Clear all saved activations."""
        self.activations_map.clear()
        self.transfer_events.clear()
        self.cpu_allocator.deallocate()

    def _extract_tensors(self, activations: Any, indices: List[int]) -> List[torch.Tensor]:
        """Extract tensors from activations by indices."""
        if isinstance(activations, torch.Tensor):
            return [activations] if 0 in indices else []
        elif isinstance(activations, (tuple, list)):
            return [activations[i] for i in indices if i < len(activations)]
        elif isinstance(activations, dict):
            keys = list(activations.keys())
            return [activations[keys[i]] for i in indices if i < len(keys)]
        return []

    def _replace_tensor(self, activations: Any, index: int, new_tensor: torch.Tensor) -> None:
        """Replace a tensor in activations structure."""
        if isinstance(activations, torch.Tensor) and index == 0:
            # Can't replace in-place, caller must handle
            pass
        elif isinstance(activations, list):
            if index < len(activations):
                activations[index] = new_tensor
        elif isinstance(activations, dict):
            keys = list(activations.keys())
            if index < len(keys):
                activations[keys[index]] = new_tensor

    def _get_layer_for_call(self, call_index: int) -> int:
        """Map call_index to layer_index."""
        # Simple mapping: call_index == layer_index for most cases
        return call_index % self.conductor.num_layers if hasattr(self.conductor, 'num_layers') else call_index


# Update OffloadConductor to include activation offloader
class OffloadConductor:
    def __init__(self, ...):
        # ... existing init ...
        self.activation_offloader: Optional[ActivationOffloader] = None

    def enable_activation_offload(self) -> None:
        """Enable activation offloading."""
        self.activation_offloader = ActivationOffloader(self)

    @property
    def num_layers(self) -> int:
        return len(self.layers)
```

### Step 2: Custom checkpoint function

**File:** `toolkit/checkpoint_offload.py`

```python
"""
Checkpoint with activation offloading instead of recomputation.
"""

import torch
from torch import nn
from typing import Any, Callable, Tuple
from toolkit.memory_management.offload_conductor import ActivationOffloader


class OffloadCheckpointFunction(torch.autograd.Function):
    """Autograd function that offloads activations instead of discarding."""

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        offloader: ActivationOffloader,
        call_index: int,
        layer_index: int,
        *args
    ):
        ctx.run_function = run_function
        ctx.offloader = offloader
        ctx.call_index = call_index
        ctx.layer_index = layer_index
        ctx.save_for_backward(*args)

        with torch.no_grad():
            outputs = run_function(*args)

        # Save outputs to CPU
        offloader.save_activations(call_index, layer_index, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Restore activations from CPU
        inputs = ctx.saved_tensors

        # Restore outputs if needed for gradient computation
        # (For most transformer blocks, we need to rerun forward anyway)

        with torch.enable_grad():
            # Detach inputs and enable grad
            detached_inputs = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)
            outputs = ctx.run_function(*detached_inputs)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs,
            detached_inputs,
            grad_outputs,
            allow_unused=True
        )

        # Return None for run_function, offloader, call_index, layer_index, then input grads
        return (None, None, None, None) + grads


def checkpoint_with_offload(
    function: Callable,
    offloader: ActivationOffloader,
    call_index: int,
    layer_index: int,
    *args,
    **kwargs
) -> Any:
    """Run function with activation offloading for checkpointing.

    Args:
        function: The function to checkpoint.
        offloader: ActivationOffloader instance.
        call_index: Unique call identifier.
        layer_index: Layer index for configuration lookup.
        *args: Arguments to pass to function.
        **kwargs: Keyword arguments (not checkpointed, passed directly).

    Returns:
        Output of function.
    """
    if kwargs:
        # Wrap function to include kwargs
        def wrapper(*args):
            return function(*args, **kwargs)
        return OffloadCheckpointFunction.apply(wrapper, offloader, call_index, layer_index, *args)
    else:
        return OffloadCheckpointFunction.apply(function, offloader, call_index, layer_index, *args)


def replace_checkpoint_with_offload(
    module: nn.Module,
    offloader: ActivationOffloader,
    block_class: type = None
) -> None:
    """Replace standard checkpointing with offload checkpointing in a module.

    Args:
        module: The module to modify.
        offloader: ActivationOffloader to use.
        block_class: Optional class type to target (e.g., TransformerBlock).
    """
    call_index = [0]  # Mutable counter

    def make_forward_hook(layer_index: int):
        def hook(module, args, kwargs, output):
            nonlocal call_index
            if module.training:
                offloader.save_activations(call_index[0], layer_index, output)
                call_index[0] += 1
            return output
        return hook

    for i, (name, child) in enumerate(module.named_modules()):
        if block_class is None or isinstance(child, block_class):
            child.register_forward_hook(make_forward_hook(i), with_kwargs=True)
```

### Step 3: Integration

```python
# In SDTrainer setup:
if self.train_config.activation_offload:
    if not self.offload_conductor:
        raise ValueError("activation_offload requires use_offload_conductor=True")

    self.offload_conductor.enable_activation_offload()

    # Configure which tensors to offload per layer
    for i in range(len(self.offload_conductor.layers)):
        # Offload hidden_states (index 0) by default
        self.offload_conductor.activation_offloader.configure_layer(i, [0])
```

---

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| PCIe 4.0/5.0 GPU | Consider activation offload |
| PCIe 3.0 GPU | Prefer recompute |
| Complex attention | Activation offload |
| Simple FFN-heavy model | Recompute |
| Abundant CPU RAM | Activation offload |
| Limited CPU RAM | Recompute |

Benchmark both on your hardware to determine optimal choice.

---

## Configuration

```yaml
memory_management:
  use_offload_conductor: true  # Required
  activation_offload: false    # default: false
  # Tensor indices to offload per layer (model-specific)
  activation_offload_indices: [0]  # hidden_states
```

---

## Verification

### Test 1: VRAM comparison

```python
# With recompute (standard checkpoint)
torch.cuda.reset_peak_memory_stats()
train_step_recompute()
peak_recompute = torch.cuda.max_memory_allocated()

# With offload
torch.cuda.reset_peak_memory_stats()
train_step_offload()
peak_offload = torch.cuda.max_memory_allocated()

print(f"Recompute peak: {peak_recompute / 1e9:.2f} GB")
print(f"Offload peak: {peak_offload / 1e9:.2f} GB")
```

### Test 2: Speed comparison

```python
import time

# Recompute timing
start = time.time()
for _ in range(100):
    train_step_recompute()
recompute_time = time.time() - start

# Offload timing
start = time.time()
for _ in range(100):
    train_step_offload()
offload_time = time.time() - start

print(f"Recompute: {recompute_time:.1f}s")
print(f"Offload: {offload_time:.1f}s")
print(f"Speedup: {recompute_time / offload_time:.2f}x")
```

---

## Success Criteria

- [ ] VRAM usage comparable to recompute
- [ ] Faster than recompute on PCIe 4.0+ systems
- [ ] Activations correctly restored during backward
- [ ] No training divergence vs standard checkpoint
- [ ] CPU memory usage bounded by allocator

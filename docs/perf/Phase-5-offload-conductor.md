# Phase 5: Coordinated Layer Offload Conductor

**Impact:** Reduces PCIe contention, enables overlapped compute/transfer
**Risk:** Medium — core training path change
**Effort:** 1-2 weeks
**Dependencies:** Phase 2 (ring allocator), Phase 4 (architecture-agnostic gating)

---

## Problem

Current approach in `manager_modules.py`:
- Each `_BouncingLinearFn`/`_BouncingConv2dFn` independently bounces weights CPU↔GPU
- Uses 2 streams (`transfer_stream`, `transfer_grad_stream`) with ping-pong double buffers
- **No layer knows what other layers are doing** — they all compete for PCIe bandwidth

Result: PCIe transfers are uncoordinated, causing congestion and stalls.

### OneTrainer Approach

`LayerOffloadConductor.py:524-950`:
- Centralized conductor pre-computes which layers to load/offload at each step
- Uses 3 dedicated streams: compute, layer transfer, activation transfer
- Knows about forward→backward transitions
- **Loads layer N+1 while computing layer N** (overlapped transfer)

---

## Files to Create/Modify

| File | Change |
|------|--------|
| NEW: `toolkit/memory_management/offload_conductor.py` | OffloadStrategy, OffloadConductor |
| `toolkit/memory_management/manager.py` | Integration entry points for layer transfer allocators |
| `jobs/process/BaseSDTrainProcess.py` | Lifecycle setup/teardown and safety gating |
| Model setup/checkpointing integration | Register layer order and wrap checkpointed blocks |

---

## Correctness Guardrails

Before implementing, preserve these invariants from OneTrainer and existing AI Toolkit behavior:

1. **Integrate at checkpoint wrapper/model layer level, not SDTrainer block loop.**
   - In this repo, model forward paths are extension-specific and not centralized in `SDTrainer`.
   - Matching OneTrainer means patching checkpointed layer wrappers (`before_layer` / `after_layer`) where execution order is explicit.

2. **Forward/backward transition detection assumes `use_reentrant=True` checkpointing.**
   - `torch.is_grad_enabled()` is only a reliable backward-recompute signal in that mode.
   - If a non-reentrant path is used, transition detection must use explicit phase state, not grad-enabled checks.

3. **Keep offload ordering consistent with OneTrainer strategy.**
   - Forward pass offload/load ordering should prioritize `>= layer_index` first, then `< layer_index`.
   - Backward pass ordering is the inverse.

4. **Do not silently defer every offload when gradients are present.**
   - Gradient-present offload deferral is a narrow multi-GPU async-reduce exception.
   - Otherwise this should fail fast (or be gated behind fused-backward guarantees), not degrade silently.

5. **Use existing config schema names unless explicitly adding new keys.**
   - Current repo uses `model.layer_offloading_*` knobs, not `memory_management.use_offload_conductor`.
   - If new conductor flags are added, document migration and defaults clearly.

## Implementation

### Step 1: OffloadStrategy

**File:** `toolkit/memory_management/offload_conductor.py`

```python
"""
Coordinated layer offloading conductor.

Manages CPU↔GPU transfers for transformer blocks with:
- Pre-computed offload schedules
- 3 dedicated CUDA streams
- Overlapped compute and transfer

Based on OneTrainer's LayerOffloadConductor.
"""

import torch
from torch import nn
from typing import List, Optional, Any, Set
from dataclasses import dataclass
from .ring_allocator import RingBufferAllocator


def get_layer_bytes(layer: nn.Module) -> int:
    """Calculate total bytes for a layer's parameters."""
    total = 0
    for param in layer.parameters():
        total += param.numel() * param.element_size()
    return total


@dataclass
class OffloadSchedule:
    """Pre-computed schedule of which layers to have loaded at each step."""
    layers_to_load: List[int]
    layers_to_offload: List[int]


class OffloadStrategy:
    """Pre-computes which layers should be loaded at each execution step.

    Handles three scenarios:
    1. Forward → Backward: Keep last N layers (needed immediately for backward)
    2. Forward → Forward: Start pre-loading first layers during last layers
    3. Backward → Forward: Reverse order pre-loading

    Args:
        layers: List of transformer block modules.
        layer_offload_fraction: Fraction of layers to offload (0.0-1.0).
    """

    def __init__(self, layers: List[nn.Module], layer_offload_fraction: float):
        self.num_layers = len(layers)
        self.layer_bytes = [get_layer_bytes(layer) for layer in layers]
        self.total_bytes = sum(self.layer_bytes)

        # Target bytes to keep on GPU
        target_loaded_bytes = int(self.total_bytes * (1.0 - layer_offload_fraction))

        # Pre-compute loaded layer sets for each scenario
        self.initial_loaded = self._compute_initial_layers(target_loaded_bytes)

        self.forward_backward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=True, is_cyclic=False)
            for i in range(self.num_layers)
        ]

        self.forward_forward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=True, is_cyclic=True)
            for i in range(self.num_layers)
        ]

        self.backward_forward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=False, is_cyclic=False)
            for i in range(self.num_layers)
        ]

        # Calculate memory bounds
        all_loaded = (
            self.forward_backward_loaded +
            self.forward_forward_loaded +
            self.backward_forward_loaded
        )

        self.max_loaded_bytes = max(
            sum(self.layer_bytes[i] for i in layers)
            for layers in all_loaded
        )

        min_loaded_bytes = min(
            sum(self.layer_bytes[i] for i in layers)
            for layers in all_loaded
        )

        self.max_offloaded_bytes = (
            self.total_bytes - min_loaded_bytes + max(self.layer_bytes)
        )

    def _compute_initial_layers(self, target_bytes: int) -> List[int]:
        """Compute which layers to load initially."""
        return self._get_layers_below(0, target_bytes, is_forward=True, is_cyclic=False)

    def _get_layers_below(
        self,
        start_layer: int,
        max_bytes: int,
        is_forward: bool,
        is_cyclic: bool
    ) -> List[int]:
        """Get layers that fit within byte budget starting from a layer."""
        accumulator = 0
        layers = []

        if is_forward and is_cyclic:
            # Forward with wrap-around (e.g., end of forward, prepare for next)
            for i in range(start_layer, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

            for i in range(start_layer):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

        elif is_forward and not is_cyclic:
            # Forward without wrap (during forward pass)
            for i in range(start_layer, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

            # Also include some previous layers
            for i in range(start_layer - 1, -1, -1):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

        else:
            # Backward (reverse order)
            for i in range(start_layer, -1, -1):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

            for i in range(start_layer + 1, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

        return sorted(layers)

    def get_schedule(
        self,
        layer_index: int,
        is_forward: bool,
        is_next_forward: bool,
        loaded_layers: Set[int]
    ) -> OffloadSchedule:
        """Get the offload schedule for a given layer execution.

        Args:
            layer_index: Current layer being executed.
            is_forward: True if in forward pass.
            is_next_forward: True if next pass will be forward (not backward).
            loaded_layers: Set of currently loaded layer indices.

        Returns:
            OffloadSchedule with layers to load and offload.
        """
        # Determine target loaded set
        if is_forward and is_next_forward:
            target = set(self.forward_forward_loaded[layer_index])
        elif is_forward:
            target = set(self.forward_backward_loaded[layer_index])
        else:
            target = set(self.backward_forward_loaded[layer_index])

        # Compute deltas
        to_offload = sorted(loaded_layers - target)
        to_load = sorted(target - loaded_layers)

        # Order offloads/loads optimally
        if is_forward:
            # Match OneTrainer ordering for forward scheduling.
            to_offload = [i for i in to_offload if i >= layer_index] + \
                        [i for i in to_offload if i < layer_index]
        else:
            # Backward: offload in reverse order
            to_offload = [i for i in reversed(to_offload) if i < layer_index] + \
                        [i for i in reversed(to_offload) if i >= layer_index]

        return OffloadSchedule(layers_to_load=to_load, layers_to_offload=to_offload)


class SyncEvent:
    """Wrapper for CUDA events with optional logging."""

    def __init__(self, event: Optional[torch.cuda.Event] = None, description: str = ""):
        self.event = event
        self.description = description

    def record(self, stream: Optional[torch.cuda.Stream] = None):
        if self.event is not None:
            if stream:
                self.event.record(stream)
            else:
                self.event.record()

    def wait(self, stream: torch.cuda.Stream):
        if self.event is not None:
            stream.wait_event(self.event)

    def synchronize(self):
        if self.event is not None:
            self.event.synchronize()


class OffloadConductor:
    """Centralized coordinator for layer offloading.

    Manages:
    - 3 CUDA streams (compute, layer transfer, activation transfer)
    - Pre-computed offload strategy
    - Ring buffer allocators for GPU and CPU
    - Synchronization events per layer

    Args:
        train_device: GPU device for training.
        temp_device: CPU device for offloaded weights.
        layer_offload_fraction: Fraction of layers to offload (0.0-1.0).
    """

    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device = torch.device("cpu"),
        layer_offload_fraction: float = 0.5
    ):
        self.train_device = train_device
        self.temp_device = temp_device
        self.layer_offload_fraction = layer_offload_fraction

        # Layers to manage
        self.layers: List[nn.Module] = []
        self.layer_device_map: List[Optional[torch.device]] = []

        # 3 CUDA streams
        if train_device.type == "cuda":
            self.train_stream = torch.cuda.default_stream(train_device)
            self.layer_transfer_stream = torch.cuda.Stream(train_device)
            self.activations_transfer_stream = torch.cuda.Stream(train_device)
            self.async_transfer = True
        else:
            self.train_stream = None
            self.layer_transfer_stream = None
            self.activations_transfer_stream = None
            self.async_transfer = False

        # Allocators (initialized on activate)
        self.gpu_allocator: Optional[RingBufferAllocator] = None
        self.cpu_allocator: Optional[RingBufferAllocator] = None

        # Sync events per layer
        self.layer_train_events: List[SyncEvent] = []
        self.layer_transfer_events: List[SyncEvent] = []

        # Strategy (computed on activate)
        self.strategy: Optional[OffloadStrategy] = None

        # State
        self.is_forward_pass = True
        self.keep_graph = False
        self.is_active = False

        # Deferred offloads (for multi-GPU gradient sync)
        self._deferred_offloads: List[int] = []

    def add_layer(self, layer: nn.Module) -> int:
        """Register a transformer block with the conductor.

        Args:
            layer: The transformer block module.

        Returns:
            The layer index.
        """
        idx = len(self.layers)
        self.layers.append(layer)
        self.layer_device_map.append(None)
        self.layer_train_events.append(SyncEvent())
        self.layer_transfer_events.append(SyncEvent())
        return idx

    def activate(self) -> None:
        """Initialize strategy and allocators based on registered layers."""
        if not self.layers:
            raise RuntimeError("No layers registered. Call add_layer() first.")

        # Compute strategy
        self.strategy = OffloadStrategy(self.layers, self.layer_offload_fraction)

        # Initialize allocators
        self.gpu_allocator = RingBufferAllocator(
            self.train_device,
            self.strategy.max_loaded_bytes
        )
        self.cpu_allocator = RingBufferAllocator(
            self.temp_device,
            self.strategy.max_offloaded_bytes
        )

        # Move layers to initial positions
        for i, layer in enumerate(self.layers):
            if i in self.strategy.initial_loaded:
                self._load_layer_sync(i)
            else:
                self._offload_layer_sync(i)

        # Create events
        if self.async_transfer:
            for i in range(len(self.layers)):
                self.layer_train_events[i] = SyncEvent(
                    torch.cuda.Event(),
                    f"train_layer_{i}"
                )
                self.layer_transfer_events[i] = SyncEvent(
                    torch.cuda.Event(),
                    f"transfer_layer_{i}"
                )

        self.is_active = True

        print(f"OffloadConductor activated:")
        print(f"  Layers: {len(self.layers)}")
        print(f"  Initial loaded: {len(self.strategy.initial_loaded)}")
        print(f"  Max GPU: {self.strategy.max_loaded_bytes / 1e9:.2f} GB")
        print(f"  Max CPU: {self.strategy.max_offloaded_bytes / 1e9:.2f} GB")

    def deactivate(self) -> None:
        """Release resources and move all layers to temp device."""
        if self.async_transfer:
            self._wait_all_transfers()

        # Move all layers to CPU
        for i in range(len(self.layers)):
            if self.layer_device_map[i] == self.train_device:
                self._offload_layer_sync(i)

        # Release allocators
        if self.gpu_allocator:
            self.gpu_allocator.deallocate_cache()
        if self.cpu_allocator:
            self.cpu_allocator.deallocate_cache()

        self.is_active = False

    def start_forward(self, keep_graph: bool = True) -> None:
        """Called before forward pass begins.

        Args:
            keep_graph: True if backward will follow (training mode).
        """
        if not self.is_active:
            return

        if self.async_transfer:
            self.layer_transfer_stream.wait_stream(self.train_stream)

        self._wait_all_transfers()

        self.is_forward_pass = True
        self.keep_graph = keep_graph

    def before_layer(self, layer_index: int) -> None:
        """Called before executing a layer. Ensures layer is loaded.

        Args:
            layer_index: Index of layer about to execute.
        """
        if not self.is_active:
            return

        # Detect forward→backward transition.
        # NOTE: valid when layers are executed under reentrant checkpointing wrappers.
        if torch.is_grad_enabled() and self.is_forward_pass:
            # Gradients enabled during forward = we're in backward recompute
            self.is_forward_pass = False

        # Wait for this layer's transfer to complete
        if self.async_transfer:
            self.layer_transfer_events[layer_index].wait(self.train_stream)

        # Process any deferred offloads
        self._process_deferred_offloads(except_layer=layer_index)

        # Get schedule
        loaded = self._get_loaded_layers()
        schedule = self.strategy.get_schedule(
            layer_index,
            self.is_forward_pass,
            not self.keep_graph,
            loaded
        )

        # Schedule offloads
        for i in schedule.layers_to_offload:
            self._schedule_offload(i)

        # Schedule loads
        for i in schedule.layers_to_load:
            self._schedule_load(i)

    def after_layer(self, layer_index: int) -> None:
        """Called after executing a layer. Records completion.

        Args:
            layer_index: Index of layer that just completed.
        """
        if not self.is_active:
            return

        if self.async_transfer:
            self.layer_train_events[layer_index].record(self.train_stream)

    def _get_loaded_layers(self) -> Set[int]:
        """Get set of currently loaded layer indices."""
        return {
            i for i, dev in enumerate(self.layer_device_map)
            if dev == self.train_device
        }

    def _schedule_load(self, layer_index: int) -> None:
        """Async load a layer from CPU to GPU."""
        if self.layer_device_map[layer_index] == self.train_device:
            return

        if self.async_transfer:
            with torch.cuda.stream(self.layer_transfer_stream):
                self.layer_train_events[layer_index].wait(self.layer_transfer_stream)
                self._load_layer_impl(layer_index)
                self.layer_transfer_events[layer_index].record(self.layer_transfer_stream)
        else:
            self._load_layer_impl(layer_index)

    def _schedule_offload(self, layer_index: int) -> None:
        """Async offload a layer from GPU to CPU."""
        if self.layer_device_map[layer_index] == self.temp_device:
            return

        # Check if layer has pending gradients.
        # In production, defer only for supported multi-GPU async-reduce paths;
        # otherwise fail fast to avoid silent schedule degradation.
        layer = self.layers[layer_index]
        for param in layer.parameters():
            if param.grad is not None:
                self._deferred_offloads.append(layer_index)
                return

        if self.async_transfer:
            with torch.cuda.stream(self.layer_transfer_stream):
                self.layer_train_events[layer_index].wait(self.layer_transfer_stream)
                self._offload_layer_impl(layer_index)
                self.layer_transfer_events[layer_index].record(self.layer_transfer_stream)
        else:
            self._offload_layer_impl(layer_index)

    def _load_layer_impl(self, layer_index: int) -> None:
        """Actually load a layer to GPU."""
        layer = self.layers[layer_index]

        for param in layer.parameters():
            gpu_tensor = self.gpu_allocator.allocate_like(param.data, layer_index)
            gpu_tensor.copy_(param.data, non_blocking=self.async_transfer)
            param.data = gpu_tensor

        self.cpu_allocator.deallocate_layer(layer_index)
        self.layer_device_map[layer_index] = self.train_device

    def _offload_layer_impl(self, layer_index: int) -> None:
        """Actually offload a layer to CPU."""
        layer = self.layers[layer_index]

        for param in layer.parameters():
            cpu_tensor = self.cpu_allocator.allocate_like(param.data, layer_index)
            cpu_tensor.copy_(param.data, non_blocking=self.async_transfer)
            param.data = cpu_tensor

        self.gpu_allocator.deallocate_layer(layer_index)
        self.layer_device_map[layer_index] = self.temp_device

    def _load_layer_sync(self, layer_index: int) -> None:
        """Synchronously load a layer (for initialization)."""
        layer = self.layers[layer_index]
        layer.to(self.train_device)
        self.layer_device_map[layer_index] = self.train_device

    def _offload_layer_sync(self, layer_index: int) -> None:
        """Synchronously offload a layer (for initialization)."""
        layer = self.layers[layer_index]
        layer.to(self.temp_device)
        self.layer_device_map[layer_index] = self.temp_device

    def _wait_all_transfers(self) -> None:
        """Wait for all pending transfers to complete."""
        for event in self.layer_transfer_events:
            event.synchronize()

    def _process_deferred_offloads(self, except_layer: int) -> None:
        """Process any deferred offloads."""
        pending = self._deferred_offloads
        self._deferred_offloads = []

        for layer_index in pending:
            if layer_index == except_layer:
                continue
            self._schedule_offload(layer_index)

    def get_stats(self) -> dict:
        """Get conductor statistics."""
        loaded = self._get_loaded_layers()
        return {
            "total_layers": len(self.layers),
            "loaded_layers": len(loaded),
            "loaded_indices": sorted(loaded),
            "is_active": self.is_active,
            "is_forward": self.is_forward_pass,
            "deferred_offloads": len(self._deferred_offloads),
            "gpu_allocator": self.gpu_allocator.get_stats() if self.gpu_allocator else None,
            "cpu_allocator": self.cpu_allocator.get_stats() if self.cpu_allocator else None,
        }
```

### Step 2: Integration with model checkpoint wrappers

**Files:** model setup + checkpointing wrappers (per architecture), with lifecycle hooks in `BaseSDTrainProcess`

```python
from toolkit.memory_management.offload_conductor import OffloadConductor

# Pseudocode mirroring OneTrainer's checkpoint_util integration pattern:
#
# 1) Build conductor where checkpointed layers are registered in exact execution order.
# 2) Wrap each checkpointed layer forward:
#       args = conductor.before_layer(layer_idx, call_id, args)
#       out = orig_forward(*args)
#       conductor.after_layer(layer_idx, call_id, args)
# 3) At start of forward pass:
#       conductor.start_forward(keep_graph=True)  # training
#       conductor.start_forward(keep_graph=False) # eval/inference
#
# This avoids model-specific SDTrainer surgery and keeps execution order explicit.
```

---

## Configuration

```yaml
model:
  # Existing knobs in this repo (today):
  layer_offloading: false
  layer_offloading_transformer_percent: 0.5
  layer_offloading_text_encoder_percent: 0.0

  # If adding a conductor-specific toggle, keep it opt-in and define migration:
  # use_offload_conductor: false
```

---

## Verification

### Test 1: PCIe utilization

```bash
# Monitor PCIe transfers
nvidia-smi dmon -s t -d 1
```

Should see smoother transfer pattern vs. spiky with per-layer bouncing.

### Test 2: Stream overlap

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Run training step
    ...

prof.export_chrome_trace("trace.json")
# Open in chrome://tracing - verify compute and transfer overlap
```

### Test 3: Memory stability

```python
for step in range(1000):
    train_step()
    if step % 100 == 0:
        print(f"Step {step}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

Memory should be stable, not growing.

---

## Success Criteria

- [ ] PCIe transfers overlap with compute
- [ ] No memory growth over training
- [ ] Works with Flux transformer architecture
- [ ] Configurable via `layer_offload_fraction`
- [ ] Proper fallback when conductor disabled

# AI Toolkit Performance Improvement Plan

## Lessons from OneTrainer's Architecture

### Context

The final report (`finalreport.md`) identifies several architectural areas where OneTrainer outperforms AI Toolkit, backed by source code analysis and community-reported issues. AI Toolkit suffers from:

1. **Catastrophic VRAM leak after checkpoint saves** (Issues #504, #575, #387) causing 3.6x-64x slowdowns
2. **Uncoordinated per-layer "bouncing"** for CPU/GPU offloading — each Linear/Conv2d independently competes for PCIe bandwidth
3. **No pre-allocated memory pools** — new GPU buffers allocated per transfer via ping-pong assignment
4. **No fused backward pass** — standard separate `optimizer.step()` after `backward()`, keeping all gradients alive until step
5. **No activation offloading** — only gradient checkpointing recompute, no CPU offload path
6. **HuggingFace Accelerate dispatch overhead** on every backward call

This plan ports the most impactful architectural patterns from OneTrainer into AI Toolkit, ordered by user impact and implementation feasibility.

---

## Part 1: Fix Checkpoint Save VRAM Leak (Critical Bug Fix)

**Impact:** Eliminates the #1 reported performance issue (3.6x-64x slowdowns)
**Risk:** Low — isolated to save path
**Effort:** Small (1-2 days)
**Files:**

- `ai-toolkit/jobs/process/BaseSDTrainProcess.py` — `save()` method (line ~495)
- `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py` — save hooks
- `ai-toolkit/toolkit/memory_management/manager_modules.py` — buffer cleanup

### Problem

The `save()` method at `BaseSDTrainProcess.py:495` calls `flush()` (which does `torch.cuda.empty_cache()` + `gc.collect()`), then performs model serialization. Multiple potential leak sources:

1. **Ping-pong buffers not cleared:** `_DEVICE_STATE` maintains per-device buffers (`w_buffers`, `b_buffers`, `w_bwd_buffers`, `w_grad_buffers`, `b_grad_buffers`) that hold references to GPU tensors
2. **State dict cloning:** `copy.deepcopy(self.meta)` and state dict operations may create tensor copies that aren't released
3. **Async operations not synchronized:** Non-blocking `.to()` operations may complete after `flush()` is called

### Implementation

1. **Add buffer cleanup to `manager_modules.py`:**

   ```python
   def clear_device_state_buffers(device: torch.device = None):
       """Clear all ping-pong buffers to release GPU memory."""
       devices = [device] if device else list(_DEVICE_STATE.keys())
       for dev in devices:
           if dev in _DEVICE_STATE:
               state = _DEVICE_STATE[dev]
               for key in ['w_buffers', 'b_buffers', 'w_bwd_buffers',
                           'w_grad_buffers', 'b_grad_buffers']:
                   if key in state:
                       state[key] = [None, None]
   ```

2. **Modify `flush()` in `SDTrainer.py`:**

   ```python
   def flush(clear_bouncing_buffers=False):
       torch.cuda.synchronize()  # Wait for all async ops FIRST
       if clear_bouncing_buffers:
           from toolkit.memory_management.manager_modules import clear_device_state_buffers
           clear_device_state_buffers()
       gc.collect()  # GC before empty_cache to release Python refs
       torch.cuda.empty_cache()
   ```

3. **Update `save()` method in `BaseSDTrainProcess.py`:**

   ```python
   def save(self, step=None):
       if not self.accelerator.is_main_process:
           return

       # Clear bouncing buffers before save to release GPU memory
       flush(clear_bouncing_buffers=True)

       # ... existing save logic ...

       # Post-save cleanup
       self.post_save_cleanup()

   def post_save_cleanup(self):
       """Ensure clean state after checkpoint save."""
       torch.cuda.synchronize()
       flush(clear_bouncing_buffers=True)
   ```

4. **Add diagnostic mode for debugging:**

   ```python
   # Before save
   if os.environ.get('AITK_MEMORY_DEBUG'):
       torch.cuda.memory._record_memory_history()
   # ... save ...
   if os.environ.get('AITK_MEMORY_DEBUG'):
       torch.cuda.memory._dump_snapshot("checkpoint_memory.pickle")
       torch.cuda.memory._record_memory_history(enabled=None)
   ```

### Verification

- Train for 100 steps, save checkpoint at step 50, measure s/it before and after save
- Confirm VRAM usage returns to pre-save levels via `torch.cuda.memory_allocated()`
- Run with `AITK_MEMORY_DEBUG=1` to capture memory snapshot if issues persist

---

## Part 2: Static Ring Buffer Memory Allocator

**Impact:** Eliminates per-transfer allocation overhead, reduces fragmentation
**Risk:** Low-Medium — memory management change
**Effort:** Medium (3-4 days)
**Files:**

- NEW: `ai-toolkit/toolkit/memory_management/ring_allocator.py`
- MODIFY: `ai-toolkit/toolkit/memory_management/manager_modules.py`

### Problem

Current approach: Each bouncing forward/backward creates new GPU tensors via `.to(device, non_blocking=True)`. The ping-pong buffers (`w_buffers[idx]`) are reassigned each call, but old buffers linger until GC. This causes:

- Per-transfer allocation overhead (CUDA malloc)
- Memory fragmentation over thousands of steps
- Unpredictable VRAM usage patterns

OneTrainer approach (`LayerOffloadConductor.py:45-221`): `StaticLayerAllocator` + `StaticLayerTensorAllocator` pre-allocate int8 byte tensors with 16-byte alignment. All layer weights are sliced from this pre-allocated pool. Zero allocation overhead after initialization.

### Implementation

1. **`RingBufferAllocator` class** (adapted from OneTrainer's pattern):

   ```python
   class RingBufferAllocator:
       """Pre-allocated ring buffer for layer weight transfers.

       Inspired by OneTrainer's StaticLayerAllocator with simplified API
       for AI Toolkit's bouncing module pattern.
       """
       def __init__(self, device: torch.device, target_bytes: int, num_cache_tensors: int = 4):
           self.device = device
           self.is_pinned = device.type == "cpu"

           # Calculate tensor size with overhead for alignment
           max_tensor_bytes = target_bytes // num_cache_tensors
           self.cache_tensor_size = self._ceil_16(max_tensor_bytes + 4096)

           # Lazy allocation - tensors created on first use
           self.cache_tensors: list[torch.Tensor | None] = [None] * num_cache_tensors

           # Ring buffer tracking
           self.allocation_start = 0
           self.allocation_end = 0
           self._layer_allocations: dict[int, tuple[int, int]] = {}  # layer_idx -> (start, end)

       @staticmethod
       def _ceil_16(n: int) -> int:
           return n + (16 - (n % 16)) % 16

       @staticmethod
       def _floor_16(n: int) -> int:
           return n - (n % 16)

       def _ensure_tensor(self, index: int):
           """Lazily allocate cache tensor on first use."""
           if self.cache_tensors[index] is None:
               torch.cuda.empty_cache()
               self.cache_tensors[index] = torch.zeros(
                   self.cache_tensor_size, dtype=torch.int8, device=self.device
               )
               if self.is_pinned:
                   self.cache_tensors[index] = self.cache_tensors[index].pin_memory()

       def allocate_like(self, source_tensor: torch.Tensor, layer_index: int = -1) -> torch.Tensor:
           """Allocate a view into the ring buffer matching source shape/dtype."""
           num_bytes = source_tensor.numel() * source_tensor.element_size()

           # Find slot in ring buffer
           cache_idx = self.allocation_end // self.cache_tensor_size
           offset = self._ceil_16(self.allocation_end % self.cache_tensor_size)

           # Check if we need to wrap to next tensor
           if offset + num_bytes > self.cache_tensor_size:
               cache_idx = (cache_idx + 1) % len(self.cache_tensors)
               offset = 0

           self._ensure_tensor(cache_idx)

           # Get view with correct dtype/shape
           byte_view = self.cache_tensors[cache_idx][offset:offset + num_bytes]
           tensor_view = byte_view.view(dtype=source_tensor.dtype).view(source_tensor.shape)

           # Track allocation
           new_end = cache_idx * self.cache_tensor_size + offset + num_bytes
           if layer_index >= 0:
               self._layer_allocations[layer_index] = (self.allocation_end, new_end)
           self.allocation_end = new_end

           return tensor_view

       def deallocate_layer(self, layer_index: int):
           """Mark a layer's allocation as reclaimable."""
           if layer_index in self._layer_allocations:
               start, end = self._layer_allocations.pop(layer_index)
               # Move allocation_start forward if this was the oldest allocation
               if start == self.allocation_start:
                   self.allocation_start = end

       def clear(self):
           """Reset all allocations."""
           self.allocation_start = 0
           self.allocation_end = 0
           self._layer_allocations.clear()

       def deallocate_cache(self):
           """Release all pre-allocated memory."""
           self.cache_tensors = [None] * len(self.cache_tensors)
           self.clear()
   ```

2. **Integrate with existing bouncing in `manager_modules.py`:**

   ```python
   # Add to _DEVICE_STATE initialization
   _DEVICE_STATE[device] = {
       # ... existing fields ...
       "ring_allocator_gpu": None,  # Created lazily based on model size
       "ring_allocator_cpu": None,
   }

   # In _BouncingLinearFn.forward, replace:
   # OLD: w_bufs[idx] = weight_cpu.to(device, non_blocking=True)
   # NEW:
   allocator = state.get("ring_allocator_gpu")
   if allocator:
       w_bufs[idx] = allocator.allocate_like(weight_cpu)
       w_bufs[idx].copy_(weight_cpu, non_blocking=True)
   else:
       w_bufs[idx] = weight_cpu.to(device, non_blocking=True)
   ```

3. **Add allocator initialization to `MemoryManager`:**

   ```python
   def initialize_ring_allocators(self, model_size_bytes: int, device: torch.device):
       """Initialize pre-allocated buffers based on model size."""
       state = _get_device_state(device)
       # Allocate ~25% of model size for GPU-side weight staging
       state["ring_allocator_gpu"] = RingBufferAllocator(
           device, target_bytes=model_size_bytes // 4
       )
       # CPU-side pinned for offloaded weights
       state["ring_allocator_cpu"] = RingBufferAllocator(
           torch.device("cpu"), target_bytes=model_size_bytes // 2
       )
   ```

### Verification

- Profile CUDA memory allocation calls before/after — should drop to near-zero after first epoch
- Measure VRAM fragmentation via `torch.cuda.memory_stats()['active_bytes.all.peak']`
- Benchmark allocation time: `torch.cuda.synchronize(); time.time()` around allocation code

---

## Part 3: Coordinated Layer Offload Conductor

**Impact:** Reduces PCIe contention, enables overlapped compute/transfer
**Risk:** Medium — core training path change, needs careful testing
**Effort:** Large (1-2 weeks)
**Files:**

- NEW: `ai-toolkit/toolkit/memory_management/offload_conductor.py`
- MODIFY: `ai-toolkit/toolkit/memory_management/manager_modules.py`
- MODIFY: `ai-toolkit/toolkit/memory_management/manager.py`
- MODIFY: `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py`

### Problem

Current approach (`_BouncingLinearFn`/`_BouncingConv2dFn` in `manager_modules.py`): Each layer independently bounces weights CPU→GPU→CPU via custom autograd functions. Uses 2 streams (`transfer_stream`, `transfer_grad_stream`) and ping-pong double buffers per layer. **No layer knows what other layers are doing** — they all compete for PCIe bandwidth simultaneously.

OneTrainer approach (`LayerOffloadConductor.py:524-950`): A centralized conductor pre-computes which layers to load/offload at each step. Uses 3 dedicated streams and knows about forward→backward transitions. Loads layer N+1 while computing layer N.

### Key Design from OneTrainer

1. **`LayerOffloadStrategy` (lines 376-521):** Pre-computes load/offload schedules for 3 scenarios:
   - Forward→Backward: Keep last N layers loaded (needed immediately for backward)
   - Forward→Forward: Start pre-loading first layers during last layer execution
   - Backward→Forward: Reverse order pre-loading
   - Uses byte-budget approach: calculate how many layers fit in target VRAM

2. **3 CUDA streams (lines 587-594):**
   - `train_stream`: Default compute stream
   - `layer_transfer_stream`: H2D/D2H for model weights
   - `activations_transfer_stream`: H2D/D2H for activations (if offloading)

3. **Static allocators (lines 596-598):**
   - `train_device_layer_allocator`: GPU-side ring buffer
   - `temp_device_layer_allocator`: CPU-side pinned ring buffer
   - `temp_device_activations_allocator`: CPU-side for activation offload

### Implementation

1. **`OffloadStrategy` class:**

   ```python
   class OffloadStrategy:
       """Pre-computes which layers should be loaded at each step."""

       def __init__(self, layers: list[nn.Module], layer_offload_fraction: float):
           self.layer_bytes = [self._get_layer_bytes(layer) for layer in layers]
           total_bytes = sum(self.layer_bytes)
           target_loaded_bytes = int(total_bytes * (1.0 - layer_offload_fraction))

           # Pre-compute loaded layer sets for each scenario
           self.forward_backward_loaded = self._compute_loaded_layers(
               start_forward=True, is_cyclic=False, target_bytes=target_loaded_bytes
           )
           self.forward_forward_loaded = self._compute_loaded_layers(
               start_forward=True, is_cyclic=True, target_bytes=target_loaded_bytes
           )
           self.backward_forward_loaded = self._compute_loaded_layers(
               start_forward=False, is_cyclic=False, target_bytes=target_loaded_bytes
           )

           # Track max memory needed
           all_loaded = self.forward_backward_loaded + self.forward_forward_loaded + self.backward_forward_loaded
           self.max_loaded_bytes = max(
               sum(self.layer_bytes[i] for i in layers) for layers in all_loaded
           )
           self.max_offloaded_bytes = total_bytes - min(
               sum(self.layer_bytes[i] for i in layers) for layers in all_loaded
           ) + max(self.layer_bytes)

       def get_layers_to_offload(self, layer_index: int, is_forward: bool,
                                   is_next_forward: bool, loaded_layers: list[int]) -> list[int]:
           """Returns layers that should be offloaded after executing layer_index."""
           if is_forward and is_next_forward:
               target = self.forward_forward_loaded[layer_index]
           elif is_forward:
               target = self.forward_backward_loaded[layer_index]
           else:
               target = self.backward_forward_loaded[layer_index]
           return sorted([i for i in loaded_layers if i not in target])

       def get_layers_to_load(self, layer_index: int, is_forward: bool,
                              is_next_forward: bool, loaded_layers: list[int]) -> list[int]:
           """Returns layers that should be pre-loaded before executing layer_index."""
           # ... similar logic, returns layers to load ...
   ```

2. **`OffloadConductor` class:**

   ```python
   class OffloadConductor:
       """Centralized coordinator for layer offloading."""

       def __init__(self, train_device: torch.device, temp_device: torch.device,
                    layer_offload_fraction: float = 0.5):
           self.layers: list[nn.Module] = []
           self.layer_device_map: list[torch.device | None] = []

           self.train_device = train_device
           self.temp_device = temp_device

           # 3 CUDA streams
           self.train_stream = torch.cuda.default_stream(train_device)
           self.layer_transfer_stream = torch.cuda.Stream(train_device)
           self.activations_transfer_stream = torch.cuda.Stream(train_device)

           # Static allocators (from Part 2)
           self.gpu_allocator: RingBufferAllocator = None
           self.cpu_allocator: RingBufferAllocator = None

           # Sync events per layer
           self.layer_train_events: list[torch.cuda.Event] = []
           self.layer_transfer_events: list[torch.cuda.Event] = []

           self.strategy: OffloadStrategy = None
           self.is_forward_pass = True
           self.keep_graph = False  # True if backward will follow this forward
           self.is_active = False

       def add_layer(self, layer: nn.Module):
           """Register a layer (transformer block) with the conductor."""
           self.layers.append(layer)
           self.layer_device_map.append(None)
           self.layer_train_events.append(torch.cuda.Event())
           self.layer_transfer_events.append(torch.cuda.Event())

       def activate(self):
           """Initialize strategy and allocators based on registered layers."""
           self.strategy = OffloadStrategy(self.layers, self.layer_offload_fraction)
           self.gpu_allocator = RingBufferAllocator(
               self.train_device, self.strategy.max_loaded_bytes
           )
           self.cpu_allocator = RingBufferAllocator(
               self.temp_device, self.strategy.max_offloaded_bytes
           )
           self.is_active = True

       def start_forward(self, keep_graph: bool):
           """Called before forward pass begins."""
           self.layer_transfer_stream.wait_stream(self.train_stream)
           self.is_forward_pass = True
           self.keep_graph = keep_graph

       def before_layer(self, layer_index: int) -> None:
           """Ensure layer is on GPU, schedule next layer transfer."""
           if not self.is_active:
               return

           # Wait for this layer's transfer if pending
           self.train_stream.wait_event(self.layer_transfer_events[layer_index])

           # Schedule offloads for layers no longer needed
           loaded = self._get_loaded_layers()
           for i in self.strategy.get_layers_to_offload(
               layer_index, self.is_forward_pass, not self.keep_graph, loaded
           ):
               self._schedule_layer_offload(i)

           # Schedule loads for upcoming layers
           for i in self.strategy.get_layers_to_load(
               layer_index, self.is_forward_pass, not self.keep_graph, loaded
           ):
               self._schedule_layer_load(i)

       def after_layer(self, layer_index: int) -> None:
           """Record completion event for this layer."""
           if not self.is_active:
               return
           self.layer_train_events[layer_index].record(self.train_stream)

       def _schedule_layer_load(self, layer_index: int):
           """Async load layer from CPU to GPU."""
           with torch.cuda.stream(self.layer_transfer_stream):
               self.layer_transfer_stream.wait_event(self.layer_train_events[layer_index])
               layer = self.layers[layer_index]
               for param in layer.parameters():
                   # Allocate from GPU ring buffer and copy
                   gpu_param = self.gpu_allocator.allocate_like(param.data, layer_index)
                   gpu_param.copy_(param.data, non_blocking=True)
                   param.data = gpu_param
               self.layer_device_map[layer_index] = self.train_device
               self.layer_transfer_events[layer_index].record()

       def _schedule_layer_offload(self, layer_index: int):
           """Async offload layer from GPU to CPU."""
           # ... similar pattern with cpu_allocator ...
   ```

3. **Integration with training loop in `SDTrainer.py`:**

   ```python
   # In hook_train_loop or similar:
   if self.offload_conductor and self.offload_conductor.is_active:
       self.offload_conductor.start_forward(keep_graph=self.training)

   # Wrap transformer block execution:
   for i, block in enumerate(self.sd.unet.transformer_blocks):
       self.offload_conductor.before_layer(i)
       hidden_states = block(hidden_states, ...)
       self.offload_conductor.after_layer(i)
   ```

### Key Design Decisions

- The conductor operates at the **transformer block level** (not individual Linear/Conv layers) — same granularity as OneTrainer
- Existing per-module bouncing is preserved as fallback for non-transformer architectures
- The conductor is optional and controlled by a config flag: `layer_offload_conductor: true`
- Requires Part 2 (ring allocators) to be implemented first

### Verification

- Compare VRAM usage profiles with/without conductor on Flux training
- Measure PCIe utilization via `nvidia-smi dmon` — should see smoother transfer pattern
- Benchmark s/it on 12GB, 16GB, 24GB GPUs
- Profile with `torch.profiler` to verify stream overlap

---

## Part 4: Fused Backward Pass (Optimizer-Step-During-Backward)

**Impact:** Reduces peak memory by discarding gradients immediately after consumption
**Risk:** Medium — optimizer integration change
**Effort:** Medium (3-5 days)
**Files:**

- MODIFY: `ai-toolkit/jobs/process/BaseSDTrainProcess.py` — optimizer setup
- MODIFY: `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py` — training loop
- NEW: `ai-toolkit/toolkit/fused_backward.py`

### Problem

Current flow: `accelerator.backward(loss)` → all gradients accumulate → `optimizer.step()` → `optimizer.zero_grad()`. All parameter gradients must coexist in VRAM simultaneously.

OneTrainer flow (`GenericTrainer.py:554-598`): Uses `register_post_accumulate_grad_hook` to run the optimizer step on each parameter immediately after its gradient is computed during backward. Gradients are set to `None` immediately after the optimizer consumes them → lower peak VRAM.

### Key Implementation from OneTrainer (lines 554-598)

```python
def __apply_fused_back_pass(self, scaler):
    fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
    # ...

    for param_group in self.model.optimizer.param_groups:
        for i, parameter in enumerate(param_group["params"]):
            if parameter.requires_grad:
                if scaler:
                    def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                        scaler.unscale_parameter_(tensor, self.model.optimizer)
                        if self.config.clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                        scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                        tensor.grad = None  # <-- KEY: immediately release gradient
                else:
                    def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                        if self.config.clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                        self.model.optimizer.step_parameter(tensor, param_group, i)
                        tensor.grad = None  # <-- KEY: immediately release gradient

                def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                    if self.__is_update_step(self.model.train_progress):
                        __optimizer_step(tensor)

                handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                self.grad_hook_handles.append(handle)
```

### Implementation

1. **`FusedBackwardManager` class** (in `fused_backward.py`):

   ```python
   class FusedBackwardManager:
       """Manages fused backward pass with per-parameter optimizer steps."""

       def __init__(self, optimizer, parameters: list[nn.Parameter],
                    clip_grad_norm: float | None = None,
                    grad_scaler: torch.cuda.amp.GradScaler | None = None):
           self.optimizer = optimizer
           self.clip_grad_norm = clip_grad_norm
           self.grad_scaler = grad_scaler
           self.hooks: list[RemovableHandle] = []
           self._is_update_step = True  # Controlled externally

       def attach(self):
           """Register post_accumulate_grad_hook on all trainable parameters."""
           for param_group in self.optimizer.param_groups:
               for i, param in enumerate(param_group['params']):
                   if not param.requires_grad:
                       continue

                   def make_hook(pg=param_group, idx=i):
                       def hook(tensor: torch.Tensor):
                           if not self._is_update_step:
                               return

                           if self.grad_scaler:
                               self.grad_scaler.unscale_parameter_(tensor, self.optimizer)

                           if self.clip_grad_norm:
                               nn.utils.clip_grad_norm_(tensor, self.clip_grad_norm)

                           if self.grad_scaler:
                               self.grad_scaler.maybe_opt_step_parameter(
                                   tensor, pg, idx, self.optimizer
                               )
                           else:
                               self.optimizer.step_parameter(tensor, pg, idx)

                           tensor.grad = None  # Immediately release gradient memory
                       return hook

                   handle = param.register_post_accumulate_grad_hook(make_hook())
                   self.hooks.append(handle)

       def set_update_step(self, is_update_step: bool):
           """Control whether hooks should run (for gradient accumulation)."""
           self._is_update_step = is_update_step

       def detach(self):
           """Remove all hooks."""
           for h in self.hooks:
               h.remove()
           self.hooks.clear()
   ```

2. **Extend GradScaler for per-parameter operations:**

   ```python
   # May need custom scaler methods if torch.amp.GradScaler doesn't support:
   # - unscale_parameter_(tensor, optimizer)
   # - maybe_opt_step_parameter(tensor, param_group, idx, optimizer)
   ```

3. **Integration in training setup:**

   ```python
   # In BaseSDTrainProcess or SDTrainer setup:
   if self.train_config.fused_back_pass:
       if self.train_config.gradient_accumulation_steps > 1:
           print("Warning: fused_back_pass with gradient_accumulation > 1 "
                 "does not reduce VRAM usage")

       self.fused_backward_manager = FusedBackwardManager(
           optimizer=self.optimizer,
           parameters=list(self.get_trainable_params()),
           clip_grad_norm=self.train_config.max_grad_norm,
           grad_scaler=self.grad_scaler if self.train_config.mixed_precision else None
       )
       self.fused_backward_manager.attach()
   ```

4. **Modified training loop:**

   ```python
   # Before backward:
   if self.fused_backward_manager:
       is_update = (step + 1) % self.train_config.gradient_accumulation_steps == 0
       self.fused_backward_manager.set_update_step(is_update)

   # Backward pass (hooks fire automatically)
   self.accelerator.backward(loss)

   # After backward:
   if self.fused_backward_manager and is_update:
       # Skip regular optimizer.step() - already done in hooks
       self.lr_scheduler.step()
       # zero_grad not needed - grads already None
   else:
       # Regular path
       self.optimizer.step()
       self.optimizer.zero_grad()
   ```

### Constraints

- **Not compatible with gradient accumulation > 1** for memory savings (same as OneTrainer — warn in config)
- Requires optimizer to support per-parameter stepping (AdamW, Lion, Prodigy all do)
- With mixed precision, requires custom GradScaler extension

### Verification

- Measure peak VRAM with/without fused back pass on Flux LoRA training
- Verify training loss curves are identical (same math, different execution order)
- Test with gradient accumulation disabled
- Profile to confirm gradients are freed during backward

---

## Part 5: Activation Offloading to CPU

**Impact:** Enables training with less VRAM by offloading activations instead of recomputing
**Risk:** Medium — requires checkpoint hook integration
**Effort:** Large (1-2 weeks)
**Files:**

- NEW: `ai-toolkit/toolkit/memory_management/activation_offload.py`
- MODIFY: `ai-toolkit/toolkit/memory_management/manager_modules.py`
- MODIFY: Gradient checkpointing integration

### Problem

Current approach: Gradient checkpointing recomputes activations during backward pass. This trades compute for memory — activations are never stored, but each layer's forward pass runs twice.

OneTrainer approach (`LayerOffloadConductor.py:770-784` + `StaticActivationAllocator`): Activations are saved to pre-allocated CPU pinned memory during forward, then transferred back to GPU during backward on a dedicated stream. This trades PCIe bandwidth for compute — each layer runs once, but activations travel CPU↔GPU.

### When Activation Offload > Recompute

- When PCIe bandwidth is high relative to compute (newer GPUs with PCIe 4.0/5.0)
- When recompute cost is high (complex attention mechanisms)
- When CPU RAM is plentiful but VRAM is constrained

### Implementation

1. **`ActivationOffloader` class:**

   ```python
   class ActivationOffloader:
       """Offloads activations to CPU instead of recomputing."""

       def __init__(self, conductor: OffloadConductor):
           self.conductor = conductor
           self.activations_map: dict[int, Any] = {}  # call_index -> activations
           self.transfer_events: dict[int, torch.cuda.Event] = {}

           # Use conductor's activation stream
           self.transfer_stream = conductor.activations_transfer_stream
           self.cpu_allocator = StaticActivationAllocator(torch.device("cpu"))

       def save_activations(self, call_index: int, activations: Any, tensor_indices: list[int]):
           """Called after forward layer - offload to CPU."""
           self.activations_map[call_index] = activations

           with torch.cuda.stream(self.transfer_stream):
               # Wait for compute to finish
               self.transfer_stream.wait_stream(self.conductor.train_stream)

               # Reserve space and copy
               tensors = self._get_tensors(activations, tensor_indices)
               self.cpu_allocator.reserve_cache(tensors)
               for t in tensors:
                   cpu_t = self.cpu_allocator.allocate_like(t)
                   cpu_t.copy_(t, non_blocking=True)

               self.transfer_events[call_index] = torch.cuda.Event()
               self.transfer_events[call_index].record()

       def restore_activations(self, call_index: int) -> Any:
           """Called before backward layer - restore from CPU."""
           # Wait for offload to complete
           if call_index in self.transfer_events:
               self.conductor.train_stream.wait_event(self.transfer_events[call_index])

           activations = self.activations_map.pop(call_index, None)
           if activations is None:
               return None

           # Pre-fetch from CPU (could be overlapped with previous layer backward)
           # ... copy back to GPU ...
           return activations

       def clear(self):
           """Reset state between forward/backward passes."""
           self.activations_map.clear()
           self.transfer_events.clear()
           self.cpu_allocator.deallocate()
   ```

2. **Integration with gradient checkpointing:**

   Replace `torch.utils.checkpoint.checkpoint()` with custom function that offloads instead of discarding:

   ```python
   def checkpoint_with_offload(function, *args, offloader: ActivationOffloader,
                               call_index: int, tensor_indices: list[int], **kwargs):
       """Checkpoint that offloads activations to CPU instead of recomputing."""

       # Forward: run function, offload activations
       with torch.no_grad():
           outputs = function(*args, **kwargs)
       offloader.save_activations(call_index, outputs, tensor_indices)

       # Create autograd function for backward
       class OffloadCheckpointFunction(torch.autograd.Function):
           @staticmethod
           def forward(ctx, *inputs):
               ctx.call_index = call_index
               return outputs

           @staticmethod
           def backward(ctx, *grad_outputs):
               # Restore activations from CPU
               restored = offloader.restore_activations(ctx.call_index)
               # Rerun forward with grad enabled to get gradients
               with torch.enable_grad():
                   outputs = function(*args, **kwargs)
               return torch.autograd.grad(outputs, args, grad_outputs)

       return OffloadCheckpointFunction.apply(*args)
   ```

3. **Config option:** `activation_offload: true/false` (default: `false`)

### Dependencies

- Requires the conductor from Part 3 to coordinate streams
- Works best with Part 2's static allocators

### Verification

- Compare VRAM peak with checkpoint-recompute vs checkpoint-offload
- Measure s/it impact — offload should be faster if PCIe bandwidth > recompute cost
- Test on 12GB GPU where recompute is the bottleneck
- Profile to verify stream overlap

---

## Part 6: Optional Accelerate Bypass for Single-GPU

**Impact:** Eliminates Accelerate dispatch overhead (~1-5% per step)
**Risk:** Low — only for single-GPU, Accelerate path preserved for multi-GPU
**Effort:** Small (1-2 days)
**Files:**

- MODIFY: `ai-toolkit/jobs/process/BaseSDTrainProcess.py`
- MODIFY: `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py`

### Problem

Every `self.accelerator.backward(loss)` call at `SDTrainer.py:2242` goes through Accelerate's dispatch layer. For single-GPU training (the majority use case), this adds overhead with zero benefit.

OneTrainer calls `loss.backward()` directly (`GenericTrainer.py:750`).

### Implementation

1. **Add config option and detection:**

   ```python
   # In train config
   bypass_accelerate: Optional[bool] = None  # None = auto-detect

   # In setup
   if self.train_config.bypass_accelerate is None:
       self.use_accelerate = torch.cuda.device_count() > 1
   else:
       self.use_accelerate = not self.train_config.bypass_accelerate
   ```

2. **Conditional backward:**

   ```python
   # In training loop
   if self.use_accelerate:
       self.accelerator.backward(loss)
   else:
       if self.grad_scaler:
           self.grad_scaler.scale(loss).backward()
       else:
           loss.backward()
   ```

3. **Manual gradient scaling when bypassing:**

   ```python
   if not self.use_accelerate:
       if self.grad_scaler:
           self.grad_scaler.unscale_(self.optimizer)
       if self.train_config.max_grad_norm:
           nn.utils.clip_grad_norm_(self.get_trainable_params(),
                                     self.train_config.max_grad_norm)
       if self.grad_scaler:
           self.grad_scaler.step(self.optimizer)
           self.grad_scaler.update()
       else:
           self.optimizer.step()
   ```

### Verification

- Benchmark 1000 steps with/without Accelerate on single GPU
- Verify identical training loss
- Ensure multi-GPU path still uses Accelerate

---

## Implementation Order & Dependencies

```
Part 1 (VRAM leak fix)     ← No dependencies, highest impact, do first
     ↓
Part 2 (Ring buffer)       ← Independent, enables Parts 3 & 5
     ↓
Part 4 (Fused backward)    ← Independent of 2/3, can parallel with Part 2
     ↓
Part 3 (Offload conductor) ← Depends on Part 2
     ↓
Part 5 (Activation offload) ← Depends on Parts 2 & 3
     ↓
Part 6 (Accelerate bypass) ← Independent, lowest priority
```

**Recommended phases:**

- **Phase A** (immediate, 1 week): Part 1 (bug fix) — addresses user-reported critical issue
- **Phase B** (foundation, 2 weeks): Parts 2 + 4 (allocator + fused backward) — can be developed in parallel
- **Phase C** (core, 2 weeks): Part 3 (conductor) — biggest architectural change
- **Phase D** (advanced, 2 weeks): Parts 5 + 6 (activation offload + Accelerate bypass)

---

## Additional Considerations from OneTrainer

### torch_util.py Helper Functions

OneTrainer has utility functions that simplify tensor operations across the codebase:

- `pin_tensor_()` / `unpin_tensor_()` — safe pinning with error handling
- `tensors_to_device_()` — batch move with allocator support
- `tensors_record_stream()` — ensure tensors aren't freed while async ops pending
- `torch_gc()` — comprehensive garbage collection

Consider adding similar utilities to `ai-toolkit/toolkit/` for consistency.

### Multi-GPU Considerations

OneTrainer's fused backward has special handling for multi-GPU (`GenericTrainer.py:876-882`):
- Gradients may still be in flight during async gradient reduction
- Layers can be deferred for offload until gradients complete
- Need to track `deferred_layers` list for later offload

### Quantization Offload Support

OneTrainer's `quantization_util.py` has `get_offload_tensor_bytes()` and `offload_quantized()` for handling quantized tensors specially during offload. AI Toolkit's `_is_quantized_tensor()` and `_materialize_linear_weight()` patterns align well but may need extension.

---

## User Decisions

- **Scope:** Full plan — all 6 parts
- **Migration:** Keep existing bouncing as fallback, add conductor as opt-in via config flag
- **Quantization:** Ensure all parts handle bitsandbytes/torchao quantized weights

## Testing Strategy

For each part:

1. **Unit test**: Isolated component test (allocator, conductor, fused backward manager)
2. **Integration test**: Run a short LoRA training (50 steps) on SD 1.5 (fast) and Flux (representative)
3. **Regression test**: Compare loss curves before/after — must be numerically equivalent
4. **VRAM profiling**: `torch.cuda.memory_stats()` snapshots at key points
5. **Speed benchmark**: Measure s/it on RTX 3090 (24GB) and RTX 4070 Super (12GB) configurations

## Config Options Summary

```yaml
# New training config options
memory_management:
  # Part 1: Always enabled (bug fix)

  # Part 2: Ring buffer allocation
  use_ring_allocator: true  # default: true
  ring_allocator_gpu_fraction: 0.25  # fraction of model size
  ring_allocator_cpu_fraction: 0.50

  # Part 3: Coordinated offloading
  use_offload_conductor: false  # default: false (opt-in)
  layer_offload_fraction: 0.5  # 0.0 = all on GPU, 1.0 = all offloaded

  # Part 4: Fused backward
  fused_back_pass: false  # default: false
  # Warning: gradient_accumulation_steps > 1 negates memory benefit

  # Part 5: Activation offloading
  activation_offload: false  # default: false
  # Requires: use_offload_conductor: true

  # Part 6: Accelerate bypass
  bypass_accelerate: null  # null = auto-detect, true/false = manual
```

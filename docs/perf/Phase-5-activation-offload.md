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
| `toolkit/memory_management/offload_conductor.py` | Extend conductor with activation offload state + transfer scheduling |
| Model checkpoint wrapper integration | Register `call_index -> layer_index`, invoke conductor `before_layer/after_layer` |
| `jobs/process/BaseSDTrainProcess.py` | Lifecycle setup/teardown and config gating |

---

## Correctness Guardrails

Before implementation, preserve these invariants:

1. **Keep activation offload inside the conductor/checkpoint-wrapper path (OneTrainer pattern).**
   - Do not add a separate custom autograd checkpoint stack.
   - The same wrapped layer execution order must drive both layer offload and activation offload.

2. **Activation offload requires the same checkpoint assumptions as Phase 4.**
   - `call_index -> layer_index` mapping must be exact for every wrapped call.
   - Reentrant checkpoint mode is required for the `torch.is_grad_enabled()` transition heuristic used by conductor logic.

3. **Allocator API must match this repo’s implementation.**
   - Current allocator is `RingBufferAllocator`; there is no `ActivationAllocator` in `ring_allocator.py`.
   - If adding activation allocator support, define it explicitly in conductor/memory modules first.

4. **Offload only selected activation tensors, not arbitrary outputs.**
   - Layer wrappers should declare `included_offload_param_indices`/tensor indices explicitly.
   - Text encoder paths may need no activation offload (model-specific exceptions).

## Implementation

### Step 1: Extend conductor activation path

**File:** `toolkit/memory_management/offload_conductor.py`

Implement activation handling directly in `OffloadConductor` (mirroring OneTrainer):
- Maintain:
  - `activations_map: dict[int, Any]`
  - `call_index_layer_index_map: dict[int, int]`
  - `activations_transfer_event_map: dict[int, SyncEvent]`
- Add a dedicated activation CPU allocator (pinned CPU buffers) in conductor memory modules.
- In `before_layer(layer_index, call_index, activations)`:
  - On backward path, wait for activation transfer completion for `call_index`
  - Restore offloaded activation tensors to train device
  - Optionally prefetch previous call activations.
- In `after_layer(layer_index, call_index, activations)`:
  - On forward path with `keep_graph=True`, save activations and schedule async transfer to CPU.

### Step 2: Integrate through checkpoint wrappers, not SDTrainer block loops

Use the same wrapper pattern used for conductor layer offloading:
- Register each wrapped layer with conductor in exact execution order.
- Pass `call_index` and activation args through:
  - `args = conductor.before_layer(layer_idx, call_index, args)`
  - `out = orig_forward(*args)`
  - `conductor.after_layer(layer_idx, call_index, args)`
- Start each forward via `conductor.start_forward(keep_graph=True/False)` at wrapper boundaries.

### Step 3: Gate by config + lifecycle

In process setup/teardown:
- Require conductor enabled before activation offload.
- Initialize and clear activation caches on forward boundaries and on teardown.
- Add runtime fallback to recompute path if activation offload is unsupported for a specific model wrapper.

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
model:
  # Existing keys in this repo:
  layer_offloading: true
  layer_offloading_transformer_percent: 0.5

  # Proposed conductor extension keys:
  # use_offload_conductor: true
  # activation_offload: false
  # activation_offload_indices: [0]
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

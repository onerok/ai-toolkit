# AI Toolkit Performance Improvement Plan

## Overview

This plan ports memory management and performance patterns from [OneTrainer](https://github.com/Nerogar/OneTrainer) into AI Toolkit to address:

1. **Catastrophic VRAM leak after checkpoint saves** (Issues #504, #575, #387)
2. **Uncoordinated per-layer CPU/GPU offloading** causing PCIe contention
3. **No pre-allocated memory pools** leading to fragmentation
4. **No fused backward pass** keeping all gradients alive simultaneously
5. **No activation offloading** alternative to gradient checkpointing recompute
6. **HuggingFace Accelerate dispatch overhead** on single-GPU systems

---

## Lightweight Testing (No Full Training Required)

To iterate quickly without running full training sessions, use these `just` commands:

### Quick Commands

```bash
# Memory-only benchmark (~5 sec, no dataset needed)
just perf-memory

# Training benchmark (requires datasets/perf_test/)
just perf-train                                           # 20 steps, save at 10, generation validation + CLIPScore
just perf-train steps=30 save_at=15 clipscore_threshold=0.22  # Custom threshold

# Capture baseline
just perf-baseline "phase-2-before"

# Compare against baseline
just perf-compare docs/perf/baselines/phase-2-before.json

# List saved baselines
just perf-list
```

### Direct Script Usage

```bash
# Run benchmark with generation validation + CLIPScore threshold
uv run python scripts/perf_benchmark.py \
    --steps 20 --resolution 256 --save-at 10 \
    --clean-output --validate-generation --clipscore-threshold 0.20

# Memory-only test (no actual training, just allocation patterns)
uv run python scripts/perf_benchmark.py --memory-only

# Capture baseline with notes + generation validation + CLIPScore threshold
uv run python scripts/collect_baselines.py \
    --tag "my-baseline" --notes "Before Phase 2" \
    --clean-output --validate-generation --clipscore-threshold 0.20

# Compare two baselines
uv run python scripts/compare_baselines.py \
    --baseline docs/perf/baselines/phase-1.json \
    --compare docs/perf/baselines/phase-2.json
```

### Minimal Training Config

A pre-configured minimal config exists at `config/perf_test.yaml`:
- 20 steps, 256px resolution, batch size 1
- Saves checkpoint at step 10 (for VRAM leak testing)
- Sampling disabled for speed

**Note:** Requires a test dataset at `datasets/perf_test/` with images + captions.

---

## Phase 0: Performance Test Harness

**Purpose:** Reusable test infrastructure for measuring and comparing performance across all phases.

This is **not** a one-time baseline capture—it's the test harness you'll use throughout the entire implementation to:
- Capture baselines before each phase
- Verify improvements after each phase
- Detect regressions
- Compare different configurations

### Test Harness Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `perf_benchmark.py` | Core benchmark runner | `scripts/perf_benchmark.py` |
| `collect_baselines.py` | Capture metrics to JSON | `scripts/collect_baselines.py` |
| `compare_baselines.py` | Compare two baseline files | `scripts/compare_baselines.py` |
| `perf_test.yaml` | Minimal training config | `config/perf_test.yaml` |
| Baseline storage | Historical metrics | `docs/perf/baselines/<tag>.json` |

### Metrics Captured

| Metric | Description | Key For |
|--------|-------------|---------|
| `peak_vram_mb` | Maximum VRAM during training | All phases |
| `post_save_vram_mb` | VRAM 2 steps after checkpoint | Phase 1 |
| `leaked_mb` | VRAM not recovered after save | Phase 1 |
| `avg_step_time_s` | Mean step time (20 steps) | All phases |
| `post_save_step_time_s` | Mean step time after save | Phase 1 |
| `slowdown_ratio` | post_save / pre_save time | Phase 1 |
| `num_alloc_retries` | CUDA allocator pressure | Phase 2 |
| `fragmentation_ratio` | Reserved vs allocated | Phase 2 |
| `pcie_transfer_mb` | CPU↔GPU data movement | Phase 5, 6 |

### Workflow Per Phase

```
┌─────────────────────────────────────────────────────────────────┐
│  Before Phase N                                                 │
│  ─────────────────                                              │
│  1. git checkout <pre-phase-N-commit>                           │
│  2. uv run python scripts/collect_baselines.py \                │
│       --clean-output \                                          │
│       --validate-generation --clipscore-threshold 0.20 \        │
│       --tag "phase-N-baseline" --output docs/perf/baselines/    │
│                                                                 │
│  Implement Phase N                                              │
│  ─────────────────                                              │
│  3. Make changes                                                │
│  4. Quick iteration: uv run python scripts/perf_benchmark.py \  │
│       --clean-output --validate-generation \                    │
│       --clipscore-threshold 0.20                                │
│                                                                 │
│  After Phase N                                                  │
│  ─────────────────                                              │
│  5. uv run python scripts/compare_baselines.py \                │
│       --baseline docs/perf/baselines/phase-N-baseline.json \    │
│       --current                                                 │
│  6. Verify expected improvements, no regressions                │
│  7. Capture new baseline for Phase N+1                          │
└─────────────────────────────────────────────────────────────────┘
```

### Usage Examples

```bash
# Quick benchmark during development (< 2 min)
uv run python scripts/perf_benchmark.py --steps 20 --resolution 256
uv run python scripts/perf_benchmark.py --steps 20 --resolution 256 --clean-output --validate-generation --clipscore-threshold 0.20

# Full benchmark with save cycle
uv run python scripts/perf_benchmark.py --steps 30 --save-at 15 --clean-output --validate-generation --clipscore-threshold 0.20

# Memory-only test (no training, just allocation patterns)
uv run python scripts/perf_benchmark.py --memory-only

# Capture baseline with tag
uv run python scripts/collect_baselines.py --tag "main-baseline" --clean-output --validate-generation --clipscore-threshold 0.20

# Compare current vs baseline
uv run python scripts/compare_baselines.py \
    --baseline docs/perf/baselines/main-baseline.json \
    --current \
    --clean-output

# Compare two saved baselines
uv run python scripts/compare_baselines.py \
    --baseline docs/perf/baselines/phase-1-before.json \
    --compare docs/perf/baselines/phase-1-after.json
```

### Baseline JSON Format

```json
{
  "tag": "phase-1-baseline",
  "commit": "abc123",
  "timestamp": "2025-02-13T10:00:00Z",
  "hardware": {
    "gpu": "RTX 3090",
    "vram_gb": 24,
    "cuda_version": "12.4",
    "driver_version": "550.54"
  },
  "config": {
    "steps": 20,
    "resolution": 256,
    "batch_size": 1,
    "save_at": 10
  },
  "metrics": {
    "peak_vram_mb": 18432,
    "post_save_vram_mb": 18500,
    "leaked_mb": 68,
    "avg_step_time_s": 1.23,
    "post_save_step_time_s": 1.25,
    "slowdown_ratio": 1.02,
    "num_alloc_retries": 0,
    "fragmentation_ratio": 0.05
  }
}
```

### Regression Thresholds

Comparison script flags regressions if:

| Metric | Regression Threshold |
|--------|---------------------|
| `peak_vram_mb` | > 5% increase |
| `avg_step_time_s` | > 5% slower |
| `slowdown_ratio` | > 1.10x |
| `leaked_mb` | > 100 MB |

### Current Status

| Phase | Baseline Captured | Implemented | Verified |
|-------|-------------------|-------------|----------|
| 0 (Harness) | N/A | ✅ Done | N/A |
| 1 (VRAM Leak) | ✅ `phase1-before.json` | ✅ Done | ✅ `phase1-fixed.json` |
| 2 (Ring Buffer) | ⬜ | ⬜ | ⬜ |
| 3 (Fused Backward) | ⬜ | ⬜ | ⬜ |
| 4 (Arch-Agnostic) | ⬜ | ⬜ | ⬜ |
| 5 (Conductor) | ⬜ | ⬜ | ⬜ |
| 6 (Activation) | ⬜ | ⬜ | ⬜ |
| 7 (Accelerate) | ⬜ | ⬜ | ⬜ |

---

## Phase Overview

| Phase | Name | Impact | Risk | Effort | Dependencies |
|-------|------|--------|------|--------|--------------|
| **0** | Performance Test Harness ✅ | Foundation | None | 1 day | None |
| **1** | [VRAM Leak Fix](perf/Phase-1-vram-leak-fix.md) ✅ | Critical | Low | 1-2 days | Phase 0 |
| **2** | [Ring Buffer Allocator](perf/Phase-2-ring-allocator.md) | High | Low-Med | 3-4 days | Phase 1 |
| **3** | [Fused Backward](perf/Phase-3-fused-backward.md) | High | Medium | 3-5 days | Phase 1 |
| **4** | [Architecture-Agnostic Training](perf/Phase-4-architecture-agnostic-training.md) | High | Medium | 3-5 days | Phase 3 |
| **5** | [Offload Conductor](perf/Phase-5-offload-conductor.md) | High | Medium | 1-2 weeks | Phase 2, 4 |
| **6** | [Activation Offload](perf/Phase-6-activation-offload.md) | Medium | Medium | 1-2 weeks | Phase 5 |
| **7** | [Accelerate Bypass](perf/Phase-7-accelerate-bypass.md) | Low | Low | 1-2 days | None |

---

## Dependency Graph

```
Phase 0 (Test harness) ✅
         │
         ▼
Phase 1 (VRAM leak fix) ✅
         │
         ├───────────────────────────┐
         │                           │
         ▼                           ▼
Phase 2 (Ring buffer)        Phase 3 (Fused backward)
         │                           │
         ▼                           │
Phase 4 (Architecture-agnostic training)
         │
         ▼
Phase 5 (Conductor) ←────────────────────┘
         │                    (Phase 2 + 4 converge here)
         ▼
Phase 6 (Activation offload)


Phase 7 (Accelerate bypass) ← Independent (can run anytime)
```

**Notes:**
- Phases 2 and 3 can be implemented in parallel after Phase 1
- Phase 4 establishes cross-architecture capability gating before conductor rollout
- Phase 5 integrates work from both Phase 2 (ring allocator) and Phase 4 (architecture gating)
- Phase 7 has no dependencies and can be done at any point

---

## Implementation Timeline

### Week 1: Test Harness + Critical Bug Fix
- **Phase 0:** Build performance test harness ✅
  - `scripts/perf_benchmark.py` — core benchmark runner
  - `scripts/collect_baselines.py` — metric capture
  - `scripts/compare_baselines.py` — regression detection
  - `config/perf_test.yaml` — minimal training config
  - `justfile` commands: `just perf-*`

- **Phase 1:** Fix VRAM leak after checkpoint saves ✅
  - Add `torch.cuda.synchronize()` before cleanup
  - Clear ping-pong buffers in `_DEVICE_STATE`
  - Add post-save cleanup
  - ✅ Baseline captured and improvement verified (see Implementation Log)

### Weeks 2-3: Foundation
- **Phase 2:** Static ring buffer allocator (parallel track A)
  - Pre-allocated GPU staging buffer
  - 16-byte aligned allocations
  - Integration with bouncing modules

- **Phase 3:** Fused backward pass (parallel track B)
  - `register_post_accumulate_grad_hook` integration
  - Per-parameter optimizer stepping
  - Immediate gradient release

### Weeks 4-5: Core Architecture
- **Phase 4:** Architecture-agnostic training contract and gating
- Include handoff-required parity work:
  - optimizer `step_parameter` support for common optimizers
  - config-time compatibility guards for fused/offload
  - stable-loss capability gating across model families
- **Phase 5:** Coordinated offload conductor
  - 3-stream architecture
  - Pre-computed offload strategy
  - Overlapped compute/transfer

### Weeks 6-7: Advanced Features
- **Phase 6:** Activation offloading
  - CPU pinned memory for activations
  - Replace recompute with transfer
  - Custom checkpoint integration

- **Phase 7:** Accelerate bypass
  - Single-GPU optimization
  - Direct `loss.backward()` calls

### Regression Track (Required)
- Add LoRA-strength regression checks to all Phase 4-6 rollouts:
  - control run with fused/conductor/layer_offloading/stable_loss all disabled
  - one-knob-at-a-time re-enable sequence
  - compare image strength at fixed LoRA weight and LoRA norm/delta stats
  - if divergence is isolated to layer offloading, inspect `toolkit/memory_management/manager.py` and `toolkit/memory_management/manager_modules.py`

### Latest Phase 4 Ablation Status (2026-02-14)

- Runner/analyzer automation:
  - `scripts/run_phase4_ablations.py`
    - fixed-seed config generation (`training_seed`, `train.seed`, `sample.seed`)
    - auto optimizer compatibility swap (`adamw8bit` -> `adamw8`)
    - optional `--double-control` and `--analyze`
  - `scripts/analyze_phase4_ablations.py`
    - reports `relative_delta`, `cosine_similarity`, and `lora_cosine_similarity`
    - explicit control-repeat sanity check

- Verified run:
  - Run root: `output/phase4_ablations/20260214_110624`
  - Control repeat sanity:
    - `relative_delta=0.0677`
    - `cosine=0.9977`

- Per-knob deltas vs control:
  - `fused_back_pass`: `relative_delta=0.0619`, `cosine=0.9981`
  - `offload_conductor`: `relative_delta=0.0705`, `cosine=0.9975`
  - `stable_loss`: `relative_delta=0.0914`, `cosine=0.9958`
  - `layer_offloading`: `relative_delta=0.1528`, `cosine=0.9884`

- Current interpretation:
  - Baseline determinism is now acceptable for ablation decisions.
  - `layer_offloading` is the clear outlier and should be treated as the primary regression to investigate before Phase 5/6 progression.
  - `fused_back_pass` and `offload_conductor` currently track control-repeat noise band closely.

---

## Configuration Summary

```yaml
model:
  # Phase 1: Always enabled (bug fix)

  # Phase 2: Ring buffer allocation
  # (via MemoryManager.attach params)
  layer_offloading: false
  layer_offloading_transformer_percent: 1.0
  layer_offloading_text_encoder_percent: 1.0

  # Phase 4: Architecture-agnostic gating
  # use_arch_capability_gates: true

  # Phase 5: Coordinated offloading
  # Proposed extension if conductor is added:
  # use_offload_conductor: false
  # layer_offload_fraction: 0.5

  # Phase 6: Activation offloading
  # Proposed extension:
  # activation_offload: false
  # Requires: use_offload_conductor: true

  # Phase 7: Accelerate bypass
  # Proposed extension:
  # bypass_accelerate: null  # null = auto-detect

train:
  # Phase 3: Fused backward
  fused_back_pass: false
  # Note: strongest memory benefit when gradient_accumulation=1 and gradient_accumulation_steps=1
```

---

## Testing Strategy

Each phase includes:

1. **Unit tests:** Isolated component validation
2. **Integration tests:** 50-step training runs on SD 1.5 and Flux
3. **Regression tests:** Loss curve comparison before/after
4. **Memory profiling:** `torch.cuda.memory_stats()` snapshots
5. **Benchmarking:** s/it on RTX 3090 (24GB) and RTX 4070 Super (12GB)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Post-save VRAM recovery | < 100 MB leaked |
| Post-save s/it ratio | < 1.05x (no slowdown) |
| Peak VRAM with fused backward | -10-30% |
| CUDA allocations after warmup | Near zero |
| PCIe utilization smoothness | Visible improvement in `nvidia-smi dmon` |

---

## References

- [OneTrainer Source](https://github.com/Nerogar/OneTrainer)
- Key files:
  - `modules/util/LayerOffloadConductor.py` (lines 45-950)
  - `modules/trainer/GenericTrainer.py` (lines 554-598)
  - `modules/util/torch_util.py`

---

## User Decisions

- **Scope:** Full plan — all 7 phases
- **Migration:** Keep existing bouncing as fallback, add conductor as opt-in
- **Quantization:** All phases must handle bitsandbytes/torchao quantized weights

---

## Implementation Log

Historical notes from a prior iteration. Use the phase docs and phase overview above as the current execution guide.

### 2026-02-13: Phase 1 Implementation & Benchmark Enhancement

#### Summary
- Implemented Phase 1 VRAM leak fix
- Enhanced benchmark instrumentation to capture save-specific metrics
- Captured baseline and verified Phase 1 fix works correctly

#### Phase 1: VRAM Leak Fix — IMPLEMENTED

**Files Modified:**
| File | Changes |
|------|---------|
| `jobs/process/BaseSDTrainProcess.py` | Enhanced `flush()` with `clear_bouncing_buffers`, `synchronize`, `sync_device` params; added `post_save_cleanup()` |
| `toolkit/memory_management/manager_modules.py` | Added `clear_device_state_buffers()` and `get_device_state_devices()` |
| `extensions_built_in/diffusion_models/flux2/flux2_model.py` | Minor fix for None batch handling |

**Implementation Details:**

1. **`flush()` signature change** (`BaseSDTrainProcess.py:77-101`):
   ```python
   def flush(clear_bouncing_buffers=False, synchronize=False, sync_device=None):
       if torch.cuda.is_available() and synchronize:
           # Synchronize CUDA devices before cleanup
           cuda_devices = get_device_state_devices(cuda_only=True) or [sync_device]
           for dev in cuda_devices:
               torch.cuda.synchronize(dev)

       if clear_bouncing_buffers:
           clear_device_state_buffers(sync_device)

       gc.collect()
       torch.cuda.empty_cache()
   ```

2. **`clear_device_state_buffers()`** (`manager_modules.py`):
   - Clears ping-pong buffers (`w_buffers`, `b_buffers`, `w_bwd_buffers`, `w_grad_buffers`, `b_grad_buffers`)
   - Sets each buffer slot to `[None, None]` to release tensor references

3. **Save path cleanup** (`BaseSDTrainProcess.py:514-527`):
   - Pre-save: `flush(clear_bouncing_buffers=True, synchronize=True, sync_device=self.device_torch)`
   - Post-save: `self.post_save_cleanup()` calls same flush

**Status:** ✅ Implemented, ✅ Tested, ✅ Verified

---

#### Benchmark Instrumentation Enhancement

**Problem:** Original benchmark only captured pre/post training memory, not save-specific metrics needed to verify Phase 1.

**Solution:** Added hook-based instrumentation to capture metrics around checkpoint saves.

**New Metrics Captured:**
| Metric | Description |
|--------|-------------|
| `pre_save_vram_mb` | VRAM reserved immediately before save |
| `save_peak_vram_mb` | Peak VRAM during save operation |
| `post_save_vram_mb` | VRAM reserved after save completes |
| `pre_save_step_time_s` | Average step time before save |
| `post_save_step_time_s` | Average step time after save |
| `save_duration_s` | How long the save operation took |
| `slowdown_ratio` | `post_save_step_time / pre_save_step_time` |

**Memory Snapshots Added:**
- `pre_save` — Right before `save()` starts
- `during_save_peak` — Peak memory during save (uses `torch.cuda.reset_peak_memory_stats()`)
- `post_save` — Immediately after save completes
- `post_save_step_1` — After first training step post-save
- `post_save_step_2` — After second training step post-save
- `post_save_settled` — Memory stabilized state

**Implementation (`scripts/perf_benchmark.py`):**
- `_instrument_training_processes()` — Wraps `process.timer.start/stop`, `process.save`, `process.end_step_hook`
- Uses `time.perf_counter()` (monotonic) to avoid negative durations from clock jumps
- `MemorySnapshot.capture()` enforces monotonic timestamps

**Bug Fixed:** Initial implementation had timing issues causing negative step durations. Fixed by:
- Using `time.perf_counter()` instead of `time.time()` for elapsed calculation
- Clamping durations with `max(0.0, elapsed)`
- Enforcing monotonic timestamps in `MemorySnapshot`

---

#### Test Results

**Hardware:** NVIDIA GeForce RTX 5090 (31.8 GB VRAM)

**Baseline Captured:** `docs/perf/baselines/phase1-before.json`
- Pre-Phase 1 state (before applying fix)

**Phase 1 Results:** `docs/perf/baselines/phase1-fixed.json`

| Metric | Value | Status |
|--------|-------|--------|
| Pre-save step time | 0.651s | — |
| Post-save step time | 0.642s | — |
| **Slowdown ratio** | **0.99x** | ✅ No regression |
| Save duration | 0.722s | — |
| Pre-save VRAM | 20,202 MB | — |
| Post-save VRAM | 17,396 MB | ✅ Drops after flush |
| Post-save settled | 20,202 MB | Returns to normal |
| VRAM leaked | 0 MB | ✅ No leak |

**Key Finding:** The `flush()` enhancement works — VRAM drops from 20,202 MB to 17,396 MB immediately after save, then returns to normal during subsequent training steps. No performance regression (0.99x ratio means steps are essentially the same speed before/after save).

---

#### Config Changes

**`config/perf_test.yaml`:**
- Changed model from `FLUX.1-dev` to `FLUX.2-klein-base-9B` (gated model access issue)
- Added `arch: "flux2_klein_9b"` for proper model class selection

---

#### Current Branch State

**Branch:** `feat/system-metrics-graph`

**Uncommitted Changes:**
- Phase 1 VRAM leak fix (from stash)
- Benchmark instrumentation enhancement
- Config updates

**Files Modified:**
```
M config/perf_test.yaml
M extensions_built_in/diffusion_models/flux2/flux2_model.py
M jobs/process/BaseSDTrainProcess.py
M scripts/perf_benchmark.py
M toolkit/memory_management/manager_modules.py
```

**Baselines Captured:**
- `docs/perf/baselines/phase1-before.json` — Pre-fix baseline
- `docs/perf/baselines/phase1-after.json` — Post-fix (incomplete instrumentation)
- `docs/perf/baselines/phase1-instrumented.json` — With save-specific metrics (had timing bug)
- `docs/perf/baselines/phase1-fixed.json` — Final verified results

---

#### Next Steps

1. **Commit Phase 1 changes** — All files are tested and verified
2. **Update status table** — Mark Phase 1 as fully verified
3. **Begin Phase 2** — Ring buffer allocator (see `docs/perf/Phase-2-ring-allocator.md`)

---

#### Commands to Resume

```bash
# Check current state
git status
git diff

# Run benchmark to verify
just perf-baseline-train 'verification' 20 10

# Compare against Phase 1 baseline
just perf-diff docs/perf/baselines/phase1-before.json docs/perf/baselines/phase1-fixed.json

# View all baselines
just perf-list
```

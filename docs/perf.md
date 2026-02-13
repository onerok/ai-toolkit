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
just perf-train                      # 20 steps, save at 10
just perf-train steps=30 save_at=15  # Custom

# Capture baseline
just perf-baseline "phase-2-before"

# Compare against baseline
just perf-compare docs/perf/baselines/phase-2-before.json

# List saved baselines
just perf-list
```

### Direct Script Usage

```bash
# Run benchmark with options
uv run python scripts/perf_benchmark.py --steps 20 --resolution 256 --save-at 10

# Memory-only test (no actual training, just allocation patterns)
uv run python scripts/perf_benchmark.py --memory-only

# Capture baseline with notes
uv run python scripts/collect_baselines.py --tag "my-baseline" --notes "Before Phase 2"

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
| Baseline storage | Historical metrics | `docs/perf/baselines/<commit>.json` |

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
| `pcie_transfer_mb` | CPU↔GPU data movement | Phase 4, 5 |

### Workflow Per Phase

```
┌─────────────────────────────────────────────────────────────────┐
│  Before Phase N                                                 │
│  ─────────────────                                              │
│  1. git checkout <pre-phase-N-commit>                           │
│  2. uv run python scripts/collect_baselines.py \                │
│       --tag "phase-N-baseline" --output docs/perf/baselines/    │
│                                                                 │
│  Implement Phase N                                              │
│  ─────────────────                                              │
│  3. Make changes                                                │
│  4. Quick iteration: uv run python scripts/perf_benchmark.py    │
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

# Full benchmark with save cycle
uv run python scripts/perf_benchmark.py --steps 30 --save-at 15

# Memory-only test (no training, just allocation patterns)
uv run python scripts/perf_benchmark.py --memory-only

# Capture baseline with tag
uv run python scripts/collect_baselines.py --tag "main-baseline"

# Compare current vs baseline
uv run python scripts/compare_baselines.py \
    --baseline docs/perf/baselines/main-baseline.json \
    --current

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
| 1 (VRAM Leak) | ⬜ Need baseline | ✅ Done | ⬜ Pending |
| 2 (Ring Buffer) | ⬜ | ⬜ | ⬜ |
| 3 (Fused Backward) | ⬜ | ⬜ | ⬜ |
| 4 (Conductor) | ⬜ | ⬜ | ⬜ |
| 5 (Activation) | ⬜ | ⬜ | ⬜ |
| 6 (Accelerate) | ⬜ | ⬜ | ⬜ |

---

## Phase Overview

| Phase | Name | Impact | Risk | Effort | Dependencies |
|-------|------|--------|------|--------|--------------|
| **0** | Performance Test Harness ✅ | Foundation | None | 1 day | None |
| **1** | [VRAM Leak Fix](perf/Phase-1-vram-leak-fix.md) ✅ | Critical | Low | 1-2 days | Phase 0 |
| **2** | [Ring Buffer Allocator](perf/Phase-2-ring-allocator.md) | High | Low-Med | 3-4 days | Phase 1 |
| **3** | [Fused Backward](perf/Phase-3-fused-backward.md) | High | Medium | 3-5 days | Phase 1 |
| **4** | [Offload Conductor](perf/Phase-4-offload-conductor.md) | High | Medium | 1-2 weeks | Phase 2 |
| **5** | [Activation Offload](perf/Phase-5-activation-offload.md) | Medium | Medium | 1-2 weeks | Phase 2, 4 |
| **6** | [Accelerate Bypass](perf/Phase-6-accelerate-bypass.md) | Low | Low | 1-2 days | None |

---

## Dependency Graph

```
Phase 0 (Test harness) ✅ ────────────────────────────────────┐
         │                                                    │
         ▼                                                    │
Phase 1 (VRAM leak fix) ✅ ───────────────────────────────────┤
         │                                                    │
         ▼                                                    │
Phase 2 (Ring buffer) ─────────┬──────────────────────────────┤
         │                     │                              │
         │                     ▼                              │
         │             Phase 3 (Fused backward) ──────────────┤
         │                     │                              │
         ▼                     │                              │
Phase 4 (Conductor) ───────────┘                              │
         │                                                    │
         ▼                                                    │
Phase 5 (Activation offload) ─────────────────────────────────┘
                                                              │
Phase 6 (Accelerate bypass) ← Independent ────────────────────┘
```

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
  - **TODO:** Capture baseline and verify improvement

### Weeks 2-3: Foundation
- **Phase 2:** Static ring buffer allocator (parallel track A)
  - Pre-allocated GPU/CPU buffers
  - 16-byte aligned allocations
  - Integration with bouncing modules

- **Phase 3:** Fused backward pass (parallel track B)
  - `register_post_accumulate_grad_hook` integration
  - Per-parameter optimizer stepping
  - Immediate gradient release

### Weeks 4-5: Core Architecture
- **Phase 4:** Coordinated offload conductor
  - 3-stream architecture
  - Pre-computed offload strategy
  - Overlapped compute/transfer

### Weeks 6-7: Advanced Features
- **Phase 5:** Activation offloading
  - CPU pinned memory for activations
  - Replace recompute with transfer
  - Custom checkpoint integration

- **Phase 6:** Accelerate bypass
  - Single-GPU optimization
  - Direct `loss.backward()` calls

---

## Configuration Summary

```yaml
memory_management:
  # Phase 1: Always enabled (bug fix)

  # Phase 2: Ring buffer allocation
  use_ring_allocator: true
  ring_allocator_gpu_fraction: 0.25
  ring_allocator_cpu_fraction: 0.50

  # Phase 3: Fused backward
  fused_back_pass: false
  # Note: gradient_accumulation_steps=1 required for memory benefit

  # Phase 4: Coordinated offloading
  use_offload_conductor: false
  layer_offload_fraction: 0.5

  # Phase 5: Activation offloading
  activation_offload: false
  # Requires: use_offload_conductor: true

  # Phase 6: Accelerate bypass
  bypass_accelerate: null  # null = auto-detect
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

- **Scope:** Full plan — all 6 phases
- **Migration:** Keep existing bouncing as fallback, add conductor as opt-in
- **Quantization:** All phases must handle bitsandbytes/torchao quantized weights

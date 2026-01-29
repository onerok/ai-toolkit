# Feature: LoRA Training Visualization

**Status:** Proposed
**Priority:** Medium
**Complexity:** Medium
**Category:** UI / Training Observability

---

## Summary

Add real-time visualization of LoRA weight statistics during training, showing per-block learning progress, weight magnitudes, and impact analysis. This helps users understand *what* the model is learning and *where* in the network the changes are happening.

---

## Problem Statement

When training a LoRA, users currently see:
- Overall loss curves
- System metrics (VRAM, RAM, CPU)
- Periodic sample images

What they **don't** see:
- Which model blocks are learning the most
- Whether weight magnitudes are healthy or exploding/collapsing
- How learning is distributed across the network architecture

**Why this matters:**

1. **Debugging overfitting** — If certain blocks have disproportionately high weights, the LoRA may be memorizing rather than generalizing
2. **Understanding model behavior** — Different blocks handle different aspects (composition, style, fine details). Seeing where learning concentrates helps interpret results
3. **Early stopping decisions** — Weight saturation or instability visible before loss plateaus
4. **Architecture insights** — Compare block importance across FLUX, SDXL, WAN, etc.

**Inspiration:** [ShootTheSound-Realtime-Lora](https://github.com/ShootTheSound/comfyUI-Realtime-Lora) provides post-training LoRA analysis with block-level impact scoring and color-coded visualization. This feature brings similar insights to the *training* phase.

---

## Proposed Solution

### Core Concept

Log LoRA weight statistics per block at each logging step, then visualize in the existing Metrics tab. Three main visualizations:

| Visualization | Purpose | Location |
|---------------|---------|----------|
| **Weight Norms Graph** | Track per-block weight magnitudes over time | Metrics tab, collapsible section |
| **Block Impact Heatmap** | Show relative learning intensity across blocks | Metrics tab, collapsible section |
| **Current State Panel** | Snapshot of all block weights at current step | Metrics tab, collapsible section |

---

## Metrics to Log

### Per-Block Statistics (logged every N steps)

For each LoRA block/layer, compute and log:

```python
# For a LoRA with up/down weight matrices:
lora_up_norm = lora_up.float().norm().item()      # Frobenius norm
lora_down_norm = lora_down.float().norm().item()
combined_norm = lora_up_norm * lora_down_norm     # Effective magnitude

# Metric key format:
# lora/block_{name}/norm       -> combined_norm
# lora/block_{name}/up_norm    -> lora_up_norm (optional, detailed mode)
# lora/block_{name}/down_norm  -> lora_down_norm (optional, detailed mode)
```

### Block Naming Conventions

Map internal weight keys to human-readable block names:

| Architecture | Block Pattern | Display Name |
|--------------|---------------|--------------|
| **FLUX** | `double_blocks.{N}` | `double_{N}` |
| **FLUX** | `single_transformer_blocks.{N}` | `single_{N}` |
| **SDXL/SD15** | `down_blocks.{N}` | `unet_down_{N}` |
| **SDXL/SD15** | `up_blocks.{N}` | `unet_up_{N}` |
| **SDXL/SD15** | `mid_block` | `unet_mid` |
| **WAN** | `blocks.{N}` | `block_{N}` |
| **Z-Image** | `layers.{N}` | `layer_{N}` |

### Logging Frequency

- Default: Log every `log_every` steps (same as loss)
- Optional: Separate `lora_stats_every` config for finer/coarser control
- Trade-off: More frequent = larger log files, but better resolution

---

## Technical Architecture

### Backend Changes

#### 1. New Utility: `toolkit/lora_stats.py`

```python
"""LoRA weight statistics extraction."""

from typing import Dict, Optional
import torch
from collections import defaultdict


def extract_lora_block_stats(
    network: torch.nn.Module,
    architecture: str = "auto"
) -> Dict[str, Dict[str, float]]:
    """
    Extract per-block weight statistics from a LoRA network.

    Args:
        network: The LoRA network module (e.g., LoRANetwork from toolkit/lora_special.py)
        architecture: Model architecture hint for block naming

    Returns:
        Dict mapping block names to their statistics:
        {
            "double_0": {"norm": 0.0234, "up_norm": 0.12, "down_norm": 0.195},
            "double_1": {"norm": 0.0312, ...},
            ...
        }
    """
    ...


def detect_architecture_from_keys(keys: list[str]) -> str:
    """Detect model architecture from LoRA weight key patterns."""
    ...


def normalize_block_scores(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Normalize block norms to 0-100 scale for visualization.

    Returns dict mapping block_name -> score (0-100).
    """
    ...
```

#### 2. Integration Point: `jobs/process/BaseSDTrainProcess.py`

Add logging in the training loop (near line 2334):

```python
# Log LoRA weight statistics
if self.train_config.log_lora_stats and self.step_num % self.lora_stats_log_every == 0:
    if hasattr(self, 'network') and self.network is not None:
        lora_stats = extract_lora_block_stats(self.network)
        for block_name, stats in lora_stats.items():
            self.logger.log({
                f'lora/{block_name}/norm': stats['norm'],
            })
```

#### 3. Config Addition: `toolkit/config_modules.py`

```python
class LoggingConfig(BaseModel):
    # ... existing fields ...

    log_lora_stats: bool = True
    lora_stats_log_every: Optional[int] = None  # None = use log_every
```

### Frontend Changes

#### 1. New Component: `ui/src/components/LoraWeightGraph.tsx`

Multi-line chart showing weight norms over training steps:

```typescript
interface LoraWeightGraphProps {
  job: Job;
  defaultCollapsed?: boolean;
}

// Features:
// - Line per block (color-coded)
// - Toggle individual blocks on/off
// - Log/linear Y-axis option
// - Smoothing slider
// - Architecture-aware block grouping (double vs single for FLUX)
```

#### 2. New Component: `ui/src/components/LoraBlockHeatmap.tsx`

Grid heatmap showing relative block importance:

```typescript
interface LoraBlockHeatmapProps {
  job: Job;
  defaultCollapsed?: boolean;
}

// Features:
// - Grid layout matching architecture (e.g., 19 double + 38 single for FLUX)
// - Color gradient: blue (low) -> red (high)
// - Click block to highlight in weight graph
// - Animate over time (scrub through training)
```

#### 3. New Component: `ui/src/components/LoraBlockPanel.tsx`

Current state snapshot with per-block details:

```typescript
interface LoraBlockPanelProps {
  job: Job;
  defaultCollapsed?: boolean;
}

// Features:
// - List of all blocks with current norm value
// - Mini bar chart per block (relative to max)
// - Architecture presets: "Face Focus", "Style Only" (informational labels)
// - Export current state as JSON
```

#### 4. Update: `ui/src/components/JobMetricsPage.tsx`

Add new components to Metrics tab:

```typescript
export default function JobMetricsPage({ job }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <JobLossGraph job={job} />
      <JobSystemMetricsGraph job={job} defaultCollapsed={true} />
      <LoraWeightGraph job={job} defaultCollapsed={true} />
      <LoraBlockHeatmap job={job} defaultCollapsed={true} />
      <LoraBlockPanel job={job} defaultCollapsed={true} />
    </div>
  );
}
```

#### 5. New Hook: `ui/src/hooks/useLoraStats.tsx`

```typescript
interface UseLoraStatsResult {
  series: Record<string, MetricPoint[]>;
  blocks: string[];
  architecture: string;
  hasStats: boolean;
  status: 'loading' | 'success' | 'error';
  refreshStats: () => void;
}

export function useLoraStats(jobID: string, pollInterval: number = 2000): UseLoraStatsResult;
```

---

## User Interface

### Weight Norms Graph (Expanded)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LoRA Weight Norms                                            [▼ Collapse]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Smoothing: [====○=====] 50%     Y-Axis: [Linear ▼]     Stride: [1 ▼]     │
│                                                                             │
│  0.05 │                                                    ╭───────         │
│       │                                              ╭─────╯                │
│  0.04 │                                        ╭─────╯                      │
│       │                     ╭──────────────────╯                            │
│  0.03 │               ╭─────╯                                               │
│       │         ╭─────╯                                                     │
│  0.02 │   ╭─────╯                                                           │
│       │ ╭─╯                                                                 │
│  0.01 │╭╯                                                                   │
│       │                                                                     │
│  0.00 └─────────────────────────────────────────────────────────────────    │
│        0        500       1000      1500      2000      2500      3000      │
│                                    Step                                     │
│                                                                             │
│  ● double_0  ● double_1  ● double_2  ○ single_0  ○ single_1  [Show All]    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Block Impact Heatmap (Expanded)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Block Impact Heatmap                                         [▼ Collapse]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Architecture: FLUX.1-dev          Step: [=======○==] 2847 / 3000          │
│                                                                             │
│  Double Blocks (0-18):                                                      │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐        │
│  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │ 12 │        │
│  │░░░░│░░░░│▒▒▒▒│▒▒▒▒│▓▓▓▓│▓▓▓▓│████│████│████│▓▓▓▓│▓▓▓▓│▒▒▒▒│▒▒▒▒│        │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘        │
│  ┌────┬────┬────┬────┬────┬────┐                                            │
│  │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │                                            │
│  │▒▒▒▒│░░░░│░░░░│▒▒▒▒│░░░░│░░░░│                                            │
│  └────┴────┴────┴────┴────┴────┘                                            │
│                                                                             │
│  Single Blocks (0-37):                                                      │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐             │
│  │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│13│14│15│16│17│18│19│             │
│  │░░│░░│░░│▒▒│▒▒│▒▒│▓▓│██│██│▓▓│▓▓│▒▒│▒▒│░░│░░│░░│▒▒│▒▒│░░│░░│             │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘             │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                    │
│  │20│21│22│23│24│25│26│27│28│29│30│31│32│33│34│35│36│37│                    │
│  │░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│                    │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                    │
│                                                                             │
│  Legend: ░ Low (0-25%)  ▒ Medium (25-50%)  ▓ High (50-75%)  █ Very High    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current State Panel (Expanded)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LoRA Block Weights (Step 2847)                               [▼ Collapse]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sort: [By Impact ▼]     Filter: [All Blocks ▼]     [Export JSON]          │
│                                                                             │
│  Block             Norm     Impact   Bar                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  double_7          0.0487   100%     ████████████████████████████████████   │
│  double_8          0.0451    93%     ██████████████████████████████████░░   │
│  double_6          0.0423    87%     ████████████████████████████████░░░░   │
│  single_7          0.0398    82%     ██████████████████████████████░░░░░░   │
│  single_8          0.0371    76%     ████████████████████████████░░░░░░░░   │
│  double_9          0.0345    71%     ██████████████████████████░░░░░░░░░░   │
│  ...                                                                        │
│  single_37         0.0021     4%     ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  double_0          0.0018     4%     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│                                                                             │
│  Total Blocks: 57    Max Norm: 0.0487    Mean Norm: 0.0198                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Color Scheme

Use a consistent color gradient for impact visualization:

| Impact Range | Color | Hex |
|--------------|-------|-----|
| 0-10% | Deep Blue | `#0066ff` |
| 10-20% | Blue | `#0088ff` |
| 20-30% | Light Blue | `#00aaff` |
| 30-40% | Cyan | `#00cccc` |
| 40-50% | Teal | `#00cc66` |
| 50-60% | Yellow-Green | `#88cc00` |
| 60-70% | Yellow | `#cccc00` |
| 70-80% | Orange | `#ff9900` |
| 80-90% | Orange-Red | `#ff6600` |
| 90-100% | Red | `#ff3300` |

---

## Implementation Phases

### Phase 1: Backend Logging (MVP)
- [ ] Create `toolkit/lora_stats.py` with `extract_lora_block_stats()`
- [ ] Add architecture detection from weight keys
- [ ] Integrate logging into `BaseSDTrainProcess.py`
- [ ] Add `log_lora_stats` config option
- [ ] Test with FLUX, SDXL, WAN architectures

### Phase 2: Basic UI Visualization
- [ ] Create `useLoraStats` hook
- [ ] Create `LoraWeightGraph` component (multi-line chart)
- [ ] Add to `JobMetricsPage.tsx`
- [ ] Test real-time updates during training

### Phase 3: Advanced Visualizations
- [ ] Create `LoraBlockHeatmap` component
- [ ] Create `LoraBlockPanel` component
- [ ] Add step scrubber for heatmap animation
- [ ] Add JSON export functionality

### Phase 4: Polish
- [ ] Architecture-specific layout presets
- [ ] Block grouping and filtering
- [ ] Informational labels (Face Focus, Style, etc.)
- [ ] Documentation and tooltips

---

## Performance Considerations

### Logging Overhead
- Weight norm computation: O(n) per block, ~50-100 blocks typical
- Estimated overhead: <1ms per logging step
- Negligible compared to forward/backward pass

### Frontend Performance
- Limit displayed points with stride/windowing (existing pattern)
- Use `useMemo` for expensive computations
- Virtualize block list if >100 blocks

### Storage
- Additional ~500 bytes per step (57 blocks * ~9 chars per metric)
- At 3000 steps: ~1.5MB additional log data
- Acceptable given existing loss/system metrics logging

---

## API Endpoints

Uses existing `/api/jobs/{jobID}/loss` endpoint with new metric keys:

```
GET /api/jobs/{jobID}/loss?key=lora/double_0/norm&limit=1000
GET /api/jobs/{jobID}/loss?key=lora/*&limit=1000  (if wildcard supported)
```

May need new endpoint for efficient bulk retrieval:

```
GET /api/jobs/{jobID}/lora-stats?since_step=2000
```

Returns:
```json
{
  "architecture": "flux",
  "blocks": ["double_0", "double_1", ...],
  "points": [
    { "step": 2000, "values": { "double_0": 0.0234, "double_1": 0.0312, ... } },
    { "step": 2010, "values": { "double_0": 0.0241, "double_1": 0.0319, ... } },
    ...
  ]
}
```

---

## Success Metrics

- Users can identify which blocks are learning most within 10 seconds
- Weight explosion/collapse visible before loss curve indicates problems
- Feature used in >50% of training sessions (opt-out rather than opt-in)
- Positive feedback on understanding model behavior

---

## Out of Scope (Future Features)

These are intentionally excluded from this spec:

1. **Selective LoRA Loading** — Post-training feature to load specific blocks
2. **LoRA Editing** — Modify block weights after training
3. **Block Strength Presets** — Apply presets like "Face Focus" during inference
4. **Cross-Training Comparison** — Compare block weights across different runs

These could be separate features building on the visualization foundation.

---

## Open Questions

1. **Logging granularity**: Should we log up_norm/down_norm separately, or just combined norm? Combined is simpler but less informative.

2. **Architecture auto-detection**: Detect from model or require config specification?

3. **Historical comparison**: Show comparison to previous checkpoint? Adds complexity.

4. **Alert thresholds**: Auto-detect weight explosion (>10x growth) and show warning?

---

## References

- [ShootTheSound-Realtime-Lora](https://github.com/ShootTheSound/comfyUI-Realtime-Lora) — Block-level LoRA analysis for ComfyUI
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- Existing ai-toolkit visualization: `ui/src/components/JobLossGraph.tsx`, `JobSystemMetricsGraph.tsx`

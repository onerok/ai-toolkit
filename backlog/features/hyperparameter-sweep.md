# Feature: Hyperparameter Sweep / A/B Testing

**Status:** Proposed
**Priority:** High
**Complexity:** Medium-High
**Category:** Training Enhancement

---

## Summary

Enable users to define ranges of hyperparameters and automatically run multiple training experiments, generating comparison galleries and metrics tables to identify optimal settings.

---

## Problem Statement

Training a quality LoRA requires finding the right combination of hyperparameters:
- Network rank (4, 8, 16, 32, 64)
- Learning rate (1e-4, 5e-4, 1e-3)
- Training steps (500, 1000, 2000, 4000)
- Network alpha (1, rank/2, rank)
- Optimizer (AdamW8bit, Prodigy, AdaFactor)
- Batch size (1, 2, 4)

**Current workflow:**
1. Pick settings based on guides or intuition
2. Train for hours
3. Test results manually
4. Tweak config and repeat

This is time-consuming, doesn't systematically explore the parameter space, and makes it difficult to attribute quality differences to specific settings.

**User quotes from community:**
- "I've trained the same character 15 times trying different ranks"
- "Is rank 16 or 32 better for styles? No one agrees"
- "Wasted 3 days finding out my LR was too high"

---

## Proposed Solution

### Core Concept

A sweep configuration layer that wraps existing training configs, defining parameter ranges and evaluation criteria. The system orchestrates multiple training runs and produces comparison artifacts.

### Sweep Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Grid** | All combinations of all parameters | Small parameter spaces (< 20 runs) |
| **Random** | Random sampling from parameter ranges | Large spaces, time-limited exploration |
| **Bayesian** | Guided search using prior run results | Optimizing for specific metric |

---

## Configuration Schema

### YAML Config Example

```yaml
job: sweep
name: "character_lora_sweep_v1"

# Base training config - all runs inherit from this
base_config:
  job: extension
  model:
    name_or_path: "black-forest-labs/FLUX.1-dev"
    quantize: true

  network:
    type: lora
    # rank defined in sweep params

  train:
    batch_size: 1
    gradient_accumulation_steps: 1
    optimizer: adamw8bit
    # lr and steps defined in sweep params

  dataset:
    folder_path: "/path/to/dataset"
    caption_ext: ".txt"
    resolution: 1024

# Sweep-specific configuration
sweep:
  strategy: grid  # grid | random | bayesian

  # Parameters to sweep over
  parameters:
    network.rank:
      values: [8, 16, 32]

    train.lr:
      values: [1e-4, 5e-4, 1e-3]

    train.steps:
      values: [1000, 2000]

  # For random/bayesian strategies
  max_runs: 20  # Optional limit on total runs

  # Early stopping criteria (optional)
  early_stopping:
    enabled: true
    metric: loss
    patience: 200  # steps without improvement
    min_delta: 0.001  # minimum improvement threshold

  # Evaluation settings
  evaluation:
    # Prompts to generate for comparison
    prompts:
      - "photo of [trigger], professional headshot, studio lighting"
      - "photo of [trigger], full body, standing outdoors, natural lighting"
      - "painting of [trigger] in impressionist style"
      - "[trigger] as a cartoon character"

    # Seeds for reproducibility
    seeds: [42, 123, 456]

    # When to generate samples during training
    sample_at_steps: [250, 500, 1000, 1500, 2000]

    # Final evaluation at end of each run
    final_samples: 8

    # Inference settings for evaluation
    inference:
      guidance_scale: 3.5
      num_inference_steps: 28
      lora_strength: 1.0

# Output configuration
output:
  path: "output/sweeps/[name]"

  # What to save
  save_all_checkpoints: false  # Only save final checkpoint per run
  save_comparison_grid: true
  save_metrics_csv: true
  save_tensorboard: true

  # Cleanup
  delete_failed_runs: true
  keep_top_n: 3  # Keep only top N runs by final loss
```

### Parameter Range Types

```yaml
# Explicit values
network.rank:
  values: [8, 16, 32]

# Linear range
train.lr:
  min: 1e-5
  max: 1e-3
  num: 5  # 5 evenly spaced values

# Log scale range (for learning rates)
train.lr:
  min: 1e-5
  max: 1e-3
  num: 5
  scale: log

# Categorical
train.optimizer:
  values: ["adamw8bit", "prodigy", "adafactor"]
```

---

## Technical Architecture

### Directory Structure

```
toolkit/
  sweep/
    __init__.py
    config.py           # Pydantic models for sweep configuration
    runner.py           # SweepRunner orchestration class
    strategies/
      __init__.py
      grid.py           # Grid search implementation
      random.py         # Random sampling
      bayesian.py       # Bayesian optimization (optional v2)
    evaluation.py       # Sample generation and comparison
    metrics.py          # Metrics collection and aggregation

extensions_built_in/
  sweep_trainer/
    __init__.py
    SweepJob.py         # Job class for sweep execution

ui/src/
  components/
    sweep/
      SweepBuilder.tsx        # Parameter range configuration UI
      SweepDashboard.tsx      # Multi-run progress monitoring
      SweepComparison.tsx     # Side-by-side result gallery
      SweepMetricsTable.tsx   # Sortable metrics table
  app/
    sweeps/
      page.tsx                # Sweep list page
      [sweepId]/
        page.tsx              # Individual sweep detail page
```

### Core Classes

#### SweepConfig (Pydantic Model)

```python
from pydantic import BaseModel
from typing import Literal, Optional
from enum import Enum

class SweepStrategy(str, Enum):
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"

class ParameterRange(BaseModel):
    values: Optional[list] = None
    min: Optional[float] = None
    max: Optional[float] = None
    num: Optional[int] = None
    scale: Literal["linear", "log"] = "linear"

class EarlyStoppingConfig(BaseModel):
    enabled: bool = False
    metric: str = "loss"
    patience: int = 200
    min_delta: float = 0.001

class EvaluationConfig(BaseModel):
    prompts: list[str]
    seeds: list[int] = [42]
    sample_at_steps: list[int] = []
    final_samples: int = 4
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    lora_strength: float = 1.0

class SweepConfig(BaseModel):
    strategy: SweepStrategy = SweepStrategy.GRID
    parameters: dict[str, ParameterRange]
    max_runs: Optional[int] = None
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    evaluation: EvaluationConfig
```

#### SweepRunner

```python
class SweepRunner:
    """Orchestrates multiple training runs for hyperparameter sweeps."""

    def __init__(self, config: SweepConfig, base_config: dict):
        self.config = config
        self.base_config = base_config
        self.runs: list[SweepRun] = []
        self.strategy = self._get_strategy()

    def _get_strategy(self) -> SweepStrategy:
        """Returns the appropriate strategy implementation."""
        ...

    def generate_run_configs(self) -> list[dict]:
        """Generate all run configurations based on strategy."""
        ...

    def run(self) -> SweepResult:
        """Execute the sweep."""
        # 1. Generate run configurations
        # 2. Load base model once (shared across runs)
        # 3. For each run config:
        #    a. Apply parameter overrides
        #    b. Execute training
        #    c. Generate evaluation samples
        #    d. Collect metrics
        #    e. Check early stopping
        # 4. Generate comparison artifacts
        # 5. Return aggregated results
        ...

    def generate_comparison_grid(self) -> Path:
        """Generate side-by-side comparison image grid."""
        ...

    def export_metrics(self) -> Path:
        """Export metrics to CSV."""
        ...
```

---

## User Interface

### 1. Sweep Builder Page

**URL:** `/sweeps/new`

```
┌─────────────────────────────────────────────────────────────────┐
│  Create New Sweep                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base Config: [Select existing config ▼] or [Upload YAML]      │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  PARAMETERS TO SWEEP                                            │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  ┌─────────────┬──────────────────────────────────────────┐    │
│  │ Rank        │ ☑ 8   ☑ 16   ☑ 32   ☐ 64                │    │
│  ├─────────────┼──────────────────────────────────────────┤    │
│  │ Learning    │ ☑ 1e-4   ☑ 5e-4   ☐ 1e-3                │    │
│  │ Rate        │                                          │    │
│  ├─────────────┼──────────────────────────────────────────┤    │
│  │ Steps       │ ☑ 1000   ☑ 2000   ☐ 4000                │    │
│  └─────────────┴──────────────────────────────────────────┘    │
│                                                                 │
│  [+ Add Parameter]                                              │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  SWEEP SUMMARY                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Strategy: [Grid ▼]                                             │
│  Total Runs: 12  (3 ranks × 2 LRs × 2 steps)                   │
│  Est. Time: ~6 hours (based on 30 min/run)                     │
│  Est. VRAM: 22 GB peak                                         │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  EVALUATION PROMPTS                                             │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  1. [photo of [trigger], headshot________________]  [×]        │
│  2. [photo of [trigger], full body_______________]  [×]        │
│  [+ Add Prompt]                                                 │
│                                                                 │
│                              [Cancel]  [Start Sweep]            │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Sweep Dashboard

**URL:** `/sweeps/[sweepId]`

```
┌─────────────────────────────────────────────────────────────────┐
│  Sweep: character_lora_sweep_v1                    [⏸ Pause]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Progress: ████████████░░░░░░░░░░░░░░░░░  8/12 runs (67%)     │
│  Elapsed: 4h 12m  |  Remaining: ~2h 06m                        │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  RUN STATUS                                                     │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  │ Run │ Rank │ LR    │ Steps │ Status      │ Loss   │ Time  │ │
│  ├─────┼──────┼───────┼───────┼─────────────┼────────┼───────┤ │
│  │ 1   │ 8    │ 1e-4  │ 1000  │ ✓ Complete  │ 0.042  │ 28m   │ │
│  │ 2   │ 8    │ 1e-4  │ 2000  │ ✓ Complete  │ 0.038  │ 52m   │ │
│  │ 3   │ 8    │ 5e-4  │ 1000  │ ✓ Complete  │ 0.051  │ 27m   │ │
│  │ 4   │ 8    │ 5e-4  │ 2000  │ ✓ Complete  │ 0.044  │ 51m   │ │
│  │ 5   │ 16   │ 1e-4  │ 1000  │ ✓ Complete  │ 0.039  │ 31m   │ │
│  │ 6   │ 16   │ 1e-4  │ 2000  │ ✓ Complete  │ 0.035  │ 58m   │ │
│  │ 7   │ 16   │ 5e-4  │ 1000  │ ✓ Complete  │ 0.048  │ 30m   │ │
│  │ 8   │ 16   │ 5e-4  │ 2000  │ ▶ Running   │ 0.041  │ 34m   │ │
│  │ 9   │ 32   │ 1e-4  │ 1000  │ ○ Pending   │ -      │ -     │ │
│  │ 10  │ 32   │ 1e-4  │ 2000  │ ○ Pending   │ -      │ -     │ │
│  │ 11  │ 32   │ 5e-4  │ 1000  │ ○ Pending   │ -      │ -     │ │
│  │ 12  │ 32   │ 5e-4  │ 2000  │ ○ Pending   │ -      │ -     │ │
│                                                                 │
│  [View Comparison Gallery]  [Export Metrics CSV]                │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Comparison Gallery

**URL:** `/sweeps/[sweepId]/compare`

```
┌─────────────────────────────────────────────────────────────────┐
│  Comparison: character_lora_sweep_v1                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prompt: [photo of [trigger], headshot ▼]  Seed: [42 ▼]        │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│              │  LR = 1e-4      │  LR = 5e-4      │              │
│  ────────────┼─────────────────┼─────────────────┤              │
│              │                 │                 │              │
│  Rank = 8    │   [  IMG  ]     │   [  IMG  ]     │              │
│              │   loss: 0.042   │   loss: 0.051   │              │
│              │                 │                 │              │
│  ────────────┼─────────────────┼─────────────────┤              │
│              │                 │                 │              │
│  Rank = 16   │   [  IMG  ]     │   [  IMG  ]     │              │
│              │   loss: 0.035 ★ │   loss: 0.048   │              │
│              │                 │                 │              │
│  ────────────┼─────────────────┼─────────────────┤              │
│              │                 │                 │              │
│  Rank = 32   │   [  IMG  ]     │   [  IMG  ]     │              │
│              │   loss: 0.033   │   loss: 0.046   │              │
│              │                 │                 │              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  ★ = Best by loss   [Sort by: Loss ▼]                          │
│                                                                 │
│  Selected: Run 6 (Rank 16, LR 1e-4, 2000 steps)                │
│  [Promote to Final] [Download LoRA] [View Training Logs]        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Output Artifacts

### Directory Structure

```
output/sweeps/character_lora_sweep_v1/
├── sweep_config.yaml           # Full sweep configuration
├── sweep_results.json          # Aggregated results
├── metrics.csv                 # All runs metrics
├── comparison_grid.png         # Visual comparison
├── tensorboard/                # TensorBoard logs for all runs
│   ├── run_001/
│   ├── run_002/
│   └── ...
├── runs/
│   ├── run_001_rank8_lr1e-4_steps1000/
│   │   ├── config.yaml
│   │   ├── model.safetensors
│   │   ├── training_log.txt
│   │   └── samples/
│   │       ├── step_0500/
│   │       ├── step_1000/
│   │       └── final/
│   ├── run_002_rank8_lr1e-4_steps2000/
│   └── ...
└── best/
    └── model.safetensors       # Symlink to best run
```

### metrics.csv

```csv
run_id,rank,lr,steps,optimizer,final_loss,min_loss,vram_peak_gb,training_time_sec,file_size_mb
run_001,8,1e-4,1000,adamw8bit,0.042,0.041,18.2,1680,24.1
run_002,8,1e-4,2000,adamw8bit,0.038,0.036,18.2,3120,24.1
run_003,8,5e-4,1000,adamw8bit,0.051,0.049,18.2,1650,24.1
...
```

### sweep_results.json

```json
{
  "sweep_id": "character_lora_sweep_v1",
  "strategy": "grid",
  "started_at": "2025-01-28T10:00:00Z",
  "completed_at": "2025-01-28T16:30:00Z",
  "total_runs": 12,
  "completed_runs": 12,
  "failed_runs": 0,
  "best_run": {
    "run_id": "run_006",
    "parameters": {
      "rank": 16,
      "lr": 1e-4,
      "steps": 2000
    },
    "metrics": {
      "final_loss": 0.035,
      "vram_peak_gb": 19.1,
      "training_time_sec": 3480
    }
  },
  "parameter_insights": {
    "rank": {
      "best_value": 16,
      "trend": "diminishing_returns_above_16"
    },
    "lr": {
      "best_value": 1e-4,
      "trend": "lower_is_better_for_this_dataset"
    }
  }
}
```

---

## Implementation Phases

### Phase 1: Core Sweep Engine (MVP)
- [ ] `SweepConfig` Pydantic model
- [ ] Grid search strategy
- [ ] `SweepRunner` orchestration
- [ ] CLI support (`uv run python run.py sweep_config.yaml`)
- [ ] Basic metrics collection (loss, time, VRAM)
- [ ] Metrics CSV export
- [ ] Sample generation at completion

### Phase 2: UI Integration
- [ ] Sweep Builder page
- [ ] Sweep Dashboard with live progress
- [ ] Comparison Gallery
- [ ] Metrics table with sorting

### Phase 3: Advanced Features
- [ ] Random sampling strategy
- [ ] Early stopping
- [ ] Multi-GPU parallel runs
- [ ] Bayesian optimization
- [ ] Parameter importance analysis
- [ ] Auto-generated insights

---

## Performance Considerations

### Memory Optimization
- Load base model once, reuse across runs
- Unload LoRA weights between runs, keep base model
- Share dataset loading across runs

### Time Optimization
- Parallel runs on multi-GPU setups
- Skip redundant evaluation if loss plateaus
- Cache text encoder outputs for evaluation prompts

### Checkpointing
- Save sweep state after each run completes
- Resume interrupted sweeps from last completed run
- Option to re-run specific failed runs

---

## API Endpoints (for UI)

```
POST   /api/sweeps              # Create new sweep
GET    /api/sweeps              # List all sweeps
GET    /api/sweeps/:id          # Get sweep details
POST   /api/sweeps/:id/pause    # Pause sweep
POST   /api/sweeps/:id/resume   # Resume sweep
POST   /api/sweeps/:id/cancel   # Cancel sweep
GET    /api/sweeps/:id/runs     # Get all runs for sweep
GET    /api/sweeps/:id/compare  # Get comparison data
GET    /api/sweeps/:id/metrics  # Download metrics CSV
POST   /api/sweeps/:id/promote  # Promote run to final output
```

---

## Success Metrics

- Users can define and run a sweep in under 5 minutes
- Comparison gallery helps users identify best settings visually
- 50% reduction in time to find optimal hyperparameters
- Feature generates shareable content for social media (comparison grids)

---

## Open Questions

1. **Multi-GPU parallelism**: Run multiple experiments in parallel on multi-GPU setups? Complexity vs. value tradeoff.

2. **Bayesian optimization**: Worth implementing in v1 or defer? Requires a metric to optimize against.

3. **Cross-model sweeps**: Allow sweeping across base models (Flux vs SDXL)? Probably out of scope.

4. **Cloud integration**: Tie into cloud GPU provisioning for large sweeps? Defer to separate feature.

---

## References

- [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) - Inspiration for sweep configuration
- [Optuna](https://optuna.org/) - Bayesian optimization library (potential integration)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) - Distributed hyperparameter tuning

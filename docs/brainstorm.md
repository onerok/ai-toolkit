# UX Brainstorm: Guided LoRA Training Experience

Based on the methodology from [flux2_klein.md](./flux2_klein.md), this document explores UX improvements that could bridge the gap between expert knowledge and beginner users.

Core insight from the source document: **"measure don't guess, change one variable at a time, and stop training when the loss curve tells you to."**

Most LoRA training UX today throws users into a sea of hyperparameters with no feedback loop. These features aim to change that.

---

## 1. Training Session: Multi-Job Experiment Container

The biggest architectural addition. A **Session** encapsulates multiple training runs, a shared dataset, and comparison tools.

```
Session: "Character LoRA - Alice"
├── Dataset (defined once)
│   ├── Training images (20)
│   └── Validation images (5) ← automatically held out
├── Baseline Run (rank 16, LR 1e-4, decay 0.00001)
│   ├── checkpoint-500  (val_loss: 0.42)
│   ├── checkpoint-1000 (val_loss: 0.38)
│   ├── checkpoint-1500 (val_loss: 0.35) ★ minimum
│   └── checkpoint-2000 (val_loss: 0.37) ↑ rising
├── Experiment 2: Higher rank (rank 32)
│   └── ...
├── Experiment 3: Lower LR (5e-5)
│   └── ...
└── Comparison Dashboard
    ├── Loss curves overlaid
    ├── Side-by-side inference samples (same seed)
    └── Recommendation: "Baseline, checkpoint-1500"
```

**Why this matters:** The source document emphasizes changing one variable at a time. A Session makes this natural—clone a run, change one param, compare.

### Session Features

- **Dataset inheritance**: Define training/validation split once, reuse across experiments
- **Run cloning**: "Clone this run and change X" workflow
- **Cross-run comparison**: Overlay loss curves, compare samples at same step counts
- **Best checkpoint tracking**: Automatic recommendation based on validation loss

---

## 2. Validation Loss Dashboard (The Key Feature)

The source document's core methodology is **deterministic validation loss**. The UI should make this first-class:

- **Real-time loss curve** with training vs validation clearly separated
- **Automatic minimum detection** with visual marker ("Stop here")
- **Sample grid at each checkpoint** (same prompts, same seeds across all checkpoints)
- **Overfitting warning** when validation loss starts rising while training loss drops

```
┌─────────────────────────────────────────────────────────────┐
│  Loss Curve                                     [Pause]     │
│  ──────────────────────────────────────────────────────     │
│  0.5 │                                                      │
│      │ ●─●                                                  │
│  0.4 │     ●─●         ← Training loss                      │
│      │         ●─●─●─●─●─●                                  │
│  0.3 │   ○─○                                                │
│      │       ○─○   ★ ← Validation minimum (step 1500)       │
│  0.2 │           ○   ○─○ ← Rising = overfitting             │
│      └──────────────────────────────────────────────────    │
│         500  1000  1500  2000  2500  3000                   │
│                                                             │
│  ⚠️  Validation loss rising since step 1500. Consider       │
│     stopping or using checkpoint-1500.                      │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Notes

- Validation requires held-out images with fixed seeds, timesteps, and noise
- Evaluation frequency: every 100-200 steps (configurable)
- Store validation metrics in job metadata for cross-run comparison

---

## 3. Guided Setup Wizard

Ask 3-4 questions, set 15+ hyperparameters correctly:

### Q1: What are you training?

| Type | Recommended Settings |
|------|---------------------|
| Character/Face | rank 16, 15-30 images, high-likeness tips |
| Style | rank 32, 50-200 images, natural language captions emphasized |
| Object/Concept | rank 16-32, 20-40 images |

### Q2: How many training images?

- Shows warning if <15: "Below 15 images you're memorizing, not learning"
- Calculates recommended step range based on dataset size
- Auto-suggests validation split (e.g., "Hold out 4 of your 24 images")

### Q3: What GPU/VRAM?

| VRAM | Configuration |
|------|--------------|
| 16GB | Aggressive caching, FP8, rank ≤16, warning about limitations |
| 24GB | Standard caching, gradient checkpointing, rank 16-32 |
| 40GB+ | Optional caching, batch size options, rank 32-64 |

**Output:** Pre-populated config with explanations for each choice.

---

## 4. Hyperparameter Guidance Panel

Contextual help for every parameter, drawing from the source document:

```
┌─────────────────────────────────────────────────────────────┐
│  Weight Decay: [0.0001    ▼]                                │
│  ─────────────────────────────────────────────────────────  │
│  ⚠️  Current value may be too high                          │
│                                                             │
│  Recommended: 0.00001                                       │
│                                                             │
│  Why it matters:                                            │
│  Weight decay had a larger impact on output quality than    │
│  learning rate in controlled experiments. At 0.0001+,       │
│  you'll see crushed shadows and contrast spikes. At         │
│  0.00001, you get clean tonal separation.                   │
│                                                             │
│  [Apply Recommended]                                        │
└─────────────────────────────────────────────────────────────┘
```

### Parameters to Document

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| Learning Rate | 1e-4 | Klein is LR-sensitive; changes of "five thousandths of a percent" can destroy quality |
| Weight Decay | 0.00001 | Larger impact than LR; controls tonal response |
| LoRA Rank | 16-32 | 16 for simple concepts, 32 for complex; >64 rarely helps |
| Alpha | rank/2 | Decouples rank from effective LR |
| Network Dims | 128/64/64/32 | 4:2:2:1 ratio outperformed all other combinations tested |

---

## 5. Pre-flight Checks

Before training starts, validate everything:

```
┌─────────────────────────────────────────────────────────────┐
│  Pre-flight Check                               [Start →]   │
│  ─────────────────────────────────────────────────────────  │
│  ✅ Model: Klein 4B Base (correct for training)             │
│  ✅ VRAM estimate: ~21GB (fits your 24GB GPU)               │
│  ⚠️  Dataset: 12 images (minimum is 15, consider adding)    │
│  ⚠️  Captions: Tag-style detected in 8/12 images            │
│     → Klein uses Qwen3-4B, natural language works better    │
│  ✅ Validation split: 3 images held out                     │
│  ✅ Weight decay: 0.00001 (optimal)                         │
│  ✅ LR: 1e-4 (recommended starting point)                   │
│  ❌ Evaluation steps: 4 (should be 50 for Base model)       │
│     → [Fix: Set to 50]                                      │
└─────────────────────────────────────────────────────────────┘
```

### Checks to Implement

- **Model selection**: Warn if training on distilled model
- **VRAM estimation**: Calculate based on model + caching + gradient checkpointing settings
- **Dataset size**: Warn if below minimum for training type
- **Caption quality**: Detect tag-style vs natural language
- **Validation split**: Ensure held-out images exist
- **Hyperparameter sanity**: Flag values outside recommended ranges
- **Inference settings mismatch**: Catch Base model with distilled eval settings

---

## 6. Checkpoint Comparison View

Within a session, compare checkpoints side-by-side:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Checkpoint Comparison (seed: 42, prompt: "alice in a garden")         │
│  ──────────────────────────────────────────────────────────────────    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  step    │  │  step    │  │  step    │  │  step    │               │
│  │  500     │  │  1000    │  │  1500 ★  │  │  2000    │               │
│  │  [img]   │  │  [img]   │  │  [img]   │  │  [img]   │               │
│  │          │  │          │  │          │  │          │               │
│  │ val: 0.42│  │ val: 0.38│  │ val: 0.35│  │ val: 0.37│               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
│                              ↑ Best                                    │
│  Recommendation: Use checkpoint-1500 (lowest validation loss)          │
│  [Export checkpoint-1500]                                              │
└────────────────────────────────────────────────────────────────────────┘
```

### Features

- **Deterministic sampling**: Same seed, same prompt across all checkpoints
- **Validation loss overlay**: Show metric alongside visual
- **A/B comparison mode**: Select two checkpoints for detailed comparison
- **Export best**: One-click export of recommended checkpoint

---

## 7. Live Training Coach

Contextual notifications during training:

| Condition | Message |
|-----------|---------|
| Loss spike detected | "This often indicates LR is too high. Consider pausing and reducing to 5e-5." |
| Validation loss rising | "You may be overfitting. Step 1500 was your minimum—consider stopping." |
| High repeat count | "3000 steps with 15 images means each image seen ~200 times. Check samples for memorization." |
| Stable training | "Looking good! Validation loss still dropping." |
| Plateau detected | "Loss hasn't improved in 500 steps. Consider stopping or adjusting LR." |

### Implementation Notes

- Non-intrusive notifications (dismissible)
- Link to relevant documentation/explanation
- Optional "auto-pause on overfitting" setting

---

## 8. Dataset Quality Analyzer

Before training, analyze the dataset:

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset Analysis                                           │
│  ─────────────────────────────────────────────────────────  │
│  Images: 28 total                                           │
│  ─────────────────────────────────────────────────────────  │
│  ⚠️  2 near-duplicates detected (IMG_023.jpg, IMG_024.jpg)  │
│     → Duplicates bias training and accelerate overfitting   │
│                                                             │
│  Caption Quality:                                           │
│  ├── Natural language: 12 (43%) ✅                          │
│  ├── Tag-style: 14 (50%) ⚠️                                 │
│  └── Missing: 2 (7%) ❌                                     │
│                                                             │
│  Recommendation: Convert tag captions to natural language.  │
│  Klein's Qwen3-4B text encoder understands full sentences.  │
│  [Auto-expand tags to sentences]                            │
│                                                             │
│  Suggested validation split: Hold out 4 images              │
│  [Apply split]                                              │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Features

- **Duplicate detection**: Perceptual hashing to find near-duplicates
- **Caption classification**: Detect tag-style vs natural language
- **Missing caption detection**: Flag images without captions
- **Resolution analysis**: Warn about low-res or inconsistent sizes
- **Validation split suggestion**: Based on dataset size and type
- **Tag-to-sentence conversion**: Use VLM to expand tags into natural language

---

## 9. Smart Presets with Tradeoff Explanations

Not just "24GB preset" but presets with context:

```
┌─────────────────────────────────────────────────────────────┐
│  Choose a Starting Point                                    │
│  ─────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Character LoRA (24GB GPU)                    [Use]  │    │
│  │ For: Faces, characters, specific people             │    │
│  │ Settings: rank 16, LR 1e-4, ~2000-4000 steps        │    │
│  │ Why: Lower rank captures likeness without           │    │
│  │      overfitting style/pose flexibility             │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Style LoRA (24GB GPU)                        [Use]  │    │
│  │ For: Art styles, aesthetic treatments               │    │
│  │ Settings: rank 32, LR 1e-4, ~4000-6000 steps        │    │
│  │ Why: Styles need higher rank—"style" is a broader   │    │
│  │      concept than "this specific face"              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Preset Categories

| Preset | Use Case | Key Settings |
|--------|----------|--------------|
| Character (24GB) | Faces, people | rank 16, 2000-4000 steps |
| Style (24GB) | Art styles | rank 32, 4000-6000 steps |
| Object (24GB) | Products, items | rank 16-32, 2000-4000 steps |
| Memory-constrained (16GB) | Any, limited hardware | rank 16, FP8, aggressive caching |
| Quality-focused (40GB+) | Best results | rank 32, 4:2:2:1 dims, batch 2 |

---

## 10. Post-Training Report

When training completes:

```
┌─────────────────────────────────────────────────────────────┐
│  Training Complete: alice_lora_v1                           │
│  ─────────────────────────────────────────────────────────  │
│  Best Checkpoint: step-1500 (validation loss: 0.35)         │
│                                                             │
│  Training Summary:                                          │
│  • Ran 2500 steps, optimal was step 1500                    │
│  • Each image seen ~125 times (within normal range)         │
│  • No loss spikes detected                                  │
│                                                             │
│  Recommended Inference Settings:                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Steps: 50        CFG: 4.0        Strength: 0.73     │    │
│  │ (Copy these to ComfyUI/Diffusers)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ⚠️  Using 4 steps or CFG 1.0 will produce garbage—you     │
│     trained on Base, not distilled.                         │
│                                                             │
│  [Export LoRA]  [Compare in Session]  [Start New Experiment]│
└─────────────────────────────────────────────────────────────┘
```

### Report Contents

- Best checkpoint identification with validation loss
- Training dynamics summary (spikes, plateaus, overfitting point)
- Repeat count analysis
- Recommended inference settings for the trained model
- Common pitfalls warning (e.g., Base vs distilled settings)
- Export options and next-step suggestions

---

## Implementation Priority

If prioritizing for a first release:

| Priority | Feature | Impact |
|----------|---------|--------|
| 1 | Validation Loss Dashboard | The single most impactful feature |
| 2 | Pre-flight Checks | Catches common mistakes before wasting GPU time |
| 3 | Guided Setup Wizard | Lowers barrier to entry dramatically |
| 4 | Session/Experiment Container | Enables the "change one variable" methodology |
| 5 | Hyperparameter Guidance Panel | Teaches while configuring |

The rest are valuable but these five would transform the experience from "hope this works" to "I understand what's happening and can iterate intelligently."

---

## Data Model Sketch

### Session

```typescript
interface TrainingSession {
  id: string;
  name: string;
  createdAt: Date;

  // Dataset (shared across runs)
  dataset: {
    trainingImages: DatasetImage[];
    validationImages: DatasetImage[];  // held out
  };

  // Runs within this session
  runs: TrainingRun[];

  // Comparison state
  comparison?: {
    selectedRuns: string[];
    comparisonPrompts: string[];
    comparisonSeed: number;
  };
}

interface TrainingRun {
  id: string;
  sessionId: string;
  name: string;
  config: TrainingConfig;

  // Checkpoints with metrics
  checkpoints: Checkpoint[];

  // Validation metrics over time
  validationHistory: ValidationPoint[];

  // Best checkpoint (auto-detected)
  bestCheckpoint?: string;

  status: 'pending' | 'running' | 'completed' | 'failed';
}

interface Checkpoint {
  step: number;
  path: string;
  trainingLoss: number;
  validationLoss?: number;
  sampleImages?: string[];  // paths to generated samples
}

interface ValidationPoint {
  step: number;
  loss: number;
  timestamp: Date;
}
```

---

## Next Steps

1. Review with team for feasibility and prioritization
2. Design detailed UI mockups for top 3 features
3. Identify backend changes needed (validation loss computation, session storage)
4. Prototype validation loss dashboard with existing training infrastructure

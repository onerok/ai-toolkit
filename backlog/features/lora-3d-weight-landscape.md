# Feature: 3D LoRA Weight Landscape

**Status:** Prototype Complete
**Priority:** Medium
**Complexity:** Medium-High
**Category:** UI / Training Observability / Post-Training Analysis
**Prototype:** `tmp/lora-viz-prototype/`

---

## Summary

An interactive 3D terrain visualization of LoRA weight statistics across blocks and training time, built with React Three Fiber. Shows per-block weight norms and stable rank as a heightmap landscape, plus individual weight matrix surfaces. Produces actionable insights that map directly to the `network` YAML training config (`block_dims`, `block_alphas`, `only_if_contains`, `ignore_if_contains`).

Companion to the 2D [LoRA Training Visualization](./lora-visualization.md) feature — this adds a 3D perspective for post-training and cross-checkpoint analysis.

---

## Problem Statement

After training a LoRA, users have no way to answer:

- **"Am I wasting rank?"** — If a rank-16 LoRA only uses 1.5 effective dimensions in some blocks, the other 14.5 are noise. This noise degrades output quality at higher LoRA strengths (1.2+) and bloats file size.
- **"Which blocks actually learned something?"** — Different blocks handle different aspects (composition, style, detail). Knowing where learning concentrated helps interpret results and guides per-block config.
- **"Should I change my `network` config?"** — Users set `linear: 16` globally but have no data to inform per-block rank allocation via `block_dims`, or block exclusion via `ignore_if_contains`.

### Why 3D?

2D line charts (per the companion feature) show individual block trajectories well, but struggle to show the *relative shape* across 32 blocks simultaneously. The 3D terrain encodes blocks on X, time on Z, and magnitude on Y — the whole training story in one view. Patterns like "double blocks plateau while single blocks keep growing" or "stable rank collapses uniformly" become immediately visible as terrain features.

---

## What the Prototype Shows

### View 1: Block Landscape

A terrain surface where:
- **X axis** = block index (D0-D7 double blocks, S0-S23 single blocks for Flux)
- **Z axis** = training step (front = early, back = late)
- **Y axis (height)** = metric value
- **Color** = metric-appropriate colormap

Two metrics, toggled in the UI:

| Metric | What it measures | Colormap | Actionable insight |
|--------|-----------------|----------|-------------------|
| **Weight Norm** | `mean(||B_i||_F * ||A_i||_F)` per block — effective magnitude of the LoRA update | Inferno (black → purple → red → yellow) | Identifies which blocks learned the most; detects explosion or dead blocks |
| **Stable Rank** | `||BA||_F² / σ_max²` — how many of r dimensions carry real information (1 = concentrated, r = uniform) | Viridis (purple → teal → yellow) | Reveals rank waste; guides `block_dims` configuration |

A cyan time-cursor plane slices through the terrain at the current step. A divider separates double and single block groups.

### View 2: Weight Matrix Surface

A heightmap of a single weight matrix (e.g., `double_blocks.4.img_attn.proj.lora_B`):
- **X** = rank dimension (16 columns)
- **Z** = feature dimension (128 rows, downsampled)
- **Y (height)** = raw weight value
- **Color** = diverging blue → white → red (signed values)

Morphs smoothly between training step snapshots via lerp interpolation.

### Controls

- Orbit (drag), zoom (scroll)
- Timeline slider with play/pause
- View toggle: Block Landscape / Weight Matrix
- Metric toggle: Weight Norm / Stable Rank (landscape only)
- Live stats: min/mean/max at current step

---

## Key Findings from Prototype Data

Tested on a Flux LoRA (`f2k_retrousse1`, rank 16, 5000 steps, 20 checkpoints):

### Weight Norms
- All blocks grew 7-8x from step 250 to 5000 — steady, healthy training
- Double blocks and single blocks grew at similar rates
- No explosion or collapse detected

### Stable Rank (the interesting finding)
- **Double blocks (D0-D7)**: stable rank dropped from ~3-4 at step 250 to ~1.4-1.9 by step 5000
- **Late single blocks (S16-S23)**: maintained high stable rank throughout training

This means the double blocks converged to using only ~1.5 out of 16 available dimensions — the other 14+ are noise. The single blocks are genuinely using more of their capacity. This is consistent with Hu et al. (2021) who found that "even at rank 64, nearly all meaningful information concentrates in the top 1 singular vector."

---

## Connection to `network` YAML Config

The visualization directly informs these training config options:

### `block_dims` / `block_alphas` — Per-Block Rank

```yaml
network:
  type: "lora"
  linear: 16          # default rank
  linear_alpha: 16
  network_kwargs:
    # Informed by stable rank analysis:
    # Double blocks only use ~1.5 dimensions → rank 4 is sufficient
    # Single blocks use more → keep rank 16
    block_dims: "4,4,4,4,4,4,4,4,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16"
    block_alphas: "4,4,4,4,4,4,4,4,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16"
```

**When the visualization helps:** Stable rank terrain shows which blocks are wasting capacity. If a block's stable rank sits at ~1 throughout training, lowering its rank removes noise without losing signal. Results in smaller LoRA files and potentially cleaner output at higher inference strengths.

### `ignore_if_contains` / `only_if_contains` — Block Targeting

```yaml
network_kwargs:
  ignore_if_contains:
    - "single_blocks.20"    # skip blocks with near-zero norms
    - "single_blocks.21"
  # or
  only_if_contains:
    - "double_blocks"       # only train double blocks
```

**When the visualization helps:** Weight norm terrain reveals blocks with negligible learning. Excluding them reduces training time and LoRA size with no quality impact.

### LBW Inference Weights (ComfyUI)

The block norm landscape maps conceptually to ComfyUI's LoRA Block Weight (LBW) syntax:

```
LBW = 1.0, 1.0, 0.8, 0.8, 0.6, ...
```

Where each value scales a block's contribution at inference. The key difference:
- **LBW scales signal and noise together** — it's a volume knob
- **`block_dims` eliminates noise at the source** — it's choosing how many microphones to use

The visualization helps with both: norm data informs LBW tuning, stable rank data informs `block_dims` decisions.

---

## Technical Architecture

### Data Pipeline

```
safetensors checkpoints
    → extract_lora_data.py (Python, offline)
        → Pairs lora_A/lora_B per layer
        → Computes per-layer ||B||_F * ||A||_F (norm product)
        → Computes stable rank via efficient r×r SVD trick
        → Aggregates per block (mean across layers within block)
    → block_stats.json + matrix_snapshots.json
        → Vite dev server / Next.js API
            → React Three Fiber components
```

### Efficient Stable Rank Computation

Computing `stable_rank(B @ A)` without forming the full product (which can be [24576 x 12288] = 1.2GB):

```python
# B: [m, r], A: [r, n] where r=16 is small
# SVD of B (m×r): O(m*r²) — fast since r is tiny
U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)
U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)

# r×r core matrix whose singular values = those of B@A
M = diag(S_B) @ Vh_B @ U_A @ diag(S_A)   # 16×16 matrix
sv = svdvals(M)

stable_rank = sum(sv²) / sv_max²
```

Total cost: two thin SVDs + one 16x16 SVD. Runs in <1ms per layer.

### Frontend Stack

| Package | Purpose |
|---------|---------|
| `@react-three/fiber` | React renderer for Three.js |
| `@react-three/drei` | OrbitControls, Camera, Environment |
| `three` | WebGL scene graph, PlaneGeometry with vertex displacement |

Terrain is a `PlaneGeometry(width, height, numBlocks-1, numSteps-1)` with vertex Y positions set from metric data and vertex colors from colormaps (inferno for norms, viridis for stable rank). ~640 vertices for 32 blocks x 20 steps — trivially fast.

---

## Integration Plan

### Phase 1: Standalone Tool (current prototype)

- [x] Python extraction script (`extract_lora_data.py`)
- [x] Block landscape with weight norm metric
- [x] Block landscape with stable rank metric
- [x] Weight matrix surface with diverging colormap
- [x] Timeline scrubber with play/pause
- [x] Per-block labels with double/single grouping
- [x] Live stats overlay (min/mean/max)
- [ ] Accept arbitrary safetensors directory as input (currently hardcoded)
- [ ] Command-line args for extraction script (rank, architecture)

### Phase 2: Integration with Main UI

- [ ] Add R3F dependencies to `ui/package.json`
- [ ] New API endpoint: `GET /api/jobs/{jobID}/weight-landscape`
- [ ] Extract weight stats during training (hook into checkpoint saving)
- [ ] Add 3D landscape tab to `JobMetricsPage.tsx`
- [ ] Lazy-load R3F to avoid bundle size impact on other pages

### Phase 3: Config Recommendations

- [ ] "Suggested block_dims" generator based on stable rank analysis
- [ ] "Suggested ignore_if_contains" based on near-zero norm blocks
- [ ] Export YAML snippet for recommended `network` config
- [ ] Side-by-side comparison of two training runs

### Phase 4: Real-Time During Training

- [ ] Incremental extraction (compute stats at each checkpoint save)
- [ ] WebSocket push for live terrain updates
- [ ] Integrate with companion 2D feature for synchronized timeline

---

## Prototype File Structure

```
tmp/lora-viz-prototype/
├── extract_lora_data.py      # Python: safetensors → JSON
├── package.json               # Vite + R3F dependencies
├── vite.config.ts
├── index.html
├── public/
│   ├── block_stats.json       # Per-block norms + stable rank (32 blocks × 20 steps)
│   └── matrix_snapshots.json  # Weight matrix snapshots (128×16 × 20 steps)
└── src/
    ├── main.tsx               # Entry point
    ├── App.tsx                # Canvas, controls, UI overlay, metric toggle
    ├── BlockNormLandscape.tsx # 3D terrain with vertex displacement
    ├── MatrixSurface.tsx      # Weight matrix heightmap
    ├── colormap.ts            # Inferno, viridis, diverging colormaps
    └── types.ts               # BlockStatsData, MetricKey, MatrixSnapshotsData
```

Run: `cd tmp/lora-viz-prototype && npm install && npx vite --host`

---

## Performance Considerations

### Extraction
- 20 checkpoints × 112 layer pairs × 2 SVDs each = ~4480 SVD operations
- Each SVD is on a matrix with small dimension r=16 — <1ms each
- Total extraction: ~30 seconds for 20 checkpoints

### Frontend
- 32 × 20 = 640 vertices — any GPU handles this trivially
- Weight matrix: 128 × 16 = 2048 vertices — also trivial
- Smooth interpolation via `lerp()` in vertex buffer — no React re-renders
- JSON data: ~200KB for block_stats, ~2MB for matrix_snapshots

### Bundle Size
- Three.js + R3F + Drei adds ~1.1MB gzipped
- Should be code-split / lazy-loaded when integrated into main UI

---

## Academic Basis

| Metric | Source | Key Finding |
|--------|--------|-------------|
| Stable rank / singular value concentration | Hu et al. 2021, "LoRA" (ICLR 2022) [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Even at rank 64, meaningful information concentrates in top 1 singular vector |
| Weight norm as divergence detector | Kim et al. 2025, "LoRA Training Provably Converges..." (ICML 2025) [arXiv:2502.09376](https://arxiv.org/abs/2502.09376) | Spurious local minima have conspicuously large weight magnitude — "fails loudly" |
| Per-block rank allocation | Zhang et al. 2023, "AdaLoRA" (ICLR 2023) [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) | FFN layers need higher rank than attention; allocation varies by layer |
| Rank vs. module breadth tradeoff | Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (TMLR) [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) | Targeting more modules at low rank beats fewer modules at high rank |
| Intruder dimensions from low rank | Shuttleworth et al. 2024, "LoRA vs Full Fine-tuning" [arXiv:2410.21228](https://arxiv.org/abs/2410.21228) | LoRA introduces singular vectors dissimilar to pretrained model; correlates with forgetting |

---

## Inspiration

- [ShootTheSound/comfyUI-Realtime-Lora](https://github.com/ShootTheSound/comfyUI-Realtime-Lora) — Post-training block-level LoRA analysis for ComfyUI
- [Loss Landscape Library](https://github.com/tomgoldstein/loss-landscape) — 3D loss surface visualization (Goldstein et al., NeurIPS 2018)
- [hako-mikan/sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) — Per-block weight adjustment at inference

---

## Open Questions

1. **Block indexing for Flux**: The existing `block_dims` system in `kohya_lora.py` uses a 25-slot UNet index (12 down + 1 mid + 12 up). Flux's architecture (8 double + 24 single) doesn't map to this. Need to verify whether `block_dims` works for Flux at all, or if `only_if_contains` / `ignore_if_contains` is the only viable per-block mechanism for transformer architectures.

2. **Stable rank threshold for rank recommendation**: At what stable rank value should we recommend lowering rank? If stable_rank < 2 and rank is 16, suggesting rank 4 seems safe. But the threshold needs validation across different LoRA types and architectures.

3. **Comparison mode**: Would comparing two training runs side-by-side (e.g., rank 4 vs rank 16 on the same dataset) be more valuable than single-run analysis?

4. **Integration with real-time training**: Computing SVD at every logging step adds overhead. Should stable rank be computed only at checkpoint saves, or is per-step monitoring valuable enough to justify the cost?

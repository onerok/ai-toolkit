# Advanced Job Config (YAML-Only Knobs)

This document covers **advanced job config parameters that are not exposed in the Simple Job UI**.

Scope and filter:
- Included: parameters that are real user tuning knobs (used in examples and/or intended for manual tuning).
- Excluded: internal/plumbing/migration fields (for example `model.is_flux`, `device`, SQLite paths, old aliases).

## Before You Change Advanced Knobs

1. Change only 1-2 knobs per run.
2. Keep a baseline config that worked.
3. For first runs, prefer defaults and only tune what you need.

## Core Advanced Knobs

| Config Path | Type | What It Does | Typical Values / Default |
|---|---|---|---|
| `save.save_format` | enum string | Output format for saved checkpoints. | `"safetensors"` (default), `"diffusers"` |
| `save.push_to_hub` | boolean | Push saved model to Hugging Face Hub after save. | `false` default |
| `save.hf_repo_id` | string or null | HF repo name used when push is enabled. | e.g. `"username/model-name"` |
| `save.hf_private` | boolean | Create/push as private HF repo. | `false` default |
| `logging.log_every` | integer | Training log interval (steps). | `100` default |
| `logging.use_wandb` | boolean | Enable Weights & Biases logging. | `false` default |
| `logging.verbose` | boolean | Print extra debug-level logs. | `false` default |
| `logging.project_name` | string | Tracker project grouping name. | `"ai-toolkit"` default |
| `logging.run_name` | string or null | Optional explicit run name in trackers. | `null` default |
| `train.noise_scheduler` | string | Training noise schedule algorithm; model-dependent. | often `"flowmatch"` or `"ddpm"` |
| `train.train_unet` | boolean | Train UNet/transformer backbone weights. | usually `true` |
| `train.train_text_encoder` | boolean | Train text encoder weights (more VRAM/time). | usually `false` |
| `train.gradient_checkpointing` | boolean | Reduce VRAM by recomputation (slower training). | usually `true` |
| `train.gradient_accumulation_steps` | integer (legacy) | Older accumulation key; avoid using with `train.gradient_accumulation`. | default `1` |
| `train.bypass_guidance_embedding` | boolean | Model-specific speed/compat toggle for guidance embeddings. | often `true` for Flex/Flux-like configs |
| `train.lr_scheduler` | string | Learning-rate scheduling policy. | `"constant"` default |
| `train.min_snr_gamma` | number or null | SNR-based loss reweighting strength. | `null` default; slider example uses `5.0` |
| `train.max_denoising_steps` | integer | Upper timestep bound used in train sampling. | `999` default |
| `train.noise_offset` | number | Adds global noise offset during training. | `0.0` default |
| `train.dtype` | string | Training compute dtype. | config-module default `"fp32"`; many examples override |
| `train.ema_config.use_feedback` | boolean | Advanced EMA feedback mode. | `false` default |
| `train.ema_config.param_multiplier` | number | Per-step parameter multiplier (advanced EMA control). | `1.0` default |
| `network.network_kwargs.ignore_if_contains` | list of strings | Skip LoRA attachment for layers matching substrings. | default `[]` |
| `network.network_kwargs.only_if_contains` | list of strings | Train only layers whose names match one of these substrings. | default `null` / unset |
| `datasets.caption_ext` | string | Caption file extension in dataset. | usually `"txt"` |
| `datasets.shuffle_tokens` | boolean | Shuffle comma-separated caption tags during training. | usually `false` |
| `datasets.cache_latents` | boolean | Cache latents in RAM (faster, high memory). | `false` default |
| `datasets.cache_text_embeddings` | boolean | Precompute/cache text embeddings for dataset captions. | `false` default |
| `sample.neg` | string | Global negative prompt used for sample previews. | often `""` |
| `sample.network_multiplier` | number | Global LoRA strength for sample previews. | `1.0` default |
| `sample.guidance_rescale` | number | Rescale applied to CFG guidance behavior. | `0.0` default |
| `sample.format` | enum string | Sample image output format. | `"jpg"` default (`"webp"` for animated outputs) |
| `sample.adapter_conditioning_scale` | number | Strength of adapter/control conditioning in samples. | `1.0` default |
| `sample.refiner_start_at` | number (0-1) | Fraction of denoising steps where refiner starts. | `0.5` default |
| `sample.extra_values` | list of numbers | Extra numeric conditioning values passed into sampling pipeline. | `[]` default |
| `sample.do_cfg_norm` | boolean | Model-specific CFG normalization switch for sampling. | `false` default |

## High-Impact Expert LoRA Knobs

These are advanced but very useful when you want precise control over what LoRA trains.

| Config Path | Type | What It Does | Typical Values / Default |
|---|---|---|---|
| `network.network_kwargs.block_dims` | comma-separated int list | Per-block LoRA rank. Lets you allocate rank where it matters most. Use `0` to skip a block entirely. | e.g. `"4,4,...,16,16,..."`; default is global `network.linear` for all blocks |
| `network.network_kwargs.block_alphas` | comma-separated float/int list | Per-block alpha scaling (usually matched to `block_dims`). | often same pattern as `block_dims`; default is global alpha |
| `network.network_kwargs.conv_block_dims` | comma-separated int list | Per-block rank for conv blocks (if conv LoRA is enabled). | optional; unset by default |
| `network.network_kwargs.conv_block_alphas` | comma-separated float/int list | Per-block alpha for conv blocks. | optional; unset by default |
| `network.network_kwargs.down_lr_weight` | string keyword or comma-separated float list | LR multiplier curve/list for "down" blocks. Can emphasize or de-emphasize groups of blocks. | keywords like `"cosine"` or explicit list |
| `network.network_kwargs.mid_lr_weight` | float | LR multiplier for the middle block. | e.g. `0.5`; default `1.0` behavior |
| `network.network_kwargs.up_lr_weight` | string keyword or comma-separated float list | LR multiplier curve/list for "up" blocks. | keywords like `"sine"` or explicit list |
| `network.network_kwargs.block_lr_zero_threshold` | float | Any block LR weight at or below this threshold is effectively zeroed (can disable blocks without editing dims directly). | default `0.0` |
| `network.network_kwargs.only_if_contains` | list of strings | Train only matching module names (block/group targeting by name). | e.g. `["single_blocks"]` |
| `network.network_kwargs.ignore_if_contains` | list of strings | Exclude matching module names from training. | e.g. `["double_blocks.0"]` |

Quick example:

```yaml
network:
  type: "lora"
  linear: 16
  linear_alpha: 16
  network_kwargs:
    block_dims: "4,4,4,4,4,4,4,4,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16"
    block_alphas: "4,4,4,4,4,4,4,4,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16"
    down_lr_weight: "cosine"
    up_lr_weight: "sine"
    mid_lr_weight: 0.5
    only_if_contains:
      - "single_blocks"
```

Notes:
- The list length for `block_dims`/`block_alphas` must match the model's expected block count.
- `block_dims=0` disables that block.
- Prefer changing one of: rank layout, LR weighting, or name targeting at a time, then compare against a baseline run.

## Concept Slider (YAML-Only in Simple UI)

When job `type` is `concept_slider`, these slider fields are advanced-only in the form workflow.

| Config Path | Type | What It Does | Typical Values / Default |
|---|---|---|---|
| `slider.targets` | list of objects | Defines concept targets/prompts and direction/weights for slider learning. | required for real slider behavior |
| `slider.resolutions` | list of `[int, int]` | Explicit training resolutions for slider targets. | e.g. `[[512, 512]]` |
| `slider.batch_full_slide` | boolean | Batch strategy for applying full slider direction per step. | `true` default |
| `slider.anchors` | list of objects | Anchor prompts to stabilize slider direction and strength. | optional |
| `slider.prompt_file` | string or null | File path with prompts for slider training. | optional |
| `slider.prompt_tensors` | string or null | Path to precomputed prompt tensors. | optional |
| `slider.use_adapter` | boolean or null | Use adapter-conditioned slider path (model-dependent). | optional |
| `slider.adapter_img_dir` | string or null | Image dir used by adapter-conditioned slider training. | optional |
| `slider.low_ram` | boolean | Lower-memory slider mode. | `false` default |

## Safe Defaults (Non-Expert)

Start by changing only these:
- `save.save_format`
- `save.push_to_hub`
- `datasets.caption_ext`
- `datasets.shuffle_tokens`
- `train.train_text_encoder`
- `train.gradient_checkpointing`
- `train.noise_scheduler`
- `sample.neg`

## Caution / Easy-to-Misuse Knobs

Only change these if you have a specific reason:
- `train.gradient_accumulation_steps` (legacy behavior)
- `train.ema_config.use_feedback`
- `train.ema_config.param_multiplier`
- `sample.extra_values`
- `sample.do_cfg_norm`
- `network.network_kwargs.ignore_if_contains`

## Notes

- The Advanced YAML editor can set many more fields than listed here.
- This page intentionally focuses on knobs likely to be useful to end users.
- If you need a complete raw field inventory, use backend config classes in `toolkit/config_modules.py`.

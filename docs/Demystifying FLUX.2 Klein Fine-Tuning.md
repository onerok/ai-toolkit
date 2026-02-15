# Demystifying FLUX.2 Klein Fine-Tuning

A practical, experiment-informed guide to LoRA training on FLUX.2 Klein. Inspired by [spacepxl/demystifying-sd-finetuning](https://github.com/spacepxl/demystifying-sd-finetuning) — same philosophy, new architecture. Klein-specific findings draw heavily from [Calvin Herbst's](https://www.youtube.com/@CalvinHerbst) 50+ isolated-variable training experiments across FLUX.2 dev and Klein 9B.

The original repo proved that most fine-tuning advice is vibes-based and wrong, and that you can cut through it with deterministic validation loss and controlled experiments. Everything here follows that principle: measure don't guess, change one variable at a time, and stop training when the loss curve tells you to.

---

## Table of Contents

1. [What is FLUX.2 Klein and why does it matter](#1-what-is-flux2-klein-and-why-does-it-matter)
2. [The variants: which one you actually train on](#2-the-variants-which-one-you-actually-train-on)
3. [Architecture differences that affect your training](#3-architecture-differences-that-affect-your-training)
4. [Methodology: deterministic validation still wins](#4-methodology-deterministic-validation-still-wins)
5. [LoRA configuration](#5-lora-configuration)
6. [Learning rate and optimizer](#6-learning-rate-and-optimizer)
7. [Weight decay: the parameter nobody talks about](#7-weight-decay-the-parameter-nobody-talks-about)
8. [Network dimensions: linear_dim and linear_alpha](#8-network-dimensions-linear_dim-and-linear_alpha)
9. [Dataset preparation](#9-dataset-preparation)
10. [Training steps and when to stop](#10-training-steps-and-when-to-stop)
11. [VRAM management](#11-vram-management)
12. [Inference settings for evaluation](#12-inference-settings-for-evaluation)
13. [Quick-start configs](#13-quick-start-configs)
14. [Common failures and how to diagnose them](#14-common-failures-and-how-to-diagnose-them)
15. [Tools and environment](#15-tools-and-environment)

---

## 1. What is FLUX.2 Klein and why does it matter

FLUX.2 Klein is a family of compact DiT (Diffusion Transformer) models from Black Forest Labs, released January 15, 2026. It's the first model in the Flux lineage that's both small enough to train on consumer hardware AND released under Apache 2.0 (the 4B variant).

That combination — trainable on a single 24GB GPU, commercially licensable, and producing genuinely good images — didn't exist before Klein. FLUX.2 dev is 32B and non-commercial. Flux.1 dev was 12B and non-commercial. SD 1.5/SDXL are commercially friendly but architecturally obsolete. Klein 4B Base fills the gap.

## 2. The variants: which one you actually train on

There are four Klein models. Only two of them should ever be used for training.

| Variant | Params | Distilled? | Inference Steps | License | Train on this? |
|---------|--------|-----------|-----------------|---------|---------------|
| Klein 4B | 4B | Yes (4-step) | 4 | Apache 2.0 | **No** |
| **Klein 4B Base** | **4B** | **No** | **~50** | **Apache 2.0** | **Yes** |
| Klein 9B | 9B | Yes (4-step) | 4 | Non-commercial | **No** |
| Klein 9B Base | 9B | No | ~50 | Non-commercial | Maybe |

**Always train on a Base variant.** This cannot be overstated.

The distilled 4-step models have been compressed through step distillation and guidance distillation. When you fine-tune them, you're fighting the distillation — the model was specifically optimized to produce good outputs in exactly 4 steps with no classifier-free guidance, and your LoRA training destabilizes that optimization. The result is LoRAs that appear to "work" at low step counts but produce increasingly incoherent outputs as you push them.

This is the same lesson the community learned the hard way with Flux.1 Schnell: don't train on distilled models. Train on the undistilled Base, then optionally apply the distillation at inference if you need speed.

**4B Base vs 9B Base:** Start with 4B. It's Apache 2.0 (commercially usable), requires ~24GB VRAM for LoRA training, and converges faster. The 9B model produces marginally better output at the cost of non-commercial licensing, ~32GB+ VRAM, and notably worse training stability. Multiple community reports describe 9B training runs collapsing under memory-saving configurations that work fine on 4B. Unless you specifically need the 9B's quality ceiling and have the hardware headroom, 4B Base is the correct default.

## 3. Architecture differences that affect your training

If you're coming from SD 1.5 / SDXL, basically everything is different under the hood. If you're coming from Flux.1, most things are similar but the text encoder changed.

### UNet → DiT

SD used a UNet (convolutional encoder-decoder with skip connections). Klein uses a Diffusion Transformer — a sequence of transformer blocks operating on patched latent tokens. This means:

- **No convolution layers to target.** The spacepxl repo showed that adding resnet convolutions to SD LoRA targets doubled parameters with no quality improvement. On Klein there are no convolutions to add — it's attention and MLP all the way down. This is good; the architecture naturally matches the optimal LoRA target strategy.
- **Attention + MLP are your LoRA targets.** The transformer blocks contain attention projections and MLP/feedforward layers — these are the modules that get LoRA adapters. This is what the original repo recommended for SD after extensive testing, and it's the only option here. Note that the exact layer names vary by framework (e.g., AI Toolkit uses `qkv`/`proj` internally, diffusers uses `to_q`/`to_k`/`to_v`/`to_out`); most training tools handle targeting automatically.

### CLIP → Qwen3 LLM

SD used CLIP ViT-L/14 (~400M params, ~500K concept vocabulary). Klein replaces CLIP with a Qwen3 LLM as its text encoder — the size scales with the model variant:

- **Klein 4B** uses **Qwen3-4B** (~8GB in BF16)
- **Klein 9B** uses **Qwen3-8B** (~16GB in BF16)

Practical implications:

- **Captioning strategy shifts from tags to natural language.** CLIP was trained on short tag-like descriptions. Qwen3 understands full sentences and paragraphs. Write captions like you're describing the image to someone, 40–100 words. Danbooru-style tag dumps still work but leave quality on the table.
- **Text encoder training is even less necessary.** The spacepxl repo showed text encoder fine-tuning was unnecessary for SD because CLIP already knew ~500K concepts. Qwen3 knows orders of magnitude more. You'd need to be training a truly alien concept to justify touching it. Keep it frozen.
- **The text encoder is a significant VRAM cost.** On the 4B variant, Qwen3-4B is roughly the same size as the diffusion model itself. On the 9B variant, Qwen3-8B is nearly as large. Cache your text embeddings (encode all captions once, save to disk, free the encoder from VRAM). This is the single biggest VRAM optimization available.

### SD VAE → FLUX VAE (32 channels)

Klein uses a 32-channel VAE (vs SD's 4-channel). This means the latent space is much richer, which is partly why these models handle fine detail better. For training purposes, the main impact is that **latent caching saves more VRAM** than it did on SD, because the VAE is larger. Cache latents alongside text embeddings.

### Flow matching → still flow matching

Like Flux.1, Klein uses rectified flow matching rather than DDPM-style diffusion. The noise schedule is different from SD — it's a linear interpolation between noise and data rather than a cosine/linear beta schedule. Your training scripts handle this, but it's why SD-era intuitions about timestep weighting and noise schedules don't directly transfer.

## 4. Methodology: deterministic validation still wins

The spacepxl repo's most important contribution wasn't any specific hyperparameter — it was the methodology of **stable (deterministic) loss measurement.** The approach:

1. Split your dataset into training images and held-out validation images (even just 3–5 validation images help).
2. Periodically pause training.
3. Run validation images through the model with **fixed seeds, fixed timesteps, fixed noise.**
4. Record the average loss.
5. Resume training.

This produces smooth loss curves instead of the noisy mess you get from training loss alone. You'll see a clear U-shape: loss drops as the model learns generalizable features, hits a minimum, then rises as it begins memorizing your specific training images (overfitting).

**The minimum of the validation loss curve is your optimal stopping point.** Not a predetermined step count. Not "when the images look good to me." The loss curve.

This methodology transfers directly to Klein. The architecture changed; the math of overfitting did not. If you're training without validation loss, you're guessing when to stop, and you will guess wrong.

**A note on complementary approaches:** Herbst's testing methodology is different from spacepxl's but valuable in its own right. Rather than deterministic loss curves, he evaluates visually: generating images from **out-of-distribution prompts** (e.g., "dog on a log" — tokens that appear nowhere in his training data) to test whether the LoRA transfers style rather than memorizing content. He also uses **RGB waveform analysis** to objectively measure tonal characteristics across training runs. Both approaches — quantitative loss curves and structured visual evaluation — are useful. The loss curve tells you *when* to stop; visual evaluation tells you *what* the model actually learned.

### How to implement it

Most training tools don't natively support deterministic validation the way spacepxl's custom scripts did. Your options:

- **SimpleTuner** has built-in validation image generation at configurable intervals.
- **Kohya SS / Musubi** can generate sample images at intervals, but you'll need to manually track loss.
- **AI Toolkit (Ostris)** supports eval steps with configurable frequency.
- **Manual approach:** Save checkpoints every N steps. Write a script that loads each checkpoint, runs inference on your validation set with fixed seeds, and computes the loss. Plot it. This takes more effort but gives you the cleanest signal.

Set evaluation frequency to every 100–200 steps. More frequent than that wastes time; less frequent and you might miss the minimum.

## 5. LoRA configuration

Full fine-tuning of even the 4B model requires multi-GPU setups and isn't practical for most people. LoRA is the standard approach.

### Rank

Start at **16** for simple concepts (single character, single style). Move to **32** if results are underfitting (validation loss plateaus high). **64** for complex multi-concept training.

The spacepxl repo showed that for SD, higher ranks didn't help once attention + MLP layers were included — the bottleneck was which layers you targeted, not how many parameters each adapter had. This holds partially for Klein, but the transformer architecture has more layers that benefit from adaptation, so ranks above 16 can genuinely help where they didn't on SD.

Don't go above 64 unless you have a specific reason and are monitoring validation loss to confirm it helps.

### Alpha

The spacepxl repo made a strong case for **alpha = 1 always**, because it decouples rank from effective learning rate — you can change rank without needing to retune LR. This remains the cleanest approach.

The Klein community (including Herbst's configs) commonly uses **alpha = rank/2**, which is the default in many training tools. This works fine but means your effective learning rate changes whenever you change rank, which makes it harder to isolate variables.

The relationship:

- Effective LoRA learning rate scales as `alpha / rank × base_lr`
- With alpha=1, rank=32, LR=1e-4: effective LR = 3.125e-6
- With alpha=16, rank=32, LR=1e-4: effective LR = 5e-5

**Recommendation: use alpha = 1 if you're experimenting with different ranks** (so LR stays constant). Use alpha = rank/2 if you've settled on a rank and want the standard LR recommendations (like 1e-4) to apply directly. Don't switch between conventions mid-experiment.

### Target modules

Target attention projections and MLP/feedforward layers. The exact layer names differ between frameworks:

- **AI Toolkit**: Targets all `nn.Linear` modules within the `Flux2` transformer class automatically (internally: `qkv`, `proj` for attention; `linear1`/`linear2`, `img_mlp`/`txt_mlp` for MLP). You don't need to specify individual layer names — just select the Klein model and LoRA is applied to the right modules.
- **Diffusers / Kohya**: May use different naming conventions like `to_q`, `to_k`, `to_v`, `to_out`, `ff.net.*`. Check your tool's documentation or inspect the model's `state_dict` keys.

The important thing is that both attention and MLP layers are targeted. Most training tools handle this automatically when you select standard LoRA targets for Klein.

**Do not train the text encoder** unless you have a compelling reason (introducing vocabulary that doesn't exist in Qwen3's training data, which is rare).

### Other LoRA variants

**LOKR:** Herbst briefly tested LOKR on Klein 9B but did not find it beneficial. Stick with standard LoRA unless you have a specific reason to experiment.

**EMA (Exponential Moving Average):** Herbst tested EMA on Klein 9B and found it produced worse results than standard training. Not recommended.

## 6. Learning rate and optimizer

### Learning rate

**Start at 1e-4.** This is the convergent recommendation from HuggingFace's official training script, Ostris AI Toolkit defaults, and Calvin Herbst's 50+ isolated-variable experiments.

Klein is significantly more LR-sensitive than SD 1.5. The spacepxl repo demonstrated that on SD, learning rate mainly affected convergence speed — all tested rates reached the same minimum loss, just at different speeds. On Klein, this is not true. Herbst found that changing the default LR of 0.0001 by as little as five one-thousandths of a percent (i.e., ~5e-6) "totally ripped apart the image." Note that the spacepxl experiments measured LR sensitivity via validation loss curves, while Herbst evaluated primarily via visual/aesthetic quality — it's possible that visual degradation precedes measurable loss divergence, making Klein appear more sensitive depending on how you measure.

**If training looks unstable** (loss spikes, artifacts in validation images): reduce to 5e-5.
**If convergence is very slow** (thousands of steps with minimal validation loss change): increase to 2e-4, but monitor closely.

### Optimizer

**AdamW8bit.** Same conclusion as the spacepxl repo, same reasoning: identical performance to full AdamW at lower memory cost. This has been re-validated on transformer architectures extensively.

Adafactor is usable on Klein if VRAM is extremely tight, but historically needs a higher LR (1.5–2× AdamW's rate). If you use it, adjust accordingly.

SGD is not recommended. The spacepxl repo showed it underperformed on SD, and there's no evidence it's better on transformers.

### Scheduler

**Cosine with warmup** or **constant with warmup.** 200 warmup steps is standard. The choice between cosine and constant matters less than getting the base LR right — cosine gives you a graceful tail-off that can help in the late stages of training, but if you're stopping at the validation loss minimum anyway (as you should be), the tail behavior is irrelevant.

### Batch size and LR scaling

The spacepxl repo demonstrated that **square-root scaling** (`new_lr = base_lr × √(new_batch / old_batch)`) is correct for AdamW with small batch sizes, not linear scaling. This almost certainly still holds for Klein, though it hasn't been independently re-validated.

In practice, most Klein training runs use batch size 1 (VRAM-constrained), so this only matters if you're using gradient accumulation as a batch size substitute. With gradient accumulation of 4 (effective batch 4), you'd scale from 1e-4 to 2e-4, not 4e-4.

## 7. Weight decay: the parameter nobody talks about

This is the sleeper parameter that most guides either ignore or leave at the default.

Interestingly, the spacepxl repo found weight decay had **zero measurable effect** on SD 1.5 validation loss across the range 0 to 0.1 — it simply didn't matter for short fine-tuning runs on UNet architectures. On Klein, the story is completely different: Herbst's research (50+ isolated-variable runs across FLUX.2 dev and Klein 9B) found that **weight decay had a larger impact on output quality than learning rate.** His optimal value: **0.00001** — one-tenth the typical default of 0.0001.

Why the divergence? Herbst's evaluation was primarily visual/aesthetic (grain texture, highlight bloom, shadow detail) rather than loss-curve-based. It's likely that weight decay affects perceptual quality on flow-matching transformers in ways that don't surface as clearly in aggregate loss metrics. Regardless, the visual differences in his tests are striking.

Weight decay acts as a regularizer on parameter magnitudes. On these flow-matching transformers, it manifests as changes in tonal response:

- **Too low** (0 or near-0): Parameters grow unchecked, causing subtle color channel bleed and lifted blacks. Can look "analog" in a pleasant way but isn't controllable.
- **Sweet spot** (0.00001): Clean tonal separation, accurate color reproduction, natural contrast.
- **Too high** (0.0001+): Aggressive parameter suppression causes contrast spikes, crushed shadows, and at 0.001 the images get visibly degraded.

**Set weight decay to 0.00001 and leave it there.** This was consistent across Herbst's tests on FLUX.2 dev and Klein 9B (he did not separately test 4B, but the same principle should apply). If your training tool defaults to a higher value, override it.

Herbst also visualized the decay effect using RGB waveform analysis in DaVinci Resolve, measuring how the black point, white point, and channel separation changed across decay values. Higher decay crushed the waveform into a narrow band (high contrast, clipped shadows); lower decay spread it wider (lifted blacks, softer tonality). This is one of the few training parameters where the effect is directly measurable in the output pixels.

## 8. Network dimensions: linear_dim and linear_alpha

Most people train LoRAs with flat dimensions — rank 32 for everything. Herbst's research revealed this is significantly suboptimal.

The winning configuration across his ~64 tested combinations: **linear_dim = 128, linear_alpha = 64.**

```
linear_dim    = 128   (rank for nn.Linear layers)
linear_alpha  = 64    (alpha for nn.Linear layers)
```

The flat default of 32/32 was measurably worse. Going higher to 256/256 destroyed images entirely. The sweet spot is a higher rank with alpha at half the rank.

**A note on conv_dim / conv_alpha:** Many training tools (including AI Toolkit and Kohya) expose four network dimension fields: `linear_dim`, `linear_alpha`, `conv_dim`, and `conv_alpha`. These exist because SD 1.5 / SDXL used UNet architectures with convolution layers. Herbst's configs included `conv_dim = 64` and `conv_alpha = 32` — however, Klein is a pure DiT with **no convolution layers** (as noted in Section 3). The conv settings create zero LoRA modules on Klein; they are silently ignored. The "4:2:2:1 ratio" (128/64/64/32) that Herbst reports is effectively just **2 active values**: `linear_dim = 128` and `linear_alpha = 64`. The conv values are vestigial from UNet-era configs.

If your tool only offers a single rank setting, use 32 (or 128 if your VRAM allows) and accept the suboptimality — it still works, just not as well.

## 9. Dataset preparation

### How many images

The spacepxl repo definitively proved that **"less is more" is false.** Every dataset size tested on SD showed that more images → better generalization, period. This holds on Klein.

Practical minimums and targets:

- **Single subject (face/character):** 15–30 images minimum. 50+ is better. Variety of angles, lighting, expressions.
- **Style LoRA:** 40–200 images. Styles need more examples to generalize because "style" is a broader concept than "this specific face." (Herbst's HerbstPhoto style LoRA used 41 images and produced strong results, though more images would likely improve generalization further.)
- **Object/concept:** 20–40 images showing the object in varied contexts.

**Absolute minimum:** 8–10 images. Below this you're memorizing, not learning.

### Captioning

Write natural language descriptions. Klein's Qwen3 text encoder is an LLM — it processes language like GPT, not like CLIP. Good captions are specific, descriptive, and 40–100 words.

**Good caption:** "A woman with short red hair and green eyes, wearing a blue denim jacket, standing in a sunlit garden with white roses in the background. She is smiling and looking slightly to the left. Soft natural lighting, shallow depth of field."

**Bad caption:** "woman, red hair, green eyes, denim jacket, garden, roses, smiling, natural light"

The tag-style caption "works" in the sense that the model will train, but you're wasting the text encoder's capacity. It was trained on full sentences and understands compositional descriptions that tags cannot express.

For subject LoRAs, use a **trigger word** consistently: "a photo of [triggername], a woman with short red hair..." This gives you a reliable activation token at inference.

**Caption dropout:** Some training tools support randomly dropping captions during training (replacing them with empty strings). This can improve the LoRA's ability to activate without the trigger word and may improve style transfer. Herbst tested caption dropout rates around 0.1–0.2 on Klein 9B. If your tool supports it and you're not caching text embeddings, consider `caption_dropout_rate: 0.1` as a starting point. Note that caption dropout is incompatible with text encoder caching (since cached embeddings can't be dynamically emptied).

### Captioning tools

- **Florence-2** — Fast, decent quality, works locally. The spacepxl repo included `caption_florence.py` for this.
- **Joy Caption** — Higher quality than Florence, especially for aesthetic descriptions.
- **Qwen-VL / InternVL** — Vision-language models that produce very detailed captions. Since Klein's text encoder is Qwen3, using a Qwen-family captioner can produce naturally aligned descriptions.
- **Manual review is always worth it.** Auto-captioners hallucinate. Read every caption, correct errors, add details the model missed.

### Image preparation

- **Resolution:** Klein handles aspect ratios well via bucket training, and all dimensions must be divisible by 16. Training at your images' native resolution is fine — Herbst trained at 1920×1080 and evaluated at 1024×1024 to test resolution generalization. If VRAM is tight, 1024×1024 is a reasonable target.
- **Quality:** Higher resolution source images are better. Upscale only as a last resort — upscaler artifacts become training artifacts.
- **Variety:** The spacepxl repo showed that random cropping improves results by simulating a larger dataset. If your training tool supports it, enable it.
- **No duplicates or near-duplicates.** They bias the training distribution and accelerate overfitting.

### Validation split

Hold out 3–5 images from your training set. These should be representative (similar distribution to training data). Use them exclusively for deterministic validation loss measurement, never for training.

## 10. Training steps and when to stop

### Step count guidelines

These are starting points, not gospel. Your validation loss curve is the actual answer.

| Dataset size | Approximate steps to evaluate |
|-------------|-------------------------------|
| 10–20 images | 1,000–2,500 |
| 20–40 images | 2,000–4,000 |
| 50–120 images | 3,000–6,000 |
| 200+ images | 5,000–10,000+ |

Save checkpoints every 200–500 steps (Herbst saved every 1,000 for his initial sweeps, then narrowed to 250-step intervals around the optimum). Compare validation loss across checkpoints. The optimal checkpoint is often earlier than you'd expect — the spacepxl repo found the "usable range" on a 22-image SD dataset was steps 2,000–6,000 with the minimum around 2,000.

However, for style LoRAs the optimal step count can be significantly higher. Herbst found 7,000 steps optimal for a 41-image style dataset on Klein 9B — well above what the table above would suggest for that dataset size. Style requires more steps because the model needs to learn subtle texture patterns (grain, halation, tonal response) that take longer to converge than subject identity. When in doubt, train longer and save more checkpoints.

### Repeats

For small datasets, your training tool will cycle through images multiple times per epoch. The "repeats" setting controls this. With 20 images and 2,000 steps at batch 1, each image is seen 100 times. This is fine — as long as you're monitoring validation loss.

For high-likeness character training (where you specifically want the model to memorize facial features), community reports suggest pushing to 50–120 repeats before overfitting degrades pose/style flexibility.

### Signs you've gone too far

- Validation loss rising while training loss continues to drop
- Generated images from validation prompts look increasingly "same-y"
- The model produces your exact training images when given related prompts
- Backgrounds and contexts become less varied
- Faces/subjects become "waxy" or over-smoothed

### Signs you haven't gone far enough

- Validation loss still dropping
- Trigger word barely activates the concept
- Generated images have the right idea but wrong details
- Mixing with other prompts overpowers the LoRA easily

### Advanced: high-noise / low-noise split training

An experimental technique Herbst explored: training two separate LoRAs from the same dataset — one with `timestep_bias: high_noise` (learns composition, structure, broad tonal relationships) and one with `timestep_bias: low_noise` (learns fine texture, grain, detail). At inference, the high-noise LoRA is applied during early sampling steps and the low-noise LoRA during late steps, similar to how Wan 2.2's dual-model architecture works.

This is uncharted territory — Herbst described it as a theoretical experiment. If you try it, save aggressive checkpoints and expect to iterate on the step-switching point.

## 11. VRAM management

### Budget by GPU

| GPU | 4B Base LoRA | 9B Base LoRA | Notes |
|-----|-------------|-------------|-------|
| RTX 3090 (24GB) | ✅ Comfortable | ⚠️ Tight | Cache everything, rank ≤32 |
| RTX 4090 (24GB) | ✅ Comfortable | ⚠️ Tight | Same VRAM, faster training |
| RTX 4080 (16GB) | ⚠️ Possible | ❌ | Aggressive optimization required |
| L40S / A6000 (48GB) | ✅ Easy | ✅ Comfortable | Herbst's GPU of choice for 9B |
| A100 40GB | ✅ Easy | ⚠️ Tight | 9B may need caching + grad ckpt |
| H100 80GB | ✅ Trivial | ✅ Easy | Can increase batch size |

### Optimization techniques, ranked by impact

**1. Cache text embeddings** (saves ~8GB for 4B with Qwen3-4B, ~16GB for 9B with Qwen3-8B)
Encode all captions once with the Qwen3 text encoder, save embeddings to disk, unload the encoder entirely. This is the single biggest win. Most training tools support this natively (SimpleTuner: `--cache_text_encoder_outputs`, AI Toolkit: `cache_text_encoder_outputs: true`).

Downside: You can't do dynamic caption augmentation (randomly dropping/modifying words during training). For most use cases this doesn't matter.

**2. Cache latents** (saves ~1–2GB)
Same idea for the VAE. Encode images once, cache, unload. Combined with text encoder caching, the only model in VRAM during training is the DiT itself + LoRA adapters.

**3. Gradient checkpointing** (saves ~30–40% of remaining VRAM, costs ~15% speed)
Recomputes intermediate activations during backward pass instead of storing them. Always enable this on 24GB GPUs. On 40GB+ you can leave it off for speed.

**4. Mixed precision (BF16)**
Klein was trained in BF16. Always use BF16 for training. FP16 can cause numerical issues with the flow matching loss. **FP32 is actively worse, not just wasteful** — Herbst tested FP32 training on Klein 9B and found it "cooked the images too much," producing over-saturated, overly contrasty outputs compared to BF16 (though he noted FP32 was beneficial on other architectures like Wan 2.2, so this is Klein-specific). If your GPU doesn't support BF16 natively (pre-Ampere), you can use FP16 but expect occasional instability.

**5. Quantized base model (FP8 or NF4)**
Keep the frozen base model in FP8 or NF4 while training LoRA adapters in BF16. This further reduces VRAM but can subtly affect gradient quality. Use FP8 if available (requires compute capability ≥ 8.9, i.e., RTX 4090 / H100). Use NF4 (via bitsandbytes) on older GPUs. This is a last resort — prefer caching-based savings first.

## 12. Inference settings for evaluation

**This section exists because it's the #1 source of "my LoRA doesn't work" reports.**

Klein Base and Klein (distilled) use completely different inference settings. If you train on Base but evaluate with distilled settings, your outputs will look broken. The LoRA is fine. Your settings are wrong.

### Klein Base inference settings

| Setting | Value |
|---------|-------|
| Steps | **50** (range: 28–50) |
| CFG scale | **4.0** (range: 3.0–5.0) |
| Sampler | Euler / DPM++ 2M |
| Scheduler | Normal / Karras |
| LoRA strength | **0.6–0.8** (start at 0.7) |

### Klein distilled inference settings (for reference — don't train on this)

| Setting | Value |
|---------|-------|
| Steps | 4 |
| CFG scale | 1.0 |

If you evaluate your Base-trained LoRA at 4 steps with CFG 1.0, the output will be a noisy, undercooked mess. This does not mean your LoRA failed. It means you're running the model at 8% of the steps it needs.

### LoRA strength at inference

Start at **0.7–0.8** and adjust. Herbst found 0.73 optimal for his specific style LoRA (filmic grain, tested with `dpmpp_2s_a + sgm_uni` sampler), which is a reasonable starting point but not universal — your optimal strength will depend on your LoRA's subject matter, training duration, and sampler choice. At 1.0, LoRAs on Klein tend to overpower the base model — you'll see style collapse and reduced prompt adherence. Below 0.4, the LoRA barely activates.

If you're getting overpowered results even at low strength, your LoRA is overtrained. Go back to your validation loss curve and use an earlier checkpoint.

## 13. Quick-start configs

### Minimal viable config (4B Base, character LoRA, 24GB GPU)

```yaml
# Dataset
images: 20-30 subject photos, varied angles/lighting/expression
captions: Natural language, 40-80 words each, with trigger word
validation_split: 3-5 held-out images

# Model
base_model: black-forest-labs/FLUX.2-klein-base-4B
precision: bf16

# LoRA
rank: 16
alpha: 8  # or alpha: 1 with lr: 4e-4
# target_modules handled automatically by training tool

# Training
optimizer: adamw8bit
learning_rate: 1e-4
weight_decay: 0.00001
lr_scheduler: cosine
lr_warmup_steps: 200
max_train_steps: 3000
batch_size: 1
gradient_accumulation: 1
gradient_checkpointing: true

# Caching
cache_text_encoder_outputs: true
cache_latents: true

# Evaluation
eval_every_n_steps: 200
eval_steps: 50        # NOT 4
eval_cfg_scale: 4.0   # NOT 1.0
save_every_n_steps: 500
```

### Higher-quality config (4B Base, style LoRA, 40GB+ GPU)

```yaml
# Dataset
images: 80-200 style reference images
captions: Detailed natural language descriptions of content AND style

# LoRA
rank: 32
alpha: 16
# If tool supports separate dims:
linear_dim: 128
linear_alpha: 64
# conv_dim/conv_alpha not needed — Klein has no conv layers

# Training
optimizer: adamw8bit
learning_rate: 1e-4
weight_decay: 0.00001
lr_scheduler: cosine
lr_warmup_steps: 200
max_train_steps: 6000
batch_size: 2
gradient_accumulation: 2
gradient_checkpointing: false  # enough VRAM without it

# Caching
cache_text_encoder_outputs: true
cache_latents: true

# Evaluation
eval_every_n_steps: 200
save_every_n_steps: 500
```

## 14. Common failures and how to diagnose them

### "My LoRA produces garbage"

**First check:** Are you evaluating at 50 steps, CFG 4.0? If not, fix that before debugging anything else.

**Second check:** Is your LoRA strength at inference between 0.5–0.8? Strength 1.0 frequently overpowers Klein.

**Third check:** Did you train on a Base variant? If you trained on the distilled 4-step Klein, the LoRA may be genuinely broken. Retrain on Base.

### "Images are washed out / colors are wrong"

Weight decay is probably too high. Check if it's at the default 0.0001 and reduce to 0.00001.

### "The concept is there but weak"

Undertrained. Check validation loss — if it's still dropping, keep training. If your dataset is small (<15 images), add more images.

### "The model only produces my training images"

Overtrained. Your validation loss curve would show this as a rise after the minimum. Use an earlier checkpoint. If all checkpoints show this, reduce rank, reduce LR, or add more images.

### "Training loss keeps spiking"

LR too high. Reduce from 1e-4 to 5e-5. If using 9B Base, this model is inherently more brittle — consider switching to 4B Base.

### "Out of memory during training"

In order of things to try:

1. Enable text encoder caching
2. Enable latent caching
3. Enable gradient checkpointing
4. Reduce rank to 16
5. Reduce resolution to 768×768
6. Enable FP8/NF4 quantization of frozen weights
7. Use a cloud GPU

### "My Flux.1 LoRAs don't work on Klein"

Correct. They are completely incompatible — different architecture, different text encoder, different latent space. You need to retrain from scratch.

### "My Klein 4B LoRA doesn't work on FLUX.2 dev"

Also correct. Klein uses Qwen3 text encoders (Qwen3-4B for Klein 4B, Qwen3-8B for Klein 9B); dev uses Mistral Small 3.1 24B. They're different model families despite sharing the Flux name. LoRAs don't transfer between them.

## 15. Tools and environment

### Training tools with confirmed Klein Base support

**Ostris AI Toolkit** — Best out-of-box experience for Klein. Explicit Klein Base model selection in config. Supports all parameters discussed here. Actively maintained.

**HuggingFace Diffusers** — Official FLUX.2 pipeline support and training scripts. Check diffusers documentation for the current pipeline class name. Most control, least convenience. Good if you want to modify training loops.

**SimpleTuner** — Full-featured, supports validation image generation, separate dimension configs, aggressive VRAM optimization. Good for the spacepxl-style deterministic validation workflow.

**Kohya SS / Musubi Tuner** — Community standard for SD/SDXL, Klein support is emerging. May have issues with Qwen3 text encoder loading. Check for recent updates before committing.

**fal.ai** — Cloud training API. $0.0046/step for 4B. Good if you don't have local hardware. Less control over training dynamics.

**RunComfy** — Cloud platform with guided Klein training workflows. Convenient but less configurable.

### Inference / evaluation tools

**ComfyUI** — Full Klein support for all variants. The standard for local evaluation. Make sure you're using Klein-specific workflow nodes, not Flux.1 nodes.

**HuggingFace Diffusers** — Programmatic inference via the FLUX.2 pipeline. Good for scripted batch evaluation of checkpoints.

---

## Summary: The shortest possible version

1. **Train on Klein 4B Base** (not distilled, not 9B unless you need it)
2. **AdamW8bit, LR 1e-4, weight decay 0.00001**
3. **LoRA rank 16–32, alpha = rank/2, target attention + MLP**
4. **Network dims linear_dim=128, linear_alpha=64** if your tool supports it (conv settings are no-ops on Klein)
5. **Cache text embeddings and latents** to fit in 24GB
6. **15–30 images minimum**, natural language captions
7. **Deterministic validation loss** — split your data, measure periodically, stop at the minimum
8. **Evaluate at 50 steps, CFG 4.0, LoRA strength 0.7–0.8** — not distilled settings
9. **Weight decay matters more than learning rate** — get it right (0.00001)
10. **When in doubt, check the validation loss curve.** Everything else is vibes.

# Pipeline Orchestration

Main entry:
- `sd::runPipeline()`

---

# Execution stages

1. Prompt preparation
2. Text encoding (CLIP / SDXL dual encoder)
3. Latent initialization
   - **txt2img**: pure Gaussian noise at `sigmas[0]`
   - **img2img**: `encodeImage()` → posterior mean latent → add noise at `sigmas[startStep]`
4. Denoising loop (steps `startStep … numSteps-1`)
5. CFG guidance
6. Scheduler step updates
7. VAE decode
8. Image output

---

# Denoising loop

For each timestep:

- Run UNet (uncond)
- Run UNet (cond)
- Apply CFG blending
- Update latent

---

# Context object

All state is stored in:
- `GenerationContext`

Includes:
- latents
- embeddings
- model sessions
- dtype flags
- cancellation state

---

# Img2img

Controlled by two fields in `GenerationParams`:

| Field | Type | Meaning |
|---|---|---|
| `initImagePath` | `std::string` | Path to input image; empty = txt2img |
| `strength` | `float` [0, 1] | Denoising fraction; 1.0 = full noise (= txt2img) |

**Start step**: `startStep = int((1 - strength) * numSteps)`, clamped to `[0, numSteps-1]`.

**Latent init**: the input image is loaded and encoded **once before the per-image loop** via `sd::encodeImage()` with `sample=false` (posterior mean — deterministic). The same `initLatent` is reused for every image in the batch; per-image variety comes from the noise added at `sigmas[startStep]`, not from re-encoding. Noise is added at `sigmas[startStep]` rather than `sigmas[0]`.

**Requires**: `vae_encoder.onnx` in the model directory. If absent, `runPipeline()` logs a warning and falls back to txt2img. Use `scripts/export_vae_encoder.py` to add it to an existing model.

---

# Determinism

Pipeline is deterministic given:
- same seed
- same scheduler
- same model + LoRA set
- same `initImagePath` and `strength` (img2img)

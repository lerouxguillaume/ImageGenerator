# Pipeline Orchestration

Main entry:
- `sd::runPipeline()`

---

# Execution stages

| Stage | `GenerationStage` value | Notes |
|---|---|---|
| Model load | `LoadingModel` | Fast when cached by `ModelManager` |
| Text encoding | `EncodingText` | CLIP or SDXL dual encoder |
| Image encoding | `EncodingImage` | img2img only; absent for txt2img |
| Denoising | `Denoising` | `progressStep` counter is live here |
| VAE decode | `DecodingImage` | |
| Post-processing | `PostProcessing` | Alpha cutout, processed PNG, metadata sidecar |
| Complete | `Done` | Set by `GenerationService::run()` before calling `onResult` |

`runPipeline()` writes `stage` atomically at each transition. `GenerationService::run()` owns `PostProcessing` and `Done` since post-processing happens after the pipeline returns. Callers that don't need stage reporting pass `nullptr`.

Exceptions thrown anywhere inside `run()` or `runCandidateRun()` are caught by the service, which calls `onError(message)` instead of propagating. The caller thread is always left in a clean state.

For **candidate runs** `CandidateRunPipeline` owns the stage and sets coarser transitions (`Exploring → Scoring → Refining → WritingManifest → Done`). It does **not** pass the stage into its inner `generateFromPrompt` calls, so fine-grained pipeline stages do not bleed into a candidate run.

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

**Requires**: `vae_encoder.onnx` in the model directory. New full exports include it by default. If absent, `runPipeline()` logs a warning and falls back to txt2img; use `scripts/export_vae_encoder.py` only for older model directories.

---

# Determinism

Pipeline is deterministic given:
- same seed
- same scheduler
- same model + LoRA set
- same `initImagePath` and `strength` (img2img)

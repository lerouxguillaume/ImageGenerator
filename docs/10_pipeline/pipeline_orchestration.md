# Pipeline Orchestration

Main entry:
- `sd::runPipeline()`

---

# Execution stages

1. Prompt preparation
2. Text encoding (CLIP / SDXL dual encoder)
3. Latent initialization
4. Denoising loop
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

# Determinism

Pipeline is deterministic given:
- same seed
- same scheduler
- same model + LoRA set
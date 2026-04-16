# Architecture Summary

## What this file explains
How the codebase is structured and how components interact.

## When to use this
- Locating where to implement a change
- Understanding module boundaries
- Avoiding architectural violations

---

## High-level structure

The application follows a modular architecture with clear separation between UI, orchestration, and computation.

### Layers

- **View (SFML)**  
  Rendering only. No logic.

- **Controller**  
  Handles input, triggers actions, manages state transitions.

- **Presenter**  
  Stateless transformations applied to views.

---

## Core modules

### Pipeline (`src/portraits/sd/`)
Responsible for image generation.

Subcomponents:
- Scheduler — timestep / sigma progression
- Text encoder — converts prompt to embeddings
- UNet — denoising loop with CFG
- VAE — latent → image decoding

---

### Model system

- Loads ONNX models
- Builds execution sessions
- Handles caching via ModelManager

---

### LoRA system

- Parses LoRA weights
- Matches keys to ONNX tensors
- Applies deltas before session creation

---

### LLM subsystem (`src/llm/`)

- Optional
- Transforms prompts via local model
- Stateless interface (`transform()`)

---

### UI system (`src/views/`, `src/ui/widgets/`)

- SFML-based rendering
- Custom widgets (e.g. MultiLineTextArea)
- Controller-driven interaction

---

## Data flow

User input  
→ Controller  
→ (optional) LLM transform  
→ Pipeline  
→ ModelManager (load/cache)  
→ UNet loop  
→ VAE decode  
→ Image output

---

## Key abstractions

- `GenerationContext` — runtime state for pipeline
- `ModelInstance` — loaded sessions
- `ModelCacheKey` — cache identity
- `LoraInjector` — LoRA application engine

---

## Architectural invariants

- No business logic in views
- Pipeline is stateless per run
- ModelManager owns session lifecycle
- LoRA is resolved before session creation

---

## Related files

- [system_overview.md](system_overview.md)
- [build_system.md](build_system.md)
- [../10_pipeline/pipeline_orchestration.md](../10_pipeline/pipeline_orchestration.md)
- [../10_pipeline/cancellation.md](../10_pipeline/cancellation.md)
- [../10_pipeline/gpu_fallback.md](../10_pipeline/gpu_fallback.md)
- [../20_models/model_loading.md](../20_models/model_loading.md)
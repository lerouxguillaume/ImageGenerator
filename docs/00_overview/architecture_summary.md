# Architecture Summary

Image generator is structured around a strict separation of:

- UI layer (SFML)
- Control layer (controllers)
- Presentation layer (state mutation)
- Inference layer (ONNX runtime pipeline)
- Preset layer (file-backed configuration persistence)

---

# MVC structure

View:
- Pure rendering
- No business logic
- SFML drawing only

Controller:
- Input handling
- State transitions
- Triggering SD generation

Presenter:
- Stateless mutations on view state
- Formatting and UI updates

---

# SD pipeline architecture

The pipeline is a linear graph:

CLIP → UNet (CFG loop) → Scheduler → VAE → Image

Each component is:
- stateless per call
- driven by GenerationContext
- cached via ModelManager

---

# Model system architecture

Two-layer model system:

1. SdLoader
  - Resolves model.json
  - Builds ModelInstance

2. ModelManager
  - Caches ModelInstances by ModelCacheKey
  - Prevents redundant session creation

---

# LoRA architecture

LoRA injection occurs at model load time:

- ONNX external initializer patching
- weight merging via safetensors
- suffix-based tensor matching

No runtime graph mutation occurs.

---

# Execution isolation model

Each generation run:
- clones session options
- resets run_opts
- isolates cancellation state

---

# Key design decision

The system prioritizes:
- determinism over flexibility
- caching over recomputation
- static graphs over dynamic graph editing
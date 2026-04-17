# Architecture Summary

Image generator is structured around a strict separation of:

- UI layer (SFML)
- Control layer (controllers)
- Inference layer (ONNX runtime pipeline)
- Preset layer (file-backed configuration persistence)

---

# Component structure

View (`ImageGeneratorView`):
- Thin composition root — holds five panel instances
- `render()` delegates to each panel; no logic

Panels (own state + rendering + event handling):
- `MenuBar` — top bar: Back, title, Presets dropdown, Settings
- `SettingsPanel` — left panel: model, prompts, sliders, seed, LoRA
- `ResultPanel` — right panel: image display, Generate/Cancel, progress
- `LlmBar` — bottom bar: LLM toggle, instruction, Enhance
- `SettingsModal` — settings overlay modal

Controller (`ImageGeneratorController`):
- Thin coordinator: routes events to panels, acts on their action flags
- Owns async operations: model/LoRA scan, LLM load, folder browse, generation thread
- Accesses panels directly via `view.panelName.*`

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
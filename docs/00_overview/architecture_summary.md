# Architecture Summary

Image generator is structured around a strict separation of:

- UI layer (SFML)
- Control layer (controllers)
- Prompt DSL layer (parse / compile / merge)
- Inference layer (ONNX runtime pipeline)
- Preset layer (file-backed configuration persistence)

---

# Component structure

Views:
- `MenuView` — top-level launcher with `Generate Images` and `Edit Image`
- `ImageGeneratorView(Generate)` — prompt-first txt2img composition root
- `ImageGeneratorView(Edit)` — image-first img2img composition root
- Each `ImageGeneratorView` remains thin and delegates `render()` to shared panels

Panels (own state + rendering + event handling):
- `MenuBar` — top bar: Back, title, optional Presets dropdown, Settings
- `SettingsPanel` — left panel; generate mode shows prompts/preview, edit mode shows source/edit controls
- `ResultPanel` — right panel: image display, Generate/Cancel, progress, gallery actions
- `LlmBar` — bottom bar: LLM toggle, instruction, Enhance (generate screen only)
- `SettingsModal` — settings overlay modal

Controller (`ImageGeneratorController`):
- One instance per workflow mode (`Generate`, `Edit`)
- Thin coordinator: routes events to panels, acts on their action flags
- Owns async operations: model/LoRA scan, LLM load, folder browse, generation thread
- Updates DSL display state (token chips, compiled preview) every frame in generate mode
- Accesses panels directly via `view.panelName.*`

---

# Prompt DSL architecture

```
raw text (UI)
    ↓ PromptParser::parse()
Prompt DSL { subject, styles, positive[], negative[] }
    ↓ PromptMerge::merge()  ← optional LLM patch
merged Prompt DSL
    ↓ PromptCompiler::compile(dsl, ModelType)
prompt string → inference pipeline
```

The DSL is re-parsed from text areas on every relevant operation (generation, preset save,
LLM enhancement, per-frame display update). It is not stored as persistent UI state.

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

# Key design decisions

The system prioritizes:
- determinism over flexibility
- caching over recomputation
- static graphs over dynamic graph editing
- DSL as source of truth for prompts, not raw strings

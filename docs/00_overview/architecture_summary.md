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
- `MenuView` — top-level launcher with `Generate Images` and `Import Model`
- `ImageGeneratorView` — the single generate/edit composition root; txt2img by default, img2img when an input image is attached
- `ImageGeneratorView` remains thin and delegates `render()` to shared panels

Panels (own state + rendering + event handling):
- `MenuBar` — top bar: Back, title, Presets dropdown, Settings
- `SettingsPanel` — left panel: prompts/preview, model + LoRA selectors, sliders; shows the input-image/strength controls when an image is attached
- `ResultPanel` — right panel: image display, Generate/Cancel, progress, gallery actions (incl. **Edit** = attach the shown image as img2img input)
- `LlmBar` — bottom bar: LLM toggle, instruction, Enhance (shown when a prompt enhancer is loaded)
- `SettingsModal` — settings overlay modal

Controller (`ImageGeneratorController`):
- A single instance drives the one generate/edit screen
- Thin coordinator: routes events to panels, acts on their action flags
- Owns async operations: model/LoRA scan, LLM load, folder browse, generation thread
- Updates DSL display state (token chips, compiled preview) every frame
- Accesses panels directly via `view.panelName.*`

Import system:
- `MenuController` owns `ModelImporter`, `ImportedModelRegistry`, and `ImportModelModal`
- `ModelImporter` runs the full import flow on a background thread (inspect → venv setup → subprocess → validate → register)
- `ImportModelModal` is an overlay modal; controller calls `renderOverlay()` each frame after `menuScreen.render()`
- Venv is placed in `localDataDir()` (platform-appropriate local path) to avoid shared-drive limitations

UI theme system:
- `Theme` centralizes `UiColors`, `UiMetrics`, and `UiTypography`
- shared rendering helpers and newer views consume theme tokens directly
- `src/enum/constants.hpp` remains as compatibility glue while older code paths migrate

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

# Generation service architecture

`GenerationService` sits between the controller and the pipeline. It owns:
- `PortraitGeneratorAi::generateFromPrompt()` invocation
- The final `Done` stage transition after the pipeline returns
- Error handling — exceptions are caught internally and routed through typed callbacks

**Progress** is reported via `GenerationProgress` — three nullable atomic pointers:

| Field | Type | Owner |
|---|---|---|
| `step` | `atomic<int>*` | denoising step counter (1…numSteps) |
| `currentImage` | `atomic<int>*` | 1-based image index |
| `stage` | `atomic<GenerationStage>*` | typed pipeline phase (see `src/enum/enums.hpp`) |

**Typed result/error events** are delivered via callbacks passed alongside progress:

| Struct | Callback | Fired when |
|---|---|---|
| `GenerationCallbacks` | `onResult(GenerationResult)` | generation completed successfully |
| `GenerationCallbacks` | `onError(std::string)` | exception caught during generation |

`GenerationService::run()` returns `void`. The controller registers `onError` to write to `ResultPanel` atomics; `onResult` is optional (gallery refresh is driven by `generationDone` polled in `update()`).

Generation stage sequence: `LoadingModel → EncodingText → (EncodingImage) → Denoising → DecodingImage → Done`

**Controller thread boundary**: `ImageGeneratorController::startGenerationTask` is a minimal thread spawner — it owns the `jthread` lifecycle and sets `generationDone` after the task returns. Exception handling and result routing are entirely the service's responsibility.

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

# System Overview

Image generator is a C++17 / SFML application embedding a full Stable Diffusion inference stack using ONNX Runtime (no Python runtime dependency).

It supports:
- Stable Diffusion 1.5
- Stable Diffusion XL (SDXL)
- LoRA injection via external initializer patching
- Optional LLM prompt transformation via ORT GenAI

---

# Core runtime flow

1. User enters prompt (UI layer)
2. Optional LLM transforms prompt
3. SD pipeline executes:
    - CLIP encoding
    - UNet denoising loop
    - CFG guidance
    - Scheduler (DPM++ 2M Karras)
    - VAE decode
4. Image saved via OpenCV

Entry point:
- `PortraitGeneratorAi::generateFromPrompt()`
- `sd::runPipeline()`

---

# Execution model

The system is fully synchronous per generation request:

- One active pipeline per request
- Cached models reused via ModelManager
- GPU/CPU fallback handled per session

---

# Key design constraints

- No Python dependency at runtime
- ONNX Runtime is the only inference backend
- LoRA is injected at session creation (not runtime graph editing)
- SDXL and SD1.5 share pipeline but diverge at encoding stage

---

# Major subsystems

- SD Pipeline → `sd/`
- Model system → `ModelManager`, `SdLoader`
- LoRA system → `LoraInjector`
- ONNX parsing → external tensor resolution
- UI system → SFML widgets
- LLM system → optional ORT GenAI wrapper
- Preset system → `PresetManager`, file-backed reusable generation configs

---

# Critical invariants

- All models must be loaded via `ModelManager`
- All sessions are cached by `ModelCacheKey`
- All LoRA modifications occur before session creation
- All dtype selection must use runtime detection
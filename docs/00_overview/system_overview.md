# System Overview

Image generator is a C++20 / SFML application embedding a full Stable Diffusion inference stack using ONNX Runtime (no Python runtime dependency).

It supports:
- Stable Diffusion 1.5
- Stable Diffusion XL (SDXL)
- LoRA injection via external initializer patching
- Optional LLM prompt transformation via ORT GenAI
- Structured Prompt DSL with model-specific compilation

---

# Core runtime flow

1. User enters prompt text (UI layer)
2. `PromptParser::parse()` converts raw text → `Prompt` DSL
3. Optional LLM transforms the DSL (merge, not replace)
4. At generation time, `PromptCompiler::compile(dsl, modelType)` produces the final string
5. SD pipeline executes:
    - CLIP encoding
    - UNet denoising loop
    - CFG guidance
    - Scheduler (DPM++ 2M Karras)
    - VAE decode
6. Image saved via OpenCV

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
- Prompt DSL is model-agnostic; model-specific logic lives only in the compiler

---

# Major subsystems

- SD Pipeline → `sd/`
- Model system → `ModelManager`, `SdLoader`
- LoRA system → `LoraInjector`
- ONNX parsing → external tensor resolution
- UI system → SFML widgets (resizable window, min 700×550)
- LLM system → optional ORT GenAI wrapper
- Prompt DSL → `src/prompt/` — parse, compile, merge, JSON serialisation
- Preset system → `PresetManager`, DSL-backed reusable generation configs

---

# Critical invariants

- All models must be loaded via `ModelManager`
- All sessions are cached by `ModelCacheKey`
- All LoRA modifications occur before session creation
- All dtype selection must use runtime detection
- Prompt string is output only — `Prompt` DSL is the internal representation

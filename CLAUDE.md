# CLAUDE.md

Guidance for Claude Code when working in this repository.

This file is a **retrieval router**, not a full reference.  
For implementation details, always follow links into `/docs`.

---

# What this project is

**ImageGenerator** — C++17 / SFML game with embedded Stable Diffusion inference (ONNX Runtime, no Python).  
Supports:
- SD 1.5
- SDXL
- Local LoRA injection
- Optional local LLM prompt enhancement (ORT GenAI)

---

# Core architecture (high level)

MVC per screen:

- View → `src/views/` (render only)
- Controller → `src/controllers/` (input + orchestration)
- Presenter → `src/presenters/` (state mutations only)

---

# Major subsystems

## 1. SD Pipeline
Entry:
- `sd/SdPipeline.cpp → runPipeline()`

Key responsibilities:
- CLIP encode
- UNet denoising loop
- CFG guidance
- Scheduler (DPM++ 2M Karras)
- VAE decode

📘 Details:
→ docs/10_pipeline/pipeline_orchestration.md  
→ docs/10_pipeline/scheduler.md  
→ docs/10_pipeline/cfg_guidance.md  
→ docs/10_pipeline/sdxl_specifics.md  
→ docs/10_pipeline/dtype_handling.md  
→ docs/10_pipeline/cancellation.md  
→ docs/10_pipeline/gpu_fallback.md

---

## 2. Model system
- Model loading via `SdLoader.cpp`
- Session cache via `ModelManager`
- Model selection via `model.json`

📘 Details:
→ docs/20_models/model_loading.md  
→ docs/20_models/model_cache.md  
→ docs/20_models/adding_new_model.md

---

## 3. LoRA system
- Injected at model load time via ORT external initializers
- No runtime graph modification
- Cached via `LoraInjector`

📘 Details:
→ docs/30_lora/lora_overview.md  
→ docs/30_lora/lora_key_format.md  
→ docs/30_lora/lora_matching.md  
→ docs/30_lora/lora_caching.md  
→ docs/30_lora/lora_debugging.md

---

## 4. ONNX / external weights system
- External tensors parsed from ONNX protobuf
- `.onnx.data` handled by ORT
- LoRA uses suffix-index matching

📘 Details:
→ docs/40_onnx/external_data.md  
→ docs/40_onnx/onnx_indexing.md  
→ docs/40_onnx/onnx_parsing.md

---

## 5. Export pipeline
- SD 1.5 + SDXL export scripts
- ONNX validation step included
- fp16 fixes for SDXL required

📘 Details:
→ docs/50_export/export_overview.md  
→ docs/50_export/export_validation.md  
→ docs/50_export/sdxl_fp16_fixes.md

---

## 6. LLM prompt enhancement (optional)
- ORT GenAI backend (`OrtLlmEnhancer`)
- Stateless transform API
- JSON-based output parsing

📘 Details:
→ docs/60_llm/llm_overview.md  
→ docs/60_llm/prompt_transform.md  
→ docs/60_llm/json_parsing.md

---

## 7. UI system
- SFML-based
- Custom widgets (no immediate mode GUI)
- `MultiLineTextArea` owns full input state

📘 Details:
→ docs/70_ui/ui_architecture.md  
→ docs/70_ui/multiline_textarea.md

---

# Build system

- CMake project
- Uses `GLOB_RECURSE`
- Requires rebuild when adding `.cpp` files

Flags:
- `USE_CUDA=ON` → GPU EP for all models
- `USE_DML=ON` → DirectML for UNet only

📘 Details:
→ docs/00_overview/build_system.md

---

# Model behavior

Model selection via `model.json`:
```json
{ "type": "sdxl" }
```
Absent = SD 1.5 assumed. Drives resolution, encoder count, and UNet input shape.

---

# Debugging

→ docs/90_debug/common_failures.md  
→ docs/90_debug/performance.md

---

# Overview docs

→ docs/00_overview/system_overview.md  
→ docs/00_overview/architecture_summary.md  
→ docs/00_overview/build_system.md

---

# Critical rules

- Never hard-code fp16 dtype — use `ctx.unetExpectsFp32` / `ctx.vaeExpectsFp32`
- Never pass `ctx.run_opts` to VAE `Run()` — use `Ort::RunOptions{nullptr}`
- Never reset `ctx.run_opts` manually in `runPipeline()` — `ModelManager::get()` owns this
- Never call `cpu_unet` with `ctx.session_opts` — must use `ctx.cpu_session_opts`
- Never call `AddExternalInitializers` on shared opts — `Clone()` first
- Never use `operator[]` on `ModelManager` cache — use `emplace + std::move`
- Never create local `LoraInjector` — use references from `s_injectors`
- Never destroy `LoraOverrides` before `Ort::Session` constructor returns
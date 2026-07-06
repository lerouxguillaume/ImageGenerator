# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**ImageGenerator** — C++20 / SFML application with embedded Stable Diffusion inference (ONNX Runtime). Python is used only by the model import pipeline, never at runtime.

This file is a **retrieval router** only. All implementation details live in `docs/`.

---

# Build commands

```bash
# Configure (required after adding/removing .cpp files — project uses GLOB_RECURSE)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (incremental)
cmake --build build

# CPU + CUDA (Linux)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
      -DONNXRUNTIME_ROOT=~/libs/onnxruntime/onnxruntime-linux-x64-gpu-1.24.4
cmake --build build

# Run
./build/image_generator
```

There are no automated tests. Verification is done by running the binary.

The binary expects `models/vocab.json` and `models/merges.txt` in the working directory (CMake copies them post-build). Logs go to `image_generator.log`.

---

# Retrieval rules

- `docs/` is the source of truth — always follow links
- CLAUDE.md is an index only — never treat it as documentation
- Prefer `docs/` over CLAUDE.md for any implementation detail
- Priority order: lowest-numbered folder wins (00 > 10 > 20 > 30 > 90)
- If no doc covers a topic, treat behavior as unknown

---

# Subsystem index

## Overview & architecture
→ docs/00_overview/system_overview.md  
→ docs/00_overview/architecture_summary.md

## Build system
→ docs/00_overview/build_system.md

## Generation service — `GenerationService`
→ `src/controllers/GenerationService.hpp` (typed job structs)

Key facts:
- `GenerationJob` carries prompt, negative prompt, output path, params, model dir, and the `loraCompatible` capability flag (VAE-encoder availability is derived inside the pipeline from `model.json`, not passed on the job)
- `GenerationService::run()` handles single/multi-image generation; it returns `void`
- `GenerationService` uses `job.loraCompatible` to clear any LoRA entries defensively — it does not silently fall back
- `GenerationProgress` holds three nullable atomic pointers: `step` (denoising step counter), `currentImage` (1-based image index), and `stage` (`atomic<GenerationStage>*`) — all owned by `ResultPanel`
- `GenerationCallbacks` carry typed `onResult` / `onError` functors; the service catches all exceptions internally and routes them through `onError` — exceptions never reach the caller thread
- `ImageGeneratorController::startGenerationTask` is a minimal thread spawner — it owns the `jthread` lifecycle and sets `generationDone`; it has no exception handling or business logic
- `cachedModelType_` is updated in `update()` on model change; `launchGeneration` and `launchEnhancement` use it — `inferModelType()` is only called from `update()`
- `GenerationStage` enum is in `src/enum/enums.hpp`; generation: `LoadingModel → EncodingText → (EncodingImage) → Denoising → DecodingImage → Done`
- `GenerationResult` returns raw output paths
- The gallery is a flat, newest-first scan of `config.outputDir` (`.png`/`.jpg`); generated images are written directly there as `img_<timestamp>.png`

## SD Pipeline — `sd/SdPipeline.cpp → runPipeline()` (txt2img + img2img)
→ docs/10_pipeline/pipeline_orchestration.md  
→ docs/10_pipeline/scheduler.md  
→ docs/10_pipeline/cfg_guidance.md  
→ docs/10_pipeline/sdxl_specifics.md  
→ docs/10_pipeline/dtype_handling.md  
→ docs/10_pipeline/cancellation.md  
→ docs/10_pipeline/gpu_fallback.md

## Model system — `SdLoader.cpp`, `ModelManager`
→ docs/20_models/model_loading.md  
→ docs/20_models/model_cache.md  
→ docs/20_models/adding_new_model.md

## LoRA system — `LoraInjector`
→ docs/30_lora/lora_overview.md  
→ docs/30_lora/lora_key_format.md  
→ docs/30_lora/lora_matching.md  
→ docs/30_lora/lora_caching.md  
→ docs/30_lora/lora_debugging.md

## ONNX / external weights
→ docs/40_onnx/external_data.md  
→ docs/40_onnx/onnx_indexing.md  
→ docs/40_onnx/onnx_parsing.md

## Export pipeline
→ docs/50_export/export_overview.md  
→ docs/50_export/export_validation.md  
→ docs/50_export/sdxl_fp16_fixes.md

## Import pipeline — `src/import/` (`SafetensorsInspector`, `ModelImporter`, `PythonEnvManager`, `ImportedModelRegistry`)
→ docs/55_import/import_pipeline.md

Key facts:
- Entry point: Menu → "Import Model" button → `ImportModelModal`
- Python is required to import, not to run — managed venv is set up automatically on first use
- Venv lives in `localDataDir()`, not next to the binary, to avoid shared-drive failures on Windows (`%LOCALAPPDATA%\ImageGenerator\python_env\`)
- Full setup log at `<localDataDir>/python_env_setup.log` — check this when import fails
- Imported models land in `models/imported/<id>/` with an extended `model.json` (`capabilities` block)
- Registry at `models/imported/registry.json` — updated after each successful import
- Never read `ModelImporter::getOutputDir()` / `getModelId()` before `State::Done`
- **Model discovery is registry-only** — `ImageGeneratorController` reads `ImportedModelRegistry` to populate the model list; there is no filesystem scan of a model base directory
- The registry is watched by mtime in `ImageGeneratorController::update()` — newly imported models appear automatically when the user navigates to the generator screen
- `ImportedModel` carries `ModelCapabilities` (`vaeEncoderAvailable`, `loraCompatible`) populated from `model.json` at registry load time; defaults to `true` for models without a capabilities block
- `SettingsPanel` exposes `modelVaeEncoderAvailable` / `modelLoraCompatible` vectors (parallel to `availableModels`) and helpers `currentModelVaeEncoderAvailable()` / `currentModelLoraCompatible()`; the LoRA button is hidden and the strength slider is replaced with a note when the selected model lacks the capability

## LLM prompt enhancement — `OrtLlmEnhancer` (optional)
→ docs/60_llm/llm_overview.md  
→ docs/60_llm/prompt_transform.md  
→ docs/60_llm/json_parsing.md

## UI system — SFML, `MultiLineTextArea` (resizable window, token chips, compiled preview)
→ docs/70_ui/ui_architecture.md  
→ docs/70_ui/multiline_textarea.md

UI theme facts:
- `Theme` is now the authoritative UI styling source for colors, metrics, and typography
- `constants.hpp` is compatibility glue, not the long-term styling source
- prefer reading current UI behavior from `docs/70_ui/ui_architecture.md` before inferring from older helper usage

## Preset system — `PresetManager` (DSL-backed)
→ docs/80_presets/preset_overview.md

## Prompt DSL — `src/prompt/` (parse / compile / merge / JSON)
→ docs/85_prompt/prompt_dsl.md

## Debugging
→ docs/90_debug/common_failures.md  
→ docs/90_debug/performance.md

---

# Critical invariants

- Never hard-code fp16 dtype — use `ctx.unetExpectsFp32` / `ctx.vaeExpectsFp32` / `ctx.vaeEncoderExpectsFp32`
- Never hard-code `0.18215f` as the VAE scaling factor — use `ctx.vaeScalingFactor` (SDXL default is `0.13025`)
- Never pass `ctx.run_opts` to VAE `Run()` — use `Ort::RunOptions{nullptr}`
- Never call `encodeImage()` without checking `ctx.vaeEncoderAvailable` first — session is null when `vae_encoder.onnx` is absent
- Never reset `ctx.run_opts` manually in `runPipeline()` — `ModelManager::get()` owns this
- Never call `cpu_unet` with `ctx.session_opts` — must use `ctx.cpu_session_opts`
- Never call `AddExternalInitializers` on shared opts — `Clone()` first
- Never use `operator[]` on `ModelManager` cache — use `emplace + std::move`
- Never create local `LoraInjector` — use references from `s_injectors`
- Never destroy `LoraOverrides` before `Ort::Session` constructor returns
- Never store a raw prompt string as internal state — use `Prompt` DSL
- Never replace LLM-enhanced text directly — always merge via `PromptMerge::merge()`
- Never scan a model base directory for `unet.onnx` — use `ImportedModelRegistry::list()` for all model discovery
- Never derive model display name from `path.filename()` — use `availableModelNames` from `SettingsPanel` (parallel to `availableModels`)

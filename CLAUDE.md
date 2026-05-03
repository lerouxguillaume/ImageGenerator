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

## Generation service — `GenerationService`, `CandidateRunPipeline`
→ `src/controllers/GenerationService.hpp` (typed job structs)  
→ `src/controllers/CandidateRunPipeline.cpp` (candidate-run execution)

Key facts:
- `GenerationJob` carries prompt, negative prompt, output path, params, model dir, and post-process spec
- `CandidateRunJob` carries the full candidate pipeline spec (patron path, run dirs, explore/refine counts, scoring spec)
- `GenerationService::run()` handles reference normalization, single/multi-image generation, and post-processing; both `run()` and `runCandidateRun()` return `void`
- `GenerationProgress` holds three nullable atomic pointers: `step` (denoising step counter), `currentImage` (1-based image index), and `stage` (`atomic<GenerationStage>*`) — all owned by `ResultPanel`
- `GenerationCallbacks` / `CandidateRunCallbacks` carry typed `onResult` / `onError` functors; the service catches all exceptions internally and routes them through `onError` — exceptions never reach the caller thread
- `ImageGeneratorController::startGenerationTask` is a minimal thread spawner — it owns the `jthread` lifecycle and sets `generationDone`; it has no exception handling or business logic
- `GenerationStage` enum is in `src/enum/enums.hpp`; standard generation: `LoadingModel → EncodingText → (EncodingImage) → Denoising → DecodingImage → PostProcessing → Done`; candidate runs: `Exploring → Scoring → Refining → WritingManifest → Done`
- Candidate runs do NOT forward `stage` into inner `generateFromPrompt` calls — coarse outer stages own the label
- `GenerationResult` returns raw output paths and a `referenceUsed` flag

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

## Project system — `src/projects/` (`Project`, `AssetType`, `ProjectManager`)
→ docs/75_projects/project_overview.md  
→ docs/features/auto_generate.md (candidate-run generation workflow)

Key facts:
- `ProjectView` is a first-class themed asset-pack workspace, not just a launcher into the standalone generator
- The embedded project workspace reuses `ImageGeneratorController` for generation, settings modal, and gallery behavior, but owns its own theme/asset authoring UI
- Built-in asset templates live in `AssetTypeTemplate.*` and are applied only at asset-type creation time
- Gallery is scoped to the currently selected asset type subfolder in the project workspace; project asset selection happens in `ProjectView`, not through `ResultPanel::tabs`
- `ProjectController` still uses `ResolvedProjectContext` instead of reading `ProjectManager` from `ImageGeneratorController`
- Never access `ProjectManager` from `ImageGeneratorController` — use `ResolvedProjectContext` as the data carrier
- Never bake constraint tokens into `stylePrompt` on the project struct — keep them separate so text areas show only user-authored content
- Current `wall_left` generation uses `GenerationWorkflow::CandidateRun`
- Candidate runs write to `runs/<run_id>/explore/` and `runs/<run_id>/refine/`, plus `manifest.json`
- Candidate scoring lives in `src/assets/CandidateScorer.*`; it applies `AssetExportSpec::alphaCutout` before geometric scoring
- Generated asset processing lives in `src/assets/GeneratedAssetProcessor.*`; standard and candidate-run flows share this for alpha cutout, processed output, standalone transparent derivatives when applicable, and metadata sidecars
- Asset output paths and gallery discovery live in `src/assets/AssetArtifactStore.*`; do not rebuild `raw/`, `processed/`, `runs/`, `.reference_cache/`, or `patron.png` paths in controllers
- Candidate-run counts, strengths, and score threshold are persisted on `AssetType::candidateRun` and copied through `ResolvedProjectContext`
- Scoring coordinates are in the asset generation canvas space, e.g. the `wall_left` template uses 512×768
- Each `CandidateRun` asset type has a patron at `output/<project>/<asset>/patron.png` — generated from `AssetSpec`, used as img2img seed for exploration; regenerated on orientation or bounds change
- `PatronGenerator` draws orientation-specific patrons for `LeftWall`, `RightWall`, `FloorTile`, and `Character`; `Unset` and `Prop` use the rectangle fallback
- Never regenerate the patron inside the generation thread — `launchCandidateRun` reads it; `ProjectController::refreshPatron` writes it

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

# CLAUDE.md

**ImageGenerator** — C++20 / SFML application with embedded Stable Diffusion inference (ONNX Runtime, no Python).

This file is a **retrieval router** only. All implementation details live in `docs/`.

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

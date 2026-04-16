# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

**GuildMaster** — a C++17 / SFML fantasy game with an embedded Stable Diffusion pipeline for generating character portraits locally. The SD inference is implemented from scratch on top of ONNX Runtime (no Python at runtime). Supports SD 1.5 and SDXL.

## Architecture

```
MVC pattern per screen:
  View   (src/views/)       — SFML rendering only; no logic
  Controller (src/controllers/) — input dispatch, per-frame update
  Presenter  (src/presenters/)  — stateless state mutations on the view

UI widgets (src/ui/widgets/):
  MultiLineTextArea.hpp/.cpp — self-contained text area widget (cursor, word-wrap, copy/paste, scroll)

LLM prompt transform (src/llm/):
  IPromptEnhancer.hpp        — pure interface: LLMRequest/LLMResponse, transform(), isAvailable()
  OrtLlmEnhancer.hpp/.cpp    — ORT GenAI-backed implementation (compiled with -DUSE_GENAI)
  NullPromptEnhancer.hpp     — no-op stub when LLM is disabled

SD pipeline (src/portraits/):
  PortraitGeneratorAi.cpp   — thin public shell; delegates to sd::runPipeline()
  sd/SdPipeline.cpp         — denoising loop + top-level orchestration
  sd/ModelManager.cpp       — multi-entry session cache keyed by ModelCacheKey
  sd/SdLoader.cpp           — model config detection + ORT session loading + external-index cache
  sd/SdTextEncoder.cpp      — text encoding (SD 1.5 single, SDXL dual)
  sd/SdUNet.cpp             — CFG UNet passes + DML GPU fallback
  sd/SdVae.cpp              — VAE latent decode
  sd/SdScheduler.cpp        — DPM++ 2M Karras sigma schedule
  sd/SdTypes.hpp            — ModelConfig + GenerationContext + ModelInstance structs
  sd/SdUtils.hpp            — inline helpers (timing, fp16, RNG, image conversion)
  sd/SdOnnxPatcher.hpp      — shared types: OnnxModelBundle, ExternalTensorMeta, OnnxExternalIndex, ParsedLora…
  sd/OnnxParser.cpp         — resolveBundle + parseExternalIndex (protobuf wire-format parsing)
  sd/OnnxIndex.cpp          — buildExternalSuffixIndex (O(1) suffix lookup table)
  sd/LoraParser.cpp         — parseLoraLayers (kohya key → LoraLayer grouping)
  sd/LoraMath.cpp           — computeLoraDelta (matmul + scale in fp32)
  sd/SdLoraMatch.hpp/.cpp   — matchExternalLoraKey (suffix-index lookup + ambiguity detection)
  sd/LoraInjector.hpp/.cpp  — LoraInjector: loads companion _weights.safetensors, applies deltas, caches merged tensors
  sd/SdSafetensors.hpp      — safetensors loader + fp16/bf16 conversion helpers
  Pipeline: CLIP tokenize → text encode → DPM++ 2M Karras loop → VAE decode → cv::imwrite
```

## Build system

CMake with `USE_CUDA=ON` (all models) and `USE_DML=ON` (UNet only) flags. CMake uses `GLOB_RECURSE` — **re-run `cmake -B build` after adding any new `.cpp` file**.  
→ [`docs/00_overview/build_system.md`](docs/00_overview/build_system.md)

## Key files

| File | Role |
|---|---|
| `src/portraits/PortraitGeneratorAi.hpp` | Public API: `GenerationParams`, `generateFromPrompt()` |
| `src/portraits/PortraitGeneratorAi.cpp` | Thin shell — builds prompt, calls `sd::runPipeline()` |
| `src/portraits/sd/SdTypes.hpp` | `ModelConfig`, `GenerationContext`, `ModelInstance` structs |
| `src/portraits/sd/ModelManager.hpp/.cpp` | `unordered_map<ModelCacheKey, ModelInstance>` session cache |
| `src/portraits/sd/SdLoader.cpp` | `loadModelConfig()`, `loadModels()` → `ModelInstance`, external-index cache, `logModelIO()` |
| `src/portraits/sd/SdTextEncoder.cpp` | `encodeText()` (SD 1.5), `encodeTextSDXL()` (SDXL) |
| `src/portraits/sd/SdUNet.cpp` | `runUNetSingle()`, `runUNetCFG()`, DML GPU fallback |
| `src/portraits/sd/SdVae.cpp` | `decodeLatent()` |
| `src/portraits/sd/SdScheduler.cpp` | `buildAlphasCumprod()`, `buildKarrasSchedule()`, `sigmaToTimestep()` |
| `src/portraits/sd/SdPipeline.cpp` | `denoiseSingleLatent()`, `runPipeline()` |
| `src/portraits/sd/SdOnnxPatcher.hpp` | Shared types: `OnnxModelBundle`, `ExternalTensorMeta`, `OnnxExternalIndex`, `ParsedLora` |
| `src/portraits/sd/OnnxParser.cpp` | `resolveBundle()`, `parseExternalIndex()` — filesystem probe + protobuf wire-format parser |
| `src/portraits/sd/OnnxIndex.cpp` | `buildExternalSuffixIndex()` — O(1) suffix lookup table |
| `src/portraits/sd/LoraParser.cpp` | `parseLoraLayers()` — kohya key grouping into `LoraLayer` triplets |
| `src/portraits/sd/LoraMath.cpp` | `computeLoraDelta()` — matmul + scale in fp32 |
| `src/portraits/sd/SdLoraMatch.hpp/.cpp` | `matchExternalLoraKey()` — suffix-index lookup with ambiguity detection |
| `src/portraits/sd/LoraInjector.hpp/.cpp` | `LoraInjector` — loads `_weights.safetensors`, applies deltas, caches merged tensors, injects via `AddExternalInitializers` |
| `src/portraits/sd/SdUtils.hpp` | Inline helpers: `fmtMs`, `toFp16`, `randNormal`, `latentToImage` |
| `src/portraits/ClipTokenizer.cpp/.hpp` | BPE tokenizer (no Python dependency) |
| `src/portraits/PromptBuilder.hpp` | Weighted A1111-style prompt builder |
| `src/config/AppConfig.hpp/.cpp` | `AppConfig` (modelBaseDir, outputDir, loraBaseDir, …) + JSON load/save |
| `src/enum/enums.hpp` | All enums (`ModelType`, `Race`, `Gender`, …) |
| `src/enum/constants.hpp` | Colour palette (`Col::`) and layout constants |
| `src/views/ImageGeneratorView.hpp` | All generation state including atomics; holds `positiveArea`, `negativeArea`, `instructionArea` |
| `src/llm/IPromptEnhancer.hpp` | `LLMRequest`, `LLMResponse`, `IPromptEnhancer` interface |
| `src/llm/OrtLlmEnhancer.hpp/.cpp` | ORT GenAI LLM backend; `transform()`, `isAvailable()` |
| `src/llm/NullPromptEnhancer.hpp` | Passthrough stub when LLM is disabled |
| `src/ui/widgets/MultiLineTextArea.hpp/.cpp` | Reusable text area widget: cursor, word-wrap, copy/paste, scroll |
| `scripts/export_onnx_models.py` | SD 1.5 → ONNX export (`--resume`, `--validate`) |
| `scripts/sdxl_export_onnx_models.py` | SDXL → ONNX export (`--resume`, `--validate`) |
| `scripts/export_common.py` | Shared export utilities: `ExportComponentSpec`, policies, graph fixes, ORT validation |
| `scripts/check_lora_compat.py` | LoRA ↔ ONNX compatibility checker (no inference required) |

## Model detection

The application reads `<modelDir>/model.json` at load time:
```json
{ "type": "sdxl" }
```
If absent, SD 1.5 is assumed. This drives resolution (512 vs 1024), dual-encoder loading, and SDXL UNet extra inputs.

## SD pipeline internals

**Scheduler:** DPM++ 2M Karras — `buildKarrasSchedule()` returns `num_steps+1` sigmas (last = 0).  
→ [`docs/10_pipeline/scheduler.md`](docs/10_pipeline/scheduler.md)

**CFG:** Two separate batch=1 UNet passes per step (uncond then cond), blended as `eps = eps_uncond + guidance_scale × (eps_cond - eps_uncond)`. Required for DML which only supports static batch size 1.

**Cancellation:** Watcher thread polls `cancelToken` every 20 ms, calls `ctx.run_opts.SetTerminate()`. VAE is exempt (uses `Ort::RunOptions{nullptr}`). `ModelManager::get()` resets `run_opts` before every run.  
→ [`docs/10_pipeline/cancellation.md`](docs/10_pipeline/cancellation.md)

**GPU fallback:** If GPU UNet throws, `dmlFailed = true` and all subsequent steps use `cpu_unet`. Persists in cached `GenerationContext`.  
→ [`docs/10_pipeline/gpu_fallback.md`](docs/10_pipeline/gpu_fallback.md)

**dtype:** Detected at load time via `GetInputTypeInfo(0)`. Use `ctx.unetExpectsFp32` / `ctx.vaeExpectsFp32` — never hard-code.  
→ [`docs/10_pipeline/dtype_handling.md`](docs/10_pipeline/dtype_handling.md)

**SDXL:** 128×128 latent, dual encoders, extra UNet inputs (`text_embeds`, `time_ids`).  
→ [`docs/10_pipeline/sdxl_specifics.md`](docs/10_pipeline/sdxl_specifics.md)

## Logging

`Logger::info()` writes to `image_generator.log`. All pipeline stages log timing and tensor stats (min/max/mean). NaN or extreme values in eps or latent stats indicate numerical issues; wrong dtype errors surface as ORT exceptions.

## LoRA support

LoRA is applied at model load time via `AddExternalInitializers` — no LoRA code runs at inference time. LoRAs are selected from the UI panel; selected files + scales are passed into `GenerationParams.loras`.

Key docs:
- Architecture and flow → [`docs/30_lora/lora_overview.md`](docs/30_lora/lora_overview.md)
- Kohya key format and matching → [`docs/30_lora/lora_key_format.md`](docs/30_lora/lora_key_format.md)
- Suffix index (O(1) lookup) → [`docs/30_lora/lora_matching.md`](docs/30_lora/lora_matching.md)
- Caching (session + injector) → [`docs/30_lora/lora_caching.md`](docs/30_lora/lora_caching.md)
- Debugging mismatches → [`docs/30_lora/lora_debugging.md`](docs/30_lora/lora_debugging.md)

## Export

→ [`docs/50_export/export_overview.md`](docs/50_export/export_overview.md) — scripts, `--resume`, LoRA companion weights  
→ [`docs/50_export/export_validation.md`](docs/50_export/export_validation.md) — `--validate`, manifest validation  
→ [`docs/50_export/sdxl_fp16_fixes.md`](docs/50_export/sdxl_fp16_fixes.md) — fp32 contamination fixes, pre/post-export passes  
→ [`docs/40_onnx/external_data.md`](docs/40_onnx/external_data.md) — `.onnx.data` layout and consolidation

## LLM prompt transform

Optional local LLM prompt enhancement via ORT GenAI. Compiled with `-DUSE_GENAI`; degrades to `NullPromptEnhancer` without it. Interface: `IPromptEnhancer::transform(LLMRequest) → LLMResponse`. No `enhance()` method.  
→ [`docs/60_llm/llm_overview.md`](docs/60_llm/llm_overview.md)  
→ [`docs/60_llm/prompt_transform.md`](docs/60_llm/prompt_transform.md)  
→ [`docs/60_llm/json_parsing.md`](docs/60_llm/json_parsing.md)

## UI widgets

`MultiLineTextArea` — self-contained widget owning all cursor/scroll/text state. Use public API only (`setText`, `setActive`, `handleEvent`, `handleScroll`, `handleClick`).  
→ [`docs/70_ui/multiline_textarea.md`](docs/70_ui/multiline_textarea.md)

## How to navigate docs

**00_overview/** — [system_overview](docs/00_overview/system_overview.md) · [architecture_summary](docs/00_overview/architecture_summary.md) · [build_system](docs/00_overview/build_system.md)

**10_pipeline/** — [pipeline_orchestration](docs/10_pipeline/pipeline_orchestration.md) · [scheduler](docs/10_pipeline/scheduler.md) · [dtype_handling](docs/10_pipeline/dtype_handling.md) · [sdxl_specifics](docs/10_pipeline/sdxl_specifics.md) · [cancellation](docs/10_pipeline/cancellation.md) · [gpu_fallback](docs/10_pipeline/gpu_fallback.md)

**20_models/** — [model_loading](docs/20_models/model_loading.md) · [model_cache](docs/20_models/model_cache.md) · [adding_new_model](docs/20_models/adding_new_model.md)

**30_lora/** — [lora_overview](docs/30_lora/lora_overview.md) · [lora_key_format](docs/30_lora/lora_key_format.md) · [lora_caching](docs/30_lora/lora_caching.md) · [lora_debugging](docs/30_lora/lora_debugging.md)

**40_onnx/** — [onnx_indexing](docs/40_onnx/onnx_indexing.md) · [external_data](docs/40_onnx/external_data.md)

**50_export/** — [export_overview](docs/50_export/export_overview.md) · [export_validation](docs/50_export/export_validation.md) · [sdxl_fp16_fixes](docs/50_export/sdxl_fp16_fixes.md)

**60_llm/** — [llm_overview](docs/60_llm/llm_overview.md) · [prompt_transform](docs/60_llm/prompt_transform.md) · [json_parsing](docs/60_llm/json_parsing.md)

**70_ui/** — [ui_architecture](docs/70_ui/ui_architecture.md) · [multiline_textarea](docs/70_ui/multiline_textarea.md)

**90_debug/** — [common_failures](docs/90_debug/common_failures.md) · [performance](docs/90_debug/performance.md)

**Retrieval hints:**
- dtype mismatch ORT error at inference → `10_pipeline/dtype_handling.md`
- scheduler sigma or timestep question → `10_pipeline/scheduler.md`
- cancelled generation still ran / cancel had no effect → `10_pipeline/cancellation.md`
- generation silently switched to CPU → `10_pipeline/gpu_fallback.md`
- SDXL shape error or dual-encoder question → `10_pipeline/sdxl_specifics.md`
- LoRA 0 patches / high miss count → `30_lora/lora_debugging.md` + `30_lora/lora_key_format.md`
- fp16 ORT error during export → `50_export/sdxl_fp16_fixes.md`
- first image slow after model load → `90_debug/performance.md`
- adding a new model family → `20_models/adding_new_model.md`

## What NOT to do

- The SD 1.5 VAE export uses `dynamic_axes` for height/width. Do not remove those dynamic axes.
- Do not use `ctx.run_opts` for the VAE `Run()` call — it would be affected by `SetTerminate()`.
- Do not call `cpu_unet` with `ctx.session_opts` — it must always use `ctx.cpu_session_opts` (no GPU EP).
- Do not hard-code fp16 for VAE input — use `ctx.vaeExpectsFp32`.
- Do not reset `ctx.run_opts` manually in `runPipeline()` — `ModelManager::get()` already does this.
- Do not call `SessionOptions::AddExternalInitializers` on the shared `unetOpts`/`auxOpts` — always `Clone()` first.
- Do not destroy a `LoraOverrides` before the `Ort::Session` constructor returns — `Ort::Value` objects hold non-owning views into its backing buffers.
- Do not use `operator[]` to insert into `ModelManager`'s internal cache — use `emplace` with `std::move`.
- Do not create local `LoraInjector` instances inside `loadModels()` — always use references from `s_injectors`.
- Do not add `dataOffset` / `dataLength` back to `ExternalTensorMeta` — base weights come from `_weights.safetensors`, not `.onnx.data`.
- Do not add an `enhance()` method to `IPromptEnhancer` — `transform()` is the only mode.
- Do not render or route clicks to `instructionArea` when `promptEnhancerAvailable` is false.
- Do not call `computeLines()` or `drawPromptField()` — those functions have been removed.
- Do not manipulate `MultiLineTextArea` cursor/scroll fields directly — use the public API.

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

CMake with two key options:
- `USE_CUDA=ON` — enables CUDA EP for all models (UNet, VAE, text encoders)
- `USE_DML=ON` — enables DirectML EP for UNet only (VAE and text encoders stay on CPU due to Reshape node issues)

Default ORT paths are set in CMakeLists.txt but can be overridden with `-DONNXRUNTIME_ROOT=`.

CMake uses `GLOB_RECURSE` for sources — **re-run `cmake -B build` after adding new `.cpp` files**.

Linux builds: `cmake -B build && cmake --build build -j$(nproc)`  
Windows cross-compile from Linux: uses `cmake/mingw-w64.cmake` toolchain.

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

**Scheduler:** DPM++ 2M Karras (`sd/SdScheduler.cpp`). `buildKarrasSchedule()` returns `num_steps+1` sigmas (last = 0).

**CFG:** Two separate batch=1 UNet passes per step (uncond then cond), blended as:
`eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)`
This is required for DirectML which only supports static batch size 1.

**Cancellation:** A watcher `std::thread` polls `cancelToken` every 20 ms. When set, it calls `ctx.run_opts.SetTerminate()` which aborts any in-flight ORT `Run()` using those RunOptions. The VAE uses `Ort::RunOptions{nullptr}` (not `run_opts`) so it is not affected by cancellation.  
`ModelManager::get()` resets `ctx.run_opts = Ort::RunOptions{}` at the start of every run (cache hit or miss) so a previous `SetTerminate()` never carries over.

**GPU fallback:** If the GPU UNet throws, `dmlFailed` is set to true and all subsequent steps run on `cpu_unet` (a separate CPU session loaded at startup). `dmlFailed` persists in the cached `GenerationContext` across runs — once failed, always CPU.

## dtype handling

Models are queried at load time for their input dtype:
- `ctx.unetExpectsFp32` — detected from `unet.GetInputTypeInfo(0)`
- `ctx.vaeExpectsFp32` — detected from `vae_decoder.GetInputTypeInfo(0)`

SD 1.5 export: UNet = fp16, VAE = fp16  
SDXL export: UNet = fp16, VAE = fp16

Always use the detected flag rather than hard-coding a dtype — the export scripts may change.

## SDXL fp16 export: fp32 contamination

Diffusers 0.37.x injects fp32 operations in several places when tracing an fp16 UNet. ORT rejects the resulting mixed-type graph at model load time. The export pipeline applies fixes in two layers: **pre-export patches** (monkey-patch diffusers modules before tracing so the tracer never emits mixed types) and **post-export passes** (targeted ONNX graph surgery for issues that cannot be fixed before tracing).

### Pre-export patches (applied in `sdxl_export_onnx_models.py` before `from_single_file`)

| Source | Location | Fix |
|---|---|---|
| `get_timestep_embedding` | `torch.arange(..., dtype=float32)` and `timesteps[:, None].float()` | `patch_fp32_upcasts_for_tracing()` replaces both with the input tensor's dtype |
| `FP32SiLU` | `F.silu(inputs.float()).to(inputs.dtype)` in the timestep MLP | `patch_fp32_upcasts_for_tracing()` replaces its forward with plain `F.silu(inputs)` |
| `Attention.upcast_attention/upcast_softmax` | Per-module flags that cast query/key/scores to fp32 | `disable_attention_upcasting(pipe.unet)` sets both flags to `False` on every `Attention` module before `unet.to(float16)` |

### Post-export passes (applied by `export_component_to_dir`, enabled via `ExportComponentSpec` flags)

| Issue | ORT error | Fix function | Flag |
|---|---|---|---|
| Stray scalar fp32 Constant nodes | `"Type parameter (T) … bound to different types"` on Mul/Add | `fix_fp32_constants` — converts fp32 `t` (TensorProto), `value_float`, and `value_floats` attributes on Constant nodes to fp16 | `fix_fp32_constants=True` |
| Attention scale `Sqrt → Cast(fp32)` | `"tensor(float16) and tensor(float)"` on Div in attention blocks | `fix_attention_sqrt_cast_fp32` — rewrites any `Cast(to=float32)` whose direct input is a `Sqrt` node to `Cast(to=float16)` | `fix_attention_sqrt_cast=True` |
| Resize with fp16 data or scales input | `"Type 'tensor(float16)' … of operator (Resize) is invalid"` | `fix_resize_fp16_input` — (1) wraps data input (input[0]) with `Cast(fp16→fp32)` + `Cast(fp32→fp16)`; (2) wraps dynamic scales input (input[2]) with `Cast(fp16→fp32)` when non-empty (ONNX Resize requires scales to be fp32) | `fix_resize_fp16=True` |

All three post-export passes are enabled for the SDXL UNet via `SDXLExportPolicy`.

### Why not a blanket Cast-to-fp32 rewrite?

A broad pass that rewrites all `Cast(to=float32)` → `Cast(to=float16)` nodes breaks the `Resize` operator: ONNX `Resize` requires its `scales` input (input[2]) to be `float32`. The post-export passes above are each scoped to a specific structural pattern that cannot appear in Resize scale paths:
- `fix_fp32_constants` only touches `Constant` nodes (literal values, never dynamic scale tensors)
- `fix_attention_sqrt_cast_fp32` only touches `Cast` nodes whose immediate parent is `Sqrt` (attention scale computation, never Resize scales which come from Constant nodes)
- `fix_resize_fp16_input` wraps the Resize *data* input (input[0]) with Cast(fp16↔fp32) and, when present, wraps the *scales* input (input[2]) with Cast(fp16→fp32) — dynamic scale tensors produced by Concat/Shape ops in fp16 UNets also need this treatment

### Diagnosing new fp16 type errors

If a new `"Type parameter (T) bound to different types"` or `"Type … is invalid"` error appears:

1. Find the failing node name in the ORT error
2. Use `scripts/onnx_check.py` to trace both inputs back to their producing nodes and inspect dtypes
3. Identify whether the fp32 source is a Constant node, a Cast node, or a diffusers upcast flag
4. For Constant nodes: `fix_fp32_constants` already handles all three attribute forms (`t`, `value_float`, `value_floats`)
5. For a Cast node: check its parent op — add a targeted rewrite function following the pattern of `fix_attention_sqrt_cast_fp32`
6. For a diffusers upcast flag: add a pre-export monkey-patch following the pattern of `patch_fp32_upcasts_for_tracing` or `disable_attention_upcasting`

## Export script features

### Resume (`--resume`)

Both export scripts accept `--resume`. If all output files for a component already exist (`<name>.onnx`, `<name>.onnx.data`, and `<name>_weights.safetensors` when applicable), that component is skipped. Use this to continue an interrupted export without re-running already-completed components.

```bash
python scripts/export_onnx_models.py model.safetensors --resume
```

### ORT validation (`--validate`)

After exporting each component, loads the ONNX file in ORT on CPU and runs a forward pass with the same dummy inputs used for export. Catches dtype mismatches, missing external data, and output shape errors before they surface in the C++ pipeline. Slow for large models (SDXL UNet loads all weights and may take several minutes).

```bash
python scripts/sdxl_export_onnx_models.py model.safetensors --validate
```

### LoRA companion weights

`ExportComponentSpec(export_lora_weights=True)` saves `<name>_weights.safetensors` alongside the ONNX file. Contains all 2-D (linear) parameters from `model.named_parameters()`, which match ONNX initializer names exactly when exported with `keep_initializers_as_inputs=True`. Required for `LoraInjector` to function.

### LoRA compatibility checker

```bash
python scripts/check_lora_compat.py <lora.safetensors> <model_dir> [--verbose]
```

Runs the same suffix-matching logic as the C++ `LoraInjector` and reports how many layer triplets would be applied per component. Each LoRA key is counted once globally — matched if any component claims it, missed if none do. Use `--verbose` to see unmatched keys grouped by Kohya prefix.

```
  text_encoder.onnx  :  66 matched  (180 initializers)
  unet.onnx          : 192 matched  (686 initializers)
  vae_decoder.onnx   :   0 matched  (expected — VAE has no LoRA targets)

✅  258/264 unique layers matched (97%)
```

## SDXL export: external data layout

The legacy ONNX tracer writes each large tensor as a separate sidecar file (e.g. `text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight`). `export_component_to_dir` calls `consolidate_external_data` after export, which:
1. Reads the per-tensor sidecar files and copies raw bytes (no tensor deserialisation)
2. Consolidates them into a single `<component>.onnx.data` file
3. Deletes the individual sidecar files

The output directory should contain exactly `<component>.onnx` + `<component>.onnx.data` after a successful export. If stale sidecar files remain, they are orphaned and safe to delete manually.

## Known DML limitations (Windows)

- **SDXL UNet** falls back to CPU under DML due to `node_view_1` (Reshape nodes that DML rejects at runtime even with `ORT_ENABLE_BASIC`).
- **VAE decoder** always runs on CPU under DML due to `node_view_2` (same class of issue in the VAE attention block).
- `session_opts` for DML has `ORT_ENABLE_BASIC`, `DisableMemPattern()`, and `ORT_SEQUENTIAL` — do not change these without testing on Windows.

## SDXL specifics

- Latent: 128×128 (vs 64×64 for SD 1.5)
- Text encoding: CLIP-L (768-dim) + OpenCLIP-G (1280-dim) hidden states concatenated → (1, 77, 2048)
- Extra UNet inputs: `text_embeds` (pooled output of encoder 2, shape (1, 1280)) and `time_ids` (shape (1, 6) = `[orig_h, orig_w, crop_top, crop_left, target_h, target_w]`)
- Both text encoders share the same token IDs (single tokenizer, 77 tokens)

## Logging

`Logger::info()` writes to `image_generator.log`. All pipeline stages log timing and tensor stats (min/max/mean). If something breaks silently, check the log — NaN or extreme values in eps or latent stats indicate numerical issues; wrong dtype errors surface as ORT exceptions.

## How to add a new model

The pipeline is split so that adding a new model family (e.g. SD 2.1, LCM, FLUX) only touches a few focused files. Everything else — scheduler, VAE, denoising loop, GPU/CPU fallback, cancellation — stays unchanged.

### Step 1 — Register the model type

In `src/enum/enums.hpp`, add a value to `ModelType`:
```cpp
enum class ModelType { SD15, SDXL, MyNewModel };
```

### Step 2 — Detect it from model.json

In `sd/SdLoader.cpp`, extend `loadModelConfig()`:
```cpp
if (type == "mynewmodel") {
    cfg.type    = ModelType::MyNewModel;
    cfg.image_w = 768;   // set correct native resolution
    cfg.image_h = 768;
}
```
If the new model needs a different scheduler (e.g. different beta schedule), add fields to `ModelConfig` in `sd/SdTypes.hpp` and set them here.

### Step 3 — Load the right sessions

In `sd/SdLoader.cpp`, inside `loadModels()`, add any extra session loads gated on the new type (same pattern as the SDXL `text_encoder_2` block). Store extra I/O names in `GenerationContext` if needed, adding fields to `sd/SdTypes.hpp`.

### Step 4 — Text encoding

In `sd/SdTextEncoder.cpp`, add `encodeTextMyNewModel()` if the encoder differs. Declare it in `sd/SdTextEncoder.hpp`.

### Step 5 — Wire it into the pipeline

In `sd/SdPipeline.cpp`, inside `runPipeline()`, branch on `cfg.type`:
```cpp
if (cfg.type == ModelType::SDXL) {
    ctx.text_embed = encodeTextSDXL(...);
    ctx.time_ids   = {h, w, 0, 0, h, w};
} else if (cfg.type == ModelType::MyNewModel) {
    ctx.text_embed = encodeTextMyNewModel(...);
} else {
    ctx.text_embed = encodeText(...);  // SD 1.5 default
}
```
If the new model needs extra UNet inputs, add them to `GenerationContext` and extend `runUNetSingle()` in `sd/SdUNet.cpp` with an analogous `appendMyNewModelInputs()` lambda.

### Step 6 — Export script

Add `scripts/mynewmodel_export_onnx_models.py` following the same pattern as the existing scripts. Write `model.json` with the new type key so the loader detects it automatically.

## LoRA support

### UI selection

LoRAs are selected at runtime from the image generator screen. No config file editing required.

- **LoRA directory** — configured in the Settings modal (`loraBaseDir`, default `"loras"`). The controller scans it for `*.safetensors` files on startup and after each settings save.
- **LoRA panel** — "LoRA (N)" button on the model row toggles a floating panel listing all discovered files. Each row has a checkbox toggle and an editable scale field (default `1`).
- **Generation** — selected LoRAs with their scales are passed directly into `GenerationParams.loras` when Generate is clicked.

Relevant state lives in `ImageGeneratorView`: `availableLoras`, `loraSelected`, `loraScales`, `loraScaleInputs`, `showLoraPanel`, `activeLoraScaleIdx`.  
Scanning and click handling are in `ImageGeneratorController::update()` / `handleClick()`.

### Architecture

LoRA is applied at model load time using ORT-native external data loading and `AddExternalInitializers`. No LoRA-specific code runs at inference time. No ONNX binary reconstruction is performed at runtime.

```
SdPipeline.cpp  runPipeline()
  → ModelManager::get()                — looks up ModelCacheKey in unordered_map; calls loadModels() on miss

SdLoader.cpp  loadModels() → ModelInstance / makeLoraSession()
  → LoraInjector::loadModelMetadata()  — parse .onnx once; cache external-tensor index  [LoraInjector.cpp]
       → parseExternalIndex()          — protobuf scan for field-14 EXTERNAL tensors  [OnnxParser.cpp]
       → buildExternalSuffixIndex()    — O(1) suffix lookup over the external index  [OnnxIndex.cpp]
  → LoraInjector::applyLoras()         — load _weights.safetensors + compute merged tensors (cached)
       → ensureBaseWeights()           — load <name>_weights.safetensors once per model file
       → computeMerge()                — for each LoRA: parseLoraLayers → matchExternalLoraKey → computeLoraDelta
            → parseLoraLayers()        — group safetensors keys into (down, up, alpha) triplets  [LoraParser.cpp]
            → matchExternalLoraKey()   — O(1) suffix-index lookup  [SdLoraMatch.cpp]
            → computeLoraDelta()       — matmul + scale in fp32  [LoraMath.cpp]
       → buildOverrides()              — copy CachedTensors → LoraOverrides backing buffers + Ort::Value views
  → SessionOptions::Clone()            — clone EP opts so AddExternalInitializers doesn't pollute shared state
  → cloned.AddExternalInitializers()   — inject patched tensors; ORT loads the rest from .onnx.data natively
  → Ort::Session(env, path, cloned)    — session created from .onnx path
```

Base weights come from `<name>_weights.safetensors` written by the export scripts.  The C++ code never reads `.onnx.data` directly — ORT remains the sole component responsible for loading external tensor data.

Key files:
- `sd/SdOnnxPatcher.hpp` — shared types: `ExternalTensorMeta`, `OnnxExternalIndex`, `OnnxExternalSuffixIndex`, `ParsedLora`/`LoraLayer`
- `sd/OnnxParser.cpp` — `parseExternalIndex` (field-14/EXTERNAL scan)
- `sd/OnnxIndex.cpp` — `buildExternalSuffixIndex`, `buildSuffixIndex`
- `sd/LoraParser.cpp` — `parseLoraLayers` (kohya prefix stripping + triplet grouping)
- `sd/LoraMath.cpp` — `computeLoraDelta` (fp32 matmul)
- `sd/SdLoraMatch.hpp/.cpp` — `matchExternalLoraKey`, `matchLoraKey`; optional `SD_LORA_MATCH_DEBUG` flag
- `sd/LoraInjector.hpp/.cpp` — `LoraInjector`: metadata loading, base weight caching, delta computation, merged-tensor cache (FNV-1a keyed), NaN/Inf guard, `LoraOverrides` output
- `sd/SdSafetensors.hpp` — safetensors loader + fp16/bf16 conversion helpers
- `sd/SdLoader.cpp` `makeLoraSession()` lambda — orchestrates injector→clone→session
- `sd/ModelManager.hpp/.cpp` — `unordered_map<ModelCacheKey, ModelInstance>` session cache

### Model and session caching

**Session cache (`ModelManager`)** — `SdPipeline.cpp` holds a `static ModelManager`. On each `runPipeline()` call, `ModelManager::get(cfg, modelDir, loras)` looks up a `ModelCacheKey` in an `unordered_map<ModelCacheKey, ModelInstance>`. On hit the stored `ModelInstance` is reused; on miss `loadModels()` is called, a new `ModelInstance` is emplaced, and the result is returned. Multiple distinct configurations can coexist in the cache simultaneously.

`ModelCacheKey` is constructed via `ModelCacheKey::make()` which enforces three invariants before the key is stored:
- **Canonical paths** — `modelDir` and every LoRA path are passed through `std::filesystem::weakly_canonical()` so `./models/sd15` and `models/sd15` map to the same key.
- **Sorted LoRA list** — entries are sorted by canonical path, making `{A, B}` and `{B, A}` identical keys.
- **Fixed-point scales** — scales are compared and hashed as `int(std::lround(scale * 1000))`. `lround` rounds rather than truncating, so 0.9999f → 1000 not 999. Values within 0.0005 of each other are treated as equal.

`ModelCacheKeyHash` feeds a single canonical string buffer (`modelDir\0type\0path\0scale\0...`) into **XXH64** (`libxxhash`). This avoids depending on the quality of `std::hash<std::string>` which varies across stdlibs.

**Persistent `LoraInjector` instances (`SdLoader.cpp`)** — `SdLoader.cpp` holds a `static std::unordered_map<std::string, LoraInjector> s_injectors` keyed by `.onnx` path. `loadModels()` obtains references (`LoraInjector& teInjector = s_injectors[teBundle.onnxPath]`) instead of creating local instances. This means the injector's metadata and base-weight caches persist across `loadModels()` calls (i.e. across LoRA config changes), so the expensive first-time setup runs only once per component per process lifetime.

**`LoraInjector` internal caches** — each `LoraInjector` instance (one per component: text encoder, UNet) holds two caches:
- `extIndex_` / `extSuffixIndex_` — populated once by `loadModelMetadata()` when the `.onnx` path is first seen; reused for every subsequent LoRA apply call on the same component.
- `mergeCache_` — maps a FNV-1a hash of `(onnxPath, weightsPath, loras)` → `vector<CachedTensor>`. On cache miss: `ensureBaseWeights()` loads `_weights.safetensors` once, then `computeMerge()` runs the matmuls. On cache hit: `buildOverrides()` copies the pre-computed tensors into a new `LoraOverrides` (cheap memcpy).  The cache is guarded by `cacheMutex_`.

No-LoRA path: sessions are created directly from the `.onnx` file path; ORT memory-maps the file and resolves `.onnx.data` natively. No bytes are buffered in the process.

LoRA path memory: external index (small, persistent) + base weights in `baseWeights_` (persistent, ~model-size) + `mergeCache_` entry (persistent, ~patched-tensor-count × tensor-size). Peak during first run ≈ 2× model size (base weights + merge accumulator); subsequent runs reuse the cache and skip the accumulator.

### LoRA key format (Kohya)

Safetensors keys use this naming pattern:
- `lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight`  (CLIP text encoder)
- `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight`  (UNet)
- `lora_te2_...` — second text encoder (SDXL only)

Each layer needs a down, up, and optionally alpha tensor. `parseLoraLayers()` groups these into `LoraLayer` triplets inside a `ParsedLora`.

### ONNX weight name matching

The matching pipeline has three stages:

**1. Parse** — `parseExternalIndex` scans the `.onnx` binary for TensorProto entries where `data_location == EXTERNAL` (field 14, value 1). It normalises all `.` and `/` in initializer names to `_` and records each tensor's `onnxName`, `shape`, and `dtype` (from the TensorProto fields). The `location`/`offset`/`length` values in `external_data` (field 13) are parsed but discarded — base weights come from the companion `_weights.safetensors`, not from `.onnx.data`. Returns `OnnxExternalIndex` (a `std::map<normName, ExternalTensorMeta>`).

**2. Index** — `buildExternalSuffixIndex` walks every normalised name and inserts it under every `_`-boundary suffix into an `OnnxExternalSuffixIndex`. Runs once per model file at load time; cached inside `LoraInjector` (keyed by `.onnx` path).

**3. Match** — `matchExternalLoraKey` (`SdLoraMatch.cpp`) strips the kohya prefix (`lora_te_`, `lora_unet_`, `lora_te2_`), appends `_weight` (then `_bias` on miss), and does a direct O(1) hash map lookup. When multiple candidates exist, the longest suffix match wins. True ties (same suffix length) are logged as warnings.

```
LoRA base:  text_model_encoder_layers_0_self_attn_q_proj
                          ↓  + "_weight"
suffix key: text_model_encoder_layers_0_self_attn_q_proj_weight
                          ↓  unordered_map::find
ONNX entry: text_encoder_text_model_encoder_layers_0_self_attn_q_proj_weight
```

The model-level prefix (`text_encoder_`) is absorbed by the suffix index — no knowledge of it is needed at match time.

### ONNX protobuf field numbers

Relevant TensorProto fields used by `parseExternalIndex`:
- **field 1** — `dims` (repeated varint)
- **field 2** — `data_type` (1 = float32, 10 = float16)
- **field 8** — `name` (string)
- **field 13** — `external_data` (repeated StringStringEntryProto: `location`, `offset`, `length`)
- **field 14** — `data_location` (varint: DEFAULT=0, **EXTERNAL=1**)

PyTorch `torch.onnx.export` places named initializers at **GraphProto field 5** (not field 6 as some tools use). The diagnostic log line `ONNX graph fields (ALL): f1x969 f2x1 f5x180 ...` confirms this — `f5x180` = 180 named initializers. If you see `f5x0 f6x0`, the model was exported without `keep_initializers_as_inputs=True` — re-export with the project's scripts.

### Export requirements for LoRA to work

Models **must** be exported with:
```python
keep_initializers_as_inputs=True   # weights become named field-5 initializers
do_constant_folding=False          # prevents weights becoming anonymous Constant nodes
```
Both flags are set in `scripts/export_common.py` `onnx_export()`. Third-party pre-exported ONNX models typically lack these and will show 0 layers patched.

### Delta formula

```
delta = effectiveScale × (lora_up @ lora_down)
      = (userScale × alpha / rank) × (lora_up @ lora_down)
```

`lora_down`: shape `[rank, in_feat]` — `lora_up`: shape `[out_feat, rank]`

Implemented in `computeLoraDelta(up, down, effectiveScale)` in `LoraMath.cpp`.

### Known bugs (fixed)

**`floatToFp16` subnormal bug** (`SdSafetensors.hpp`):  
The original subnormal path used `shift = 1 + (-14 - exp32)` which was 12 too small, causing values in the fp16 subnormal range (~6e-8 to ~6e-5) to overflow and produce fp16 NaN. Symptom: `LoRA WARNING: X has N NaN/Inf AFTER patching` followed by all-NaN text encoder output. Fixed to `shift = 13 + (-14 - exp32)` with `& 0x3FFu` masking.

### Diagnosing LoRA problems

The log tells you everything. Check in this order:

**1. Are external initializers detected?**
```
parseExternalIndex: text_encoder.onnx — 180 external initializer(s)
```
If you see `0 external initializer(s)`, either the model has inline weights (unsupported by this path — re-export with the project scripts) or the `.onnx` file uses a different field for `data_location` than expected (should be field 14, value 1).

**2. Is the companion weights file present?**
```
LoraInjector: loading base weights from text_encoder_weights.safetensors
LoraInjector: loaded 132 base weight tensor(s)
```
If you see `companion weights file not found`, the model was exported without `export_lora_weights=True`. Re-export with the project scripts.  LoRA injection is silently disabled for that component.

**3. Are weight names matching?**
```
LoraInjector: total 132 patch(es), 0 miss(es)
```
High miss count means the LoRA key suffixes don't align with the ONNX initializer names.  Use `scripts/check_lora_compat.py --verbose` to diagnose without running inference.

**4. Are tensor overrides correctly built?**
```
LoraInjector: 132 tensor override(s) ready
LoraInjector: injected 132 override(s) into session options
```
This confirms `Ort::Value`s were created and registered. A lower count than expected patches means some tensors failed shape validation, had unsupported dtypes, or triggered the NaN/Inf guard.

**5. Does AddExternalInitializers succeed?**
If ORT throws during session creation with a name or shape mismatch error, the `onnxName` stored in `ExternalTensorMeta` does not match what ORT expects. The name comes directly from TensorProto field 8 in the `.onnx` binary — verify with a hex dump or `onnx_check.py`.

**6. Does the text encoder output have NaN?**
```
embedding stats: min=1000000000.000000  max=-1000000000.000000  mean=nan
```
The sentinel values `min=1e9, max=-1e9` mean ALL values are NaN (NaN comparisons return false). Check `floatToFp16` in `SdSafetensors.hpp` if the LoRA delta is being applied to fp16 weights. NaN propagates: text encoder → UNet cross-attention → NaN eps → black image.

**7. Is the session cache stale?**
```
ModelManager: cache hit — reusing loaded sessions.
```
If you changed LoRA selection but this appears, check that `loraScales[i]` was updated after editing the scale input field. The cache key rounds scales to 0.001 precision (`lround(s * 1000)`), so changes smaller than 0.0005 are intentionally treated as the same key. Also verify paths are the same after `weakly_canonical` normalisation.

### Optional match-level debug logging

`SdLoraMatch.cpp` has a compile-time flag for verbose per-key logging:

```cpp
#define SD_LORA_MATCH_DEBUG 1   // add to compile flags: -DSD_LORA_MATCH_DEBUG=1
```

When enabled, every `matchExternalLoraKey()` (and `matchLoraKey()`) call logs the lookup key, all candidates with their suffix lengths, and the chosen winner. Off by default — produces one log line per layer (180+ lines for a full model).

## Session warmup

`SdLoader.cpp` calls `warmupSession()` for every ORT session immediately after creation (`text_encoder`, `unet`, `cpu_unet` if available, `vae_decoder`, `text_encoder_2` for SDXL). The helper:

1. Queries each input's shape from session metadata, replacing dynamic dims (`-1`) with `1`.
2. Allocates zero-filled tensors with the correct dtype (float16, float32, or int64).
3. Calls `session.Run()` once to trigger ORT's JIT kernel compilation.

The warmup is **non-fatal**: any exception is caught and logged as a warning so a warmup failure never blocks model load. Without warmup, the first real inference call pays the JIT cost; with it, that cost is absorbed at load time before the user clicks Generate.

Do not remove warmup calls — the latency reduction is significant for the first image after a model/LoRA change.

## Model manifest validation

`model.json` may include a `"components"` dict written by the export scripts:

```json
{
  "type": "sdxl",
  "components": {
    "text_encoder":   { "onnx": "text_encoder.onnx",   "weights": "text_encoder_weights.safetensors" },
    "text_encoder_2": { "onnx": "text_encoder_2.onnx",  "weights": "text_encoder_2_weights.safetensors" },
    "unet":           { "onnx": "unet.onnx",             "weights": "unet_weights.safetensors" },
    "vae_decoder":    { "onnx": "vae_decoder.onnx" }
  }
}
```

`validateManifest()` in `SdLoader.cpp` is called inside `loadModelConfig()` after JSON parse. It checks whether every declared `onnx` and `weights` file exists on disk and logs `[OK]` / `[MISSING]` for each. Missing files are flagged as warnings — load continues, so a partial or legacy model directory is tolerated, but the log makes it easy to diagnose "model loaded but LoRA has 0 patches" (companion weights missing).

`write_model_json(output_dir, model_type, all_specs)` in `scripts/export_common.py` writes this layout automatically when called with the list of `ExportComponentSpec` objects. Both export scripts (`export_onnx_models.py`, `sdxl_export_onnx_models.py`) pass `all_specs` so the manifest is always up to date.

## LLM prompt transform

The `src/llm/` subsystem provides optional, local LLM-backed prompt improvement. Compiled with `-DUSE_GENAI` to enable; without it everything degrades to passthrough stubs.

### Interface

```cpp
// LLMRequest — passed to transform()
struct LLMRequest {
    std::string prompt;       // original positive prompt
    std::string instruction;  // e.g. "make it cinematic"; empty → transform() uses generic quality default
    ModelType   model;        // SD15 or SDXL — drives output style guidance
    float       strength;     // 0.0–1.0: how strongly to apply the transformation
};

// LLMResponse — returned by transform()
struct LLMResponse {
    std::string prompt;
    std::string negative_prompt;
};
```

`IPromptEnhancer` has two virtual methods: `transform(req)` and `isAvailable()`. There is no `enhance()` — stateless single-turn transform is the only mode.

### Implementations

| Class | File | Behaviour |
|---|---|---|
| `OrtLlmEnhancer` | `OrtLlmEnhancer.hpp/.cpp` | Full implementation via ORT GenAI. `#ifdef USE_GENAI` required. |
| `NullPromptEnhancer` | `NullPromptEnhancer.hpp` | No-op; returns original prompt + fixed negative. Always `isAvailable() == false`. |

The controller holds a `std::unique_ptr<IPromptEnhancer>` and swaps implementations depending on whether an LLM model directory was configured.

### Prompt construction (`OrtLlmEnhancer.cpp`)

`buildTransformPrompt()` assembles a Llama 3 single-turn chat prompt with three guidance paragraphs injected into the system message:

- **`modelTypeGuidance(model)`** — SD 1.5: "write short comma-separated keywords"; SDXL: "write natural-language descriptive sentences"
- **`strengthGuidance(strength)`** — three tiers: ≤0.3 minimal, ≤0.7 moderate, >0.7 strong
- **Instruction** — `effectiveInstruction` from controller: user input > per-model `llmHint` from `AppConfig` > empty (transform() substitutes "Improve the prompt quality and detail")

Output format instructed as strict JSON:
```json
{ "prompt": "...", "negative_prompt": "..." }
```

Generation stops as soon as `}` appears in the output (avoids the model rambling past the JSON close).

### JSON parsing (`parseJsonTransform`)

Three-stage fallback:
1. Parse the full raw string.
2. Extract the substring between the first `{` and last `}`, then parse.
3. Return `nullopt` → caller uses original prompt + default negative.

### `inferModelType()` (controller)

`ImageGeneratorController.cpp` contains a local `inferModelType(modelDir)` helper that reads `<modelDir>/model.json` and returns `ModelType::SDXL` if `"type": "sdxl"` is found, `ModelType::SD15` otherwise. Called once per enhance click to pass the correct `ModelType` into `LLMRequest`.

### Instruction field

`view.instructionArea` is a 500-char, 2-visible-line `MultiLineTextArea`. It is only rendered and only accepts clicks when `view.promptEnhancerAvailable` (or `llmLoading`) is true. When the user leaves it empty, `effectiveInstruction` falls back to the per-model `llmHint` from `AppConfig`, and if that is also empty the transform() implementation applies its own generic default.

---

## UI widgets

### `MultiLineTextArea`

A self-contained text area widget in `src/ui/widgets/MultiLineTextArea.hpp/.cpp`. Owns all text state and input handling; the view and controller do not manipulate cursor/scroll directly.

**Constructor:**
```cpp
explicit MultiLineTextArea(int charLimit = 2000, int visibleLines = 4);
```
`visibleLines` drives rendering height and scroll bounds — the positive/negative areas use 4 lines; the instruction area uses 2.

**Ownership model:**
- Text content, cursor byte offset, all-selected flag
- Word-wrap layout (`VisualLine = {start, end}` byte ranges) — rebuilt each `render()` call
- Vertical scroll position
- Active/focus state

**Key methods:**
```cpp
void render(sf::RenderWindow& win, sf::Font& font) override;  // call setRect() first
void setRect(const sf::FloatRect& rect) override;
bool handleEvent(const sf::Event& e);   // returns true if consumed
void handleScroll(float delta);         // +1 = scroll down, -1 = up
void handleClick(sf::Vector2f pos);     // activates self if pos inside rect
void setText(const std::string& t);
const std::string& getText() const;
bool isActive() const;
void setActive(bool active);            // true → cursor moves to end of text
void setTextColor(sf::Color c);
```

**Keyboard handling (inside `handleEvent`):**
- Arrow keys: Left/Right by character; Up/Down using the word-wrap line layout
- Home/End: jump to start/end of current visual line
- Backspace/Delete: remove character; clears all-selected text first
- Ctrl+A: select all; next keystroke replaces
- Ctrl+C: copy full field text to clipboard
- Ctrl+V: paste from clipboard, filtered to ASCII ≥ 32, capped at `charLimit_`
- TextEntered: filtered to ASCII ≥ 32, enforces `charLimit_`

**Focus / Tab cycle (in controller):**

Tab cycles among the three fields in order: positive → negative → instruction (when LLM available) → positive. Mutual exclusion is enforced by calling `setActive(false)` on all three before activating the next.

**`setRect()` at render time:**

The view calls `setRect({LEFT_X, y, FIELD_W, FIELD_H})` immediately before each `render()` call. Layout constants `FIELD_W=700`, `FIELD_H=86` (4-line fields), `FIELD_H_SM=46` (2-line instruction field) live in `ImageGeneratorView.cpp`.

**View fields:**
```cpp
MultiLineTextArea positiveArea{2000};       // 4 visible lines, 2000 char limit
MultiLineTextArea negativeArea{2000};       // 4 visible lines, 2000 char limit
MultiLineTextArea instructionArea{500, 2};  // 2 visible lines, 500 char limit
```

The old 8 parallel view fields (`positivePrompt`, `negativePrompt`, `positiveCursor`, `negativeCursor`, `positiveActive`, `negativeActive`, `positiveLines`, `negativeLines`, `positiveScrollLine`, `negativeScrollLine`, `positiveAllSelected`, `negativeAllSelected`, `positiveField`, `negativeField`) and the `VisualLine` inner struct and `drawPromptField()` private method have all been removed.

---

## What NOT to do

- The SD 1.5 VAE export now uses `dynamic_axes` for height/width (needed for non-512 resolutions). This is safe because the VAE always loads with `cpu_session_opts` and never goes through DML. Do not remove those dynamic axes.
- Do not use `ctx.run_opts` for the VAE `Run()` call — it would be affected by `SetTerminate()` from the cancel watcher.
- Do not call `cpu_unet` with `ctx.session_opts` — it must always use `ctx.cpu_session_opts` (no GPU EP).
- Do not hard-code fp16 for VAE input — use `ctx.vaeExpectsFp32` to select the right type.
- Do not reset `ctx.run_opts` manually in `runPipeline()` — `ModelManager::get()` already does this on every call before returning the context reference.
- Do not call `SessionOptions::AddExternalInitializers` on the shared `unetOpts`/`auxOpts` references — always `Clone()` first. Mutating a shared `SessionOptions` would affect every subsequent session created with it.
- Do not destroy a `LoraOverrides` before the `Ort::Session` constructor returns — the `Ort::Value` objects hold non-owning views into `LoraOverrides::fp16Bufs`/`fp32Bufs`.
- Do not use `operator[]` to insert into `ModelManager`'s internal cache — `GenerationContext` contains non-copyable ORT sessions. Always use `emplace` with `std::move`.
- Do not create local `LoraInjector` instances inside `loadModels()` — always use references from the static `s_injectors` map so caches survive across LoRA config changes.
- Do not add `dataOffset` or `dataLength` back to `ExternalTensorMeta` — those fields were removed because base weights now come from `_weights.safetensors`, not from `.onnx.data`. The C++ code never reads `.onnx.data` directly.
- Do not add an `enhance()` method back to `IPromptEnhancer` — the stateless `transform()` is the only mode. Empty `instruction` is the signal for generic improvement; the implementation handles it internally.
- Do not render or route clicks to `instructionArea` when `promptEnhancerAvailable` is false — the field only exists in the UI when an LLM model is loaded.
- Do not call `computeLines()` or `drawPromptField()` directly — those functions have been removed; all word-wrap and rendering is owned by `MultiLineTextArea`.
- Do not manipulate `MultiLineTextArea` cursor/scroll fields directly from the view or controller — use the public API (`setText`, `setActive`, `handleEvent`, `handleScroll`, `handleClick`).

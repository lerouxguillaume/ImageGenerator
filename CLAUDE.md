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
  sd/SdLoraApply.hpp/.cpp   — buildLoraOverrides (reads .onnx.data + applies delta + builds Ort::Value overrides)
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
| `src/portraits/sd/SdLoraApply.hpp/.cpp` | `buildLoraOverrides()` — reads base weights from `.onnx.data`, applies deltas, builds `Ort::Value` overrides |
| `src/portraits/sd/SdUtils.hpp` | Inline helpers: `fmtMs`, `toFp16`, `randNormal`, `latentToImage` |
| `src/portraits/ClipTokenizer.cpp/.hpp` | BPE tokenizer (no Python dependency) |
| `src/portraits/PromptBuilder.hpp` | Weighted A1111-style prompt builder |
| `src/config/AppConfig.hpp/.cpp` | `AppConfig` (modelBaseDir, outputDir, loraBaseDir, …) + JSON load/save |
| `src/enum/enums.hpp` | All enums (`ModelType`, `Race`, `Gender`, …) |
| `src/enum/constants.hpp` | Colour palette (`Col::`) and layout constants |
| `src/views/ImageGeneratorView.hpp` | All generation state including atomics |
| `scripts/export_onnx_models.py` | SD 1.5 → ONNX export |
| `scripts/sdxl_export_onnx_models.py` | SDXL → ONNX export |

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
| Resize with fp16 data input | `"Type 'tensor(float16)' … of operator (Resize) is invalid"` | `fix_resize_fp16_input` — inserts `Cast(fp16→fp32)` before and `Cast(fp32→fp16)` after each Resize data input (input[0]) | `fix_resize_fp16=True` |

All three post-export passes are enabled for the SDXL UNet via `SDXLExportPolicy`.

### Why not a blanket Cast-to-fp32 rewrite?

A broad pass that rewrites all `Cast(to=float32)` → `Cast(to=float16)` nodes breaks the `Resize` operator: ONNX `Resize` requires its `scales` input (input[2]) to be `float32`. The post-export passes above are each scoped to a specific structural pattern that cannot appear in Resize scale paths:
- `fix_fp32_constants` only touches `Constant` nodes (literal values, never dynamic scale tensors)
- `fix_attention_sqrt_cast_fp32` only touches `Cast` nodes whose immediate parent is `Sqrt` (attention scale computation, never Resize scales which come from Constant nodes)
- `fix_resize_fp16_input` wraps the Resize *data* input (input[0]), never the scales input (input[2])

### Diagnosing new fp16 type errors

If a new `"Type parameter (T) bound to different types"` or `"Type … is invalid"` error appears:

1. Find the failing node name in the ORT error
2. Use `scripts/onnx_check.py` to trace both inputs back to their producing nodes and inspect dtypes
3. Identify whether the fp32 source is a Constant node, a Cast node, or a diffusers upcast flag
4. For Constant nodes: `fix_fp32_constants` already handles all three attribute forms (`t`, `value_float`, `value_floats`)
5. For a Cast node: check its parent op — add a targeted rewrite function following the pattern of `fix_attention_sqrt_cast_fp32`
6. For a diffusers upcast flag: add a pre-export monkey-patch following the pattern of `patch_fp32_upcasts_for_tracing` or `disable_attention_upcasting`

## SDXL export: external data layout

The legacy ONNX tracer writes each large tensor as a separate sidecar file (e.g. `text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight`). `export_component_to_dir` calls `ensure_external_data` after export, which:
1. Reads the per-tensor files into memory via `load_external_data_for_model`
2. Consolidates them into a single `<component>.onnx.data` file
3. Deletes the individual sidecar files (identified from the model's external-data references, not by prefix heuristics)

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
  → ModelManager::get()              — looks up ModelCacheKey in unordered_map; calls loadModels() on miss

SdLoader.cpp  loadModels() → ModelInstance / makeLoraSession()
  → cachedExternalIndex()            — parse .onnx once; cache {normName → offset/length/shape/dtype}
       → parseExternalIndex()        — protobuf scan for field-14 EXTERNAL tensors  [OnnxParser.cpp]
  → buildExternalSuffixIndex()       — O(1) suffix lookup over the external index  [OnnxIndex.cpp]
  → buildLoraOverrides()             — read .onnx.data + apply delta + build Ort::Value overrides  [SdLoraApply.cpp]
       → parseLoraLayers()           — group safetensors keys into (down, up, alpha) triplets  [LoraParser.cpp]
       → matchExternalLoraKey()      — O(1) suffix-index lookup  [SdLoraMatch.cpp]
       → computeLoraDelta()          — matmul + scale in fp32  [LoraMath.cpp]
  → SessionOptions::Clone()          — clone EP opts so AddExternalInitializers doesn't pollute shared state
  → cloned.AddExternalInitializers() — inject patched tensors; ORT loads the rest from .onnx.data
  → Ort::Session(env, path, cloned)  — session created from .onnx path; .onnx.data resolved natively
```

Key files:
- `sd/SdOnnxPatcher.hpp` — shared types: `ExternalTensorMeta`, `OnnxExternalIndex`, `OnnxExternalSuffixIndex`, `TensorIndex`, `ParsedLora`/`LoraLayer`, `PatchResult`
- `sd/OnnxParser.cpp` — `parseExternalIndex` (field-14/EXTERNAL scan), `parseTensorIndex` (inline-buffer scan, kept for reference)
- `sd/OnnxIndex.cpp` — `buildExternalSuffixIndex`, `buildSuffixIndex`
- `sd/LoraParser.cpp` — `parseLoraLayers` (kohya prefix stripping + triplet grouping)
- `sd/LoraMath.cpp` — `computeLoraDelta` (fp32 matmul)
- `sd/SdLoraMatch.hpp/.cpp` — `matchExternalLoraKey`, `matchLoraKey`; optional `SD_LORA_MATCH_DEBUG` flag
- `sd/SdLoraApply.hpp/.cpp` — `buildLoraOverrides`: reads base weights from `.onnx.data`, applies LoRA deltas in fp32, converts back to model dtype, returns `LoraOverrides` with backing buffers + `Ort::Value` views
- `sd/SdSafetensors.hpp` — safetensors loader + fp16/bf16 conversion helpers
- `sd/SdLoader.cpp` `makeLoraSession()` lambda — orchestrates index→match→override→session
- `sd/ModelManager.hpp/.cpp` — `unordered_map<ModelCacheKey, ModelInstance>` session cache

### Model and session caching

**Session cache (`ModelManager`)** — `SdPipeline.cpp` holds a `static ModelManager`. On each `runPipeline()` call, `ModelManager::get(cfg, modelDir, loras)` looks up a `ModelCacheKey` in an `unordered_map<ModelCacheKey, ModelInstance>`. On hit the stored `ModelInstance` is reused; on miss `loadModels()` is called, a new `ModelInstance` is emplaced, and the result is returned. Multiple distinct configurations can coexist in the cache simultaneously.

`ModelCacheKey` is constructed via `ModelCacheKey::make()` which enforces three invariants before the key is stored:
- **Canonical paths** — `modelDir` and every LoRA path are passed through `std::filesystem::weakly_canonical()` so `./models/sd15` and `models/sd15` map to the same key.
- **Sorted LoRA list** — entries are sorted by canonical path, making `{A, B}` and `{B, A}` identical keys.
- **Fixed-point scales** — scales are compared and hashed as `int(std::lround(scale * 1000))`. `lround` rounds rather than truncating, so 0.9999f → 1000 not 999. Values within 0.0005 of each other are treated as equal.

`ModelCacheKeyHash` feeds a single canonical string buffer (`modelDir\0type\0path\0scale\0...`) into **XXH64** (`libxxhash`). This avoids depending on the quality of `std::hash<std::string>` which varies across stdlibs.

**External-index cache (`SdLoader.cpp`)** — `s_extIndexCache` maps `.onnx` file path → `OnnxExternalIndex`. The first LoRA load for a given model file parses the protobuf binary once to build the metadata map (name, dtype, shape, offset, length for each external tensor). Every subsequent LoRA run (same model, different LoRA or scale) reuses the cached index — only the base-weight reads from `.onnx.data` and delta computation happen again.

No-LoRA path: sessions are created directly from the `.onnx` file path; ORT memory-maps the file and resolves `.onnx.data` natively. No bytes are buffered in the process.

LoRA path memory: external index (small, persistent) + per-tensor base weight reads during override construction (freed after session creation) + live session. Peak ≈ 1× model size + LoRA delta tensors (typically small).

### LoRA key format (Kohya)

Safetensors keys use this naming pattern:
- `lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight`  (CLIP text encoder)
- `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight`  (UNet)
- `lora_te2_...` — second text encoder (SDXL only)

Each layer needs a down, up, and optionally alpha tensor. `parseLoraLayers()` groups these into `LoraLayer` triplets inside a `ParsedLora`.

### ONNX weight name matching

The matching pipeline has three stages:

**1. Parse** — `parseExternalIndex` scans the `.onnx` binary for TensorProto entries where `data_location == EXTERNAL` (field 14, value 1). It normalises all `.` and `/` in initializer names to `_` and records each tensor's `onnxName`, `shape`, `dtype`, `dataOffset`, and `dataLength` (from the `external_data` key-value pairs at field 13). Returns `OnnxExternalIndex` (a `std::map<normName, ExternalTensorMeta>`).

**2. Index** — `buildExternalSuffixIndex` walks every normalised name and inserts it under every `_`-boundary suffix into an `OnnxExternalSuffixIndex`. Runs once per model file at load time; cached in `s_extIndexCache`.

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

PyTorch `torch.onnx.export` places named initializers at **GraphProto field 5** (not field 6 as some tools use). `parseTensorIndex` (the legacy inline scanner) accepts both. The diagnostic log line `ONNX graph fields (ALL): f1x969 f2x1 f5x180 ...` confirms this — `f5x180` = 180 named initializers. If you see `f5x0 f6x0`, the model was exported without `keep_initializers_as_inputs=True` — re-export with the project's scripts.

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

**2. Are weight names matching?**
```
buildLoraOverrides: text_encoder.onnx — 132 patch(es), 132 miss(es)
```
High miss count means the ONNX weight names don't align with the LoRA key suffixes. Look for the normalised name samples in the log to compare:
```
ONNX weight sample: text_encoder_text_model_encoder_layers_0_mlp_fc1_weight
```

**3. Are tensor overrides correctly built?**
```
buildLoraOverrides: 132 tensor override(s) prepared
```
This confirms `Ort::Value`s were created for all matched tensors. A lower count than expected patches means some tensors failed shape validation or had unsupported dtypes.

**4. Does AddExternalInitializers succeed?**
If ORT throws during session creation with a name or shape mismatch error, the `onnxName` stored in `ExternalTensorMeta` does not match what ORT expects. The name comes directly from TensorProto field 8 in the `.onnx` binary — verify with a hex dump or `onnx_check.py`.

**5. Does the text encoder output have NaN?**
```
embedding stats: min=1000000000.000000  max=-1000000000.000000  mean=nan
```
The sentinel values `min=1e9, max=-1e9` mean ALL values are NaN (NaN comparisons return false). Check `floatToFp16` in `SdSafetensors.hpp` if the LoRA delta is being applied to fp16 weights. NaN propagates: text encoder → UNet cross-attention → NaN eps → black image.

**6. Is the session cache stale?**
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

## What NOT to do

- The SD 1.5 VAE export now uses `dynamic_axes` for height/width (needed for non-512 resolutions). This is safe because the VAE always loads with `cpu_session_opts` and never goes through DML. Do not remove those dynamic axes.
- Do not use `ctx.run_opts` for the VAE `Run()` call — it would be affected by `SetTerminate()` from the cancel watcher.
- Do not call `cpu_unet` with `ctx.session_opts` — it must always use `ctx.cpu_session_opts` (no GPU EP).
- Do not hard-code fp16 for VAE input — use `ctx.vaeExpectsFp32` to select the right type.
- Do not reset `ctx.run_opts` manually in `runPipeline()` — `ModelManager::get()` already does this on every call before returning the context reference.
- Do not call `SessionOptions::AddExternalInitializers` on the shared `unetOpts`/`auxOpts` references — always `Clone()` first. Mutating a shared `SessionOptions` would affect every subsequent session created with it.
- Do not destroy a `LoraOverrides` before the `Ort::Session` constructor returns — the `Ort::Value` objects hold non-owning views into `LoraOverrides::fp16Bufs`/`fp32Bufs`.
- Do not use `operator[]` to insert into `ModelManager`'s internal cache — `GenerationContext` contains non-copyable ORT sessions. Always use `emplace` with `std::move`.

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
  sd/SdLoader.cpp           — model config detection + ORT session loading
  sd/SdTextEncoder.cpp      — text encoding (SD 1.5 single, SDXL dual)
  sd/SdUNet.cpp             — CFG UNet passes + DML GPU fallback
  sd/SdVae.cpp              — VAE latent decode
  sd/SdScheduler.cpp        — DPM++ 2M Karras sigma schedule
  sd/SdTypes.hpp            — ModelConfig + GenerationContext structs
  sd/SdUtils.hpp            — inline helpers (timing, fp16, RNG, image conversion)
  Pipeline: CLIP tokenize → text encode → DPM++ 2M Karras loop → VAE decode → cv::imwrite
```

## Build system

CMake with two key options:
- `USE_CUDA=ON` — enables CUDA EP for all models (UNet, VAE, text encoders)
- `USE_DML=ON` — enables DirectML EP for UNet only (VAE and text encoders stay on CPU due to Reshape node issues)

Default ORT paths are set in CMakeLists.txt but can be overridden with `-DONNXRUNTIME_ROOT=`.

Linux builds: `cmake -B build && cmake --build build -j$(nproc)`  
Windows cross-compile from Linux: uses `cmake/mingw-w64.cmake` toolchain.

## Key files

| File | Role |
|---|---|
| `src/portraits/PortraitGeneratorAi.hpp` | Public API: `GenerationParams`, `generateFromPrompt()` |
| `src/portraits/PortraitGeneratorAi.cpp` | Thin shell — builds prompt, calls `sd::runPipeline()` |
| `src/portraits/sd/SdTypes.hpp` | `ModelConfig` + `GenerationContext` structs |
| `src/portraits/sd/SdLoader.cpp` | `loadModelConfig()`, `loadModels()`, `logModelIO()` |
| `src/portraits/sd/SdTextEncoder.cpp` | `encodeText()` (SD 1.5), `encodeTextSDXL()` (SDXL) |
| `src/portraits/sd/SdUNet.cpp` | `runUNetSingle()`, `runUNetCFG()`, DML GPU fallback |
| `src/portraits/sd/SdVae.cpp` | `decodeLatent()` |
| `src/portraits/sd/SdScheduler.cpp` | `buildAlphasCumprod()`, `buildKarrasSchedule()`, `sigmaToTimestep()` |
| `src/portraits/sd/SdPipeline.cpp` | `denoiseSingleLatent()`, `runPipeline()` |
| `src/portraits/sd/SdUtils.hpp` | Inline helpers: `fmtMs`, `toFp16`, `randNormal`, `latentToImage` |
| `src/portraits/ClipTokenizer.cpp/.hpp` | BPE tokenizer (no Python dependency) |
| `src/portraits/PromptBuilder.hpp` | Weighted A1111-style prompt builder |
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

**GPU fallback:** If the GPU UNet throws, `dmlFailed` is set to true and all subsequent steps run on `cpu_unet` (a separate CPU session loaded at startup).

## dtype handling

Models are queried at load time for their input dtype:
- `ctx.unetExpectsFp32` — detected from `unet.GetInputTypeInfo(0)`
- `ctx.vaeExpectsFp32` — detected from `vae_decoder.GetInputTypeInfo(0)`

SD 1.5 export: UNet = fp16, VAE = fp16  
SDXL export: UNet = fp16, VAE = fp16 (after the fix in sdxl_export_onnx_models.py)

Always use the detected flag rather than hard-coding a dtype — the export scripts may change.

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

### Architecture

LoRA is applied at model load time by patching the ONNX binary in memory before creating the ORT session. No LoRA-specific code runs at inference time.

```
SdLoader.cpp  makePatchedBytes()
  → readFileBytes()          — load raw ONNX bytes
  → parseTensorIndex()       — build name→{offset, length, shape, dtype} index
  → applyLoraToBytes()       — patch weights in-place
  → Ort::Session(bytes)      — create session from patched memory
```

Key files:
- `sd/SdOnnxPatcher.hpp/.cpp` — ONNX protobuf parser + LoRA patcher
- `sd/SdSafetensors.hpp` — safetensors loader + fp16/bf16 conversion helpers
- `sd/SdLoader.cpp` `makePatchedBytes()` lambda (line ~202) — orchestrates load→patch→session
- `sd/SdLoader.cpp` `mapLoraKeyToOnnx()` — debug-only key mapping (NOT used for actual matching)

### LoRA key format (Kohya)

Safetensors keys use this naming pattern:
- `lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight`  (CLIP text encoder)
- `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight`  (UNet)
- `lora_te2_...` — second text encoder (SDXL only)

Each layer needs a down, up, and optionally alpha tensor. `applyLoraToBytes` groups these into triplets.

### ONNX weight name matching

`parseTensorIndex` normalises all `.` and `/` in ONNX initializer names to `_`, then builds a map. `applyLoraToBytes` strips the `lora_te_`/`lora_unet_` prefix, appends `_weight`, and searches by suffix:

```
LoRA base:  text_model_encoder_layers_0_self_attn_q_proj
ONNX key:   text_encoder_text_model_encoder_layers_0_self_attn_q_proj_weight
                         └──────────── suffix match ─────────────────────────┘
```

The suffix search is relaxed (no `_` boundary requirement). This works because ONNX names include a model-level prefix absent from LoRA keys.

### ONNX protobuf field numbers

PyTorch `torch.onnx.export` places named initializers at **GraphProto field 5** (not field 6 as some tools use). The parser accepts both. The diagnostic log line `ONNX graph fields (ALL): f1x969 f2x1 f5x180 ...` confirms this — `f5x180` = 180 named initializers. If you see `f5x0 f6x0` on a newly exported model, the model was exported without `keep_initializers_as_inputs=True` and all weights are anonymous Constant nodes — re-export with the project's scripts.

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

### Known bugs (fixed)

**`floatToFp16` subnormal bug** (`SdSafetensors.hpp`):  
The original subnormal path used `shift = 1 + (-14 - exp32)` which was 12 too small, causing values in the fp16 subnormal range (~6e-8 to ~6e-5) to overflow and produce fp16 NaN. Symptom: `LoRA WARNING: X has N NaN/Inf AFTER patching` followed by all-NaN text encoder output. Fixed to `shift = 13 + (-14 - exp32)` with `& 0x3FFu` masking.

### Diagnosing LoRA problems

The log tells you everything. Check in this order:

**1. Are initializers found at all?**
```
ONNX graph fields (ALL): f1x969 f2x1 f5x180 ...
ONNX patcher: indexed 397 initialisers.
```
`f5x0 f6x0` = no named weights. The model needs re-export.

**2. Are weight names matching?**
```
ONNX weight sample: text_encoder_text_model_encoder_layers_0_mlp_fc1_weight
LoRA patched: text_model_encoder_layers_0_mlp_fc1 [3072x768]
```
If you see `LoRA no match: ...` for most layers, the ONNX weight prefix or normalisation doesn't align with the LoRA key suffix.

**3. Is patching introducing NaN?**
```
LoRA WARNING: text_model_encoder_layers_0_mlp_fc1 has 212 NaN/Inf AFTER patching
```
This indicates a bug in the fp16 write-back path. Check `floatToFp16` in `SdSafetensors.hpp`.

**4. Are base weights already corrupt?**
```
LoRA WARNING: base text_model_encoder_layers_0_mlp_fc1 has N NaN/Inf BEFORE patching
```
The ONNX file itself is corrupted or the dtype detection is wrong.

**5. Are delta values sane?**
```
[diag] first patch dtype=10  offset=80961930  pre[0..3]: 0.04 0.005 ...  delta[0..3]: 0.00002 ...
[diag] post-patch[0..3]: 0.040192 ...
```
dtype should be 10 (fp16) or 1 (fp32). pre/post values should change by the delta. Huge delta values (`LoRA large delta: ... peak=15`) indicate scale or alpha issue.

**6. Does the text encoder output have NaN?**
```
embedding stats: min=1000000000.000000  max=-1000000000.000000  mean=nan
```
The sentinel values `min=1e9, max=-1e9` mean ALL values are NaN (NaN comparisons return false). NaN propagates: text encoder → UNet cross-attention → NaN eps → black image.

### Temporary diagnostics in the patcher

`SdOnnxPatcher.cpp` `applyLoraToBytes()` contains:
- Pre/post-patch NaN checks for every patched tensor (logs `LoRA WARNING`)
- First-patch dtype + sample value logging (logs `[diag]`)
- Delta magnitude warning for `peak > 1.0` (logs `LoRA large delta`)

These are safe to leave enabled; they add minimal overhead relative to session creation time. Remove them when LoRA is fully stable.

## What NOT to do

- The SD 1.5 VAE export now uses `dynamic_axes` for height/width (needed for non-512 resolutions). This is safe because the VAE always loads with `cpu_session_opts` and never goes through DML. Do not remove those dynamic axes.
- Do not use `ctx.run_opts` for the VAE `Run()` call — it would be affected by `SetTerminate()` from the cancel watcher.
- Do not call `cpu_unet` with `ctx.session_opts` — it must always use `ctx.cpu_session_opts` (no GPU EP).
- Do not hard-code fp16 for VAE input — use `ctx.vaeExpectsFp32` to select the right type.

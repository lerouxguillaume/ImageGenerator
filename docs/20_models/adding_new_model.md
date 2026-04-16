# Adding a New Model Family

## What this file explains
Step-by-step process for integrating a new SD model family (e.g. SD 2.1, LCM, FLUX) into the pipeline. Only the listed files need to change — everything else (scheduler, VAE, cancellation, GPU fallback, LoRA) is reused unchanged.

## When to use this
- Integrating a new model architecture that differs from SD 1.5 and SDXL
- Adding a new resolution tier or text encoder layout
- Wiring up model-specific UNet extra inputs

## Key invariants
- The model type is determined at runtime from `model.json` — no recompilation needed for new model directories.
- All new model-specific code must be gated on `cfg.type == ModelType::MyNewModel` checks.
- The scheduler, VAE, cancellation, and caching system are shared and must not be modified.
- The new export script must write `model.json` with the new type key.

## Mental model

```
model.json {"type": "mynewmodel"}
       │
       ▼
loadModelConfig()         → sets cfg.type, cfg.image_w/h
       │
       ▼
loadModels()              → loads extra sessions if needed
       │
       ▼
runPipeline()             → branches on cfg.type for encode + UNet inputs
       │
       ├─ encodeTextMyNewModel()    (new, if encoder differs)
       └─ appendMyNewModelInputs()  (new, if UNet needs extra inputs)
```

## Implementation details

### Step 1 — Register the model type (`src/enum/enums.hpp`)

```cpp
enum class ModelType { SD15, SDXL, MyNewModel };
```

### Step 2 — Detect from `model.json` (`sd/SdLoader.cpp — loadModelConfig()`)

```cpp
if (type == "mynewmodel") {
    cfg.type    = ModelType::MyNewModel;
    cfg.image_w = 768;
    cfg.image_h = 768;
}
```

If the new model needs scheduler parameters (e.g. different beta schedule), add fields to `ModelConfig` in `sd/SdTypes.hpp` and set them here.

### Step 3 — Load extra sessions (`sd/SdLoader.cpp — loadModels()`)

```cpp
if (cfg.type == ModelType::MyNewModel) {
    ctx.my_extra_encoder = loadSession(modelDir / "my_encoder.onnx", auxOpts);
}
```

Store extra I/O names in `GenerationContext` if needed (add fields to `sd/SdTypes.hpp`).

### Step 4 — Text encoding (`sd/SdTextEncoder.cpp`)

Add `encodeTextMyNewModel()` if the text encoder architecture differs from SD 1.5. Declare it in `sd/SdTextEncoder.hpp`.

### Step 5 — Wire into the pipeline (`sd/SdPipeline.cpp — runPipeline()`)

```cpp
if (cfg.type == ModelType::SDXL) {
    ctx.text_embed = encodeTextSDXL(...);
    ctx.time_ids   = {h, w, 0, 0, h, w};
} else if (cfg.type == ModelType::MyNewModel) {
    ctx.text_embed = encodeTextMyNewModel(...);
} else {
    ctx.text_embed = encodeText(...);   // SD 1.5 default
}
```

If the new model needs extra UNet inputs, add them to `GenerationContext` and extend `runUNetSingle()` in `sd/SdUNet.cpp` with an `appendMyNewModelInputs()` lambda (same pattern as `appendSDXLInputs`).

### Step 6 — Export script (`scripts/mynewmodel_export_onnx_models.py`)

Follow the pattern of `export_onnx_models.py`. Call `write_model_json(output_dir, "mynewmodel", all_specs)` at the end so the C++ loader detects the type automatically.

## Common pitfalls
- Forgetting to write `model.json` in the export script — the loader defaults to SD 1.5 and uses the wrong resolution and encoder path.
- Modifying the scheduler inside `runPipeline()` without adding a field to `ModelConfig` — hardcoded schedule parameters break other model types sharing the same function.
- Not adding `warmupSession()` calls for any new sessions loaded in step 3 — the first inference call on the new encoder pays the JIT cost.
- Reusing `ctx.run_opts` for any new session that should not be cancellable (e.g. a slow postprocessor) — use `Ort::RunOptions{nullptr}` for those, same as the VAE.

## Related files
- [`docs/10_pipeline/sdxl_specifics.md`](../10_pipeline/sdxl_specifics.md) — reference implementation for extra UNet inputs
- [`docs/10_pipeline/scheduler.md`](../10_pipeline/scheduler.md) — scheduler design and extension points
- [`docs/50_export/export_overview.md`](../50_export/export_overview.md) — export script structure
- `src/portraits/sd/SdTypes.hpp` — `ModelConfig`, `GenerationContext`, `ModelInstance`
- `src/portraits/sd/SdLoader.cpp` — `loadModelConfig()`, `loadModels()`
- `src/portraits/sd/SdPipeline.cpp` — `runPipeline()` branching

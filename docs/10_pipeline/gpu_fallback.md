# GPU Fallback and DML Limitations

## What this file explains
How the pipeline handles GPU UNet failures at runtime, and what DML cannot execute on Windows.

## When to use this
- Diagnosing why generation silently switched to CPU after the first image
- Understanding which models/components can use DML and which cannot
- Configuring DML session options

## Key invariants
- `dmlFailed` persists in the cached `GenerationContext` — **once a GPU failure occurs, all subsequent runs on that cache entry use CPU**. Clearing the flag requires evicting the cache entry (changing the model or LoRA config).
- `cpu_unet` must always use `ctx.cpu_session_opts` (CPU EP only). Never pass `ctx.session_opts` to it.
- VAE and text encoders always run on CPU under DML, regardless of `USE_DML`. This is a hard architectural constraint.
- DML session options (`ORT_ENABLE_BASIC`, `DisableMemPattern()`, `ORT_SEQUENTIAL`) must not be changed without testing on Windows.

## Mental model

```
runUNetCFG(step i)
    │
    ├─ try: GPU UNet (ctx.unet, ctx.session_opts)
    │       │
    │       └─ throws OrtException
    │               │
    │               ▼
    │           ctx.dmlFailed = true
    │           log warning
    │
    └─ if dmlFailed: CPU UNet (ctx.cpu_unet, ctx.cpu_session_opts)
```

The switch is permanent for the lifetime of the cached `ModelInstance`. The CPU UNet session is always loaded at startup — it is not lazily created after a failure.

## Implementation details

### `dmlFailed` flag (`SdUNet.cpp`)

```cpp
try {
    ctx.unet.Run(ctx.session_opts, ...);
} catch (const Ort::Exception& e) {
    Logger::info("UNet GPU pass failed: " + std::string(e.what()));
    ctx.dmlFailed = true;
}

if (ctx.dmlFailed) {
    ctx.cpu_unet.Run(ctx.cpu_session_opts, ...);
}
```

### CPU UNet session (`SdLoader.cpp`)

`cpu_unet` is loaded at the same time as the GPU `unet`, unconditionally when DML is enabled:

```cpp
ctx.cpu_unet = loadSession(unetPath, cpu_session_opts);
```

`cpu_session_opts` has no execution providers registered — ORT falls back to the CPU EP automatically.

### Known DML failures (Windows)

| Component | Failure node | Root cause | Behaviour |
|---|---|---|---|
| SDXL UNet | `node_view_1` | Reshape nodes rejected by DML at runtime | Falls back to `cpu_unet` |
| VAE decoder | `node_view_2` | Reshape nodes in VAE attention block | Always on CPU; no DML session loaded |

These are ORT DML EP limitations with specific Reshape node shapes — they are not bugs in the export scripts.

### DML session options

```cpp
session_opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  // higher levels break DML nodes
session_opts.DisableMemPattern();                           // required for DML
session_opts.SetExecutionMode(ORT_SEQUENTIAL);             // DML cannot handle parallel execution
```

Do not change these without testing on a Windows machine with a DML-capable GPU.

### VAE always on CPU (DML builds)

The SD 1.5 VAE uses `dynamic_axes` for height/width (needed for non-512 resolutions). DML cannot handle dynamic Reshape nodes, so the VAE always loads with `cpu_session_opts`. Do not remove the dynamic axes from the VAE export.

## Common pitfalls
- Expecting `dmlFailed = false` after the user changes settings — the flag persists in the cache entry. The only way to reset it is to trigger a cache miss (e.g. change LoRA or model path).
- Calling `cpu_unet.Run(ctx.session_opts, ...)` — `session_opts` registers the DML EP, which would attempt GPU execution on the CPU-intended session.
- Assuming DML failures are export bugs — the known `node_view_1` / `node_view_2` failures are DML EP limitations, not fixable in the ONNX file.

## Related files
- [`docs/10_pipeline/cancellation.md`](cancellation.md) — `ctx.run_opts` vs `ctx.session_opts`
- [`docs/10_pipeline/dtype_handling.md`](dtype_handling.md) — dtype detection
- [`docs/00_overview/build_system.md`](../00_overview/build_system.md) — `USE_DML` build flag
- `src/portraits/sd/SdUNet.cpp` — `runUNetSingle()`, `dmlFailed` branch
- `src/portraits/sd/SdLoader.cpp` — CPU and GPU session loading
- `src/portraits/sd/SdTypes.hpp` — `GenerationContext.dmlFailed`, `cpu_session_opts`

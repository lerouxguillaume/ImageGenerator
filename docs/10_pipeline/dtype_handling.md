# dtype Handling

## What this file explains
How the pipeline determines and applies the correct floating-point dtype (fp16 vs fp32) for UNet and VAE inputs at runtime.

## When to use this
- Debugging `"Type … is invalid"` or dtype mismatch ORT errors at inference time
- Changing export scripts and wondering whether the C++ side needs updating
- Adding a new model that uses fp32 weights

## Key invariants
- **Never hard-code a dtype.** Always use the detected flags: `ctx.unetExpectsFp32` and `ctx.vaeExpectsFp32`.
- Detection happens once per session creation by querying `GetInputTypeInfo(0)`.
- The flags live in `GenerationContext` and persist in the `ModelManager` cache.

## Mental model
Each ORT session knows its own input dtype. The pipeline queries it once at load time and stores a boolean. All inference code branches on that boolean — it never assumes a specific dtype.

## Implementation details

### Detection (in `SdLoader.cpp`)

```cpp
// UNet
const auto unetInputType = ctx.unet.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
ctx.unetExpectsFp32 = (unetInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

// VAE
const auto vaeInputType = ctx.vae_decoder.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
ctx.vaeExpectsFp32 = (vaeInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
```

### Usage (in `SdPipeline.cpp` / `SdVae.cpp`)

```cpp
// Latent passed to VAE
if (ctx.vaeExpectsFp32)
    latentTensor = toFp32(latent);
else
    latentTensor = latent;  // already fp16
```

### Current export dtypes

| Component | SD 1.5 | SDXL |
|---|---|---|
| UNet input | fp16 | fp16 |
| VAE input | fp16 | fp16 |
| Text encoder output | fp32 | fp32 |

Both flags are currently `false` for all shipped export configs. However, the detection exists precisely so this can change without touching inference code.

### Text encoder output dtype
Text encoders always output fp32. The pipeline does not query this — fp32 embeddings feed directly into the UNet without conversion because the UNet's cross-attention typically upcasts internally.

## Common pitfalls
- Calling `toFp16()` on VAE input unconditionally breaks if a future export produces an fp32 VAE. Use `ctx.vaeExpectsFp32`.
- `ctx.unetExpectsFp32` covers the main UNet (`ctx.unet`). The CPU fallback UNet (`ctx.cpu_unet`) is loaded from the same file and will have the same dtype — but it has its own separate session. Query its type independently if needed.
- Do not query dtype from the GPU session and apply the result to the CPU session blindly — they share the same `.onnx` file but are separate `Ort::Session` objects.

## Related files
- [`docs/10_pipeline/gpu_fallback.md`](gpu_fallback.md) — cpu_unet and DML fallback
- [`docs/10_pipeline/sdxl_specifics.md`](sdxl_specifics.md) — SDXL-specific tensor shapes
- `src/portraits/sd/SdLoader.cpp` — dtype detection at session creation
- `src/portraits/sd/SdVae.cpp` — `decodeLatent`, VAE dtype branch
- `src/portraits/sd/SdTypes.hpp` — `GenerationContext.unetExpectsFp32`, `vaeExpectsFp32`

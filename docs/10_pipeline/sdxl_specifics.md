# SDXL Specifics

## What this file explains
What makes SDXL different from SD 1.5 in the C++ pipeline: latent size, dual text encoders, and extra UNet inputs.

## When to use this
- Debugging SDXL generation quality or shape errors
- Extending the pipeline for a new model derived from SDXL
- Understanding why SDXL needs `text_embeds` and `time_ids` when SD 1.5 does not

## Key invariants
- SDXL latent is **128×128** (vs 64×64 for SD 1.5). Wrong latent size → wrong image size or ORT shape error.
- SDXL uses **two text encoders** that share one tokenizer. Both are required — using only one produces degraded output.
- `text_embeds` must be the **pooled output** of encoder 2 (shape `(1, 1280)`), not the sequence output.
- `time_ids` encodes `[orig_h, orig_w, crop_top, crop_left, target_h, target_w]`. For portrait generation this is typically `[1024, 1024, 0, 0, 1024, 1024]`.
- Both of these must be passed to `runUNetSingle()` on every step, including the unconditional (uncond) pass.

## Mental model

```
Tokenizer (shared, 77 tokens)
    │
    ├─→ CLIP-L (text_encoder)   → hidden states [1, 77, 768]  ─┐
    │                                                            ├─ concat → [1, 77, 2048]  → UNet cross-attention
    └─→ OpenCLIP-G (text_encoder_2) → hidden states [1, 77, 1280] ─┘
                                  └→ pooled output [1, 1280]   → text_embeds

time_ids [1, 6] = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    → injected into UNet timestep MLP as conditioning
```

## Implementation details

### Detection

`loadModelConfig()` reads `<modelDir>/model.json`. If `"type": "sdxl"` is found, `cfg.type = ModelType::SDXL`, `cfg.image_w = 1024`, `cfg.image_h = 1024`. Latent size follows: `latent_w = image_w / 8 = 128`.

### Text encoding (`SdTextEncoder.cpp — encodeTextSDXL`)

```
encoder_1_out = text_encoder.Run(tokens)          → [1, 77, 768]
encoder_2_out, pooled = text_encoder_2.Run(tokens) → [1, 77, 1280], [1, 1280]

ctx.text_embed = concat([encoder_1_out, encoder_2_out], dim=-1)  → [1, 77, 2048]
ctx.text_embeds_pooled = pooled                                   → [1, 1280]
```

Both encoders receive **identical** token ids (single tokenizer, CLIP vocabulary, 77 tokens).

### Extra UNet inputs (`SdUNet.cpp — appendSDXLInputs` lambda)

```cpp
// appended after the standard (latent, timestep, encoder_hidden_states) inputs:
inputs.push_back(text_embeds);   // shape (1, 1280), fp32
inputs.push_back(time_ids);      // shape (1, 6),    fp32
```

`time_ids` is built once per run in `SdPipeline.cpp::runPipeline()`:
```cpp
ctx.time_ids = {orig_h, orig_w, 0.f, 0.f, (float)cfg.image_h, (float)cfg.image_w};
```

### Model detection in the C++ loader

```cpp
if (type == "sdxl") {
    cfg.type    = ModelType::SDXL;
    cfg.image_w = 1024;
    cfg.image_h = 1024;
}
```

`text_encoder_2` session is loaded conditionally:
```cpp
if (cfg.type == ModelType::SDXL)
    ctx.text_encoder_2 = loadSession(modelDir / "text_encoder_2.onnx", auxOpts);
```

## Common pitfalls
- Passing `text_embeds` (pooled) where `encoder_hidden_states` (sequence) is expected — shape mismatch ORT error.
- Forgetting `time_ids` on one of the CFG passes (uncond or cond) — UNet shape error on the affected pass.
- Using SD 1.5 latent size (64×64) for SDXL — generates a 512×512 image, not 1024×1024.
- Feeding the sequence output of encoder 2 as `text_embeds` instead of the pooled output — wrong shape `(1, 77, 1280)` instead of `(1, 1280)`.

## Related files
- [`docs/10_pipeline/scheduler.md`](scheduler.md) — sigma schedule (same for SDXL and SD 1.5)
- [`docs/10_pipeline/dtype_handling.md`](dtype_handling.md) — dtype detection
- [`docs/50_export/sdxl_fp16_fixes.md`](../50_export/sdxl_fp16_fixes.md) — SDXL export fp16 issues
- `src/portraits/sd/SdTextEncoder.cpp` — `encodeTextSDXL()`
- `src/portraits/sd/SdUNet.cpp` — `appendSDXLInputs` lambda, `runUNetCFG()`
- `src/portraits/sd/SdTypes.hpp` — `GenerationContext.text_embeds_pooled`, `time_ids`

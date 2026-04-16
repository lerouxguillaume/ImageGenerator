# Export Validation

See [`export_overview.md`](export_overview.md) for script usage and output layout.

## `--validate` flag

After exporting each component, loads the ONNX file in ORT on CPU and runs a forward pass with the same dummy inputs used during export. Catches dtype mismatches, missing external data, and output shape errors before they surface in the C++ pipeline.

```bash
python scripts/export_onnx_models.py model.safetensors --validate
python scripts/sdxl_export_onnx_models.py model.safetensors --validate
```

Slow for large models: the SDXL UNet loads all weights and may take several minutes per component.

## Manifest validation (`validateManifest`)

`validateManifest()` in `SdLoader.cpp` is called inside `loadModelConfig()` after JSON parse. It checks that every file declared in the `"components"` dict exists on disk and logs `[OK]` / `[MISSING]` for each entry.

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

Missing files are logged as warnings — load continues, so a partial model directory is tolerated. This makes "model loaded but LoRA has 0 patches" diagnosable without running inference (companion weights missing).

## ONNX diagnostic script

`scripts/onnx_check.py` traces both inputs of a failing node back to their producing nodes and inspects dtypes. Use when a new `"Type parameter (T) bound to different types"` or `"Type … is invalid"` ORT error appears:

1. Find the failing node name in the ORT error.
2. Run `onnx_check.py` to trace inputs and inspect dtypes.
3. Identify whether the fp32 source is a Constant node, a Cast node, or a diffusers upcast flag.
4. Apply the appropriate targeted fix (see `docs/50_export/sdxl_fp16_fixes.md`).

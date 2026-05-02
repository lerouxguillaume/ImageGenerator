# Model Loading

Handled by:
- `SdLoader.cpp`

---

# Responsibilities

- Parse `model.json`
- Detect model type
- Load ONNX sessions
- Build ModelInstance

---

# Model selection

Configured via `model.json` at the model root:

```json
{ "type": "sdxl", "vae_scaling_factor": 0.13025 }
```

If `model.json` is absent → default = SD1.5.

Models imported via the in-app pipeline also carry a `capabilities` block (written by `scripts/import_model.py`):

```json
{
  "type": "sdxl",
  "vae_scaling_factor": 0.13025,
  "capabilities": {
    "vae_encoder_available": true,
    "lora_compatible": true,
    "components": {
      "text_encoder":   { "dtype": "fp32" },
      "text_encoder_2": { "dtype": "fp32" },
      "unet":           { "dtype": "fp16" },
      "vae_decoder":    { "dtype": "fp16" },
      "vae_encoder":    { "dtype": "fp16" }
    }
  }
}
```

`SdLoader` reads the `capabilities` block when present and stores the values in `ModelConfig` (`vaeEncoderAvailable`, `loraCompatible`). Both default to `true` so models without a capabilities block behave as before. The block is the authoritative record written by the import pipeline after all output files are validated.

The `type` field drives:
- Output resolution (512×512 for SD1.5, 1024×1024 for SDXL)
- Number of CLIP text encoders (1 for SD1.5, 2 for SDXL)
- UNet input shape (latent channels and conditioning dimensions)

The `vae_scaling_factor` field sets the latent scaling constant used by `encodeImage()` and stored in `ctx.vaeScalingFactor`. Defaults: `0.18215` for SD1.5, `0.13025` for SDXL. Written automatically by the export pipeline; only needs to be set manually for non-standard VAEs.

---

# Session creation

Each model produces:
- UNet session
- VAE decoder session
- VAE encoder session (optional — loaded only when `capabilities.vae_encoder_available` is true **and** `vae_encoder.onnx` exists)
- CLIP session(s)
- optional SDXL encoder2 session

`vaeEncoderAvailable` is set to `true` only when both conditions hold. For models without a capabilities block the file-existence check alone decides, preserving legacy behaviour.

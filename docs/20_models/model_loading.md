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

The `type` field drives:
- Output resolution (512×512 for SD1.5, 1024×1024 for SDXL)
- Number of CLIP text encoders (1 for SD1.5, 2 for SDXL)
- UNet input shape (latent channels and conditioning dimensions)

The `vae_scaling_factor` field sets the latent scaling constant used by `encodeImage()` and stored in `ctx.vaeScalingFactor`. Defaults: `0.18215` for SD1.5, `0.13025` for SDXL. Written automatically by `export_vae_encoder.py`; only needs to be set manually for non-standard VAEs.

---

# Session creation

Each model produces:
- UNet session
- VAE decoder session
- VAE encoder session (optional — loaded only when `vae_encoder.onnx` exists)
- CLIP session(s)
- optional SDXL encoder2 session

`vaeEncoderAvailable` is set to `true` only when the encoder file is found and loaded successfully. Missing the file is not an error — it simply disables img2img for that model directory.
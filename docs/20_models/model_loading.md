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
{ "type": "sdxl" }
```

If `model.json` is absent → default = SD1.5.

The `type` field drives:
- Output resolution (512×512 for SD1.5, 1024×1024 for SDXL)
- Number of CLIP text encoders (1 for SD1.5, 2 for SDXL)
- UNet input shape (latent channels and conditioning dimensions)

---

# Session creation

Each model produces:
- UNet session
- VAE session
- CLIP session(s)
- optional SDXL encoder2 session
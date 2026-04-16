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

If `model.json` missing:
→ default = SD1.5

---

# Session creation

Each model produces:
- UNet session
- VAE session
- CLIP session(s)
- optional SDXL encoder2 session
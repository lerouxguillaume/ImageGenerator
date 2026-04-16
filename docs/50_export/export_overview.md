# Export Pipeline

Converts Stable Diffusion models to ONNX format.

---

# Scripts

- SD1.5 export script
- SDXL export script

---

# Outputs

Each component produces:
- .onnx
- .onnx.data
- _weights.safetensors

---

# Rules

- Do NOT remove `dynamic_axes` from the SD 1.5 VAE export — required for non-512 resolutions; VAE always runs on CPU so DML reshape constraints don't apply
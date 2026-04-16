# Dtype Handling

ImageGenerator never hardcodes tensor precision.

---

# Detection system

At model load:

- `ctx.unetExpectsFp32`
- `ctx.vaeExpectsFp32`

Derived from:
- ONNX input tensor type inspection

---

# Supported configurations

| Component | SD1.5 | SDXL |
|----------|------|------|
| UNet     | fp16 | fp16 |
| VAE      | fp16 | fp16 |
| CLIP     | fp32/fp16 | fp32/fp16 |

---

# Why this matters

Incorrect dtype leads to:
- ORT runtime errors
- NaN outputs
- silent image corruption

---

# Rule

Never assume dtype.
Always use detected flags.
# Export Pipeline

Converts Stable Diffusion models to ONNX for use with ONNX Runtime.

---

# Scripts

| Script | Purpose |
|---|---|
| `scripts/export_onnx_models.py` | Full SD 1.5 export (text encoder + UNet + VAE decoder) |
| `scripts/sdxl_export_onnx_models.py` | Full SDXL export (2× text encoders + UNet + VAE decoder) |
| `scripts/export_vae_encoder.py` | Retrofit: add `vae_encoder.onnx` to an existing model directory |
| `scripts/patch_unet_resize.py` | Retrofit: fix fp16 Resize nodes in an already-exported directory |

---

# Outputs

Each full export produces per component:
- `.onnx` — graph structure
- `.onnx.data` — weight tensor sidecar (single consolidated file)
- `_weights.safetensors` — LoRA base weights (text encoder + UNet only)

The VAE encoder export (`export_vae_encoder.py`) adds:
- `vae_encoder.onnx`
- `vae_encoder.onnx.data`
- updates `model.json` with the checkpoint VAE scaling factor used by img2img

---

# Adding the VAE encoder to an existing model

The full export scripts do not export the VAE encoder.  Use the retrofit script:

```bash
# SD 1.5
python scripts/export_vae_encoder.py models/my_model checkpoints/my_model.safetensors

# SDXL
python scripts/export_vae_encoder.py models/my_sdxl_model checkpoints/sdxl.safetensors

# Re-export (overwrite)
python scripts/export_vae_encoder.py models/my_model checkpoints/my_model.safetensors --force
```

The script auto-detects SD1.5 vs SDXL from `model.json`.  All other files in the model directory are left untouched.

---

# Rules

- Do NOT remove `dynamic_axes` from the SD 1.5 VAE export — required for non-512 resolutions; VAE always runs on CPU so DML reshape constraints don't apply
- VAE encoder uses the same static-shape policy as the VAE decoder (no dynamic axes)
- SDXL VAE encoder requires the same `fix_fp32_constants` and `fix_resize_fp16` passes as the decoder

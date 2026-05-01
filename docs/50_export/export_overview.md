# Export Pipeline

Converts Stable Diffusion models to ONNX for use with ONNX Runtime.

---

# Scripts

| Script | Purpose |
|---|---|
| `scripts/import_model.py` | **Unified in-app entry point** — auto-detects arch and delegates to the scripts below; emits `PROGRESS:` protocol for the C++ import pipeline |
| `scripts/export_onnx_models.py` | Full SD 1.5 export (text encoder + UNet + VAE decoder + VAE encoder) |
| `scripts/sdxl_export_onnx_models.py` | Full SDXL export (2× text encoders + UNet + VAE decoder + VAE encoder) |
| `scripts/export_vae_encoder.py` | Retrofit: add or replace `vae_encoder.onnx` in an older existing model directory |
| `scripts/patch_unet_resize.py` | Retrofit: fix fp16 Resize nodes in an already-exported directory |

`import_model.py` is the only script the application calls directly. The per-architecture scripts are invoked through it and should not be called directly when using the in-app import flow.

---

# Outputs

Each full export produces per component:
- `.onnx` — graph structure
- `.onnx.data` — weight tensor sidecar (single consolidated file)
- `_weights.safetensors` — LoRA base weights (text encoder + UNet only)

Full exports include `vae_encoder.onnx` and `vae_encoder.onnx.data` by default so img2img is available immediately. `export_vae_encoder.py` remains only as a retrofit tool for older model directories.

---

# Adding the VAE encoder to an existing model

Use the retrofit script only for model directories exported before the default pipeline included `vae_encoder.onnx`:

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

- VAE encoder uses the same static-shape policy as the VAE decoder (no dynamic axes)
- SD 1.5 VAE decoder/encoder use static 512×512 shapes; SDXL VAE decoder/encoder use static 1024×1024 shapes
- fp16 Resize patching is applied by default to fp16 UNet/VAE components; `patch_unet_resize.py` is only for older exports
- SDXL VAE decoder/encoder require the same `fix_fp32_constants` and `fix_resize_fp16` passes

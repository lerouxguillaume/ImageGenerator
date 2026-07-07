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
| `scripts/validate_dynamic_sdxl.py` | Standalone GPU-box validator for an **experimental** dynamic-spatial SDXL export (see Rules) |

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

- SD 1.5 VAE **decoder** exports with dynamic H/W axes (`{0:batch, 2:height, 3:width}`, legacy tracer — dynamo cannot express spatial axes) so it can decode hires/second-pass latents larger than native; the VAE **encoder** stays static, as does the SDXL VAE decoder/encoder (static 1024×1024) **by default**
- **SDXL dynamic-spatial (experimental, opt-in — SDXL hires groundwork):** `sdxl_export_onnx_models.py --dynamic-spatial` exports the **UNet + VAE decoder** with dynamic H/W axes (legacy tracer, traced at a tiny latent — UNet 16×16, decoder 8×8 — so the fp16-CPU forward stays cheap; the VAE **encoder** stays static-1024 on dynamo). Default is unchanged (static). `hires_capable` in `model.json` is derived from the actual graphs (`import_model.py::_detect_hires_capable`) — it flips to `true` only when **both** the UNet and VAE decoder carry dynamic H/W axes, so it is arch-agnostic; if a graph can't be probed it **fails closed to `false`** (never mislabels a static-VAE model as capable). Validate a dynamic export on a GPU box with `scripts/validate_dynamic_sdxl.py` before promoting it to the default. Not yet wired into the in-app import flow.
- **SDXL fp32 VAE-decoder hedge (dynamic-spatial only):** the same `--dynamic-spatial` run *also* emits `vae_decoder_fp32.onnx` (+`.data`) beside the fp16 `vae_decoder.onnx` — an unregistered spare (absent from `model.json` `components`, inert to the runtime). SDXL fp16-VAE can produce NaN/washed-out output at >native latents; shipping the fp32 fallback in the same run means the fix is a **file swap, never a second export**. The C++ runtime auto-detects decode dtype from the decoder's input element type (`SdLoader.cpp:475-477 → ctx.vaeExpectsFp32`, consumed in `SdVae.cpp:30`), so the swap needs **zero C++ changes**. **Swap procedure** (only if fp16 decode fails validation): the runtime loads the hard-coded path `vae_decoder.onnx` (`SdLoader.cpp:279` — it does *not* read the filename from `model.json`), so put the fp32 content there. The VAE is plain-loaded by ORT (never LoRA-injected), so ORT resolves external data via the location string embedded in the `.onnx`:
  ```bash
  cd models/<name>
  mv vae_decoder.onnx      vae_decoder.onnx.fp16bak
  mv vae_decoder.onnx.data vae_decoder.onnx.data.fp16bak   # orphaned fp16 weights (optional tidy)
  cp vae_decoder_fp32.onnx vae_decoder.onnx                # embeds vae_decoder_fp32.onnx.data, which stays put
  # model.json is unchanged (filename still vae_decoder.onnx); no re-export, no re-import needed
  ```
- SD 1.5 VAE encoder uses static 512×512 shapes; SDXL VAE decoder/encoder use static 1024×1024 shapes
- fp16 Resize patching is applied by default to fp16 UNet/VAE components (the dynamic SD 1.5 VAE decoder's Resize nodes are exactly what it targets); `patch_unet_resize.py` is only for older exports
- SDXL VAE decoder/encoder require the same fp32-constant and fp16-Resize fixes as the UNet (all applied in a single `fix_fp16_graph` pass)

---

# SDXL dynamic-spatial: GPU-box run sequence

The dev box has no torch/GPU/SDXL checkpoint, so a `--dynamic-spatial` export is
validated on the GPU box. Guillaume's flow — **one export, validate, then one import
decision** (there is no static SDXL to fall back to):

```bash
# 1. Export (dynamic UNet + fp16 decoder + fp32 decoder hedge; encoder static-1024).
#    Traces UNet at 16×16 and both decoders at 8×8, so it is far faster than a full
#    static SDXL export (see docs/90_debug/performance.md).
python scripts/sdxl_export_onnx_models.py <sdxl.safetensors> --name sdxl_dyn --dynamic-spatial

# 2. Validate the graphs + numerics. Run nvidia-smi alongside to capture VRAM peaks
#    (the validator prints a ">>> ... watch nvidia-smi now" marker before each forward).
nvidia-smi -l 1 &                 # in another terminal, or: nvidia-smi dmon
python scripts/validate_dynamic_sdxl.py --model models/sdxl_dyn \
       --torch-reference <sdxl.safetensors> --sizes 128,160,192

# 3. Read the checks:
#    - unet.dynamic / vae_decoder.dynamic  -> latent H/W are symbolic
#    - unet.run@{128,160,192}              -> finite, time_ids scaled per size accepted
#    - vae.run@{...} std                   -> finite, non-degenerate (NaN/near-constant = fp16 VAE fails)
#    - *.numerics                          -> ONNX matches torch to fp16 precision
#    RESULT: PASS = go.

# 4a. If vae.run showed NaN/near-constant (SDXL fp16 VAE instability at >native): swap in
#     the fp32 decoder hedge (see the "SDXL fp32 VAE-decoder hedge" rule above), then re-run
#     the validator to confirm vae_fp32 decodes cleanly. No re-export.
# 4b. Import ONCE into the app (in-app import produces a static SDXL; to keep the dynamic
#     model, point the app's model dir at models/sdxl_dyn — it already has a full model.json
#     with the derived hires_capable flag).
```

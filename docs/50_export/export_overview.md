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
| `scripts/validate_dynamic_sdxl.py` | Standalone GPU-box validator for a dynamic-spatial SDXL export (see Rules) |

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

# Additional CLI knobs (standalone export scripts)

These flags exist on `export_onnx_models.py` / `sdxl_export_onnx_models.py` for
manual/standalone use. The in-app import never passes them (it calls the export
functions directly), so they are documented here rather than in the import flow:

| Flag | Script(s) | Effect |
|---|---|---|
| `--optimize-memory` | SDXL | Turns on attention slicing — lower peak RAM, slower export. For machines that OOM on the default fast path. |
| `--simplify-vae` | SDXL | Runs `onnxsim` on `vae_decoder.onnx` after export (slower; rarely needed). |
| `--output-dir` | both | Override the output directory path (default: `../models/<name>`). |
| `--validate` | both | Run a full ORT CPU forward pass after **each** component export to catch dtype/shape errors early. Slow on large models; a debugging aid. |

---

# Rules

- SD 1.5 VAE **decoder** *and* **encoder** export with dynamic H/W axes (`{0:batch, 2:height, 3:width}`, legacy tracer — dynamo cannot express spatial axes) so the decoder can decode hires latents larger than native and the encoder can re-encode the upscaled image for **pixel-mode** hires. The SDXL VAE decoder/encoder are static 1024×1024 **by default** (dynamo), dynamic only under `--dynamic-spatial`.
- **SDXL dynamic-spatial (the in-app SDXL import default):** the **UNet + VAE decoder + VAE encoder** export with dynamic H/W axes (legacy tracer, traced at a tiny input — UNet 16×16, decoder 8×8 latent, encoder 64×64 image — so the fp16-CPU forward stays cheap). This is what `import_model.py` runs for SDXL (`export_sdxl(..., dynamic_spatial=True)`): it enables hires (incl. sharp pixel mode), produces identical output at native 1024, and is **far faster to export** than the static path (the full-size fp16 UNet trace is the ~1 h bottleneck; the tiny trace makes it ~2–3 min). The standalone `sdxl_export_onnx_models.py` still defaults to the **static** export unless `--dynamic-spatial` is passed. Two capability flags in `model.json` are derived from the actual graphs (fail closed to `false` if a graph can't be probed): `hires_capable` (`_detect_hires_capable`) is `true` only when **both** the UNet and VAE decoder carry dynamic H/W axes; `pixel_hires_capable` (`_detect_pixel_hires_capable`) is `true` only when the VAE **encoder** carries them. A model can be `hires_capable` (latent mode) but not `pixel_hires_capable` (static encoder) — that is exactly the pre-dynamic-encoder SDXL case. `scripts/validate_dynamic_sdxl.py` is the GPU-box validator for such an export.
  - **fp32 encoder hedge — deliberately NOT emitted.** The decoder ships an fp32 hedge because fp16 *decode* (latent→pixel, the amplifying direction) can NaN/wash-out at >native latents. The encoder runs image→latent (contracting/normalizing, the numerically stable direction), and the SD1.5 dynamic encoder ships fp16-only and works. Adding an encoder hedge would be dead weight against no established failure mode. If GPU validation shows fp16 encoder blow-up at 1536, add it then by mirroring the decoder hedge (cheap, ~1.3 s).
- **SDXL fp32 VAE-decoder hedge (standalone CLI `--dynamic-spatial` only, NOT in-app):** the CLI `--dynamic-spatial` run *also* emits `vae_decoder_fp32.onnx` (+`.data`) beside the fp16 `vae_decoder.onnx` — an unregistered spare (absent from `model.json` `components`, inert to the runtime). The **in-app import ships a lean fp16-only model** (`emit_fp32_hedge=False`): fp16 was validated stable, and there is no in-app swap mechanism, so the ~200 MB spare would be dead weight. The hedge is an investigation / manual-swap tool. SDXL fp16-VAE can produce NaN/washed-out output at >native latents; shipping the fp32 fallback in the same run means the fix is a **file swap, never a second export**. The C++ runtime auto-detects decode dtype from the decoder's input element type (`SdLoader.cpp:475-477 → ctx.vaeExpectsFp32`, consumed in `SdVae.cpp:30`), so the swap needs **zero C++ changes**. **Swap procedure** (only if fp16 decode fails validation): the runtime loads the hard-coded path `vae_decoder.onnx` (`SdLoader.cpp:279` — it does *not* read the filename from `model.json`), so put the fp32 content there. The VAE is plain-loaded by ORT (never LoRA-injected), so ORT resolves external data via the location string embedded in the `.onnx`:
  ```bash
  cd models/<name>
  mv vae_decoder.onnx      vae_decoder.onnx.fp16bak
  mv vae_decoder.onnx.data vae_decoder.onnx.data.fp16bak   # orphaned fp16 weights (optional tidy)
  cp vae_decoder_fp32.onnx vae_decoder.onnx                # embeds vae_decoder_fp32.onnx.data, which stays put
  # model.json is unchanged (filename still vae_decoder.onnx); no re-export, no re-import needed
  ```
- SD 1.5 VAE encoder is dynamic H/W (traced at a tiny 64×64 image); SDXL VAE decoder/encoder are static 1024×1024 by default, dynamic H/W under `--dynamic-spatial` (decoder traced at 8×8 latent, encoder at 64×64 image)
- **Encoder fp16 fix-pass selection (deliberate, not inherited):** the dynamic (legacy-traced) SDXL encoder runs `fix_resize` but **NOT** `fix_fp32_constants` — the same choice as the dynamic decoder and the SD1.5 encoder. `fix_fp32_constants` converts an fp32 initializer feeding an otherwise-fp32 `Div` (GroupNorm variance) to fp16, leaving a mixed-operand `Div` that ORT rejects at load (`Div bound to different types (float16 and float)`). The **static dynamo** encoder keeps `fix_fp32_constants` unchanged. See `SDXLExportPolicy.should_fix_fp32_constants`.
- fp16 Resize patching is applied by default to fp16 UNet/VAE components (the dynamic SD 1.5 VAE decoder's Resize nodes are exactly what it targets); `patch_unet_resize.py` is only for older exports
- SDXL VAE decoder/encoder require the same fp32-constant and fp16-Resize fixes as the UNet (all applied in a single `fix_fp16_graph` pass)

---

# SDXL dynamic-spatial: GPU-box run sequence

The dev box has no torch/GPU/SDXL checkpoint, so a `--dynamic-spatial` export is
validated on the GPU box. Guillaume's flow — **one export, validate, then one import
decision** (there is no static SDXL to fall back to):

```bash
# 1. Export (dynamic UNet + fp16 decoder + fp32 decoder hedge + dynamic encoder).
#    Traces UNet at 16×16, both decoders at 8×8, encoder at 64×64 image, so it is far
#    faster than a full static SDXL export (see docs/90_debug/performance.md).
python scripts/sdxl_export_onnx_models.py <sdxl.safetensors> --name sdxl_dyn --dynamic-spatial

# 2. Validate the graphs + numerics. Run nvidia-smi alongside to capture VRAM peaks
#    (the validator prints a ">>> ... watch nvidia-smi now" marker before each forward).
nvidia-smi -l 1 &                 # in another terminal, or: nvidia-smi dmon
python scripts/validate_dynamic_sdxl.py --model models/sdxl_dyn \
       --torch-reference <sdxl.safetensors> --sizes 128,160,192

# 3. Read the checks:
#    - unet.dynamic / vae_decoder.dynamic / vae_encoder.dynamic -> H/W are symbolic
#    - unet.run@{128,160,192}              -> finite, time_ids scaled per size accepted
#    - vae.run@{...} std                   -> finite, non-degenerate (NaN/near-constant = fp16 VAE fails)
#    - *.numerics                          -> ONNX matches torch to fp16 precision
#    RESULT: PASS = go.

# 4. PIXEL-MODE hires additions (this change — the encoder is now dynamic):
#    - Confirm vae_encoder input "image" H/W are symbolic (pixel_hires_capable=true):
#        python -c "import onnx;m=onnx.load('models/sdxl_dyn/vae_encoder.onnx',load_external_data=False);\
#          d=[i for i in m.graph.input if i.name=='image'][0].type.tensor_type.shape.dim;\
#          print('image dims:',[x.dim_param or x.dim_value for x in d])"
#    - Encoder forward at the hires image sizes with VRAM sampled (NEW profile, never
#      measured): feed 1216×1216 and 1536×1536 images, watch nvidia-smi peaks.
#    - In-app A/B at scale=1.2, SAME seed: pixel mode vs latent mode. The staircase
#      artifact must be ABSENT in pixel and PRESENT in latent (confirms the fallback
#      diagnosis and the fix). model.json capabilities show pixel_hires_capable=true.

# 4a. If vae.run showed NaN/near-constant (SDXL fp16 VAE instability at >native): swap in
#     the fp32 decoder hedge (see the "SDXL fp32 VAE-decoder hedge" rule above), then re-run
#     the validator to confirm vae_fp32 decodes cleanly. No re-export.
# 4b. Import ONCE into the app (in-app import produces a static SDXL; to keep the dynamic
#     model, point the app's model dir at models/sdxl_dyn — it already has a full model.json
#     with the derived hires_capable flag).
```

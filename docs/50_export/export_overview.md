# Export Overview

## Scripts

| Script | Target |
|---|---|
| `scripts/export_onnx_models.py` | SD 1.5 → ONNX |
| `scripts/sdxl_export_onnx_models.py` | SDXL → ONNX |
| `scripts/export_common.py` | Shared utilities: `ExportComponentSpec`, policies, graph fixes, ORT validation |

Both scripts write a `model.json` in the output directory via `write_model_json(output_dir, model_type, all_specs)`. The C++ loader reads this file at startup to detect model type and validate component presence.

## `--resume`

If all output files for a component already exist (`<name>.onnx`, `<name>.onnx.data`, and `<name>_weights.safetensors` when applicable), that component is skipped. Use after an interrupted export without re-running completed components.

```bash
python scripts/export_onnx_models.py model.safetensors --resume
python scripts/sdxl_export_onnx_models.py model.safetensors --resume
```

## LoRA companion weights

`ExportComponentSpec(export_lora_weights=True)` saves `<name>_weights.safetensors` alongside the ONNX file. This file contains all 2-D (linear) parameters from `model.named_parameters()`, which match ONNX initializer names exactly when exported with `keep_initializers_as_inputs=True`.

Required for `LoraInjector` to function. Without it, LoRA injection is silently disabled for that component.

## `ExportComponentSpec` flags

The SDXL UNet uses `SDXLExportPolicy` which enables all three post-export passes:

| Flag | Pass |
|---|---|
| `fix_fp32_constants=True` | Converts stray fp32 Constant nodes to fp16 |
| `fix_attention_sqrt_cast=True` | Rewrites `Cast(to=float32)` on `Sqrt` outputs to fp16 |
| `fix_resize_fp16=True` | Wraps Resize data/scales inputs with correct Cast nodes |

See `docs/50_export/sdxl_fp16_fixes.md` for the full rationale.

## Output layout

```
<output_dir>/
  model.json
  text_encoder.onnx
  text_encoder.onnx.data
  text_encoder_weights.safetensors
  unet.onnx
  unet.onnx.data
  unet_weights.safetensors
  vae_decoder.onnx
  vae_decoder.onnx.data
  # SDXL only:
  text_encoder_2.onnx
  text_encoder_2.onnx.data
  text_encoder_2_weights.safetensors
```

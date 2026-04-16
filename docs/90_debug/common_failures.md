# Common Failures

## What this file explains
Index of failure symptoms mapped to the right diagnosis doc. Each entry names the symptom, the most likely cause, and where to look.

## When to use this
- Something is broken and you need to find the right doc fast

---

## Black or NaN image

**Symptom:** Generated image is pure black; log shows `min=1e9 max=-1e9 mean=nan`.  
**Cause:** NaN propagated from text encoder → UNet cross-attention → eps → latent.  
**Where to look:** [`../30_lora/lora_debugging.md`](../30_lora/lora_debugging.md) — NaN section (fp16 conversion bug in LoRA delta, or missing companion weights).

---

## ORT type mismatch at inference

**Symptom:** `"Type parameter (T) bound to different types"` or `"tensor(float16) and tensor(float)"` during `Run()`.  
**Cause:** Hard-coded dtype instead of detected flag; or wrong VAE input type.  
**Where to look:** [`../10_pipeline/dtype_handling.md`](../10_pipeline/dtype_handling.md)

---

## ORT type error during export

**Symptom:** ORT validation fails with type error on a Cast or Mul node.  
**Cause:** fp32 contamination in SDXL UNet graph.  
**Where to look:** [`../50_export/sdxl_fp16_fixes.md`](../50_export/sdxl_fp16_fixes.md)

---

## LoRA has no effect / 0 patches

**Symptom:** `LoraInjector: total N patch(es), 0 miss(es)` but image unchanged; or log shows `0 external initializer(s)`.  
**Cause:** Missing companion weights file, wrong export flags, or key prefix mismatch.  
**Where to look:** [`../30_lora/lora_debugging.md`](../30_lora/lora_debugging.md), [`../30_lora/lora_key_format.md`](../30_lora/lora_key_format.md)

---

## LoRA high miss count

**Symptom:** `check_lora_compat.py` reports < 80% matched; log shows many misses.  
**Cause:** LoRA trained on different architecture, or non-standard Kohya prefix.  
**Where to look:** [`../30_lora/lora_key_format.md`](../30_lora/lora_key_format.md) — prefix table and pitfalls

---

## Cancel had no effect

**Symptom:** Clicking Cancel doesn't stop generation; or generation stops but VAE decode still runs.  
**Cause:** `run_opts` not being passed to UNet; or VAE is expected not to cancel (by design).  
**Where to look:** [`../10_pipeline/cancellation.md`](../10_pipeline/cancellation.md)

---

## Generation silently running on CPU

**Symptom:** Generation is very slow after first image; log shows `UNet GPU pass failed`.  
**Cause:** `dmlFailed = true` after a DML runtime error; flag persists in session cache.  
**Where to look:** [`../10_pipeline/gpu_fallback.md`](../10_pipeline/gpu_fallback.md)

---

## LoRA change not applied after UI update

**Symptom:** `ModelManager: cache hit — reusing loaded sessions.` after LoRA change.  
**Cause:** Scale not propagated to `GenerationParams`; or change < 0.0005 treated as same key.  
**Where to look:** [`../20_models/model_cache.md`](../20_models/model_cache.md)

---

## First image very slow

**Symptom:** First generation after model/LoRA change takes much longer than subsequent ones.  
**Cause:** ORT JIT compilation on first `Run()` call.  
**Where to look:** [`performance.md`](performance.md) — session warmup section

---

## Model load error / 0 patches on third-party model

**Symptom:** Sessions load but LoRA shows 0 patches; manifest logs `[MISSING]`.  
**Cause:** Third-party ONNX lacks `keep_initializers_as_inputs=True` or companion weights file.  
**Where to look:** [`../40_onnx/external_data.md`](../40_onnx/external_data.md) — export requirements section; [`../50_export/export_validation.md`](../50_export/export_validation.md)

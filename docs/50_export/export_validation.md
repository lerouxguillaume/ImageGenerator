# Export & Import Validation

Validation happens in **two layers** after an export completes. Both run inside
`scripts/import_model.py` during the import pipeline; the second layer gates
registration (see `docs/55_import/import_pipeline.md`).

---

# Layer 1 — file / structure validation

`import_model.py::validate_output()` — cheap, no inference.

Detects:
- missing component files (`*.onnx`, `*.onnx.data`) for the arch
- missing per-component LoRA weight sidecars
- (during per-component export) dtype mismatch and invalid graph structure via a
  dummy ORT forward pass in `export_common.py`

This proves the files exist and load — it does **not** prove they produce useful
output.

---

# Layer 2 — inference smoke test (`scripts/verify_model.py`)

`PROGRESS:verifying` phase. Loads the exported components in ONNX Runtime (CPU)
and runs a real end-to-end pass. Arch-aware (sd15 / sdxl); reads each session's
own input metadata so it feeds the exact dtype/shape the graph expects (no
hard-coded fp16/fp32).

| Check | Gate | Catches |
|---|---|---|
| `text_encoder` finite + two different prompts diverge (cosine < 0.9995) | **hard** | fp16 attention collapse (all prompts → identical embeddings, CFG dead) |
| `text_encoder_2` (SDXL) same | **hard** | same, on OpenCLIP-G |
| `unet` single forward is finite | **hard** if it runs | fp16 overflow producing NaN/Inf latents |
| `vae_decoder` finite + non-degenerate variance | **hard** on NaN | dead/constant VAE |

### Execution model — isolation, CPU/GPU gating

- `verify_model.py` runs in a **separate process** spawned by `import_model.py`
  (`run_verification`), not in-process. The export step loaded torch + a full
  pipeline; loading the multi-GB ONNX on top of that resident memory can OOM-kill
  the process. A fresh interpreter (ORT + numpy only) is lighter and crash-isolated.
- Exit codes: `0` = pass, `3` = genuine hard failure (reject import), anything
  else (signal / OOM / access violation / timeout) = **inconclusive → import
  proceeds with a warning**. This is why a verifier crash never blocks a clean
  export.
- The **text-encoder** check runs on the **CPU** EP (fp32, always safe) and is the
  hard gate.
- The **UNet/VAE forward** passes run **only when a GPU ORT provider is present**
  (`CUDA` / `DirectML` / `ROCm`). On a CPU-only `onnxruntime` build they are
  **skipped** (`VERIFY:skip`), because the CPU EP can hard-crash on fp16 UNet
  graphs (access violation). Install `onnxruntime-gpu` / DirectML in the import
  venv to enable them.

Near-identical cond/uncond or low prompt separation warn rather than fail.

`verify_model()` returns `False` if any **hard** check failed; `import_model.py`
then emits `ERROR:` and exits `3`, so the model is **never added to the
registry**.

## Protocol

Each check emits one line consumed by the C++ `ModelImporter`:

```
VERIFY:ok:<check>:<detail>      passed
VERIFY:warn:<check>:<detail>    non-fatal
VERIFY:skip:<check>:<detail>    not run (e.g. no GPU ORT for fp16 UNet/VAE)
VERIFY:fail:<check>:<detail>    fatal — import rejected
```

The modal renders these as a colour-coded verification list.

---

# Why a real forward pass

File checks cannot see the two failure modes we have actually hit:

1. **Attention collapse** — an fp16 CLIP encoder run natively in fp16 maps every
   prompt to nearly the same embedding. Files are all present and load fine; the
   model is useless. Only a divergence probe between two different prompts
   catches it. (Fix: export text encoders in fp32 — see
   `docs/50_export/sdxl_fp16_fixes.md`.)
2. **fp16 UNet overflow** — a bad export produces NaN/Inf on the first denoise
   step. A single real forward surfaces it.

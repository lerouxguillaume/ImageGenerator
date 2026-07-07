# SDXL Hires — Investigation & Export Groundwork (Findings)

Status: **INVESTIGATION COMPLETE — fully verified end-to-end on Guillaume's Windows CUDA box**
(Animagine XL 3.1, 2026-07-07). Export 141s; structural checks PASS; **fp16 UNet + VAE both
finite through 1536px (1.5×) on a 12 GB card**; one real export bug (fp16 VAE `Div` type
mismatch) found *and fixed*; VRAM ceiling measured; tiled-decode need scoped. No unknowns
remain that block the implementation phase. Deliverables: additive, opt-in export changes
(non-breaking) + a GPU-box validation harness + this report. **No C++ feature code, no pipeline
changes** (per scope). Companion to `hires_fix_plan.md` §7 (which deferred this track).

Legend: ✅ verified on dev box · ✅🖥️ verified on Guillaume's GPU box (export/structure) ·
⛔ blocked (needs GPU ONNX Runtime: fp16 forwards / VRAM / numerics) · 📎 committed change ·
🔬 run-instruction for Guillaume.

## Runtime status (what the successful export + validator run settled)

| Item | Status |
|---|---|
| SDXL UNet exports with dynamic H/W axes (the make-or-break) | ✅🖥️ 48.6s, verified graph has `latent`+`time_ids`, fp16 fixes applied |
| VAE decoder (fp16) dynamic H/W | ✅🖥️ 43.4s, `onnx.checker`+shape-infer clean |
| VAE decoder **fp32 hedge** emitted + dynamic H/W | ✅🖥️ 1.3s, `vae_decoder_fp32.onnx` (198 MB) |
| `time_ids (1,6)` input present on UNet | ✅🖥️ validator `unet.time_ids_input` OK |
| VAE encoder dynamo cost (the "wildcard") | ✅🖥️ 3.4s — NOT a bottleneck; dynamo does no eager fp16 forward |
| Full export wall-time | ✅🖥️ **141s** total (vs ~60 min static) |
| **fp16 UNet** finiteness at latent 128/160/192 (>native) | ✅🖥️ **finite at all three** on CUDA — range ~[-4.6, 4.7], no overflow. VRAM device-peak **6.8 / 8.9 / 12.2 GB** (1024/1280/1536) |
| **fp16 VAE decoder** finiteness (after the Div fix) | ✅🖥️ **finite at all three incl. 1536**, `std≈0.635/0.638/0.637`. VRAM peak **5.1 / 9.2 / 12.2 GB** — **fits 1536 where fp32 can't** |
| **fp16 Div bug** (decoder wouldn't load) | 🐛→✅🖥️ **fixed + confirmed.** Was `Type Error: Div bound to (float16, float)` from `fix_fp32_constants` over-converting an fp32 `Div` operand. Fix: `should_fix_fp32_constants` skips the dynamic vae_decoder (matches SD1.5). Re-export + re-validate → loads clean, finite. |
| **VRAM ceiling** (Guillaume's card ≈ **12 GB**) | ✅🖥️ full fp16 pipeline (UNet+VAE) fits **through 1536 / 1.5×** (~12.2 GB, tight). |
| **fp32** decoder | ✅🖥️ finite at 1024/1280; **OOM at 1536** — mid-block self-attention Softmax needs a **5.4 GB** buffer (O(N²)). It's the numeric-stability spare, but memory-costlier than fp16. |
| **Tiled VAE decode** | ✅🖥️ **NOT needed for fp16 ≤ 1.5×** (fp16 fits 1536). Becomes required for the **fp32** path at ≥1536, or for **fp16 at ~2×** on a 12 GB card. Keep the stub; promote when 2× is targeted. |
| ONNX-vs-torch numerics | ⏸️ optional; needs a CUDA torch build too |

---

## TL;DR — go/no-go

**GO to the implementation phase.** The one item that could have killed the track —
"can the SDXL UNet even express dynamic spatial axes?" — resolves favorably on a
structural basis that needs no exotic exporter archaeology:

- The SDXL UNet **already exports through the legacy tracer** (`ExportPolicy.unet_exporter()`
  default `"legacy"`; `SDXLExportPolicy` never overrode it) and **already traces
  successfully with `time_ids` + `text_embeds`** at 128×128. The legacy tracer is the
  same path SD1.5 uses to get dynamic H/W. So dynamic-spatial SDXL is a **declaration +
  trace-size change, not a new export mechanism.** `added_cond_kwargs` (time_ids/text_embeds)
  are (batch, 6)/(batch, 1280) vectors with no spatial tie, so they do not obstruct a
  small-latent trace.
- This is a materially lower-risk finding than `hires_fix_plan.md` §7 assumed ("needs a
  UNet re-export down the fragile fp16 legacy path"). It's the *same* legacy path already
  in production for SDXL, just with two axes added.

The genuine unknowns are all **numeric/empirical**, not structural, and all need the GPU
box: (1) does fp16 SDXL hold up at >native latents (UNet overflow, and SDXL's known fp16
**VAE** instability), and (2) VRAM ceiling. The committed groundwork lets Guillaume answer
both with one export + one script.

Three design decisions the implementation phase must make (detail in A5 + §Decisions).

---

## A1 — Current-state audit ✅

### UNet exporter & dynamic axes
- **Exporter: legacy.** `ExportPolicy.unet_exporter()` returns `"legacy"` (`export_common.py:353`);
  `SDXLExportPolicy` did not override it. The SDXL export calls `policy.unet_exporter()`
  (`sdxl_export_onnx_models.py:216` pre-change).
- **Dynamic axes (pre-change): batch-only.** `SDXLExportPolicy.unet_dynamic_axes()` declared
  `{0:"batch"}` on `latent`/`timestep`/`encoder_hidden_states`/`text_embeds`/`time_ids`/`latent_out`
  — no H/W. Traced at **128×128** (`latent_h, latent_w = 128, 128`).

### VAE decoder exporter & dynamic axes
- **Exporter: dynamo.** `SDXLExportPolicy.vae_exporter()` returned `"dynamo"`.
- **Static.** `vae_dynamic_axes()` returned `None`. Traced at 128×128. (VAE **encoder** likewise
  static 1024, and — a latent bug — was routed through `policy.vae_exporter()` not
  `vae_encoder_exporter()`; harmless while both were dynamo. Fixed as part of A2/A3, see below.)

### Where `time_ids` is computed and consumed
- **Computed once, at text-encode time:** `SdPipeline.cpp:345-350` (inside `runPipeline`):
  ```cpp
  const float h = cfg.image_h;  const float w = cfg.image_w;
  ctx.time_ids = { h, w, (float)params.cropTop, (float)params.cropLeft, h, w };
  ```
  i.e. `[orig_h, orig_w, crop_top, crop_left, target_h, target_w]` with
  **original_size == target_size == the (only) generation resolution**, crop from
  `params.cropTop/cropLeft` (`PortraitGeneratorAi.hpp:36-37`, default 0).
- **Consumed:** `SdUNet.cpp:57-81` (`appendSDXLInputs`) pushes it as `unet_in4`, shape `(1,6)`,
  fp16 or fp32 per `ctx.unetExpectsFp32`. Only for `isXL`.

### Everything resolution-dependent for SDXL
Exhaustive list (verified by grep + reading the pipeline):
1. `ctx.latent_shape` / `ctx.latent_size` — set once at load (`SdLoader.cpp:499-500`).
2. `ctx.time_ids` — set once at encode (`SdPipeline.cpp:347`).

Nothing else. `embed_shape` is token/prompt-dependent (not resolution); `memory_info`,
sigmas, `run_opts`, pooled embeds are resolution-independent. **So SDXL adds exactly one
resolution-dependent field over SD1.5 (`time_ids`)** — this is the crux of the implementation
delta versus the landed SD1.5 hires.

### How the SDXL VAE handles fp16 today
- Exported by casting the checkpoint's own VAE to fp16: `pipe.vae.to(torch.float16)`
  (`sdxl_export_onnx_models.py:233, 237`), then the fp16-graph fixes are applied —
  `should_fix_fp32_constants("vae_decoder") → True` and `should_fix_resize_fp16(...) → True`
  (`export_common.py` `SDXLExportPolicy`), both run in one `fix_fp16_graph` pass.
- **No fp16-fix VAE weights** (madebyollin-style) are used anywhere — grep for
  `madebyollin`/`fp16_fix` is empty. The VAE is the checkpoint's, merely cast.
- **Runtime decode dtype is auto-detected** from the graph input type
  (`SdLoader.cpp:475-477` → `ctx.vaeExpectsFp32`), and `SdVae.cpp:30` branches on it. Since the
  export is fp16, SDXL decodes in **fp16**. There is **no fp32-decode path wired for SDXL**
  (it would require an fp32 `vae_decoder` export variant, which is not produced).
- **Implication:** SDXL's well-known fp16-VAE instability is currently unmitigated except by
  the Resize/const fixes. At native 1024 it's fine in practice; **at >native latents it is the
  single most likely failure mode** (see A3).

---

## A2 — Dynamic-spatial SDXL UNet (make-or-break) — 📎 committed, ✅🖥️ EXPORT VERIFIED

**Structural viability: PROVEN.** No longer a prediction: the `--dynamic-spatial` export ran on
Guillaume's box and the UNet exported with dynamic H/W axes via the legacy tracer at a 16×16
trace in **48.6s**. The verified graph carries `latent`, `timestep`, `encoder_hidden_states`,
`text_embeds`, `time_ids` inputs and `latent_out`; the fp16 fixes fired (144 embedded fp32
constants → fp16, 140 `Cast(Sqrt→fp32)`→fp16, 2+2 Resize inputs Cast-wrapped). `onnx.checker` +
shape inference clean; the validator confirms `latent` H/W are symbolic. The stop-condition
("legacy fp16 SDXL UNet export fails structurally") did **not** trigger.

Committed (all behind `--dynamic-spatial`, default off = byte-identical to today):
- `SDXLExportPolicy(dynamic_spatial=False)` gains H/W axes on `latent`/`latent_out` when enabled
  (`export_common.py`). Exporter stays legacy (already was).
- Small-latent trace: UNet traced at **16×16** in dynamic mode (`sdxl_export_onnx_models.py`),
  sidestepping the ~1000× fp16-CPU-conv slowness (same trick as SD1.5's 8×8 VAE). 16 keeps SDXL's
  two UNet downsamples well-defined (16→8→4 bottleneck) and stays /8-aligned; the dynamic graph is
  trace-size-independent.
- `--dynamic-spatial` CLI flag threaded `sdxl_export_onnx_models.py → export_sdxl → SDXLExportPolicy`.

✅🖥️ **Verified via** (16×16 traced cleanly — no minimum-size issue, so the 32×32 fallback was
never needed):
```bash
python scripts/sdxl_export_onnx_models.py <sdxl.safetensors> --name sdxl_dyn --dynamic-spatial
python scripts/validate_dynamic_sdxl.py --model models/sdxl_dyn            # structural: PASS
```
The validator ran structural checks (checker + shape-inference + symbolic-dim probe +
`time_ids` input) — all OK. The fp16 runtime forwards and the `--torch-reference` numerics
still ⛔ need GPU ONNX Runtime (`onnxruntime-gpu`); on a CPU-only venv the validator cleanly
`SKIP`s them (fp16 CPU EP is ~1000× slow and can segfault the fp16 Resize graph).

---

## A3 — Dynamic SDXL VAE decoder — 📎 committed, ✅🖥️ EXPORT VERIFIED, ⛔ fp16 >native stability needs GPU

Committed (dynamic mode only), and ✅🖥️ verified: both the fp16 decoder (43.4s) and the fp32
hedge (1.3s) exported with dynamic H/W axes; `onnx.checker` + shape-inference + symbolic-dim
probe all PASS on both. The fp16 pass applied 7 fp32-const fixes + 3+3 Resize Cast-wraps.
- `SDXLExportPolicy.vae_dynamic_axes()` → H/W dict; `vae_exporter()` → **legacy** (dynamo can't
  express spatial axes — same finding as SD1.5). Decoder traced at **8×8** (mirrors SD1.5's proven
  value). `should_fix_resize_fp16("vae_decoder")` already `True`, so the dynamic Resize nodes get
  the fp16 Cast wrap.
- **Encoder stays static 1024 on dynamo.** Fixed the latent bug where the SDXL script fed the
  encoder `policy.vae_exporter()` (would become legacy → a static-1024 legacy fp16-CPU trace =
  the ~1000× slow path). It now uses `policy.vae_encoder_exporter()`, which I overrode to return
  `"dynamo"` unconditionally. SDXL hires is **latent-mode** (decode-larger only); no re-encode of
  an upscaled image, so the encoder needs no dynamic axes (unlike SD1.5 pixel-mode).

⛔ **The key risk lives here.** SDXL fp16 VAE is known to produce NaN/washed-out output at some
resolutions. 🔬 The validator explicitly checks decode finiteness **and** output std at 128 / 160 /
192 and flags near-constant output. If fp16 is unstable at >native:
- The import already has **no fp16-fix weights** and **no fp32 decode path** for SDXL, so the
  fallback is a **new fp32 `vae_decoder` export variant** (export the VAE without the `.to(fp16)`
  cast; `ctx.vaeExpectsFp32` then auto-selects fp32 decode with zero C++ changes). This is a
  design decision for the implementation phase, gated on the empirical result.

---

## A4 — Capability + import plumbing — ✅ verified arch-agnostic, 📎 made self-validating

- **C++ side is already arch-agnostic.** `hires_capable` flows generically:
  `model.json → ImportedModelRegistry.cpp:64 → ModelEntry → SettingsPanel`; `SdLoader.cpp:209`
  reads it into `cfg.hiresCapable`; `SdPipeline.cpp:451` guards hires on it. Nothing in the data
  path assumes SD1.5.
- **UI gate NOT changed** (per scope): `SettingsPanel.cpp:304-308` keeps `SD15 && hiresCapable`.
  It stays SD15-only until the implementation phase.
- 📎 **`hires_capable` is now derived from the actual exported graphs** rather than hardcoded
  `arch == "sd15"` (`import_model.py::_detect_hires_capable` + `_graph_has_dynamic_spatial`):
  it flips `true` only when **both** the UNet and VAE decoder carry dynamic H/W axes — literally
  "only when both dynamic exports succeeded," as A4 asks. SD1.5 (always dynamic) → `true`;
  static SDXL → `false`; `--dynamic-spatial` SDXL → `true` automatically. Falls back to the old
  arch heuristic if the graphs can't be probed (onnx missing / parse error), so it never
  regresses SD1.5.
- ✅ **`verify_model.py` already supports a dynamic-spatial SDXL graph** with no change: `_fixed()`
  substitutes a small test size for symbolic dims and `_build_unet_feed` scales `time_ids` from the
  chosen latent size (`verify_model.py:219-249`). So a dynamic SDXL export will pass the existing
  import-time smoke test.
- Note: the in-app import flow does **not** pass `--dynamic-spatial` (no UI knob, per scope), so
  in-app SDXL imports remain static. Guillaume drives the experimental export from the CLI.

---

## A5 — Empirical measurements — ⛔ all need the GPU box

Scripted (`validate_dynamic_sdxl.py`): finiteness/range/std at latent 128/160/192, dynamic-vs-static
numerics, and a `>>> ... watch nvidia-smi now` marker before each forward so peaks can be attributed
to a size. What the script cannot self-measure (🔬 run-instructions):

- **VRAM.** ORT does not expose CUDA peak to Python. Run `nvidia-smi -l 1` (or `nvidia-smi
  dmon`) alongside the validator; record peak at each UNet forward (128/160/192) and each VAE
  decode. SDXL UNet activations already ~4× SD1.5 at native; expect the OOM wall somewhere around
  1.5–2× on a consumer card. **Report where OOM lands** — it decides whether tiled decode is
  promoted from stub to requirement.
- **First-run shape stall.** Time the first UNet forward at a *new* latent size vs the second at
  the same size (ORT/CUDA does per-shape kernel setup once). The gap is the one-time stall the
  implementation may want to warm up.

### `time_ids` semantics for a second pass (doc-cited recommendation)
diffusers' `StableDiffusionXLImg2ImgPipeline` builds add-time-ids from
`(original_size, crops_coords_top_left, target_size)` and, in `__call__`, defaults both
`original_size` and `target_size` to `(height, width)` — the **resolution actually being
denoised** — with `crops_coords_top_left=(0,0)` by default
(`diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py`,
`_get_add_time_ids` + the `__call__` defaults). A hires pass *is* img2img against the upscaled
latent, so:

> **Recommendation:** for the hires (pass-2) UNet, set `time_ids = {H2, W2, 0, 0, H2, W2}` where
> `H2×W2` is the **upscaled** pixel resolution — original_size and target_size both track the
> current pass. Keep crop at 0. (The base pass keeps its native `time_ids` unchanged.)

This means the C++ side must **recompute `time_ids` per pass** — today it is computed once at
native (`SdPipeline.cpp:347`). That is the SDXL analogue of `ScopedLatentResolution`: a per-pass
`time_ids` swap that restores on exit. (Alternative convention — pin original_size to the base
resolution to signal "upscaled from low-res" — exists in some community pipelines but is *not* the
diffusers img2img default; treat it as a tunable, not the default.)

---

## Committed export-side diffs (all additive, default behavior unchanged)

| File | Change |
|---|---|
| `scripts/export_common.py` | `SDXLExportPolicy.__init__(dynamic_spatial=False)`; H/W axes on UNet + VAE decoder when enabled; `vae_exporter()`→legacy when dynamic; `vae_encoder_exporter()`→dynamo (encoder always static-1024) |
| `scripts/sdxl_export_onnx_models.py` | `--dynamic-spatial` flag; small-latent trace (UNet 16, VAE dec 8) in dynamic mode; encoder now uses `vae_encoder_exporter()` (bugfix) |
| `scripts/import_model.py` | `hires_capable` derived from actual graphs (`_detect_hires_capable`), true only when UNet+decoder both dynamic; safe fallback to arch heuristic |
| `scripts/validate_dynamic_sdxl.py` | **new** GPU-box validator (checker/shape-infer/dynamic-probe + forward at 128/160/192 + numerics vs static) |
| `docs/50_export/export_overview.md` | documented the experimental opt-in path + validator |

Default `export_sdxl(...)` (and the in-app import) produce the **exact same static-1024 SDXL model
as before** — verified by reading the diff: `dynamic_spatial=False` reproduces every prior axis
dict, exporter choice, and trace size. Nothing here can break Guillaume's working SDXL import.

Dev-box verification performed: `py_compile` on all four Python files ✅. Runtime behavior ⛔ (no
torch/GPU/checkpoint) — deferred to the GPU box via the validator.

---

## Decisions the implementation phase must make

1. **Per-pass `time_ids` mechanism (required).** Recompute `time_ids` for the hires pass from the
   upscaled resolution (`{H2,W2,0,0,H2,W2}`, diffusers-img2img convention, cited above), and restore
   it on the base/cancel/exception paths — the SDXL twin of `ScopedLatentResolution`. This is the
   only *new* resolution-dependent state SDXL adds over SD1.5.
2. **fp16-VAE fallback (data-driven).** If the validator shows fp16 SDXL decode is unstable at
   >native, ship an fp32 `vae_decoder` export variant for SDXL hires (auto-selected via
   `ctx.vaeExpectsFp32`, no C++ change) — vs accepting fp16 if it proves clean.
3. **VRAM ceiling → tiled decode promotion (data-driven).** SDXL at 1.5–2× may OOM where SD1.5
   didn't. If A5's peaks cross Guillaume's card, promote `VaeDecodeStrategy`'s tiled route from stub
   to requirement for SDXL hires (and cap/step-down the max hires scale).

Secondary: whether to relax the `SdPipeline.cpp:300-309` native-resolution guard for SDXL (it
currently reverts any non-native *base* resolution; hires works internally via scoped resolution, so
this only matters if arbitrary base SDXL resolutions are also wanted) — out of scope for hires itself.

---

# Update — Session 2: adapt to "no SDXL model, one export ever"

Situation change: Guillaume has **no SDXL model imported** (static or otherwise). So there is
no static export to diff against and no existing SDXL workflow to protect — the plan is now
**one `--dynamic-spatial` export, validate, one import ever.** Export-side Python only; default
(non-flag) and SD1.5 exports remain byte-identical (all new logic is flag-gated).

## S1 — fp32 VAE-decoder hedge in the same run ✅ (code) / ⛔ (runtime)
`--dynamic-spatial` now *also* emits `vae_decoder_fp32.onnx` (+`.data`) alongside the fp16
decoder (`sdxl_export_onnx_models.py` step 4a). Exported **from the original fp32 weights,
before the fp16 cast** (true fp32, not a lossy fp16→fp32 roundtrip); same dynamic H/W axes +
8×8 trace; **no fp16 graph fixes** (fp16-specific). It is an **unregistered spare** — not
appended to `exported`, so it is absent from `model.json` `components` and inert to the runtime.

**C++ auto-detect claim — VERIFIED against SdLoader.cpp:**
- `ctx.vaeExpectsFp32` is derived from the decoder's input[0] element type at load
  (`SdLoader.cpp:475-477`) and consumed in `SdVae.cpp:30`. Swapping in an fp32 graph flips it
  automatically → **zero C++ changes.** ✅
- **But the swap is a file swap, not a model.json repoint:** `SdLoader.cpp:279` loads the
  **hard-coded path `vae_decoder.onnx`**; it does *not* read the decoder filename from
  `model.json` (the `components` block is only logged by `validateManifest`, which never throws).
  So fp32 content must be placed at literal `vae_decoder.onnx`.
- The VAE decoder is **plain-loaded by ORT** (never LoRA-injected — lines 338/416 both use
  `loadSession(..., vaeBundle.onnxPath, ...)`), so ORT resolves external data via the location
  string embedded in the `.onnx`. Therefore `cp vae_decoder_fp32.onnx vae_decoder.onnx` works
  as-is: the copy still references `vae_decoder_fp32.onnx.data`, which stays in place. Exact
  commands in `docs/50_export/export_overview.md`. Gated to dynamic mode only (experiment hedge).

## S2 — validator: `--static` → `--torch-reference` ✅ (wiring) / ⛔ (numerics need GPU)
`validate_dynamic_sdxl.py` dropped the never-to-exist static-ONNX diff. New `--torch-reference
<checkpoint>` loads the **same checkpoint in PyTorch** (fp32 load → same tracing patches +
`disable_attention_upcasting` → fp16 cast, mirroring the exporter), feeds byte-identical inputs
to the ONNX UNet and to `UNetWrapper`, and gates on mean relative error (fp16 threshold 0.05,
max/mean-abs reported). Same for the VAE decoder at 128 — **both the fp16 and fp32 variants** if
present. Kept: checker/shape-infer, dynamic-axes probe, forwards at 128/160/192 with
finiteness + output-std, nvidia-smi markers. **Added an explicit `time_ids` acceptance check**:
a structural check that the UNet graph declares `time_ids` last-dim 6, plus per-size forwards
that feed `{H,W,0,0,H,W}` scaled to each latent size (diffusers img2img convention, the input
the per-pass hires swap hinges on) and confirm shape `(1,6)` is accepted with sane output.
Verified on dev box: `--help` shows `--torch-reference`, no-onnxruntime path exits 2 cleanly.

## S3 — capability fallback now fails CLOSED ✅ verified
`import_model.py::_detect_hires_capable`: when a graph can't be probed (missing file / onnx
import fail / parse error) it now returns **False + a loud `VERIFY:warn:capability:` line**,
instead of the old arch heuristic (which returned True for SD1.5 and could mislabel a
pre-dynamic **static-VAE** SD1.5 import as hires-capable → runtime shape error instead of a
disabled control). Probe-**success** paths are unchanged, so a normal SD1.5/SDXL import (onnx
present, as it always is in the venv) is byte-identical. Verified: fallback returns False and
emits the warning (dev-box run).

## S4 — dynamic-export wall-time — ✅🖥️ MEASURED (was reading-based; now actuals from the run)
Guillaume's full **static** SDXL import is ~1 hour, dominated by the **128×128 fp16 UNet eager
trace on CPU** (no fast fp16 conv kernel — [[project_fp16_export_validation]]). The dynamic path
traces the UNet at **16×16** and both decoders at **8×8**, collapsing exactly that bottleneck.
**Measured total on Guillaume's box (Animagine XL 3.1): 141s** — below the ~5–12 min estimate.

| Step | Static cost | Predicted | **Measured** | Note |
|---|---|---|---|---|
| Load checkpoint (fp32 pipeline) | ~0.5–1.5 min | unchanged | ~7 s (+ patches) | 6.9 GB, fast disk |
| Text encoder (CLIP-L, fp32) | ~10–20 s | unchanged | **5.0 s** | |
| Text encoder 2 (OpenCLIP-G, fp32) | ~0.5–1.5 min | unchanged | **26.2 s** | ~2.8 GB serialize |
| **UNet (16×16 trace + fixes + LoRA weights)** | **~30–50 min** | ~30–60 s | **48.6 s** | dynamic H/W, 4.5 GB LoRA weights, 1488 sidecars consolidated |
| VAE decoder fp32 hedge (NEW) | — | ~10–20 s | **1.3 s** | fp32 conv has a fast CPU kernel |
| VAE decoder fp16 | part of ~hour | ~10–20 s | **43.4 s** | 8×8 but fp16 CPU conv is the slow path even tiny — still fine |
| VAE encoder (dynamo, static-1024) | unchanged | "wildcard" | **3.4 s** | ✅ NOT slow — dynamo does no eager fp16 forward; wildcard resolved |

**Findings vs prediction:** total 141s, ~25× faster than static. The "wildcard" (would the
static-1024 dynamo encoder dominate?) is **resolved — no**: 3.4s. The one mild surprise is that
the **fp16** VAE decoder at 8×8 (43.4s) is far slower than the **fp32** hedge at 8×8 (1.3s) —
confirming fp16 CPU conv is the ~1000× slow path *even at tiny sizes*, which is exactly why the
tiny-trace trick is load-bearing (a 128×128 fp16 decoder trace would be hours).

## Copy-pasteable GPU-box sequence
```bash
# 1. Export (DONE — 141s). 2. Structural validate on the app venv (DONE — PASS, no install):
python scripts/sdxl_export_onnx_models.py <sdxl.safetensors> --name sdxl_dyn --dynamic-spatial
python scripts/validate_dynamic_sdxl.py --model models/sdxl_dyn         # structural: PASS

# 3. fp16 runtime + VRAM (STILL TODO) — needs GPU ONNX Runtime. Use a THROWAWAY venv so the
#    app's import venv (CPU onnxruntime) is untouched. nvidia-ml-py lets the validator sample
#    GPU VRAM at ~200 Hz around each forward and print the device-peak inline (no nvidia-smi
#    polling needed — it catches sub-second spikes nvidia-smi -l 1 would miss):
python -m venv /tmp/sdxl_val
/tmp/sdxl_val/bin/pip install onnxruntime-gpu onnx numpy nvidia-ml-py
#   onnxruntime-gpu wheels pin a CUDA major; if the CUDA/cuDNN DLLs aren't on PATH the CUDA EP
#   silently falls back to CPU. Supply them via pip to match the wheel (e.g. CUDA 13 build):
#   /tmp/sdxl_val/bin/pip install nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-cuda-runtime-cu13
#   The validator calls ort.preload_dlls() so these are found; it also SKIPs (never hangs) the
#   fp16 forwards if the GPU EP still doesn't actually load.
/tmp/sdxl_val/bin/python scripts/validate_dynamic_sdxl.py --model models/sdxl_dyn --sizes 128,160,192
#   (add --torch-reference <sdxl.safetensors> for ONNX-vs-torch numerics; needs a CUDA torch too)
# PASS -> use models/sdxl_dyn. If vae.run@* shows NaN/near-constant, swap in the fp32 hedge:
#   cd models/sdxl_dyn && mv vae_decoder.onnx vae_decoder.onnx.fp16bak \
#     && cp vae_decoder_fp32.onnx vae_decoder.onnx   # then re-run the validator
```

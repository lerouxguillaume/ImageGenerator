# SD1.5 Hires Fix — Implementation Plan

Status: **implemented on branch `hires-fix-sd15`** (Phases 0–3 landed; Phase 4
runtime validation pending on the GPU box — no model/GPU on the dev box).
Phase 1 seams merged to `main` (cd3aa6d). See the code-review checklist at the
bottom for exactly what was verified vs. what needs Guillaume's GPU box.
Scope: SD1.5 only. SDXL is a separate, larger track (see Non-goals).

---

## 1. Motivation & framing

"Hires fix" (generate at native resolution, then a second low-strength denoise
pass at a higher resolution) is the single biggest visual-quality lever for
SD1.5 output — it adds real detail rather than just enlarging.

The key reframe driving this plan: **hires fix is not a one-off feature, it is
the first instance of a general "post-base refinement" capability.** Refiner
passes, ESRGAN/pixel upscalers, tiled VAE decode, and face restoration are all
"things that happen to a latent or image after the base denoise." So this plan
invests in the *seams* that make those cheap to add later and lands hires fix as
the proof case — rather than bolting on an `if (hiresFix)` branch.

---

## 2. Investigation findings (why this is viable now)

Established while scoping (all verified against the code / empirically):

- **The SD1.5 UNet already exports dynamic spatial axes**
  (`scripts/export_common.py` → `SD15ExportPolicy.unet_dynamic_axes()`:
  `latent: {0:batch, 2:height, 3:width}`). It can already denoise at an arbitrary
  latent size. The runtime guard at `SdPipeline.cpp:141` that reverts non-native
  resolutions is *overly conservative* for SD1.5.
- **The img2img primitives already exist**: `denoiseSingleLatent(..., startStep,
  initLatent)` and the `startStep = (1-strength)*steps` truncation. A hires pass
  is literally "img2img against your own base latent."
- **The VAE decoder was the only blocker** — exported static
  `(1,4,64,64)→(1,3,512,512)`, so it rejects a larger latent. An old comment
  claimed dynamic-H/W VAE export hung `torch.onnx.export` for 1+ hour.
  **Re-tested on torch 2.11.0 (2026-07): dynamic-H/W legacy export completes in
  ~19s (vs ~29s static) — the hang is gone.** (Comment since updated in
  `export_common.py`.) So a dynamic VAE decoder re-export is cheap.
- **Only two `ctx` fields are resolution-dependent**: `ctx.latent_shape` and
  `ctx.latent_size` (set once at load, `SdLoader.cpp:497-498`). Everything else
  (`memory_info`, embeddings, sigmas, `run_opts`) is resolution-independent;
  `time_ids` is SDXL-only.
- **Sharp edge**: `ctx` is **cache-owned and returned by reference**
  (`SdPipeline.cpp:161`, `s_modelManager.get(...)`). Any per-pass mutation of its
  dims must be restored even on the exception/cancel paths, or the next
  generation inherits hires dims from the cache.

> Related tech debt: `backlog.md` item #1 ("Split cached model state from per-run
> mutable state") is the *deep* fix for this exact hazard — an immutable
> `LoadedModel` plus a fresh per-run `RunContext`. This plan's
> `ScopedLatentResolution` (below) is the *tactical* mitigation that makes hires
> safe today without that refactor; the two are compatible and the RAII guard
> becomes trivial-to-delete once the RunContext split lands.

---

## 3. The reframe: per-image generation as an explicit pipeline

Today the per-image loop body (`SdPipeline.cpp:275-327`) is an inline monolith:
`seed → denoise → decode → save`. Restructure into an explicit sequence:

```
base pass → [refinement passes…] → decode (strategy) → [image post-processors…] → save
```

Hires is one refinement pass. This structural change is what makes the feature
extensible instead of a special case.

---

## 4. Architecture — five seams

**1. `Latent` value type** — `{ std::vector<float> data; int w, h; }`.
Today resolution lives only in shared `ctx` state, the root of the
cache-corruption risk. Making resolution *travel with the data* is the biggest
correctness improvement. Passes read dims from the `Latent`, not `ctx`.
(Boundary type only — internal denoise loops keep using `ctx.latent_size`; see
seam 2.)

**2. `ScopedLatentResolution` (RAII)** — the only sanctioned way to point `ctx`
at a resolution. Constructor sets `ctx.latent_shape`/`latent_size`; destructor
restores. Every pass wraps its UNet + decode work in one. Makes the
"restore even on exception/cancel" hazard structurally impossible to forget.
`ctx` stays the runtime source of truth (minimal internal churn); only *mutation*
is constrained.

**3. `RefinementPass`** — uniform `Latent apply(Latent, GenerationContext&, ...)`.
`HiresPass` is the first. Driven by config, applied in sequence in
`generateOneImage`. Adding a refiner later = write one class.

**4. `VaeDecodeStrategy`** — `decodeLatent` dispatches on latent-vs-VAE-native
size. v1 ships the single-shot strategy (route A, dynamic VAE); the tiled
strategy is a stub with a clear interface so route B drops in when 2×+ targets
start OOMing.

**5. Image post-processor hook** — an ordered list applied after decode; identity
for now. Future home for ESRGAN / face restoration. **Deliberately just a
`std::vector<function>` seam — no registry/DSL (YAGNI).**

### Config shape

```cpp
enum class UpscaleMode { Latent /*, Pixel, Esrgan (future)*/ };

struct HiresConfig {
    bool        enabled  = false;
    float       scale    = 1.5f;   // target = native * scale, /8-aligned
    float       strength = 0.5f;   // pass-2 denoise fraction (0.3–0.7)
    int         steps    = 0;      // 0 = reuse numSteps
    UpscaleMode mode     = UpscaleMode::Latent;
};
```

`GenerationParams` gains one `HiresConfig hires;` (room for a future
`RefinerConfig` beside it). Flows automatically through
`SettingsPanel.generationParams → job.params → runPipeline`.

---

## 5. Phased plan

### Phase 0 — Export & capability (unblocks decode)
- `SD15ExportPolicy`: `vae_dynamic_axes()` → H/W dict; `vae_exporter()` →
  `"legacy"` (the ~19s path verified above).
- Emit `hiresCapable: true` in the `model.json` `capabilities` block at export;
  plumb `ModelCapabilities.hiresCapable` through
  `ImportedModelRegistry → ModelEntry → SettingsPanel` (mirrors
  `vaeEncoderAvailable`).
- **fp16 end-to-end revalidation** of the dynamic VAE (the Resize-fp16 patch
  touches exactly these nodes) + re-import existing models.

### Phase 1 — Seams (pure refactor, no behavior change)
- Introduce `Latent`, `ScopedLatentResolution`; extract the loop body into
  `generateOneImage(...)`. Migrate `denoiseSingleLatent` / `decodeLatent` /
  `encodeImage` call sites to the new boundary.
- **Acceptance: output bit-identical to today.** Ship/verify this independently
  before any feature code — it is independently valuable even if hires were cut.

### Phase 2 — Hires pass
- `upscaleLatent(Latent, mode)` — `cv::resize` per channel (bilinear).
- `HiresPass::apply` — `ScopedLatentResolution` → `denoiseSingleLatent` from
  `startStep = (1-strength)*steps` with the upscaled latent as init (reuses the
  img2img path verbatim). Wire into `generateOneImage`'s refinement list from
  `HiresConfig`.

### Phase 3 — Decode strategy + UI
- `VaeDecodeStrategy` dispatch (single now, tiled stub).
- `SettingsPanel` controls, **gated on `SD15 && hiresCapable`** (LoRA-gate
  pattern).
- `GenerationStage::HiresDenoising` (`enums.hpp`) + cumulative step accounting so
  the progress bar spans both passes.

### Phase 4 — Validation (via `verify` skill / real runs)
- Base vs hires A/B; **cancel mid-hires** (proves the RAII restore); VRAM at 1.5×
  and 2×; reproducibility for a fixed seed.

---

## 6. Extensibility payoff

| Future feature | Where it slots | Touches base loop? |
|---|---|---|
| SDXL refiner pass | new `RefinementPass` | no |
| ESRGAN / face restore | image post-processor | no |
| Tiled decode (VRAM) | new `VaeDecodeStrategy` | no |
| Pixel-space hires | new `UpscaleMode` | no |
| Arbitrary output res (SD1.5) | `ScopedLatentResolution` already enables it | relax `:141` guard only |

---

## 7. Non-goals (explicit)
- **SDXL hires** — the SDXL UNet is batch-only dynamic (static spatial), so it
  needs a UNet re-export down the fragile fp16 legacy path plus `time_ids`
  rework. Separate, larger track.
- Post-processor registry/DSL.
- Multi-seed hires (single seed for v1; `HiresConfig` leaves room).

---

## 8. Risks & validation
- **fp16 dynamic VAE**: the ~19s re-test was fp32/CPU to isolate shape
  propagation. Re-validate a real fp16 dynamic export end-to-end (dynamic Resize
  is exactly what the fp16 patch handles). No hang expected; confirm numerics.
- **VRAM at 2×**: UNet activations scale ~4×; this is the real ceiling and the
  reason hires is a toggle. Route B (tiled decode) is the mitigation.
- **First hires-shape run**: ORT/CUDA does per-shape setup on first call at a new
  size (one-time stall). Load-time `warmupSession` only warms native — optionally
  add a hires warmup, or eat the first-run cost.

---

## 9. Effort
Phase 0: S · Phase 1 (refactor): **M, highest-leverage** · Phase 2: S ·
Phase 3: M · Phase 4: S. Roughly an **M-sized feature with no export
re-architecture** — a different tier from the SDXL version.

---

## 10. Open decisions (defaults chosen)
1. **VAE route A (re-export, recommended) vs B (tiled decode)** — planned as A
   with B stubbed.
2. **Ship Phase 1 as a standalone refactor first**, then hires on top — vs one
   combined change.

---

## 11. Validation checklist (run on the GPU box — dev box has no model/GPU)

Build: `cmake --build build` (add `-DUSE_CUDA=ON …` for the GPU build). Then, from
the repo root with a hires-capable SD1.5 model imported:

**Verified on the dev box (CPU-only):** clean compile of the whole feature; headless
CLI arg wiring + the two-pass `--cancel-after-steps` flow (parses hires+cancel
flags, runs both passes, fails gracefully without a model — no crash/hang).

**Needs the GPU box (byte-exact / visual):**

1. **Inertness (non-negotiable).** Capture goldens once, then verify hires-OFF is
   unchanged by the feature:
   ```
   scripts/golden/golden_harness.sh determinism      # pick exact vs tolerance
   scripts/golden/golden_harness.sh capture          # writes goldens + hashes.txt (commit hashes.txt)
   scripts/golden/golden_harness.sh verify           # all c1–c4 must MATCH
   ```
2. **Hires reproducibility.** `scripts/golden/golden_harness.sh hires-repro`
   (same seed + hires, twice → identical bytes).
3. **Cancel mid-hires restores ctx dims.**
   `scripts/golden/golden_harness.sh cancel-restore` (cancels inside pass 2, then a
   native gen in the same process must match the c1 golden — proves the RAII restore).
4. **A/B (visual).** Same seed, hires off vs on at 1.5× — same composition, more
   detail (not a different image):
   ```
   ./build/image_generator --headless-generate --model <M> --out /tmp/off.png \
       --prompt "…" --seed 42 --steps 24 --guidance 7.5
   ./build/image_generator --headless-generate --model <M> --out /tmp/on.png \
       --prompt "…" --seed 42 --steps 24 --guidance 7.5 --hires --hires-scale 1.5 --hires-strength 0.5
   ```
5. **Snapping.** `--hires-scale 1.1` → log shows target snapped to a /64 pixel dim
   (e.g. 512→576, latent 72) and generation succeeds.
6. **VRAM.** Note peak at 1.5× and 2×; a 2× OOM is expected (tiled decode is a future
   track — `decodeLatent` has the dispatch-point comment). Don't "fix" it.

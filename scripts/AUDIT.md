# scripts/ audit — 2026-07-07

Audit only. No deletions, no refactors, no behaviour changes were made in this
session. Every deadness claim below names the searches run and the callers
checked. Confidence is stated per item.

## Method / evidence base

Reachability was checked against **all** caller classes, not just "is it grepped
in another .py":

- **C++ callers** — sub-agent grep of `src/` for every Python-script invocation.
  Result: the app invokes exactly **one** script, `import_model.py`, from
  `src/import/ModelImporter.cpp:277` with argv `--input --output --arch` plus a
  **conditional `--resume`** (`ModelImporter.cpp:313-316`). It never passes
  `--dynamic-spatial`, `--name`, `--optimize-memory`, `--simplify-vae`,
  `--output-dir`, or `--validate`. `dynamic_spatial=True` / `emit_fp32_hedge=False`
  are set *inside* `import_model.py` (line 289-290), not on the C++ argv.
- **docs/ runbooks** — sub-agent grep of all 34 `.md` files for every script and
  flag. All runnable command lines live in `docs/50_export/export_overview.md`
  and `docs/55_import/import_pipeline.md`.
- **golden harness** — `scripts/golden/golden_harness.sh` + `compare.py`, plus the
  C++ headless flags it drives (`src/headless/GoldenRun.cpp`, `src/main.cpp`).
- **cross-script imports** — grep across `scripts/`.

Confirmed the one-off helpers named in the session brief (`validate_fix.py`, fp16
microbenches, A/B decode helpers) are **already gone** — `ls | grep -iE
'validate_fix|bench|ab_|decode'` returns nothing. Nothing to clean up there.

---

## 1. Safe to delete now — high confidence

### 1.1 `ExportPolicy.text_encoder_dynamic_axes()` — dead method
`export_common.py:331-339`. **Zero callers.**
- Searched: `grep -rn text_encoder_dynamic_axes .` (scripts), `../docs`, `../src` →
  the only hit is the definition itself.
- Every export script builds the text-encoder `dynamic_axes` dict **inline**, not
  through the policy: `export_onnx_models.py:115-120`,
  `sdxl_export_onnx_models.py:167-171` and `:197-202`.
- Confidence: **high.** It is a policy-surface method that was never wired up.
  Deletable with no behaviour change.

### 1.2 Stale `--dynamic-spatial` help text — actively misleading
`sdxl_export_onnx_models.py:383-386`. The argparse help still reads:

> "Export UNet + VAE decoder with dynamic H/W axes … **encoder stays static
> 1024.** This is the in-app import default…"

This is **wrong now.** The dynamic-spatial path also makes the **VAE encoder**
dynamic (the whole pixel-hires change) — see the runtime `print` in the same file
at `:103-105` ("UNet + VAE decoder + VAE encoder export with dynamic H/W axes …
incl. pixel mode"), the policy `vae_encoder_dynamic_axes()`
(`export_common.py:496-510`), and the module docstring at `:12`. A future session
reading `--help` would conclude pixel-mode hires is impossible on SDXL.
- Fix: replace "encoder stays static 1024" with "UNet + VAE decoder + VAE encoder
  gain dynamic H/W axes (enables latent- and pixel-mode hires); also emits an fp32
  decoder hedge." This is a **comment/string edit**, not behaviour.
- Confidence: **high** that it's stale; it directly contradicts the code beside it.

### 1.3 `scripts/__pycache__/` — already ignored, just noting
Not a repo problem: `git ls-files scripts/__pycache__/` is empty and
`git check-ignore` matches, so it's untracked build output. No action needed;
listed only so it isn't re-investigated next pass.

---

## 2. Probably removable — verify the one stated thing first

### 2.1 `exporter="auto"` mode in `onnx_export()` — unreachable branch
`export_common.py:666` (accept-list) and `:669-696` (the `exporter in {"dynamo",
"auto"}` block with its dynamo→legacy fallback).
- No policy or caller ever produces `"auto"`: `unet_exporter()`→`"legacy"`;
  `vae_exporter()`→`"legacy"|"dynamo"`; `vae_encoder_exporter()`→`"legacy"|"dynamo"`;
  `export_vae_encoder.py:119` passes `"dynamo"` literally. `grep -rn '"auto"'` in
  scripts hits only the `import_model.py --arch` default (unrelated) and these two
  lines.
- So the **auto-specific behaviour** (try-dynamo-then-silently-fall-back-to-legacy)
  is dead. The `"dynamo"` and `"legacy"` paths are both live.
- **Verify before removing:** confirm you don't intend `"auto"` as a future
  ergonomic default. If not, drop `"auto"` from the accept-set and collapse the
  branch to `if exporter == "dynamo":`. Low risk, small cleanup.
- Confidence: **medium-high** it's currently dead; **your call** on whether it's a
  deliberate extension point.

### 2.2 `check_lora_compat.py` — no workflow references it
`scripts/check_lora_compat.py` (256 lines). Zero references in `docs/` (checked all
`.md`, incl. `docs/30_lora/lora_debugging.md`), zero in `src/`, not imported by any
script, not called by the golden harness.
- It's a standalone LoRA-vs-ONNX match-count debugger that mirrors the C++
  `SdLoraMatch.cpp` / `OnnxIndex.cpp` suffix logic.
- **Verify before removing:** do you still run this by hand when a LoRA silently
  under-applies? If yes → move to §3 and add a one-line pointer in
  `docs/30_lora/lora_debugging.md` so it stops looking orphaned. If you haven't
  touched it since the C++ matcher stabilised → delete.
- Risk if kept-but-drifted: it duplicates the C++ suffix rules, so it can silently
  disagree with runtime after a matcher change. That's an argument for documenting
  *or* deleting, not leaving it in limbo.
- Confidence: **workflow-orphaned = high**; keep/delete is a judgement call only you
  can make.

### 2.3 `export_llm_onnx.py` — no workflow references it, but it's the only producer
`scripts/export_llm_onnx.py` (151 lines). Zero references in `docs/` (including the
`docs/60_llm/` subsystem docs) and zero in `src/`.
- BUT it is the **only** way to produce the ort-genai LLM directory that the
  optional `OrtLlmEnhancer` consumes (`config.json → promptEnhancer.modelDir`). It
  is a genuine spare tool, not dead logic.
- **Recommended action: keep, but document** — add a line to
  `docs/60_llm/llm_overview.md` pointing at it (it's currently discoverable only by
  reading the script). Deleting it would strand the prompt-enhancement feature with
  no export path.
- Confidence: **high** it's undocumented; **do not delete.**

---

## 3. Intentional, keep — spares/hedges with trigger conditions

These are deliberate escape hatches. Documented here so they don't rot into
mystery files.

### 3.1 fp32 VAE-**decoder** hedge — `vae_decoder_fp32.onnx`
- Emitted by `sdxl_export_onnx_models.py:262-285`, gated on
  `dynamic_spatial and emit_fp32_hedge`. CLI `--dynamic-spatial` sets
  `emit_fp32_hedge=args.dynamic_spatial` (`:407`); the **in-app import passes
  `emit_fp32_hedge=False`** (`import_model.py:290`), so the ~200 MB spare ships only
  from the standalone CLI.
- **Trigger:** SDXL fp16 VAE decode produces NaN/washed-out output at >native
  latents. Fix is a file swap (`vae_decoder.onnx` ← `vae_decoder_fp32.onnx`), never
  a re-export. Full procedure in `docs/50_export/export_overview.md:57-64`.
- Consumed by the validator: `validate_dynamic_sdxl.py:540,578-580,593-594` probe
  and forward it when present. Correctly a no-op when absent (in-app models).

### 3.2 fp32 VAE-**encoder** hedge — deliberately NOT emitted
- There is intentionally **no** `vae_encoder_fp32.onnx`. Documented rationale
  (`docs/50_export/export_overview.md:56`): encode is image→latent (contracting /
  numerically stable), fp16 encoder validated fine, SD1.5 ships fp16-only encoder
  and works. Adding it would be dead weight against no observed failure mode.
- **Trigger to add it later:** GPU validation shows fp16 encoder NaN/near-constant
  at ≥1536 (`validate_dynamic_sdxl.py:_run_vae_encoder` would fire
  `vae_encoder.run@… produced NaN/Inf`). Then mirror §3.1 (~1.3 s cost).
- Keep this *absence* documented so nobody "adds the missing hedge" for symmetry.

### 3.3 `--resume` (both export scripts + import_model.py)
Live, not a spare: the C++ importer appends `--resume` when it detects a partial
export (`ModelImporter.cpp:293-316`). Skips already-complete components via
`_is_component_complete` (`export_common.py:1235-1246`). Keep.

### 3.4 `patch_unet_resize.py` / `export_vae_encoder.py` — retrofit tools
Both are documented retrofit-only tools for model dirs exported before a given fix
became default (`docs/50_export/export_overview.md:14-15,33-48,67`). Reachable from
documented command lines. Keep.

### 3.5 golden harness (`golden/golden_harness.sh` + `compare.py`)
Undocumented in `docs/` but a live workflow: every flag it drives
(`--headless-generate --hires --hires-scale --hires-strength --cancel-after-steps
--init --strength --images …`) exists in `src/headless/GoldenRun.cpp` /
`src/main.cpp` (verified by grep). Modes `capture|verify|determinism|hires-repro|
cancel-restore` all map to real binary behaviour. `goldens/.gitignore` correctly
commits only `hashes.txt`. Keep. (Optional: add a `docs/90_debug/` pointer — it's
currently discoverable only via the session brief / memory.)

### 3.6 SD1.5 `vae_dynamic_axes` "HISTORICAL NOTE"
`export_common.py:396-402`. This is the one 1-hour-hang-era comment that is
**correct** — it explicitly records that the hang is *gone* (re-tested torch 2.11.0,
2026-07). It is history that prevents a future session from re-introducing the
"dynamic H/W hangs, keep it static" assumption. Keep (optionally trim to 2 lines).

---

## 4. Recommendations (report only — not done)

### 4.1 Make dynamic-spatial the default SDXL export — blast radius
Dynamic-spatial is now validated on both archs and is already the in-app default,
so the static SDXL path only ships from the standalone CLI's *default* (no flag).
Making dynamic the default would let you delete a real amount of conditional code.

**What becomes unreachable / deletable:**
- `SDXLExportPolicy.__init__(dynamic_spatial=...)` and every `if self.dynamic_spatial`
  fork: `unet_dynamic_axes` (`export_common.py:462-475`), `vae_dynamic_axes`
  (`:477-487`), `vae_exporter` (`:489-494`), `vae_encoder_dynamic_axes` (`:496-510`),
  `vae_encoder_exporter` (`:512-519`), and the two-branch `should_fix_fp32_constants`
  (`:521-537`). Each collapses to its dynamic arm.
- The static-vs-dynamic trace-size conditionals in the exporter:
  `sdxl_export_onnx_models.py:148-149` (`unet_trace_hw`/`vae_dec_trace_hw`) and
  `:326` (`enc_dummy_hw`) lose their `else` (native-1024) arm.
- The `--dynamic-spatial` flag itself (`:380-387`) and the `dynamic_spatial` param
  threading through `export_sdxl`.
- **Fail-closed capability detection stays but simplifies conceptually:**
  `_detect_hires_capable` / `_detect_pixel_hires_capable`
  (`import_model.py:104-164`) would then always derive `True` for a fresh SDXL
  import — but **do not delete the probes.** They are ground-truth-from-graph and
  still correctly report `False` for older static models already on disk. The
  fail-closed branch is the safety net; keep it.
- `validate_dynamic_sdxl.py` largely unchanged (it validates a produced dynamic
  export regardless of how the default is set).

**What must NOT change (the guardrail the brief calls out):** the *default*
(non-flag) export currently being byte-identical to the pre-hires era is a
deliberate property of the **standalone CLI only**. The in-app path already went
dynamic. Flipping the CLI default changes that byte-identity for CLI users — a
policy decision, not a bug. If you want dynamic-by-default but keep a static escape
hatch, invert the flag to `--static` rather than deleting the static code.

- **Effort:** ~half a day (mechanical branch collapse + re-run golden `verify` and
  `validate_dynamic_sdxl.py --torch-reference`).
- **Benefit:** removes ~6 policy forks + 3 trace-size conditionals + 1 CLI flag;
  one SDXL export path to reason about instead of two.
- **Risk:** medium. The static dynamo path is the older *proven* one; deleting it
  removes your fallback if a future torch/ORT regresses the legacy-tracer dynamic
  export. Recommend inverting to `--static` (keep the code, flip the default)
  rather than hard-deleting — captures most of the simplification at low risk.
- **In-app note:** the import flow **cannot pass any flag** (§Method), so it is
  unaffected by a CLI default flip; it already calls `export_sdxl(dynamic_spatial=
  True)` explicitly. If you later want the app to offer a static/lean import, that's
  a *new* C++ argv change, out of scope here.

### 4.2 Policy-class API simplification
- **`should_simplify_vae` is a no-op pass-through.** Base
  (`export_common.py:374-375`) and the `SDXLExportPolicy` override (`:545-546`) are
  **byte-identical** (`return requested`), and SD15 doesn't override. The method
  adds no policy logic — callers could pass the bool straight to the spec. Either
  delete the redundant SDXL override (trivial) or drop the method entirely and inline
  `simplify=simplify_vae` at `sdxl_export_onnx_models.py:305` /
  `simplify=False` at `export_onnx_models.py:185`. Effort: 10 min. Benefit: −1 dead
  override, −1 indirection. Risk: none.
- **`vae_encoder_exporter` base default is unused.** Base returns
  `self.vae_exporter()` (`:359-363`) but both concrete policies override it. Not
  worth touching unless you add a third arch — the default documents intent. Leave.
- **`text_encoder_dynamic_axes`** — see §1.1 (delete).

### 4.3 `validate_dynamic_sdxl.py` vs `verify_model.py` — shared core, not a merge
Both load ONNX, build feeds from session metadata, and forward the UNet / VAE. They
duplicate: `_ORT_TO_NP` + `_np_dtype`, `_session`, `_gpu_providers`, a `_fixed`/dim
resolver, and a UNet-feed builder (`_build_unet_feed` vs `_unet_feed`) — 8
overlapping symbols each (grep count). **Do not merge the scripts** — they serve
different masters:

| Keeps this, must not lose it | `verify_model.py` | `validate_dynamic_sdxl.py` |
|---|---|---|
| Consumer | C++ `ModelImporter` (parses `VERIFY:ok\|warn\|fail`) | human on a GPU box |
| Gate | text-encoder collapse = **hard** fail, exit 3 | structural + numeric, exit 0/1/2 |
| Providers | CPU for TE; GPU only for heavy | GPU-first, `preload_dlls`, BASIC opt-level |
| Unique | `enable_cpu_mem_arena=False` OOM guard, `--arch sdxl/sd15` | `_VramSampler` NVML peak sampling, `--torch-reference` numeric diff, `time_ids` per-size scaling, fp32-hedge probe |

- **Recommendation:** extract the *pure* shared helpers (`_ORT_TO_NP`/`_np_dtype`,
  `_session`, `_gpu_providers`, dim-resolver) into a tiny `ort_probe.py` in
  `scripts/` and import from both. Leave the two `main()`s, protocols, gates, and
  feed builders alone — the feed builders *look* similar but differ (verify feeds
  real encoder output + guards SDXL concat width; validate feeds seeded noise +
  scales `time_ids`). Effort: ~1-2 h. Benefit: one dtype/session source of truth.
  Risk: low if you only lift the genuinely-identical helpers.

---

## 5. Optimizations

### 5.1 Export wall-time — essentially done
The tiny-trace discovery already killed the dominant cost (full-size fp16 CPU UNet
trace, the ~1 h bottleneck → ~2-3 min). With the session's measured actuals — 141 s
total, of which OpenCLIP-G serialize ~26 s and fp16 VAE-dec trace ~43 s — the
remaining time is **weight serialization**, not compute:
- OpenCLIP-G 26 s is protobuf/disk write of a ~1.4 GB fp32 encoder; not tractable
  without changing the on-disk dtype (you export TE in fp32 **on purpose** to avoid
  attention collapse — do not touch).
- VAE-dec 43 s is already at an 8×8 trace; the cost is the fp16 graph-fix
  load/save + external-data consolidation, both single-pass already
  (`fix_fp16_graph`, `consolidate_external_data` do raw byte copies).
- **Verdict: nothing worth attacking.** Any further win trades correctness
  (TE dtype) or robustness (skip verification) for seconds. Call it done.

### 5.2 Error-path quality — good, with one intentional convention split
- The C++-consumed scripts use the greppable `VERIFY:ok|warn|skip|fail:<check>:
  <detail>` convention **consistently**: `verify_model.py:_emit` (`:61-66`) and
  `import_model.py` capability warnings (`:129-135`, `:157-163`) and the
  inconclusive-verifier warn (`:320-324`). Single-line enforced
  (`verify_model.py:63`). This is clean.
- `validate_dynamic_sdxl.py:_emit` (`:68-71`) deliberately uses a **different**
  format (`[WARN] check: detail`) — correct, because it's human-facing on a GPU box,
  not parsed by C++. Not an inconsistency to fix; just noting it's intentional.
- Failures that die with tracebacks vs clean messages: the export/import entry
  points catch `FileNotFoundError|ImportError|RuntimeError` and emit clean
  `ERROR:`/stderr (`import_model.py:353-363`, both export scripts' `__main__`).
  `export_step` (`export_common.py:129-150`) wraps every component step and
  re-raises `MemoryError`/CUDA-OOM/`Exception` with context — good. The main gap: an
  unexpected exception *type* inside `run_import` (not File/Import/Runtime) would
  escape as a raw traceback rather than an `ERROR:` line the modal can show. Low
  probability; consider a catch-all `except Exception` → `ERROR:` at
  `import_model.py:353` if you want the modal to never show a bare traceback.

### 5.3 Shared seeded-input / hashing — minor
- `validate_dynamic_sdxl.py` and `verify_model.py` both build seeded noise feeds;
  §4.3 covers extracting the dtype/session helpers. The *feed builders* differ
  enough that sharing them would cost correctness clarity — leave separate.
- The golden harness hashes with `sha256sum` (shell) and byte-compares with `cmp`;
  `compare.py` is the tolerance-mode fallback only. No reimplementation worth
  consolidating — they're deliberately in different languages (shell driver vs
  numpy pixel diff) and share nothing that would benefit from a common core.
- **Verdict: nothing worth doing** beyond the §4.3 helper lift.

---

## Appendix — CLI knobs that are reachable but undocumented (not dead)
Reachable from the standalone CLI, never from C++ or docs/. Functional; listed so
they're not mistaken for dead code, and so you can decide whether to document them:
- `sdxl_export_onnx_models.py`: `--optimize-memory` (attention slicing, `:129-131`),
  `--simplify-vae` (onnxsim, `:305`), `--output-dir`, `--validate`.
- `export_onnx_models.py`: `--output-dir`, `--validate`.
- `--validate` drives `validate_with_ort` (`export_common.py:1249-1299`) — a real
  per-component ORT forward. Genuinely useful for manual debugging; keep.
</content>
</invoke>

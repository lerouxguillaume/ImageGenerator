# Golden-run regression harness

`scripts/golden/golden_harness.sh` is the bit-for-bit regression net for the SD
pipeline. It drives the app's headless generation mode with fixed, scripted
parameters and hash-compares the PNG outputs of two builds, so a refactor that is
meant to be output-preserving can be *proven* so.

It is a **manually-run developer tool** — not invoked by the app, the import
pipeline, or CI. It needs a built binary (`build/image_generator`) and at least
one imported model (auto-detected from `models/imported/registry.json`, or passed
with `--model`).

---

# What it protects

The pipeline is expected to be deterministic on a given execution provider. The
harness catches any change that silently perturbs output: scheduler math, latent
init, CFG, the img2img encode path, or the hires second pass.

---

# Modes

```bash
scripts/golden/golden_harness.sh <mode> [--model DIR] [--bin PATH] \
    [--mode exact|tolerance] [--tol N] [--max-frac F]
```

| Mode | What it does |
|---|---|
| `determinism` | Runs the same inputs twice and checks the two PNGs are byte-identical. **Run this first** — it tells you whether this machine/EP is bit-deterministic at all, i.e. whether to use `exact` or `tolerance` compare. |
| `capture` | Generates the golden combos and writes them to `goldens/` plus a `hashes.txt` manifest. Capture from a build made **before** the change under test. |
| `verify` | Re-runs the combos on the current build and compares against the goldens (PNG byte/tolerance compare when the PNG is present, else the committed `hashes.txt`). Non-zero exit on any mismatch. |
| `hires-repro` | Same seed + hires ON, twice → must be byte-identical (hires reproducibility). |
| `cancel-restore` | Cancels mid-hires-pass, then generates native in the **same process** and compares against the `c1` golden — proves `ScopedLatentResolution` restores ctx dims on the cancel path. |

The combos cover txt2img (×2), a multi-image seed-sequence batch, and img2img
(init is a freshly-generated output, so the whole chain reproduces from scratch).

---

# Cross-machine truth: `hashes.txt`

Golden PNGs are large and machine-specific, so they are **gitignored**
(`goldens/.gitignore` commits only `hashes.txt`). The committed
`goldens/hashes.txt` sha256 manifest is what makes goldens verifiable on a
different machine: `verify` falls back to it when the PNG itself isn't present.

If a machine is **not** bit-deterministic (`determinism` reports so), use
`--mode tolerance` — the shell driver byte-compares with `cmp`, and only then
delegates to `compare.py` for a documented per-pixel tolerance.

---

# Headless flags

The combos are expressed as flags on the binary (`--headless-generate --prompt
--neg --seed --steps --guidance --images --init --strength --hires --hires-scale
--hires-strength --cancel-after-steps --out`), parsed in `src/main.cpp` /
`src/headless/GoldenRun.cpp`. The harness is the canonical caller of that mode.

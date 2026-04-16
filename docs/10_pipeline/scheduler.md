# Scheduler (DPM++ 2M Karras)

## What this file explains
The noise schedule used during the denoising loop: how sigmas are computed, how timesteps are derived, and how the loop advances.

## When to use this
- Understanding why a specific number of steps produces a specific sigma sequence
- Debugging scheduler-related generation quality issues
- Adding a new model that requires a different beta schedule

## Key invariants
- `buildKarrasSchedule()` always returns `num_steps + 1` sigmas; the last element is always `0.0`.
- Sigma values are strictly decreasing: `σ[0] > σ[1] > ... > σ[n-1] > 0`.
- The loop runs `num_steps` iterations (indices `0..num_steps-1`), consuming pairs `(σ[i], σ[i+1])`.
- `sigmaToTimestep()` converts a continuous sigma to the integer timestep expected by the UNet.

## Mental model
The Karras schedule places sigma values non-uniformly: more sigmas near the high-noise end (early steps) and fewer near the clean-image end (late steps). This concentrates compute where structural decisions are made.

```
σ_max ──────────── (step 0) major structure
         │
         ▼
      (step k) fine detail
         │
         ▼
σ ≈ 0 ─── (last step) nearly clean image
```

## Implementation details

### `buildAlphasCumprod` (`SdScheduler.cpp`)
Computes the cumulative product of `(1 - β_t)` over all 1000 training timesteps using a cosine schedule. This is the standard DDPM `ᾱ_t` sequence.

### `buildKarrasSchedule` (`SdScheduler.cpp`)
```
σ_max = sqrt((1 - ᾱ_min) / ᾱ_min)   (from the highest-noise training timestep)
σ_min = sqrt((1 - ᾱ_max) / ᾱ_max)   (from the lowest-noise training timestep)

For i in [0, num_steps]:
    t = (i / (num_steps - 1))^(1/rho)    rho = 7 (Karras default)
    σ[i] = (σ_max^(1/rho) + t × (σ_min^(1/rho) - σ_max^(1/rho)))^rho

σ[num_steps] = 0   (appended sentinel)
```

Returns `vector<float>` of length `num_steps + 1`.

### `sigmaToTimestep` (`SdScheduler.cpp`)
```
t = argmin_t |ᾱ_t / (1 - ᾱ_t) - 1/σ²|
```
Performs a linear scan over the 1000-element `alphas_cumprod` array. Called once per step per UNet pass (2 calls per step under CFG).

### DPM++ 2M update step (`SdPipeline.cpp — denoiseSingleLatent`)
Uses the previous and current denoised predictions (`d_prev`, `d_cur`) to compute the second-order correction. First step is always first-order (no `d_prev` available).

## Common pitfalls
- Off-by-one: the loop runs `num_steps` times over `num_steps + 1` sigmas. Using `sigmas.size()` as the loop bound (instead of `num_steps`) processes the sentinel `0` and produces a NaN latent.
- `sigmaToTimestep` returns an integer in `[0, 999]`. Passing this directly as a float timestep to an SDXL UNet that expects the additional `time_ids` will produce wrong results — see `docs/10_pipeline/sdxl_specifics.md`.
- A different model family (e.g. FLUX) may use a flow-matching schedule with different `σ_min`/`σ_max` bounds. Do not reuse `buildKarrasSchedule` for it without verifying the schedule definition.

## Related files
- [`docs/10_pipeline/pipeline_orchestration.md`](pipeline_orchestration.md) — loop structure
- [`docs/10_pipeline/sdxl_specifics.md`](sdxl_specifics.md) — SDXL extra UNet inputs
- `src/portraits/sd/SdScheduler.cpp` — `buildAlphasCumprod`, `buildKarrasSchedule`, `sigmaToTimestep`
- `src/portraits/sd/SdPipeline.cpp` — `denoiseSingleLatent` (DPM++ 2M update)

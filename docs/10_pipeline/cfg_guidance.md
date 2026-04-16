# CFG Guidance

Classifier-Free Guidance blending and rescale.

---

# Standard CFG

Two UNet passes per step (batch=1 each):
- uncond pass → `u_eps`
- cond pass → `c_eps`

Blend formula:
```
eps = u_eps * (1 - neg_scale) + guidance_scale * c_eps
```

When `neg_guidance_scale == guidance_scale`, this is identical to the standard form:
```
eps = u_eps + guidance_scale * (c_eps - u_eps)
```

---

# Split CFG (`neg_guidance_scale`)

`GenerationParams.negativeGuidanceScale` (default `0.0f`):
- `<= 0` → use `guidance_scale` for both directions (standard CFG)
- `> 0` → independent scale for uncond weight

Set in `SdPipeline.cpp`:
```cpp
ctx.neg_guidance_scale = (params.negativeGuidanceScale > 0.0f)
                           ? params.negativeGuidanceScale
                           : params.guidanceScale;
```

`GenerationContext.neg_guidance_scale` defaults to `8.0f` (only used when explicitly set via params).

---

# CFG Rescale (`cfg_rescale`)

Lin et al. 2023 — prevents oversaturation at high guidance scales.

`GenerationParams.cfgRescale` (default `0.0f` = off). Typical value: `0.7`.

When enabled, rescales blended `eps` so its stddev matches conditional `eps`, then blends back:
```
phi = std(c_eps) / std(eps)
eps = (r * phi + (1 - r)) * eps
```
where `r = cfg_rescale`.

Implemented in `SdUNet.cpp:159-175`.

---

# Rules

- Do NOT set `ctx.neg_guidance_scale` directly — it is set by `SdPipeline.cpp` from `GenerationParams`
- `cfg_rescale = 0.0f` disables rescale entirely — skip the stddev computation

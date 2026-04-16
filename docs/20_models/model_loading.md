# Model Loading

## What this file explains
How models are detected and loaded.

## When to use this
- Model fails to load
- Wrong model behavior

## Key invariants
- model.json defines type
- Missing = SD1.5 fallback

## Mental model
Loader builds ModelConfig then sessions.

## Implementation details
- loadModelConfig()
- loadModels()
- builds GenerationContext

## Common pitfalls
- Missing files
- Wrong model type

## Related files
- [model_cache.md](model_cache.md) — session reuse after loading
- [adding_new_model.md](adding_new_model.md) — extending loadModelConfig and loadModels
- [../10_pipeline/sdxl_specifics.md](../10_pipeline/sdxl_specifics.md) — SDXL-specific session loading
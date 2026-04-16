# Model Cache

## What this file explains
How `ModelManager` avoids rebuilding ORT sessions on every generation: cache key construction, hash strategy, and lookup semantics.

## When to use this
- LoRA change not taking effect (cache stale)
- Unexpected session reuse across different configs
- Debugging scale precision issues

## Key invariants
- `ModelManager::get()` resets `ctx.run_opts = Ort::RunOptions{}` on **every** call (hit or miss) — a prior `SetTerminate()` is never carried over.
- `GenerationContext` contains non-copyable ORT sessions — always use `emplace` with `std::move`, never `operator[]`.
- Multiple distinct configurations coexist in the cache simultaneously; changing LoRA or model path creates a new entry, not a replacement.

## Mental model

```
runPipeline(cfg, loras)
    │
    ▼
ModelManager::get(cfg, modelDir, loras)
    │
    ├─ ModelCacheKey::make(modelDir, loras)  ← canonical paths, sorted, rounded scales
    │
    ├─ unordered_map::find(key)
    │       ├─ HIT  → reset run_opts; return cached ModelInstance&
    │       └─ MISS → loadModels() → emplace → return new ModelInstance&
```

## Implementation details

### `ModelCacheKey::make()` invariants

Three normalisations are enforced before the key is stored:

1. **Canonical paths** — `modelDir` and every LoRA path go through `std::filesystem::weakly_canonical()`. `./models/sd15` and `models/sd15` map to the same key.

2. **Sorted LoRA list** — entries are sorted by canonical path. `{A, B}` and `{B, A}` are identical keys.

3. **Fixed-point scales** — scales stored as `int(std::lround(scale * 1000))`. `lround` rounds rather than truncates: `0.9999f → 1000` not `999`. Changes smaller than `0.0005` are treated as the same key by design.

### `ModelCacheKeyHash`

Feeds a single canonical string buffer:
```
modelDir\0type\0loraPath1\0scale1\0loraPath2\0scale2\0...
```
into **XXH64** (libxxhash). Avoids depending on `std::hash<std::string>` quality which varies across stdlibs.

### `run_opts` reset

```cpp
// Inside ModelManager::get() — every call, before returning
ctx.run_opts = Ort::RunOptions{};
```

This clears any `SetTerminate()` left by a previous cancelled run so the next generation doesn't throw on its first UNet call.

## Common pitfalls
- Editing `loraScales[i]` in the UI but not propagating it to `GenerationParams` — the cache key is built from the params passed to `runPipeline()`, so an un-propagated change is invisible.
- Scale changes smaller than 0.0005 — intentionally treated as the same key. Not a bug.
- Two LoRA paths that differ only by `./` prefix — `weakly_canonical()` normalises them to the same key; both configs hit the same cache entry.

## Related files
- [model_loading.md](model_loading.md) — session creation on cache miss
- [../30_lora/lora_caching.md](../30_lora/lora_caching.md) — LoraInjector cache layer (lives inside the cached ModelInstance)
- [../10_pipeline/cancellation.md](../10_pipeline/cancellation.md) — why run_opts is reset here
- [adding_new_model.md](adding_new_model.md) — how model type affects the cache key

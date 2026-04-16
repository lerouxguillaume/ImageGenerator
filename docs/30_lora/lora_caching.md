# LoRA Injector Cache

## What this file explains
The two internal caches inside each `LoraInjector` instance: the ONNX metadata index and the merged-tensor cache. Session-level caching (ModelManager) is covered in [`model_cache.md`](../20_models/model_cache.md).

## When to use this
- Understanding why the first LoRA run is slow and subsequent runs are fast
- Diagnosing memory usage with LoRA enabled
- Understanding why injector state survives LoRA config changes

## Key invariants
- `SdLoader.cpp` holds `static std::unordered_map<std::string, LoraInjector> s_injectors` keyed by `.onnx` path. Always use references from this map — never create local `LoraInjector` instances inside `loadModels()`.
- Do not destroy a `LoraOverrides` before the `Ort::Session` constructor returns — `Ort::Value` objects hold non-owning views into `LoraOverrides::fp16Bufs`/`fp32Bufs`.
- Do not call `AddExternalInitializers` on the shared `unetOpts`/`auxOpts` references — always `Clone()` first.

## Mental model

```
loadModels() (on LoRA config change)
    │
    ▼
LoraInjector& injector = s_injectors[onnxPath]   ← static, survives across calls
    │
    ├─ loadModelMetadata()   (runs once per .onnx path)
    │       └─ populates extIndex_ + extSuffixIndex_
    │
    └─ applyLoras(loras)
            │
            ├─ mergeCache_ lookup (FNV-1a key)
            │       ├─ HIT  → buildOverrides() — memcpy only
            │       └─ MISS → ensureBaseWeights() + computeMerge() → store in mergeCache_
            │
            └─ buildOverrides() → LoraOverrides (Ort::Value views)
```

## Implementation details

### Per-instance caches

| Cache field | Key | Content | Populated by |
|---|---|---|---|
| `extIndex_` / `extSuffixIndex_` | `.onnx` path (set once) | External tensor index + suffix lookup table | `loadModelMetadata()` |
| `mergeCache_` | FNV-1a hash of `(onnxPath, weightsPath, loras+scales)` | `vector<CachedTensor>` — pre-computed merged tensors | `computeMerge()` on miss |

`mergeCache_` is guarded by `cacheMutex_`.

### Cache miss path (`computeMerge`)

1. `ensureBaseWeights()` — loads `<name>_weights.safetensors` once per `.onnx` path; result cached in `baseWeights_`.
2. For each LoRA: `parseLoraLayers → matchExternalLoraKey → computeLoraDelta` (fp32 matmul).
3. Merged tensors stored in `mergeCache_` entry.

### Cache hit path (`buildOverrides`)

Copies `CachedTensor` data into new `LoraOverrides` backing buffers (`fp16Bufs` / `fp32Bufs`), wraps each in an `Ort::Value` view. No matmuls.

### `SessionOptions::Clone()` + `AddExternalInitializers`

```cpp
auto cloned = unetOpts.Clone();             // isolate from shared opts
cloned.AddExternalInitializers(overrides);  // inject merged tensors
Ort::Session session(env, path, cloned);    // LoraOverrides must outlive this call
```

## Memory profile

| Phase | Memory |
|---|---|
| No-LoRA session | Near-zero — ORT memory-maps `.onnx.data` |
| First LoRA run | ≈ 2× model size (base weights + merge accumulator) |
| Subsequent LoRA runs | ≈ model size (base weights persist; accumulator freed after caching) |

## Related files
- [../20_models/model_cache.md](../20_models/model_cache.md) — session-level cache (ModelManager)
- [lora_key_format.md](lora_key_format.md) — key format feeding computeMerge
- [lora_debugging.md](lora_debugging.md) — diagnosing cache and injection failures
- `src/portraits/sd/LoraInjector.hpp/.cpp`
- `src/portraits/sd/SdLoader.cpp` — `s_injectors`, `makeLoraSession()`

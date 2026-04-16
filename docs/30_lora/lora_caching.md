# LoRA Caching

Two-level caching system:

---

# 1. Injector cache

Per ONNX model:
- stores external index
- stores suffix map

---

# 2. Merge cache

Keyed by:
- model path
- LoRA set
- scales

Stores:
- computed tensor deltas

---

# Benefit

Avoid recomputing:
- matmul operations
- safetensors parsing

---

# Rules

- Do NOT create local `LoraInjector` instances inside `loadModels()` — always use references from the static `s_injectors` map so metadata and merge caches survive LoRA config changes
- Do NOT call `AddExternalInitializers` on the shared `unetOpts`/`auxOpts` — always `Clone()` first; mutating shared opts corrupts all subsequent sessions
- Do NOT destroy `LoraOverrides` before `Ort::Session` constructor returns — `Ort::Value` views inside it point into `LoraOverrides` backing buffers
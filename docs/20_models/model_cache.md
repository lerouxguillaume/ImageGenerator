# Model Cache

Managed by `ModelManager`.

---

# Key concept

Models are cached by:

- ModelCacheKey

Includes:
- model path
- LoRA set
- scale values

---

# Behavior

- cache hit → reuse sessions
- cache miss → loadModel()

---

# Design goal

Avoid:
- repeated ONNX initialization
- repeated weight loading
- redundant GPU setup

---

# Scale key tolerance

`scaleKey()` = `lround(scale * 1000)`. Two LoRA scales within ±0.0005 map to the same cache key and are treated as identical.

---

# Rules

- Do NOT use `operator[]` to insert into the cache — `GenerationContext` contains non-copyable ORT sessions; use `emplace` with `std::move`
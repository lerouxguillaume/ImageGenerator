# Performance Notes

---

# Cold start cost

- first model load triggers JIT compilation

---

# Cache impact

ModelManager drastically reduces:
- ONNX load time
- GPU kernel initialization

---

# Slow first image

Expected due to:
- warmup
- graph compilation
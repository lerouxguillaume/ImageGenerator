# LoRA Debugging Guide

---

# 0 patches applied

Cause:
- ONNX naming mismatch
- missing external initializers

---

# High miss rate

Cause:
- suffix mismatch
- incorrect Kohya format

---

# NaN output

Cause:
- fp16 overflow in delta
- broken safetensors conversion

---

# Checklist

- check external index
- check weights file
- check suffix mapping
- verify ModelCacheKey match
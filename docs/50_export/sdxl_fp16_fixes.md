# SDXL FP16 Fixes

SDXL export introduces fp32 contamination issues.

---

# Problems

- attention upcasting
- timestep embedding fp32
- Resize operator dtype mismatch

---

# Fix strategy

Two phases:

1. Pre-export patching
2. Post-export ONNX graph fixes

---

# Result

Ensures pure fp16-compatible graph for ORT.
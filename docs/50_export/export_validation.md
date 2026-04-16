# Export Validation

After export:

- ONNX model is loaded in ORT
- dummy forward pass executed

---

# Purpose

Detect:
- dtype mismatch
- missing tensors
- invalid graph structure
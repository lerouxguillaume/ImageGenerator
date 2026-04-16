# External Data (.onnx.data)

ONNX models store large tensors externally.

---

# Structure

- model.onnx
- model.onnx.data

---

# Behavior

Handled automatically by ORT:
- memory mapped
- no manual loading required

---

# Important

ImageGenerator does NOT manually read .onnx.data.

---

# Rules

- Do NOT add `dataOffset` or `dataLength` fields back to `ExternalTensorMeta` — those were removed intentionally; base weights for LoRA come from `_weights.safetensors`, not `.onnx.data`
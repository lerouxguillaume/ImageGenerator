# ONNX Indexing

Builds suffix-based lookup table.

---

# Process

For each tensor name:
- split by `_`
- generate suffix map entries

---

# Purpose

Enable O(1) LoRA lookup instead of full scan.
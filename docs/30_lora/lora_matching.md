                                                                 # LoRA Matching System

Matches LoRA keys to ONNX tensors.

---

# Method

1. Strip prefix (te/unet/te2)
2. Normalize key
3. Build suffix index
4. O(1) lookup

---

# Matching rule

Longest suffix match wins.

---

# Failure cases

- no match → skipped tensor
- ambiguity → warning logged
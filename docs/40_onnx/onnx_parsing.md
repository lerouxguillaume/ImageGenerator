# ONNX Parsing

Parses protobuf graph to extract:

- tensor names
- external data references
- shapes
- dtype

---

# Key field

TensorProto:
- field 8 → name
- field 14 → data_location

---

# Output

Produces:
- OnnxExternalIndex

Used by LoRA system.
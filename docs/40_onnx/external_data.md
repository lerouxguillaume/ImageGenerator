# ONNX External Data

## What this file explains
How large tensors are stored outside the `.onnx` file, how `parseExternalIndex` scans the binary to find them, and how the export pipeline consolidates sidecar files.

## When to use this
- `parseExternalIndex` returns 0 initializers
- ORT complains about missing external data
- Debugging `keep_initializers_as_inputs` export requirements
- Understanding the `.onnx.data` file layout

## Key invariants
- ORT resolves `.onnx.data` automatically — the C++ pipeline never reads it directly.
- `location`/`offset`/`length` from field 13 are parsed but **discarded** — base weights for LoRA come from `_weights.safetensors`, not `.onnx.data`. Do not add `dataOffset`/`dataLength` back to `ExternalTensorMeta`.
- Field-5 initializers (named, `keep_initializers_as_inputs=True`) are required for LoRA injection. Field-6 or anonymous Constant nodes will show 0 patches.

## Mental model

```
.onnx binary
  └─ GraphProto (field 2 of ModelProto)
       └─ initializer[] (field 5 of GraphProto) ← named, field-5, PyTorch export
            └─ TensorProto
                 ├─ field 8  = name
                 ├─ field 2  = data_type
                 ├─ field 1  = dims[]
                 ├─ field 14 = data_location (EXTERNAL = 1)
                 └─ field 13 = external_data { location, offset, length }
                                                        ↓
                                                .onnx.data sidecar
```

## TensorProto field reference

| Field | Name | Notes |
|---|---|---|
| 1 | `dims` | repeated varint — tensor shape |
| 2 | `data_type` | 1 = float32, 10 = float16 |
| 8 | `name` | string — initializer name |
| 13 | `external_data` | repeated StringStringEntryProto: `location`, `offset`, `length` |
| 14 | `data_location` | varint: DEFAULT=0, **EXTERNAL=1** |

## Parsing (`parseExternalIndex`)

`OnnxParser.cpp` scans the `.onnx` binary for TensorProto entries where `data_location == EXTERNAL` (field 14, value 1):

1. Walk `ModelProto` → `GraphProto` (field 2) → `initializer[]` (field 5).
2. For each TensorProto: check field 14 = 1.
3. Read `name` (field 8), `dims` (field 1), `data_type` (field 2).
4. Normalise all `.` and `/` in `name` to `_`.
5. Store in `OnnxExternalIndex` (`std::map<normName, ExternalTensorMeta>`).

The scan is a raw protobuf wire-format walk — no ONNX library is used. This keeps the C++ binary free of the protobuf runtime dependency.

### Diagnosing zero initializers

```
parseExternalIndex: text_encoder.onnx — 0 external initializer(s)
```

Causes:
- Model exported without `keep_initializers_as_inputs=True` → weights at field 6 (graph nodes) or anonymous.
- Model uses inline weights (field 14 absent or = 0).

Fix: re-export with the project scripts.

### GraphProto field placement diagnostic

```
ONNX graph fields (ALL): f1x969 f2x1 f5x180 ...
```

`f5x180` = 180 named initializers at field 5 (expected). `f5x0 f6x0` = no initializers → re-export required.

## SDXL export: consolidation

The legacy ONNX tracer writes each large tensor as a separate sidecar file. `export_component_to_dir` calls `consolidate_external_data` after export:

1. Reads per-tensor sidecar files (raw byte copy, no deserialisation).
2. Writes a single `<component>.onnx.data`.
3. Deletes the individual sidecar files.

After successful export: exactly `<component>.onnx` + `<component>.onnx.data`. Stale sidecar files from an interrupted consolidation are orphaned and safe to delete.

## Export requirements for LoRA

```python
keep_initializers_as_inputs=True   # weights → named field-5 initializers
do_constant_folding=False          # prevents weights → anonymous Constant nodes
```

Both set in `scripts/export_common.py::onnx_export()`. Third-party ONNX models typically lack these and will show 0 patches.

## Related files
- [onnx_indexing.md](onnx_indexing.md) — suffix index built from the parsed output
- [../30_lora/lora_key_format.md](../30_lora/lora_key_format.md) — how parsed names feed LoRA matching
- [../50_export/export_overview.md](../50_export/export_overview.md) — export script requirements
- `src/portraits/sd/OnnxParser.cpp` — `parseExternalIndex`, `resolveBundle`
- `src/portraits/sd/SdOnnxPatcher.hpp` — `OnnxExternalIndex`, `ExternalTensorMeta`

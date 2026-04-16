# ONNX Suffix Index

## Purpose

`buildExternalSuffixIndex()` (`sd/OnnxIndex.cpp`) converts the flat `OnnxExternalIndex` (a `std::map<normName, ExternalTensorMeta>`) into an `OnnxExternalSuffixIndex` — an `unordered_map` that maps every `_`-boundary suffix of every normalised name to its candidate list.

This gives O(1) average-case lookup for LoRA key matching regardless of model depth or prefix differences.

## Construction

```
for each normName in OnnxExternalIndex:
    for each suffix starting at a '_' boundary:
        suffixIndex[suffix].push_back(&entry)
```

The index is built once per `.onnx` path at `loadModelMetadata()` time and cached inside the `LoraInjector` instance for that path.

## Lookup in `matchExternalLoraKey`

1. Strip the Kohya prefix (`lora_te_`, `lora_unet_`, `lora_te2_`).
2. Append `_weight` (then retry with `_bias` on miss).
3. Call `unordered_map::find` on the suffix index — O(1).
4. If multiple candidates exist, the **longest suffix match** wins.
5. True ties (same suffix length) are logged as warnings.

Example:
```
LoRA base:  text_model_encoder_layers_0_self_attn_q_proj
lookup key: text_model_encoder_layers_0_self_attn_q_proj_weight
ONNX entry: text_encoder_text_model_encoder_layers_0_self_attn_q_proj_weight
                          ↑ prefix absorbed by the suffix index
```

## Normalisation convention

`parseExternalIndex` normalises all `.` and `/` in initializer names to `_` before populating the index. The suffix index inherits this normalisation — LoRA key stripping must also produce `_`-separated tokens to match correctly.

## Debug flag

Set `-DSD_LORA_MATCH_DEBUG=1` at compile time to log every lookup key, all candidates with their suffix lengths, and the chosen winner. Off by default (produces 180+ lines per model load).

## Related files
- [external_data.md](external_data.md) — parseExternalIndex that produces the input to this index
- [../30_lora/lora_key_format.md](../30_lora/lora_key_format.md) — prefix stripping and lookup that consumes this index

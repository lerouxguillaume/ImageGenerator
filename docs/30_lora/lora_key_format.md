# LoRA Key Format (Kohya)

## What this file explains
The naming convention used by Kohya-format LoRA safetensors files, how keys are grouped into layer triplets, and how the C++ pipeline matches them to ONNX initializer names.

## When to use this
- Diagnosing high miss counts from `check_lora_compat.py`
- Understanding why a LoRA trained for SD 1.5 may not match an SDXL model
- Adding support for a new Kohya prefix variant

## Key invariants
- Every LoRA layer consists of exactly three tensors: `lora_down.weight`, `lora_up.weight`, and optionally `alpha`. A missing `alpha` defaults to `rank` (effective scale = 1.0).
- The prefix (`lora_te_`, `lora_unet_`, `lora_te2_`) is stripped before suffix matching. The base name after stripping must match an ONNX initializer suffix.
- `_weight` is appended first; `_bias` is tried on miss. No other suffixes are attempted.
- `lora_te2_` keys only appear in SDXL LoRAs and target `text_encoder_2`.

## Mental model

```
Kohya key:  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
                │                                                                │
          prefix stripped                                                  suffix part
                ↓
base:      down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
                ↓  + "_weight"
lookup:    down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q_weight
                ↓  suffix index O(1) lookup
ONNX:      model_down_blocks_0_attentions_0_...attn1_to_q_weight
           (model_ prefix absorbed by suffix index)
```

## Implementation details

### Key anatomy

```
lora_{component}_{layer_path}.{tensor_role}
```

| Part | Example | Values |
|---|---|---|
| `component` | `te`, `unet`, `te2` | `te` = CLIP text encoder, `unet` = UNet, `te2` = OpenCLIP (SDXL only) |
| `layer_path` | `text_model_encoder_layers_0_self_attn_q_proj` | underscore-separated path |
| `tensor_role` | `lora_down.weight` | `lora_down.weight`, `lora_up.weight`, `alpha` |

### Shapes

| Tensor | Shape |
|---|---|
| `lora_down.weight` | `[rank, in_features]` |
| `lora_up.weight` | `[out_features, rank]` |
| `alpha` | scalar (not a matrix) |

### Delta formula

```
effectiveScale = userScale × (alpha / rank)
delta = effectiveScale × (lora_up @ lora_down)
      = effectiveScale × [out_features × rank] @ [rank × in_features]
      = [out_features × in_features]
```

Implemented in `computeLoraDelta(up, down, effectiveScale)` in `LoraMath.cpp`. All arithmetic in fp32.

### `parseLoraLayers` (`LoraParser.cpp`)

Groups keys by base name into `LoraLayer` triplets:
```
{base, down_tensor, up_tensor, alpha_value}
```
Keys are scanned once; `alpha` defaults to `rank` if absent.

### Prefix stripping in `matchExternalLoraKey` (`SdLoraMatch.cpp`)

```cpp
if (starts_with(key, "lora_te2_"))  stripped = key.substr(9);   // SDXL encoder 2
else if (starts_with(key, "lora_te_")) stripped = key.substr(8); // CLIP
else if (starts_with(key, "lora_unet_")) stripped = key.substr(10); // UNet
else return nullopt;  // unknown prefix — logged as miss
```

### Suffix index lookup

See [`docs/40_onnx/onnx_indexing.md`](../40_onnx/onnx_indexing.md) for the full suffix index mechanism.

## Common pitfalls
- LoRA trained on a different model architecture (e.g. SD 1.5 LoRA applied to SDXL) — layer paths won't match ONNX names. Use `check_lora_compat.py` to verify before reporting a bug.
- Custom Kohya scripts that use non-standard prefixes (e.g. `lora_text_encoder_`) — the prefix stripper only knows `lora_te_`, `lora_unet_`, `lora_te2_`. Keys with unknown prefixes return `nullopt` and are counted as misses.
- `alpha = 0` in a LoRA file — produces `effectiveScale = 0`, applying a zero delta (LoRA has no effect). Not a bug but easy to misread as "LoRA not applied".
- Rank mismatch between down and up tensors — `computeLoraDelta` will throw on the matmul dimension check.

## Related files
- [`../40_onnx/onnx_indexing.md`](../40_onnx/onnx_indexing.md) — suffix index construction and O(1) lookup
- [`lora_overview.md`](lora_overview.md) — architecture overview
- [`lora_debugging.md`](lora_debugging.md) — diagnosing miss counts
- `src/portraits/sd/LoraParser.cpp` — `parseLoraLayers()`
- `src/portraits/sd/LoraMath.cpp` — `computeLoraDelta()`
- `src/portraits/sd/SdLoraMatch.hpp/.cpp` — prefix stripping, suffix lookup
- `scripts/check_lora_compat.py` — compatibility checker

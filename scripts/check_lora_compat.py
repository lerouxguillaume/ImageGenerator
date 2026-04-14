"""
Check LoRA adapter compatibility with an exported ONNX model directory.

Mirrors the C++ suffix-matching logic (SdLoraMatch.cpp + OnnxIndex.cpp) to
report how many LoRA layers would be applied per component, without running
inference.

Each unique LoRA layer triplet (down + up) is counted once: matched if it
hits in at least one component, missed if it hits in none.  Layers with a
lora_unet_ prefix are not counted as misses against the text_encoder, and
vice versa.

Usage:
    python check_lora_compat.py <lora.safetensors> <model_dir> [--verbose]

Example output:
    LoRA:  loras/my_style.safetensors  (528 tensors, 264 layer triplets)
    Model: models/sd15_base/

      text_encoder.onnx  :  66 matched  (180 initializers)
      unet.onnx          : 192 matched  (686 initializers)
      vae_decoder.onnx   :   0 matched  (157 initializers)

    ✅  258/264 unique layers matched (97%)
       6 unmatched  —  run with --verbose to see keys
"""
import argparse
import json
import os
import struct
import sys
from collections import defaultdict
from typing import Optional


# ── Kohya key format ──────────────────────────────────────────────────────────
KOHYA_PREFIXES = ("lora_unet_", "lora_te2_", "lora_te_")  # longest-first

_DOWN  = ".lora_down.weight"
_UP    = ".lora_up.weight"
_ALPHA = ".alpha"


# ── Safetensors header reader (stdlib only) ───────────────────────────────────

def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        raw = f.read(header_len)
    return json.loads(raw.decode("utf-8"))


def load_lora_tensors(path: str) -> dict[str, list[int]]:
    header = _read_safetensors_header(path)
    return {
        name: meta.get("shape", [])
        for name, meta in header.items()
        if name != "__metadata__"
    }


# ── LoRA layer grouping ───────────────────────────────────────────────────────

def parse_lora_layers(tensors: dict[str, list[int]]) -> dict[str, dict]:
    """Group tensors into {base: {down_shape, up_shape, has_alpha}} triplets."""
    layers: dict[str, dict] = defaultdict(dict)
    for name, shape in tensors.items():
        if name.endswith(_DOWN):
            layers[name[: -len(_DOWN)]]["down"] = shape
        elif name.endswith(_UP):
            layers[name[: -len(_UP)]]["up"] = shape
        elif name.endswith(_ALPHA):
            layers[name[: -len(_ALPHA)]]["alpha"] = True
    return dict(layers)


# ── ONNX initializer names ────────────────────────────────────────────────────

def get_initializer_names(onnx_path: str) -> list[str]:
    import onnx
    model = onnx.load_model(onnx_path, load_external_data=False)
    return [
        init.name.replace(".", "_").replace("/", "_")
        for init in model.graph.initializer
    ]


# ── Suffix index ──────────────────────────────────────────────────────────────

def build_suffix_index(names: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parts = name.split("_")
        for i in range(len(parts)):
            index["_".join(parts[i:])].append(name)
    return dict(index)


# ── Key matching ──────────────────────────────────────────────────────────────

def match_lora_base(
    suffix_index: dict[str, list[str]],
    lora_base: str,
) -> Optional[str]:
    stripped = lora_base
    for prefix in KOHYA_PREFIXES:
        if lora_base.startswith(prefix):
            stripped = lora_base[len(prefix):]
            break

    for key in (stripped + "_weight", stripped + "_bias"):
        candidates = suffix_index.get(key)
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=len)  # longest name = most specific match

    return None


# ── Per-component matching (returns set of matched base keys) ─────────────────

def match_component(
    onnx_path: str,
    valid_keys: set[str],
) -> tuple[set[str], int]:
    """Return (set of matched base keys, number of initializers)."""
    if not os.path.exists(onnx_path):
        return set(), -1

    names = get_initializer_names(onnx_path)
    if not names:
        return set(), 0

    index = build_suffix_index(names)
    matched: set[str] = set()

    for base in valid_keys:
        if match_lora_base(index, base) is not None:
            matched.add(base)

    return matched, len(names)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check LoRA ↔ ONNX model compatibility without running inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lora_path",  help="Path to the LoRA .safetensors file")
    parser.add_argument("model_dir",  help="Path to the exported ONNX model directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each unmatched LoRA key")
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("Error: 'onnx' package required.  pip install onnx", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.lora_path):
        print(f"Error: LoRA file not found: {args.lora_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.model_dir):
        print(f"Error: model directory not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load LoRA ─────────────────────────────────────────────────────────────
    tensors = load_lora_tensors(args.lora_path)
    layers  = parse_lora_layers(tensors)

    # Only consider complete triplets (down + up required; alpha optional).
    valid_keys: set[str] = {
        base for base, layer in layers.items()
        if "down" in layer and "up" in layer
    }

    lora_size_mb = os.path.getsize(args.lora_path) / 1e6
    print(f"LoRA:  {args.lora_path}  "
          f"({lora_size_mb:.0f} MB, {len(tensors)} tensors, "
          f"{len(valid_keys)} complete layer triplets)")
    print(f"Model: {args.model_dir}")
    print()

    # ── Discover components ───────────────────────────────────────────────────
    components = sorted(
        f for f in os.listdir(args.model_dir) if f.endswith(".onnx")
    )
    if not components:
        print("No .onnx files found in model directory.")
        sys.exit(1)

    # ── Match per component ───────────────────────────────────────────────────
    # Each key is counted once globally: matched if any component claims it.
    globally_matched: set[str] = set()
    col_w = max(len(c) for c in components) + 2

    for fname in components:
        onnx_path = os.path.join(args.model_dir, fname)
        matched, n_init = match_component(onnx_path, valid_keys)

        if n_init == -1:
            print(f"  {fname:{col_w}}: not found — skipped")
            continue
        if n_init == 0:
            print(f"  {fname:{col_w}}: 0 named initializers "
                  f"— re-export with keep_initializers_as_inputs=True")
            continue

        globally_matched |= matched
        print(f"  {fname:{col_w}}: {len(matched):>3} matched  "
              f"({n_init} initializers)")

    # ── Global summary ────────────────────────────────────────────────────────
    globally_missed = valid_keys - globally_matched
    total     = len(valid_keys)
    n_matched = len(globally_matched)
    n_missed  = len(globally_missed)

    print()
    if total == 0:
        print("⚠️  LoRA file has no complete layer triplets.")
        sys.exit(0)

    pct  = 100 * n_matched // total
    icon = "✅" if pct >= 90 else ("⚠️ " if pct >= 50 else "❌")
    print(f"{icon}  {n_matched}/{total} unique layers matched ({pct}%)")

    if n_missed > 0:
        if args.verbose:
            print(f"\n   Unmatched keys ({n_missed}):")
            # Group by Kohya prefix for readability
            by_prefix: dict[str, list[str]] = defaultdict(list)
            for key in sorted(globally_missed):
                prefix = next((p for p in KOHYA_PREFIXES if key.startswith(p)), "other_")
                by_prefix[prefix].append(key)
            for prefix, keys in sorted(by_prefix.items()):
                print(f"   [{prefix}*]  {len(keys)} key(s):")
                for k in keys:
                    print(f"     {k}")
        else:
            print(f"   {n_missed} unmatched  —  run with --verbose to see keys")

    if n_matched == 0:
        print()
        print("   Common causes:")
        print("   • Model exported without keep_initializers_as_inputs=True")
        print("   • LoRA trained on a different architecture")


if __name__ == "__main__":
    main()

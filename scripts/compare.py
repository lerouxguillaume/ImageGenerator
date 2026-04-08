#!/usr/bin/env python3
"""
Compare LoRA safetensor keys to ONNX model keys and show differences.
"""

from safetensors.torch import load_file
import argparse

def load_lora_keys(path: str):
    """Load LoRA tensor keys from a safetensors file."""
    return set(load_file(path).keys())

def load_onnx_keys(path: str):
    """Load tensor names from an ONNX model (named initializers only, not Constant nodes)."""
    import onnx
    model = onnx.load(path)
    return {init.name for init in model.graph.initializer}

def check_onnx(path: str, n_samples: int = 10) -> None:
    """Quick sanity check: show initializer count and sample names."""
    import onnx
    model = onnx.load(path)

    inits  = list(model.graph.initializer)
    consts = [n for n in model.graph.node if n.op_type == "Constant"]

    print(f"\n{path}")
    print(f"  Named initializers : {len(inits)}")
    print(f"  Constant nodes     : {len(consts)}")

    if inits:
        print(f"  Sample names (first {min(n_samples, len(inits))}):")
        for t in inits[:n_samples]:
            print(f"    {t.name}")
    else:
        print("  ⚠️  No named initializers — LoRA patching will not work.")
        print(f"  Constant node sample names (first {min(n_samples, len(consts))}):")
        for n in consts[:n_samples]:
            print(f"    {n.output[0]}")

def main():
    parser = argparse.ArgumentParser(description="Compare LoRA vs ONNX keys")
    parser.add_argument("--lora", help="Path to LoRA .safetensors")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--check", action="store_true",
                        help="Quick check: show initializer count and sample names (no LoRA needed)")
    args = parser.parse_args()

    if args.check or not args.lora:
        check_onnx(args.onnx)
        return

    lora_keys = load_lora_keys(args.lora)
    onnx_keys = load_onnx_keys(args.onnx)

    missing_in_onnx = lora_keys - onnx_keys
    missing_in_lora = onnx_keys - lora_keys
    matched = lora_keys & onnx_keys

    print(f"LoRA keys: {len(lora_keys)}, ONNX keys: {len(onnx_keys)}")
    print(f"Matched keys: {len(matched)}")
    print(f"Keys in LoRA but not in ONNX: {len(missing_in_onnx)}")
    for k in sorted(missing_in_onnx):
        print("  ✖", k)
    print(f"Keys in ONNX but not in LoRA: {len(missing_in_lora)}")
    for k in sorted(missing_in_lora):
        print("  ✖", k)

if __name__ == "__main__":
    main()
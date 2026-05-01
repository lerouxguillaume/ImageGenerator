"""
Retrofit tool: patch Resize fp16 inputs in an already-exported model directory.

Use this when you have models exported before fix_resize_fp16 was added to the
export pipeline.  New SD 1.5 and SDXL exports already apply the fix to fp16
UNet/VAE components automatically.

Only the .onnx proto is rewritten.  The .onnx.data weight sidecar is never
touched, so this is fast regardless of model size.  Safe to run multiple times.

Usage:
    python patch_unet_resize.py <model_dir>

Example:
    python patch_unet_resize.py /media/sf_shared_vm/models/animagineXLV31_v31
"""
import argparse
import glob
import os
import sys

from export_common import fix_resize_fp16_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Resize fp16 nodes in all ONNX components of a model directory"
    )
    parser.add_argument("model_dir", help="Path to the exported ONNX model directory")
    args = parser.parse_args()

    onnx_files = sorted(glob.glob(os.path.join(args.model_dir, "*.onnx")))
    if not onnx_files:
        print(f"Error: no .onnx files found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    for path in onnx_files:
        print(f"Patching {os.path.basename(path)} ...")
        fix_resize_fp16_input(path)
    print("Done.")


if __name__ == "__main__":
    main()

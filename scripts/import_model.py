"""
Unified model import entry point for ImageGenerator.

Usage:
    python import_model.py --input MODEL_FILE --output OUTPUT_DIR [--arch auto|sd15|sdxl]

Stdout protocol (C++ parent reads these lines to drive the progress UI):
    PROGRESS:analyzing
    PROGRESS:exporting_sd15   or   PROGRESS:exporting_sdxl
    PROGRESS:validating
    PROGRESS:done
    ERROR:<human-readable message>

All other stdout/stderr lines are treated as log output by the C++ parent.
"""
import argparse
import json
import os
import struct
import sys
import time

# ── Architecture detection (no external deps) ─────────────────────────────────

def detect_arch(model_file: str) -> str:
    """Detect SD1.5 vs SDXL by scanning the safetensors JSON header."""
    try:
        with open(model_file, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return "unknown"
            header_size = struct.unpack("<Q", raw)[0]
            if header_size == 0 or header_size > 512 * 1024 * 1024:
                return "unknown"
            header = f.read(header_size).decode("utf-8", errors="replace")
        if "conditioner.embedders.1." in header:
            return "sdxl"
        if "model.diffusion_model." in header or "cond_stage_model." in header:
            return "sd15"
        return "unknown"
    except OSError as e:
        print(f"ERROR:Could not read file for arch detection: {e}", flush=True)
        return "unknown"


# ── Output validation ─────────────────────────────────────────────────────────

SD15_REQUIRED = ["text_encoder.onnx", "unet.onnx", "vae_decoder.onnx", "vae_encoder.onnx", "model.json"]
SDXL_REQUIRED = ["text_encoder.onnx", "text_encoder_2.onnx", "unet.onnx", "vae_decoder.onnx", "vae_encoder.onnx", "model.json"]

def validate_output(output_dir: str, arch: str) -> None:
    required = SDXL_REQUIRED if arch == "sdxl" else SD15_REQUIRED
    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    missing += [
        f"{f}.data"
        for f in required
        if f.endswith(".onnx") and not os.path.exists(os.path.join(output_dir, f"{f}.data"))
    ]
    lora_weight_files = ["text_encoder_weights.safetensors", "unet_weights.safetensors"]
    if arch == "sdxl":
        lora_weight_files.append("text_encoder_2_weights.safetensors")
    missing += [
        f for f in lora_weight_files
        if not os.path.exists(os.path.join(output_dir, f))
    ]
    if missing:
        raise RuntimeError(f"Export incomplete - missing files: {', '.join(missing)}")


# ── Capabilities block ────────────────────────────────────────────────────────

def write_capabilities(output_dir: str, arch: str) -> None:
    """Extend model.json with a capabilities block after export."""
    path = os.path.join(output_dir, "model.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}

    components: dict = {}
    if arch == "sd15":
        components = {
            "text_encoder": {"dtype": "fp16"},
            "unet":         {"dtype": "fp16"},
            "vae_decoder":  {"dtype": "fp16"},
            "vae_encoder":  {"dtype": "fp16"},
        }
    else:
        components = {
            "text_encoder":   {"dtype": "fp32"},
            "text_encoder_2": {"dtype": "fp32"},
            "unet":           {"dtype": "fp16"},
            "vae_decoder":    {"dtype": "fp16"},
            "vae_encoder":    {"dtype": "fp16"},
        }

    data["capabilities"] = {
        "dynamic_shapes":       True,
        "vae_encoder_available": True,
        "lora_compatible":      True,
        "components":           components,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_import(model_file: str, output_dir: str, arch: str) -> None:
    # The scripts live in the same directory as this file.
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    print("PROGRESS:analyzing", flush=True)

    if arch == "auto":
        print(f"Detecting architecture for: {os.path.basename(model_file)}", flush=True)
        arch = detect_arch(model_file)
        print(f"Detected: {arch}", flush=True)
        if arch == "unknown":
            print(
                "ERROR:Could not detect model architecture. "
                "Re-run with --arch sd15 or --arch sdxl.",
                flush=True,
            )
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    if arch == "sd15":
        print(f"PROGRESS:exporting_sd15", flush=True)
        from export_onnx_models import export_sd15
        export_sd15(model_file, output_dir, resume=False, validate=False)
    elif arch == "sdxl":
        print(f"PROGRESS:exporting_sdxl", flush=True)
        from sdxl_export_onnx_models import export_sdxl
        export_sdxl(model_file, output_dir, resume=False, validate=False)
    else:
        print(f"ERROR:Unknown architecture '{arch}'", flush=True)
        sys.exit(1)

    print("PROGRESS:validating", flush=True)
    validate_output(output_dir, arch)
    write_capabilities(output_dir, arch)

    elapsed = time.time() - t0
    print(f"Import complete in {elapsed:.0f}s", flush=True)
    print("PROGRESS:done", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a safetensors model into ONNX format")
    parser.add_argument("--input",  required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    parser.add_argument(
        "--arch",
        default="auto",
        choices=["auto", "sd15", "sdxl"],
        help="Model architecture (default: auto-detect)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_import(args.input, args.output, args.arch)
    except (FileNotFoundError, ImportError) as e:
        print(f"ERROR:{e}", flush=True)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR:{e}", flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("ERROR:Import cancelled.", flush=True)
        sys.exit(1)

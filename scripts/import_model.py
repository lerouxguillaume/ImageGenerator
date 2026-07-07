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

def _graph_has_dynamic_spatial(onnx_path: str, input_name: str = "latent") -> bool | None:
    """Return True iff <input_name>'s H/W dims (axes 2, 3) are dynamic in the graph.

    A dim is dynamic when it is symbolic (dim_param set) or unset; a concrete
    dim_value is static. Loads graph structure only (no tensor bytes). Returns
    None when the graph cannot be inspected (onnx missing / parse error / file
    absent); the caller (`_detect_hires_capable`) treats None as fail-closed.
    """
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load_model(onnx_path, load_external_data=False)
    except Exception:
        return None

    def _is_dynamic(dim) -> bool:
        return bool(dim.dim_param) or not dim.HasField("dim_value")

    for inp in model.graph.input:
        if inp.name != input_name:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) < 4:
            return False
        return _is_dynamic(dims[2]) and _is_dynamic(dims[3])
    return False


def _detect_hires_capable(output_dir: str, arch: str) -> bool:
    """Hires needs BOTH the UNet and VAE decoder to accept a larger-than-native
    latent, i.e. both exported with dynamic H/W axes.

    This is derived from the actual exported graphs (ground truth) rather than
    assumed from the architecture, so it is correct for any arch and any export
    variant: SD1.5 (always dynamic) reports True; a static SDXL export reports
    False; an SDXL export produced with --dynamic-spatial reports True
    automatically once both graphs carry the axes.

    If a graph cannot be probed, fail CLOSED (return False) and warn loudly. The
    old behaviour fell back to the arch heuristic (SD1.5 -> True), which would
    mislabel a pre-dynamic *static-VAE* SD1.5 import as hires-capable — the
    runtime would then hit a VAE input-shape error at hires time instead of
    presenting a cleanly disabled control. A wrong "capable" is far worse than a
    wrong "not capable", so an unprobeable graph is treated as not capable.
    """
    unet_dyn = _graph_has_dynamic_spatial(os.path.join(output_dir, "unet.onnx"))
    vae_dyn = _graph_has_dynamic_spatial(os.path.join(output_dir, "vae_decoder.onnx"))
    if unet_dyn is None or vae_dyn is None:
        unprobeable = []
        if unet_dyn is None:
            unprobeable.append("unet.onnx")
        if vae_dyn is None:
            unprobeable.append("vae_decoder.onnx")
        print(
            "VERIFY:warn:capability:could not probe dynamic H/W axes on "
            f"{', '.join(unprobeable)} (missing file, onnx import failure, or "
            "parse error) - defaulting hires_capable=False to avoid mislabelling "
            "a static-VAE model as hires-capable",
            flush=True,
        )
        return False
    return bool(unet_dyn and vae_dyn)


def write_capabilities(output_dir: str, arch: str) -> None:
    """Extend model.json with a capabilities block after a successful export.

    This is the authoritative source of truth that the C++ pipeline reads to
    know what the model supports — VAE encoder availability and LoRA compatibility.
    Written only after validate_output() confirms all expected files exist.
    """
    path = os.path.join(output_dir, "model.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"model.json missing or corrupt before writing capabilities: {e}\n"
            f"Expected at: {path}"
        ) from e

    vae_encoder_available = os.path.exists(os.path.join(output_dir, "vae_encoder.onnx"))

    data["capabilities"] = {
        "vae_encoder_available": vae_encoder_available,
        "lora_compatible":       True,
        # Derived from the exported graphs, not the arch: hires needs the UNet AND
        # VAE decoder to both accept a larger-than-native latent (dynamic H/W).
        # SD1.5 always exports both dynamic → True. SDXL is True only when exported
        # with --dynamic-spatial; the default static SDXL export → False.
        "hires_capable":         _detect_hires_capable(output_dir, arch),
        # Written only after verify_model() passed — a model.json carrying this
        # flag has been confirmed to run, not just to have all its files present.
        "verified":              True,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Verification (isolated subprocess) ────────────────────────────────────────

def run_verification(scripts_dir: str, output_dir: str, arch: str,
                     timeout: int = 900) -> int:
    """Run verify_model.py in a fresh interpreter, relaying its VERIFY: lines.

    Returns the child exit code:
        0  — all hard checks passed (or only warnings)
        3  — a genuine hard failure (reject the import)
        other / negative — crash, OOM, timeout (inconclusive)
    """
    import subprocess

    script = os.path.join(scripts_dir, "verify_model.py")
    if not os.path.exists(script):
        print(f"VERIFY:warn:inference:verify_model.py not found at {script}", flush=True)
        return 0  # nothing to run — don't block the import

    proc = subprocess.Popen(
        [sys.executable, script, "--output", output_dir, "--arch", arch],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            # Swallow the child's internal end markers; relay everything else so
            # the C++ parent sees the VERIFY: check lines and log output.
            if line in ("VERIFY:done", "VERIFY:rejected"):
                continue
            print(line, flush=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("VERIFY:warn:inference:verification timed out", flush=True)
        return 124
    return proc.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def run_import(model_file: str, output_dir: str, arch: str, resume: bool = False) -> None:
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
        export_sd15(model_file, output_dir, resume=resume, validate=False)
    elif arch == "sdxl":
        print(f"PROGRESS:exporting_sdxl", flush=True)
        from sdxl_export_onnx_models import export_sdxl
        # dynamic_spatial=True is the in-app default for SDXL: it exports the UNet +
        # VAE decoder with dynamic H/W axes (enabling hires), is far faster than the
        # static export (~2-3 min vs ~1 h — the UNet/decoder trace at a tiny latent
        # instead of full 1024), and produces identical output at native 1024.
        # hires_capable is then derived true from the graphs. emit_fp32_hedge=False:
        # the in-app model is lean fp16-only (fp16 validated stable; no in-app swap
        # mechanism to use the ~200 MB spare). See docs/50_export/export_overview.md.
        export_sdxl(model_file, output_dir, resume=resume, validate=False,
                    dynamic_spatial=True, emit_fp32_hedge=False)
    else:
        print(f"ERROR:Unknown architecture '{arch}'", flush=True)
        sys.exit(1)

    print("PROGRESS:validating", flush=True)
    validate_output(output_dir, arch)

    # Inference smoke test — proves the exported graphs actually run and produce
    # finite, prompt-sensitive output. Gates registration: a model that fails
    # here (e.g. text-encoder attention collapse) is never added to the registry.
    #
    # Run it in a *separate* process from this one. The export above imported
    # torch and loaded a full pipeline; loading the multi-GB ONNX models for
    # verification on top of that resident memory can OOM-kill the process. A
    # fresh interpreter (ORT + numpy only) is both lighter and crash-isolated:
    # if it dies we treat verification as inconclusive rather than failing an
    # otherwise-clean export.
    print("PROGRESS:verifying", flush=True)
    rc = run_verification(scripts_dir, output_dir, arch)
    if rc == 3:
        print(
            "ERROR:Model failed inference verification - see VERIFY:fail lines "
            "above. The export produced files but they do not run correctly.",
            flush=True,
        )
        sys.exit(1)
    elif rc != 0:
        # Crash / OOM / timeout inside the isolated verifier — not a proven defect.
        # Import the model but flag that the full inference check did not complete.
        print(
            f"VERIFY:warn:inference:verification did not complete on this machine "
            f"(exit {rc}) - imported without full inference check",
            flush=True,
        )

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip components whose output files already exist (resume an interrupted export)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_import(args.input, args.output, args.arch, resume=args.resume)
    except (FileNotFoundError, ImportError) as e:
        print(f"ERROR:{e}", flush=True)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR:{e}", flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("ERROR:Import cancelled.", flush=True)
        sys.exit(1)

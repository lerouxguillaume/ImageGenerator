"""
Shared utilities for ONNX model export (SD 1.5 and SDXL).
"""
import gc
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager

import torch


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def check_dependencies(required: list[str], optional: list[str] | None = None) -> None:
    """Verify required packages are importable; warn about missing optional ones."""
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        pip_names = " ".join(missing)
        raise ImportError(
            f"Missing required package(s): {', '.join(missing)}\n"
            f"Install with:  pip install {pip_names}"
        )
    for pkg in (optional or []):
        if importlib.util.find_spec(pkg) is None:
            print(f"  ℹ️  Optional package '{pkg}' not installed — "
                  f"some features will be skipped")


def check_model_file(path: str) -> None:
    """Verify the source checkpoint exists and looks reasonable."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Pass the full path to a .safetensors or .ckpt checkpoint."
        )
    size_gb = os.path.getsize(path) / 1e9
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".safetensors", ".ckpt", ".pt"):
        print(f"  ⚠️  Unexpected file extension '{ext}' — expected .safetensors / .ckpt")
    print(f"  Source: {path}  ({size_gb:.1f} GB)")


# ── Step context manager ──────────────────────────────────────────────────────

@contextmanager
def export_step(name: str):
    """Times a named export step and re-raises exceptions with context.

    Usage:
        with export_step("UNet"):
            onnx_export(...)
    """
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        yield
        print(f"  ✅ Done in {time.time() - t0:.1f}s")
    except MemoryError as e:
        raise MemoryError(
            f"Step '{name}' ran out of memory.\n"
            f"Close other applications or try exporting on a machine with more RAM."
        ) from e
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"Step '{name}' ran out of GPU memory.\n"
            f"Try CPU-only export or reduce batch size."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Step '{name}' failed: {e}") from e


# ── CLIP tracing patch ────────────────────────────────────────────────────────

def patch_clip_for_tracing() -> None:
    """Replace create_causal_mask in CLIP's module namespace with a
    TorchScript-safe implementation.

    Transformers 4.51+ routes CLIP attention masking through
    create_causal_mask → eager_mask → sdpa_mask, which calls
    `q_length.shape[0]` on a value the TorchScript tracer passes as a tuple,
    causing 'IndexError: tuple index out of range'.  This replacement returns
    an identical upper-triangular causal mask without that code path.
    """
    import transformers.models.clip.modeling_clip as _clip_mod

    def _safe_causal_mask(config, inputs_embeds,
                           attention_mask=None, past_key_values=None, **_):
        seq = inputs_embeds.shape[1]
        mask = torch.full((1, 1, seq, seq), float('-inf'),
                          dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return torch.triu(mask, diagonal=1)

    _clip_mod.create_causal_mask = _safe_causal_mask


# ── ONNX export helper ────────────────────────────────────────────────────────

def onnx_export(model: torch.nn.Module,
                dummy_inputs,
                path: str,
                *,
                input_names: list,
                output_names: list,
                dynamic_axes: dict | None = None,
                do_constant_folding: bool = False) -> None:
    """Consistent torch.onnx.export call with project-wide defaults.

    Always uses dynamo=False (TorchScript path) because the dynamo path does
    not support dynamic_axes.

    keep_initializers_as_inputs=True forces PyTorch to store every model parameter
    as a named ONNX initializer (graph.initializer) in its original, non-transposed
    form with the Python dotted path as the name.  This is what the C++ LoRA patcher
    requires: parseTensorIndex finds them by name, applyLoraToBytes patches the
    un-transposed weight in-place.

    do_constant_folding=False (default) keeps the graph structure clean (weight.T
    is a runtime Transpose op, not a pre-folded constant), which is consistent with
    the initializer storing the un-transposed weight.

    Prints the output file size on success.
    """
    name = os.path.basename(path)
    print(f"  Exporting {name} ...")
    torch.onnx.export(
        model, dummy_inputs, path,
        opset_version=18,
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        do_constant_folding=do_constant_folding,
        keep_initializers_as_inputs=True,
        export_params=True,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✅ {name}  ({size_mb:.0f} MB)")


# ── Post-processing helpers ───────────────────────────────────────────────────

def fix_fp32_constants(path: str) -> None:
    """Convert any remaining fp32 initializers/constants to fp16 in-place.

    Runs in a subprocess so the main process heap is unaffected by loading the
    large ONNX protobuf a second time (would otherwise double peak RAM).
    Raises RuntimeError with subprocess stderr on failure.
    """
    script = textwrap.dedent(f"""
        import onnx, numpy as np, os
        from onnx import numpy_helper, TensorProto
        path = {repr(path)}
        model = onnx.load(path)
        changed = 0
        for init in model.graph.initializer:
            if init.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(init).astype(np.float16)
                init.CopyFrom(numpy_helper.from_array(arr, init.name))
                changed += 1
        for node in model.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.HasField("t") and attr.t.data_type == TensorProto.FLOAT:
                        arr = numpy_helper.to_array(attr.t).astype(np.float16)
                        attr.t.CopyFrom(numpy_helper.from_array(arr))
                        changed += 1
        try:
            onnx.save(model, path)
        except Exception:
            # Model exceeds protobuf's 2 GB limit — save weights as external data.
            # ORT loads external-data models transparently from the same directory.
            data_location = os.path.basename(path) + ".data"
            data_full_path = os.path.join(os.path.dirname(os.path.abspath(path)), data_location)
            if os.path.exists(data_full_path):
                os.remove(data_full_path)
            onnx.save_model(
                model, path,
                save_as_external_data=True,
                location=data_location,
                all_tensors_to_one_file=True,
                convert_attribute=True,
            )
        print(f"  → fixed {{changed}} fp32 constant(s) → float16")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fix_fp32_constants failed for {os.path.basename(path)}:\n"
            f"{result.stderr.strip()}"
        )
    if result.stdout:
        print(result.stdout.strip())


def simplify_with_onnxsim(path: str) -> None:
    """Run onnxsim to fold remaining dynamic shape ops.
    Silently skipped if onnxsim is not installed; logs a warning on other errors.
    """
    try:
        import onnx
        import onnxsim
        model = onnx.load(path)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_sim, path)
            print(f"  ✅ {os.path.basename(path)} simplified with onnxsim")
        else:
            print(f"  ⚠️  onnxsim check failed, keeping original")
    except ImportError:
        pass  # onnxsim not installed — not an error
    except Exception as e:
        print(f"  ⚠️  onnxsim failed ({e}), keeping original")


def write_model_json(output_dir: str, model_type: str) -> None:
    """Write model.json consumed by the C++ runtime to detect the model family."""
    path = os.path.join(output_dir, "model.json")
    with open(path, "w") as f:
        json.dump({"type": model_type}, f)
    print(f"  ✅ model.json  (type={model_type})")


def free(*objs) -> None:
    """Delete objects and run gc.collect() to release GPU/CPU memory between exports."""
    for obj in objs:
        del obj
    gc.collect()


# ── Shared model wrappers ─────────────────────────────────────────────────────

class VAEDecoderWrapper(torch.nn.Module):
    """Wraps the diffusers VAE decoder so it handles the 0.18215 unscaling
    internally.  Exported to vae_decoder.onnx for both SD 1.5 and SDXL."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample
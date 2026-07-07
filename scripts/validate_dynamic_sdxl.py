"""
Validate a dynamic-spatial SDXL export (the hires-capable SDXL export).

Run this on a machine with `onnxruntime-gpu` (ideally CUDA) after exporting a
model with (dynamic-spatial is the default now — no flag needed):

    python scripts/sdxl_export_onnx_models.py <sdxl.safetensors> --name my_sdxl_dyn

It answers the make-or-break questions the dev box cannot (no torch / no GPU):

  1. STRUCTURE — are the UNet, VAE decoder AND VAE encoder graphs dynamic in H/W?
     (onnx.checker + shape inference + symbolic-dim probe; the encoder probes its
     "image" input — dynamic there = pixel_hires_capable, i.e. sharp pixel-mode hires)
  2. RUNTIME  — do they RUN and produce finite, non-degenerate output at the
     native latent 128 AND larger latents (160, 192 -> 1280 / 1536 px)? The encoder
     is exercised at the matching IMAGE sizes (px = latent*8) — a NEW memory profile,
     since it was static-1024 before — with the same VRAM sampler.
  3. time_ids — at each size, is time_ids scaled to that resolution
     ({H,W,0,0,H,W}, the diffusers img2img convention) accepted as shape (1,6)
     and does the graph still produce sane output? This is exactly the input the
     implementation's per-pass time_ids swap depends on, so it is exercised at
     non-native sizes now.
  4. NUMERICS — with --torch-reference <checkpoint>, run the SAME checkpoint's
     UNet (and VAE decoder) forward in PyTorch with identical inputs and confirm
     the ONNX export agrees to fp16 precision. (There is intentionally NO static
     ONNX export to diff against — the plan is a single dynamic export, ever.)
  5. VRAM — prints a marker before each forward so `nvidia-smi -l 1` peaks can be
     attributed to a size (ORT does not expose CUDA peak to Python).

This is a validation *harness* only — it never writes into the model directory.
Exit code 0 = all hard checks passed, 1 = a hard check failed, 2 = could not run.

    python scripts/validate_dynamic_sdxl.py --model models/my_sdxl_dyn \
        [--torch-reference path/to/sdxl.safetensors] [--sizes 128,160,192]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Shared ORT primitives (see scripts/ort_probe.py). This validator runs GPU-first:
# it preloads CUDA/cuDNN DLLs and caps graph-opt at BASIC — both wired through the
# thin _session/_gpu_providers wrappers below.
from ort_probe import GPU_PROVIDERS as _GPU_PROVIDERS
from ort_probe import dim_is_dynamic as _dim_is_dynamic
from ort_probe import gpu_providers as _gpu_providers_base
from ort_probe import make_session
from ort_probe import np_dtype as _np_dtype

SEQ_LEN = 77
HIDDEN_DIM = 2048   # SDXL concat(CLIP-L 768, OpenCLIP-G 1280)
POOLED_DIM = 1280

# Numeric-agreement thresholds for the torch reference (fp16). Heuristic — fp16
# has ~1e-3 relative precision and torch/ORT use different kernels + accumulation
# order, so a few 1e-2 absolute over a deep UNet is normal. The gate is on the
# MEAN relative error (robust to a lone fp16 outlier); max-abs is informational.
_UNET_REL_MEAN_FAIL = 0.05
_VAE_REL_MEAN_FAIL = 0.05

_hard_failures: list[str] = []


def _emit(status: str, check: str, detail: str) -> None:
    print(f"[{status.upper():4}] {check}: {detail}", flush=True)
    if status == "fail":
        _hard_failures.append(check)


def _is_oom(msg: str) -> bool:
    """True if an ORT exception is a GPU out-of-memory (a VRAM ceiling, not a defect)."""
    m = msg.lower()
    return ("out of memory" in m or "failed to allocate" in m
            or "cuda failure 2" in m or "cudnn_status_alloc_failed" in m)


# GPU-box validator: preload CUDA/cuDNN DLLs (no system toolkit needed) and cap
# the graph optimizer at BASIC (ORT's full optimizer can segfault the CPU EP on
# the fp16 Resize graphs this export wraps). Both wrap the shared ort_probe core.
def _gpu_providers() -> list:
    return _gpu_providers_base(preload=True)


def _session(path: str, prefer_gpu: bool = True):
    return make_session(path, prefer_gpu=prefer_gpu, preload=True, basic_opt=True)


class _VramSampler:
    """High-frequency background sampler of GPU used-memory via NVML.

    `nvidia-smi -l 1` polls once a second and misses sub-second forward peaks.
    This spins a daemon thread sampling `nvmlDeviceGetMemoryInfo().used` every
    few ms for the duration of a `with` block and records the peak, so VRAM is
    attributed to each forward exactly. No-op (empty report) if pynvml / NVML is
    unavailable. Reports whole-device used memory (the number that decides where
    OOM lands) plus the delta over the pre-forward baseline.
    """
    _MB = 1024 * 1024

    def __init__(self, device_index: int = 0):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.ok = True
        except Exception:
            self.ok = False

    def _used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._h).used)

    def __enter__(self):
        if not self.ok:
            return self
        import threading
        self._stop = threading.Event()
        self.baseline = self.peak = self._used()

        def _loop():
            while not self._stop.is_set():
                u = self._used()
                if u > self.peak:
                    self.peak = u
                self._stop.wait(0.005)  # ~200 Hz

        self._t = threading.Thread(target=_loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if not self.ok:
            return
        self._stop.set()
        self._t.join(timeout=1.0)

    def report(self) -> str:
        if not self.ok:
            return ""
        return (f"  [VRAM device-peak {self.peak / self._MB:.0f} MiB, "
                f"+{(self.peak - self.baseline) / self._MB:.0f} MiB over pre-forward]")


def _rel_stats(onnx_out: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Return (max_abs, mean_abs, mean_rel) between an ONNX and a reference array."""
    a = onnx_out.astype(np.float64)
    b = ref.astype(np.float64)
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = float(np.abs(b).mean()) or 1.0
    return max_abs, mean_abs, mean_abs / denom


# ── structural checks ─────────────────────────────────────────────────────────

def _check_structure(path: str, label: str, input_name: str = "latent") -> bool:
    """checker + shape inference + symbolic-dim probe on <input_name> (H/W axes 2,3).

    Returns True iff the graph loaded and that input's H/W are dynamic. checker /
    shape-inference problems are reported as warnings (the ORT forward below is the
    authoritative proof); static H/W is a hard fail because it means the
    dynamic-spatial export did not take effect (e.g. the model was exported with
    --static). input_name is "latent" for the UNet / VAE decoder and "image" for
    the VAE encoder (pixel-mode hires re-encode).
    """
    try:
        import onnx
    except ImportError:
        # A missing inspector tool is a venv gap, not an export defect — don't
        # hard-fail the run over it (pip install onnx to enable these checks).
        _emit("skip", f"{label}.structure", "onnx not installed in this venv — cannot inspect graph (pip install onnx)")
        return False
    if not os.path.exists(path):
        _emit("fail", f"{label}.structure", f"missing file: {path}")
        return False

    model = onnx.load_model(path, load_external_data=False)

    try:
        onnx.checker.check_model(path)  # path form resolves the .onnx.data sidecar
        _emit("ok", f"{label}.checker", "onnx.checker passed")
    except Exception as e:
        _emit("warn", f"{label}.checker", f"checker reported: {str(e)[:200]}")

    try:
        onnx.shape_inference.infer_shapes(model)
        _emit("ok", f"{label}.shape_infer", "shape inference completed")
    except Exception as e:
        _emit("warn", f"{label}.shape_infer", f"shape inference: {str(e)[:200]}")

    for inp in model.graph.input:
        if inp.name != input_name:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) < 4:
            _emit("fail", f"{label}.dynamic", f"{input_name} has {len(dims)} dims, expected 4")
            return False
        shp = [(d.dim_param or (d.dim_value if d.HasField("dim_value") else "?")) for d in dims]
        if _dim_is_dynamic(dims[2]) and _dim_is_dynamic(dims[3]):
            _emit("ok", f"{label}.dynamic", f"{input_name} H/W are dynamic (shape={shp})")
            return True
        _emit("fail", f"{label}.dynamic",
              f"{input_name} H/W are STATIC (shape={shp}) — dynamic-spatial export did "
              f"not apply (was it exported with --static?)")
        return False

    _emit("fail", f"{label}.dynamic", f"no input named '{input_name}' found")
    return False


def _check_time_ids_input(model_dir: str) -> None:
    """Confirm the UNet graph declares a time_ids input with last dim 6 (1,6)."""
    try:
        import onnx
    except ImportError:
        return
    path = os.path.join(model_dir, "unet.onnx")
    if not os.path.exists(path):
        return
    model = onnx.load_model(path, load_external_data=False)
    for inp in model.graph.input:
        if inp.name != "time_ids":
            continue
        dims = inp.type.tensor_type.shape.dim
        last = dims[-1].dim_value if dims and dims[-1].HasField("dim_value") else None
        if last == 6 or last is None:
            _emit("ok", "unet.time_ids_input",
                  f"time_ids input present, last-dim={last or 'dynamic'} (expect 6)")
        else:
            _emit("fail", "unet.time_ids_input",
                  f"time_ids last-dim={last}, expected 6 (SDXL conditioning vector)")
        return
    _emit("fail", "unet.time_ids_input", "no 'time_ids' input on the UNet graph")


# ── feed builders ─────────────────────────────────────────────────────────────

def _unet_feed(sess, latent_hw: int, rng) -> dict:
    """Build a UNet feed at the given latent size from session metadata.

    time_ids is scaled to this size's pixel resolution ({H,W,0,0,H,W}) — the
    diffusers SDXL img2img convention (original_size == target_size == the
    resolution being denoised), which is what a per-pass hires swap will feed.
    """
    feed: dict = {}
    px = latent_hw * 8
    for i in sess.get_inputs():
        dt = _np_dtype(i.type)
        name = i.name
        if name == "latent":
            feed[name] = rng.standard_normal((1, 4, latent_hw, latent_hw)).astype(dt)
        elif name == "timestep":
            feed[name] = np.full((1,), 500.0, dtype=dt)
        elif name == "encoder_hidden_states":
            feed[name] = rng.standard_normal((1, SEQ_LEN, HIDDEN_DIM)).astype(dt)
        elif name == "text_embeds":
            feed[name] = rng.standard_normal((1, POOLED_DIM)).astype(dt)
        elif name == "time_ids":
            feed[name] = np.array([[px, px, 0, 0, px, px]], dtype=dt)
        else:
            shp = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
            feed[name] = np.zeros(shp, dtype=dt)
    return feed


# ── runtime checks ────────────────────────────────────────────────────────────

def _run_unet(model_dir: str, sizes: list[int]) -> None:
    path = os.path.join(model_dir, "unet.onnx")
    try:
        sess = _session(path)
    except Exception as e:
        _emit("fail", "unet.load", f"could not create session: {str(e)[:200]}")
        return
    # get_available_providers() lists CUDA even when its DLLs fail to load; the
    # session's ACTIVE providers are the truth. A fp16 UNet on the CPU EP is
    # ~1000x slow / crash-prone, so skip rather than hang.
    if not any(p in sess.get_providers() for p in _GPU_PROVIDERS):
        _emit("skip", "unet.run",
              "GPU EP did not actually load (see ORT error above) — skipping fp16 UNet "
              "forwards to avoid a CPU-EP hang; fix the venv's CUDA libs and re-run")
        return
    for hw in sizes:
        print(f"  >>> UNet forward at latent {hw}x{hw} ({hw*8}px)", flush=True)
        rng = np.random.default_rng(0)  # same inputs across sizes for reproducibility
        feed = _unet_feed(sess, hw, rng)
        try:
            with _VramSampler() as vram:
                out = sess.run(None, feed)[0]
        except Exception as e:
            if _is_oom(str(e)):
                _emit("warn", f"unet.run@{hw}",
                      f"CUDA OUT OF MEMORY at {hw*8}px — VRAM ceiling on this card, not a defect")
            else:
                _emit("fail", f"unet.run@{hw}", f"forward raised: {str(e)[:200]}")
            continue
        if not np.isfinite(out).all():
            _emit("fail", f"unet.run@{hw}", f"produced NaN/Inf (fp16 overflow at this size){vram.report()}")
        else:
            lo, hi = float(np.min(out)), float(np.max(out))
            _emit("ok", f"unet.run@{hw}",
                  f"finite, out shape={tuple(out.shape)}, time_ids scaled to {hw*8}px "
                  f"accepted, range [{lo:.2f}, {hi:.2f}]{vram.report()}")


def _run_vae(model_dir: str, sizes: list[int], filename: str = "vae_decoder.onnx",
             label: str = "vae") -> None:
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        return
    try:
        sess = _session(path)
    except Exception as e:
        _emit("fail", f"{label}.load", f"could not create session: {str(e)[:200]}")
        return
    inp = sess.get_inputs()[0]
    dt = _np_dtype(inp.type)
    # fp16 VAE on the CPU EP is slow/crashy; skip if the GPU EP didn't load. An
    # fp32 decoder (the hedge) is fine on CPU, so let it run either way.
    on_gpu = any(p in sess.get_providers() for p in _GPU_PROVIDERS)
    if dt == np.float16 and not on_gpu:
        _emit("skip", f"{label}.run",
              "GPU EP did not load — skipping fp16 VAE forwards (CPU EP too slow/crashy)")
        return
    for hw in sizes:
        print(f"  >>> {label} decode at latent {hw}x{hw} ({hw*8}px)", flush=True)
        rng = np.random.default_rng(1)
        latent = rng.standard_normal((1, 4, hw, hw)).astype(dt)
        try:
            with _VramSampler() as vram:
                out = sess.run(None, {inp.name: latent})[0]
        except Exception as e:
            if _is_oom(str(e)):
                _emit("warn", f"{label}.run@{hw}",
                      f"CUDA OUT OF MEMORY at {hw*8}px — VRAM ceiling for this decoder on this "
                      f"card, not a graph defect (use fp16 and/or tiled decode above this size)")
            else:
                _emit("fail", f"{label}.run@{hw}", f"decode raised: {str(e)[:200]}")
            continue
        if not np.isfinite(out).all():
            _emit("fail", f"{label}.run@{hw}", f"produced NaN/Inf — SDXL fp16 VAE instability{vram.report()}")
            continue
        std = float(np.std(out.astype(np.float32)))
        note = "" if std >= 1e-3 else " (near-constant — check for fp16 VAE collapse)"
        status = "ok" if std >= 1e-3 else "warn"
        _emit(status, f"{label}.run@{hw}",
              f"finite, image shape={tuple(out.shape)}, std={std:.4f}{note}{vram.report()}")


def _run_vae_encoder(model_dir: str, sizes: list[int]) -> None:
    """Forward the VAE ENCODER at hires IMAGE sizes (pixel-mode hires re-encode).

    Unlike the decoder (latent input), the encoder input is the UPSCALED RGB image
    [1,3,px,px] where px = latent_hw * 8, so a --sizes of 152,192 exercises 1216 and
    1536 px. This is a NEW memory profile — the encoder was static-1024 before the
    dynamic-encoder change — so the VRAM sampler matters here. Output is raw moments
    [1,8,px/8,px/8] (mean+logvar); finite + non-degenerate std confirms the fp16
    encoder is stable at that size.
    """
    path = os.path.join(model_dir, "vae_encoder.onnx")
    if not os.path.exists(path):
        _emit("warn", "vae_encoder.load", "vae_encoder.onnx not found — no pixel-mode hires")
        return
    try:
        sess = _session(path)
    except Exception as e:
        _emit("fail", "vae_encoder.load", f"could not create session: {str(e)[:200]}")
        return
    inp = sess.get_inputs()[0]
    dt = _np_dtype(inp.type)
    on_gpu = any(p in sess.get_providers() for p in _GPU_PROVIDERS)
    if dt == np.float16 and not on_gpu:
        _emit("skip", "vae_encoder.run",
              "GPU EP did not load — skipping fp16 encoder forwards (CPU EP too slow/crashy)")
        return
    for hw in sizes:
        px = hw * 8
        print(f"  >>> vae_encoder encode at image {px}x{px}", flush=True)
        rng = np.random.default_rng(2)
        image = rng.standard_normal((1, 3, px, px)).astype(dt)
        try:
            with _VramSampler() as vram:
                out = sess.run(None, {inp.name: image})[0]
        except Exception as e:
            if _is_oom(str(e)):
                _emit("warn", f"vae_encoder.run@{px}",
                      f"CUDA OUT OF MEMORY at {px}px — VRAM ceiling for the encoder on this "
                      f"card, not a graph defect")
            else:
                _emit("fail", f"vae_encoder.run@{px}", f"encode raised: {str(e)[:200]}")
            continue
        if not np.isfinite(out).all():
            _emit("fail", f"vae_encoder.run@{px}",
                  f"produced NaN/Inf — fp16 encoder instability at this size{vram.report()}")
            continue
        std = float(np.std(out.astype(np.float32)))
        note = "" if std >= 1e-3 else " (near-constant — check for fp16 encoder collapse)"
        status = "ok" if std >= 1e-3 else "warn"
        _emit(status, f"vae_encoder.run@{px}",
              f"finite, moments shape={tuple(out.shape)} (expect [1,8,{px//8},{px//8}]), "
              f"std={std:.4f}{note}{vram.report()}")


# ── torch reference (numeric fidelity) ──────────────────────────────────────────

def _compare_torch(model_dir: str, ckpt: str) -> None:
    """Compare the ONNX UNet + VAE decoder against the same checkpoint in PyTorch.

    Loads the checkpoint exactly as the exporter does (fp32 load, same tracing
    patches, disable_attention_upcasting, then cast to fp16) so the reference
    matches the exported graph's math, and feeds byte-identical inputs.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        from export_common import (
            VAEDecoderWrapper,
            disable_attention_upcasting,
            load_single_file_pipeline,
            prepare_libraries_for_export,
        )
        from sdxl_export_onnx_models import UNetWrapper
    except Exception as e:  # torch/diffusers/checkpoint infra problem — not a defect
        _emit("warn", "torch_ref", f"could not set up torch reference: {str(e)[:200]}")
        return
    if not os.path.exists(ckpt):
        _emit("warn", "torch_ref", f"checkpoint not found: {ckpt}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  >>> torch reference on {device} (loading checkpoint) ...", flush=True)
    try:
        prepare_libraries_for_export("sdxl")
        pipe = load_single_file_pipeline(
            StableDiffusionXLPipeline, ckpt, torch_dtype=torch.float32)
    except Exception as e:
        _emit("warn", "torch_ref", f"pipeline load failed: {str(e)[:200]}")
        return

    # UNet reference at native latent 128 with the SAME inputs the ONNX gets.
    try:
        disable_attention_upcasting(pipe.unet)
        unet = pipe.unet.to(torch.float16).to(device).eval()
        wrapper = UNetWrapper(unet).eval()

        sess = _session(os.path.join(model_dir, "unet.onnx"))
        feed = _unet_feed(sess, 128, np.random.default_rng(7))
        onnx_out = sess.run(None, feed)[0].astype(np.float32)

        with torch.no_grad():
            targs = [torch.from_numpy(feed[n]).to(device) for n in
                     ["latent", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]]
            ref = wrapper(*targs).float().cpu().numpy()

        max_abs, mean_abs, mean_rel = _rel_stats(onnx_out, ref)
        status = "fail" if mean_rel > _UNET_REL_MEAN_FAIL else "ok"
        _emit(status, "unet.numerics",
              f"ONNX vs torch @128: mean_rel={mean_rel:.4f} (gate {_UNET_REL_MEAN_FAIL}), "
              f"mean_abs={mean_abs:.4f}, max_abs={max_abs:.4f}")
    except Exception as e:
        _emit("warn", "unet.numerics", f"comparison failed: {str(e)[:200]}")
    finally:
        try:
            pipe.unet = None
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # VAE reference at 128 — fp32 variant first (vae still fp32), then fp16.
    def _vae_compare(filename: str, cast_fp16: bool, tag: str):
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            return
        try:
            vae = pipe.vae
            if cast_fp16:
                vae = vae.to(torch.float16)
            vae = vae.to(device).eval()
            dec = VAEDecoderWrapper(vae).eval()

            sess = _session(path)
            inp = sess.get_inputs()[0]
            dt = _np_dtype(inp.type)
            latent = np.random.default_rng(11).standard_normal((1, 4, 128, 128)).astype(dt)
            onnx_out = sess.run(None, {inp.name: latent})[0].astype(np.float32)
            with torch.no_grad():
                ref = dec(torch.from_numpy(latent).to(device)).float().cpu().numpy()
            max_abs, mean_abs, mean_rel = _rel_stats(onnx_out, ref)
            status = "fail" if mean_rel > _VAE_REL_MEAN_FAIL else "ok"
            _emit(status, f"{tag}.numerics",
                  f"ONNX vs torch @128: mean_rel={mean_rel:.4f} (gate {_VAE_REL_MEAN_FAIL}), "
                  f"mean_abs={mean_abs:.4f}, max_abs={max_abs:.4f}")
        except Exception as e:
            _emit("warn", f"{tag}.numerics", f"comparison failed: {str(e)[:200]}")

    _vae_compare("vae_decoder_fp32.onnx", cast_fp16=False, tag="vae_fp32")
    _vae_compare("vae_decoder.onnx", cast_fp16=True, tag="vae_fp16")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a dynamic-spatial SDXL export")
    ap.add_argument("--model", required=True, help="dynamic-spatial export dir")
    ap.add_argument("--torch-reference", dest="torch_reference",
                    help="path to the SAME .safetensors checkpoint, for a PyTorch "
                         "numeric-fidelity comparison of the ONNX UNet + VAE decoder")
    ap.add_argument("--sizes", default="128,160,192",
                    help="comma-separated latent sizes to exercise (default 128,160,192)")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("onnxruntime not installed — cannot run the forward-pass checks", file=sys.stderr)
        return 2

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    gpu = _gpu_providers()
    heavy = bool(gpu)
    print(f"Providers available: {gpu or ['CPUExecutionProvider']}")
    print(f"Runtime fp16 forwards + numerics: {'ENABLED' if heavy else 'SKIPPED (no GPU ONNX Runtime)'}")
    print(f"Validating dynamic-spatial SDXL export: {args.model}")
    print(f"Latent sizes: {sizes}  (px: {[s*8 for s in sizes]})\n")

    # Structural checks always run — cheap, CPU-safe, and the primary go/no-go.
    unet_ok = _check_structure(os.path.join(args.model, "unet.onnx"), "unet")
    vae_ok = _check_structure(os.path.join(args.model, "vae_decoder.onnx"), "vae_decoder")
    # Pixel-mode hires: the encoder's "image" input must be dynamic H/W too. Absent
    # or static → not a hard fail (latent-mode hires still works), but reported so
    # pixel_hires_capable is confirmed. A present-but-static encoder is the
    # pre-dynamic-encoder SDXL case this change fixes.
    enc_path = os.path.join(args.model, "vae_encoder.onnx")
    enc_ok = os.path.exists(enc_path) and _check_structure(enc_path, "vae_encoder", input_name="image")
    _check_time_ids_input(args.model)
    has_fp32 = os.path.exists(os.path.join(args.model, "vae_decoder_fp32.onnx"))
    if has_fp32:
        _check_structure(os.path.join(args.model, "vae_decoder_fp32.onnx"), "vae_decoder_fp32")

    # fp16 UNet/VAE forwards on the CPU EP are ~1000x slow and can crash — only
    # exercise them where a GPU ORT provider (CUDA/DML/ROCm) is present.
    if not heavy:
        _emit("skip", "runtime",
              "fp16 forwards + torch numerics skipped — install onnxruntime-gpu "
              "into the venv to run them (see docs/50_export/export_overview.md)")
    else:
        if unet_ok:
            _run_unet(args.model, sizes)
        if vae_ok:
            _run_vae(args.model, sizes, "vae_decoder.onnx", "vae")
        if has_fp32:
            _run_vae(args.model, sizes, "vae_decoder_fp32.onnx", "vae_fp32")
        if enc_ok:
            _run_vae_encoder(args.model, sizes)
        if args.torch_reference:
            _compare_torch(args.model, args.torch_reference)

    print()
    if _hard_failures:
        print(f"RESULT: FAIL ({len(_hard_failures)} hard check(s): {', '.join(_hard_failures)})")
        return 1
    if heavy:
        print("RESULT: PASS — structurally sound AND fp16 forwards/numerics verified")
    else:
        print("RESULT: PASS (structural only) — graphs are dynamic-H/W and load; fp16 "
              "runtime + numerics NOT checked (no GPU ONNX Runtime). Re-run with "
              "onnxruntime-gpu to verify fp16 finiteness at >native and VRAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

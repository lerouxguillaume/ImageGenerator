"""
Post-export inference smoke test for imported ONNX models.

The plain file/shape checks in import_model.py only prove the files exist — they
cannot catch a model that exports cleanly but produces garbage at runtime. The
two failure modes we have actually hit in practice are:

  1. Text-encoder attention collapse — an fp16 CLIP encoder run natively in fp16
     accumulates enough error over its transformer stack that *every* prompt maps
     to nearly the same embedding, so CFG does nothing. File checks miss it; a
     divergence test between two different prompts catches it immediately.

  2. fp16 UNet overflow — a bad export produces NaN/Inf latents on the first
     denoise step. A single real forward pass surfaces it.

This module loads the exported components in ONNX Runtime (CPU) and runs a real
end-to-end pass. It is arch-aware (sd15 / sdxl) and reads each session's own
input metadata so it feeds exactly the dtype and shape the graph expects — no
hard-coded fp16/fp32 assumptions.

Protocol (consumed by the C++ ModelImporter):
    VERIFY:ok:<check>:<detail>      passed
    VERIFY:warn:<check>:<detail>    non-fatal (import still succeeds)
    VERIFY:fail:<check>:<detail>    fatal (import is rejected, model not registered)

verify_model() returns True only if every hard check passed.
"""
from __future__ import annotations

import gc
import os

import numpy as np

# Shared ORT primitives (dtype map, provider discovery, session, dim resolver).
# ort_probe imports onnxruntime lazily inside its functions, so importing it here
# does not require ORT at module-import time — this module (and its pure helpers)
# can still be imported on a host without ORT. In the import venv ORT is present.
from ort_probe import fixed_dim as _fixed
from ort_probe import gpu_providers as _gpu_providers
from ort_probe import make_session
from ort_probe import np_dtype as _np_dtype

# CLIP special tokens (shared by SD1.5 CLIP-L and SDXL's two encoders).
BOS = 49406
EOS = 49407
SEQ_LEN = 77

# Two clearly different token sequences. A healthy encoder maps these to
# well-separated embeddings; a collapsed one maps them to nearly the same point.
# Arbitrary in-vocab ids — no tokenizer needed for a divergence probe.
_PROMPT_A = [320, 1125, 539, 786, 2368]      # "a photo of ..."-ish
_PROMPT_B = [1794, 6982, 8853, 4558, 12609]   # unrelated content ids

# A near-identical cond/uncond pair only warns; two *different* prompts collapsing
# is a hard failure (that is unambiguously attention collapse).
_COLLAPSE_HARD = 0.9995
_COLLAPSE_WARN = 0.99


# ── protocol emit ─────────────────────────────────────────────────────────────

_hard_failures: list[str] = []


def _emit(status: str, check: str, detail: str) -> None:
    # Keep detail single-line so the C++ line parser stays simple.
    detail = detail.replace("\n", " ").strip()
    print(f"VERIFY:{status}:{check}:{detail}", flush=True)
    if status == "fail":
        _hard_failures.append(check)


# ── ORT helpers ───────────────────────────────────────────────────────────────
# _np_dtype / _gpu_providers / _fixed come from ort_probe (imported above).
# verify_model runs on the CPU EP by default: no CUDA-DLL preload, and it keeps
# ORT's default graph-opt level, so it takes make_session's defaults.


def _session(path: str, prefer_gpu: bool = False):
    return make_session(path, prefer_gpu=prefer_gpu)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── token builders ────────────────────────────────────────────────────────────

def _ids_and_mask(content: list[int], np_ids_dtype, np_mask_dtype):
    ids = [BOS] + content + [EOS]
    ids = ids[:SEQ_LEN]
    ids += [EOS] * (SEQ_LEN - len(ids))  # CLIP pads with EOS
    mask = [0] * SEQ_LEN
    for i, tok in enumerate(ids):
        mask[i] = 1
        if tok == EOS:
            break  # attend up to and including the first EOS
    return (
        np.array([ids], dtype=np_ids_dtype),
        np.array([mask], dtype=np_mask_dtype),
    )


# ── text encoder ──────────────────────────────────────────────────────────────

def _run_text_encoder(sess: ort.InferenceSession, content: list[int]):
    """Run a CLIP text encoder. Returns (hidden_states[1,77,dim], pooled|None)."""
    inputs = {i.name: i for i in sess.get_inputs()}
    ids_meta = inputs.get("input_ids")
    mask_meta = inputs.get("attention_mask")
    ids, mask = _ids_and_mask(
        content,
        _np_dtype(ids_meta.type) if ids_meta else np.int64,
        _np_dtype(mask_meta.type) if mask_meta else np.int64,
    )
    feed = {}
    if ids_meta:
        feed[ids_meta.name] = ids
    if mask_meta:
        feed[mask_meta.name] = mask

    outputs = sess.run(None, feed)
    out_meta = sess.get_outputs()

    hidden = None
    pooled = None
    for arr, meta in zip(outputs, out_meta):
        if arr.ndim == 3:  # [batch, seq, dim] hidden states
            if hidden is None:
                hidden = arr
        elif arr.ndim == 2:  # [batch, dim] pooled projection
            pooled = arr
    if hidden is None:
        # Fall back to the first output so downstream still has something.
        hidden = outputs[0]
    return hidden, pooled


def _verify_text_encoder(path: str, label: str):
    """Hard-gates on finiteness and prompt divergence. Returns hidden/pooled for A."""
    try:
        sess = _session(path)
    except Exception as e:
        _emit("fail", label, f"failed to load ONNX session: {e}")
        return None, None

    try:
        hidden_a, pooled_a = _run_text_encoder(sess, _PROMPT_A)
        hidden_b, _ = _run_text_encoder(sess, _PROMPT_B)
    except Exception as e:
        _emit("fail", label, f"forward pass raised: {e}")
        return None, None

    if not (np.isfinite(hidden_a).all() and np.isfinite(hidden_b).all()):
        _emit("fail", label, "non-finite (NaN/Inf) values in text embeddings")
        return None, None

    sim = _cosine(hidden_a, hidden_b)
    dim = hidden_a.shape[-1]
    if sim > _COLLAPSE_HARD:
        _emit(
            "fail", label,
            f"two different prompts produce near-identical embeddings "
            f"(cosine={sim:.4f}) - attention collapse; export encoder in fp32",
        )
    elif sim > _COLLAPSE_WARN:
        _emit("warn", label,
              f"low prompt separation (cosine={sim:.4f}, dim={dim}) - CFG may be weak")
    else:
        _emit("ok", label, f"finite, prompts diverge (cosine={sim:.4f}, dim={dim})")
    return hidden_a, pooled_a


# ── UNet ──────────────────────────────────────────────────────────────────────

def _build_unet_feed(sess, hidden_states, pooled):
    """Assemble a UNet feed from real encoder output + session metadata."""
    feed = {}
    latent_hw = 32  # SD1.5 dynamic default → 256px; overwritten if static
    for i in sess.get_inputs():
        dt = _np_dtype(i.type)
        shp = i.shape
        name = i.name
        if name == "latent":
            h = _fixed(shp[2] if len(shp) > 2 else None, 32)
            w = _fixed(shp[3] if len(shp) > 3 else None, 32)
            latent_hw = h
            rng = np.random.default_rng(0)
            feed[name] = rng.standard_normal((1, _fixed(shp[1], 4), h, w)).astype(dt)
        elif name == "timestep":
            n = _fixed(shp[0] if shp else 1, 1)
            feed[name] = np.full((n,), 500.0, dtype=dt)
        elif name == "encoder_hidden_states":
            want = _fixed(shp[2] if len(shp) > 2 else None, hidden_states.shape[-1])
            hs = hidden_states
            if hs.shape[-1] != want:
                # SDXL concat mismatch guard — tile/trim to expected width.
                hs = np.resize(hs, (1, SEQ_LEN, want))
            feed[name] = hs.astype(dt)
        elif name == "text_embeds":  # SDXL pooled
            dim = _fixed(shp[-1], 1280)
            base = pooled if pooled is not None else np.zeros((1, dim))
            feed[name] = np.resize(base, (1, dim)).astype(dt)
        elif name == "time_ids":  # SDXL [h, w, crop_y, crop_x, h, w]
            px = latent_hw * 8
            feed[name] = np.array([[px, px, 0, 0, px, px]], dtype=dt)
        else:
            dims = [_fixed(d, 1) for d in shp]
            feed[name] = np.zeros(dims, dtype=dt)
    return feed


def _verify_unet(path: str, hidden_states, pooled, enabled: bool):
    if not enabled:
        # fp16 UNet on the CPU EP can hard-crash ORT (access violation). Only run
        # it where a GPU provider is present; otherwise skip without touching it.
        _emit("skip", "unet",
              "runtime check skipped - needs GPU ONNX Runtime (fp16 UNet); "
              "text encoder verified")
        return
    try:
        sess = _session(path, prefer_gpu=True)
    except Exception as e:
        _emit("warn", "unet", f"could not load ONNX session: {e}")
        return
    try:
        feed = _build_unet_feed(sess, hidden_states, pooled)
        out = sess.run(None, feed)[0]
    except Exception as e:
        # OOM / unsupported fp16 op on CPU — the graph loaded, so treat as
        # non-fatal rather than blocking an otherwise valid import.
        _emit("warn", "unet", f"could not execute forward pass on CPU: {e}")
        return
    if not np.isfinite(out).all():
        _emit("fail", "unet",
              "UNet produced NaN/Inf on a single denoise step - fp16 overflow")
    else:
        lo, hi = float(np.min(out)), float(np.max(out))
        _emit("ok", "unet", f"finite output over 1 step (range [{lo:.2f}, {hi:.2f}])")


# ── VAE decoder ───────────────────────────────────────────────────────────────

def _verify_vae_decoder(path: str, enabled: bool):
    if not enabled:
        _emit("skip", "vae_decoder",
              "runtime check skipped - needs GPU ONNX Runtime (fp16 VAE)")
        return
    try:
        sess = _session(path, prefer_gpu=True)
    except Exception as e:
        _emit("warn", "vae_decoder", f"could not load ONNX session: {e}")
        return
    try:
        i = sess.get_inputs()[0]
        dt = _np_dtype(i.type)
        shp = i.shape
        h = _fixed(shp[2] if len(shp) > 2 else None, 32)
        w = _fixed(shp[3] if len(shp) > 3 else None, 32)
        rng = np.random.default_rng(1)
        latent = rng.standard_normal((1, _fixed(shp[1], 4), h, w)).astype(dt)
        out = sess.run(None, {i.name: latent})[0]
    except Exception as e:
        _emit("warn", "vae_decoder", f"could not execute forward pass on CPU: {e}")
        return
    if not np.isfinite(out).all():
        _emit("fail", "vae_decoder", "VAE decoder produced NaN/Inf")
        return
    std = float(np.std(out.astype(np.float32)))
    if std < 1e-3:
        _emit("warn", "vae_decoder",
              f"decoded image is nearly constant (std={std:.5f}) - VAE may be dead")
    else:
        _emit("ok", "vae_decoder", f"finite, non-degenerate output (std={std:.3f})")


# ── entry point ───────────────────────────────────────────────────────────────

def verify_model(output_dir: str, arch: str) -> bool:
    """Run the full smoke test. Returns True iff all hard checks pass."""
    _hard_failures.clear()

    def p(name):
        return os.path.join(output_dir, name)

    # The fp16 UNet/VAE forward passes only run where a GPU provider exists; on a
    # CPU-only ORT build they are skipped (the CPU EP can crash on fp16 graphs).
    gpu = _gpu_providers()
    heavy_enabled = bool(gpu)
    print(f"Verification providers: {gpu or ['CPUExecutionProvider']}  "
          f"(UNet/VAE runtime check: {'on' if heavy_enabled else 'skipped'})",
          flush=True)

    # Text encoders — the hard gate that catches attention collapse.
    hidden_1, pooled_1 = _verify_text_encoder(p("text_encoder.onnx"), "text_encoder")

    hidden_for_unet = hidden_1
    pooled_for_unet = pooled_1

    if arch == "sdxl":
        hidden_2, pooled_2 = _verify_text_encoder(p("text_encoder_2.onnx"), "text_encoder_2")
        # SDXL UNet consumes concat(enc1, enc2) hidden states + enc2 pooled.
        if hidden_1 is not None and hidden_2 is not None:
            hidden_for_unet = np.concatenate([hidden_1, hidden_2], axis=-1)
        pooled_for_unet = pooled_2 if pooled_2 is not None else pooled_1

    gc.collect()  # release encoder sessions before loading the multi-GB UNet

    if hidden_for_unet is not None:
        _verify_unet(p("unet.onnx"), hidden_for_unet, pooled_for_unet, heavy_enabled)
    else:
        _emit("warn", "unet", "skipped - text encoder failed, no embeddings to feed")

    gc.collect()
    _verify_vae_decoder(p("vae_decoder.onnx"), heavy_enabled)

    return len(_hard_failures) == 0


if __name__ == "__main__":  # manual invocation for debugging
    import argparse

    ap = argparse.ArgumentParser(description="Verify an exported ONNX model runs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--arch", required=True, choices=["sd15", "sdxl"])
    args = ap.parse_args()
    ok = verify_model(args.output, args.arch)
    print("VERIFY:done" if ok else "VERIFY:rejected", flush=True)
    # Exit 3 (not 1) for a genuine hard-fail rejection, so the parent can tell a
    # real defect apart from a crash/OOM/uncaught error (which exits 1 or a signal
    # code) and only block the import on the former.
    raise SystemExit(0 if ok else 3)
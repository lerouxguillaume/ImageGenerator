"""
Export Stable Diffusion XL models to ONNX for use with ONNX Runtime.

Usage:
    python sdxl_export_onnx_models.py <model.safetensors> [--name MODEL_NAME]

Outputs (under models/<MODEL_NAME>/):
    text_encoder.onnx   — CLIP-L penultimate hidden states, fp32
    text_encoder_2.onnx — OpenCLIP-G hidden states + pooled embeds, fp32
    unet.onnx           — UNet, fp16, dynamic H/W
    vae_decoder.onnx    — VAE decoder, fp16, dynamic H/W
    model.json          — {"type": "sdxl"} for C++ runtime detection

Notes on dtype strategy:
    Text encoders are exported as fp32 for accuracy.
    UNet and VAE are loaded as fp32 then cast to fp16 before export to avoid
    creating a 10 GB fp32 ONNX intermediate.  fix_fp32_constants() is called
    afterwards to cast the few remaining scalar constants the tracer emits as
    fp32 (e.g. time-embedding scalars, GroupNorm eps) — these cause Concat
    type errors in ORT if left as fp32 in an otherwise fp16 graph.
"""
import argparse
import os
import sys
import time

import torch
from diffusers import StableDiffusionXLPipeline

from export_common import (
    VAEDecoderWrapper,
    check_dependencies,
    check_model_file,
    export_step,
    fix_fp32_constants,
    free,
    onnx_export,
    patch_clip_for_tracing,
    simplify_with_onnxsim,
    write_model_json,
)

SEQ_LEN = 77


# ── Model wrappers ────────────────────────────────────────────────────────────

class CLIPL_Wrapper(torch.nn.Module):
    """CLIP-L: returns penultimate hidden state [batch, 77, 768]."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        return out.hidden_states[-2]


class OpenCLIPG_Wrapper(torch.nn.Module):
    """OpenCLIP-G: returns penultimate hidden state [batch, 77, 1280]
    and pooled text embeds [batch, 1280]."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


class UNetWrapper(torch.nn.Module):
    """SDXL UNet with 5 inputs: latent, timestep, encoder_hidden_states,
    text_embeds (pooled), time_ids."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        return self.unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample


# ── Export pipeline ───────────────────────────────────────────────────────────

def export_sdxl(model_file: str, output_dir: str) -> None:
    check_dependencies(
        required=["torch", "diffusers", "transformers", "onnx"],
        optional=["onnxsim"],
    )
    check_model_file(model_file)
    os.makedirs(output_dir, exist_ok=True)
    patch_clip_for_tracing()

    t_total = time.time()
    print("Loading SDXL pipeline (fp32) ...")
    # Load as fp32 — UNet and VAE are cast to fp16 individually before export
    # to avoid a 10 GB fp32 ONNX intermediate.
    pipe = StableDiffusionXLPipeline.from_single_file(model_file, torch_dtype=torch.float32)
    pipe.enable_attention_slicing()

    hidden_dim      = pipe.unet.config.cross_attention_dim  # 2048
    pooled_dim      = 1280
    latent_h, latent_w = 128, 128  # SDXL latent for 1024 px

    # 1. CLIP-L text encoder ──────────────────────────────────────────────────
    with export_step("1/4  Text encoder (CLIP-L)"):
        clip_l = CLIPL_Wrapper(pipe.text_encoder).cpu().eval()
        dummy_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
        onnx_export(
            clip_l, (dummy_ids,),
            os.path.join(output_dir, "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["hidden_states"],
            dynamic_axes={"input_ids": {0: "batch"}, "hidden_states": {0: "batch"}},
        )
        free(clip_l, pipe.text_encoder, pipe.tokenizer)
        pipe.text_encoder = None

    # 2. OpenCLIP-G text encoder ──────────────────────────────────────────────
    with export_step("2/4  Text encoder 2 (OpenCLIP-G)"):
        clip_g = OpenCLIPG_Wrapper(pipe.text_encoder_2).cpu().eval()
        dummy_ids_2 = torch.randint(0, pipe.tokenizer_2.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
        onnx_export(
            clip_g, (dummy_ids_2,),
            os.path.join(output_dir, "text_encoder_2.onnx"),
            input_names=["input_ids"],
            output_names=["hidden_states", "text_embeds"],
            dynamic_axes={
                "input_ids":     {0: "batch"},
                "hidden_states": {0: "batch"},
                "text_embeds":   {0: "batch"},
            },
        )
        free(clip_g, pipe.text_encoder_2, pipe.tokenizer_2)
        pipe.text_encoder_2 = None

    # 3. UNet ─────────────────────────────────────────────────────────────────
    with export_step("3/4  UNet"):
        pipe.unet.to(torch.float16)
        unet = UNetWrapper(pipe.unet).cpu().eval()
        dummy_latent   = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16)
        dummy_timestep = torch.tensor([1], dtype=torch.float16)
        dummy_embeds   = torch.randn(1, SEQ_LEN, hidden_dim, dtype=torch.float16)
        dummy_pooled   = torch.zeros(1, pooled_dim, dtype=torch.float16)
        dummy_time_ids = torch.zeros(1, 6, dtype=torch.float16)
        unet_path = os.path.join(output_dir, "unet.onnx")
        onnx_export(
            unet, (dummy_latent, dummy_timestep, dummy_embeds, dummy_pooled, dummy_time_ids),
            unet_path,
            input_names=["latent", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            output_names=["latent_out"],
            dynamic_axes={
                "latent":                {0: "batch", 2: "height", 3: "width"},
                "timestep":              {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "text_embeds":           {0: "batch"},
                "time_ids":              {0: "batch"},
                "latent_out":            {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=False,  # must be False for fp16 UNet — see module docstring
        )
        fix_fp32_constants(unet_path)
        free(unet, pipe.unet)
        pipe.unet = None

    # 4. VAE decoder ──────────────────────────────────────────────────────────
    with export_step("4/4  VAE decoder"):
        pipe.vae.to(torch.float16)
        vae = VAEDecoderWrapper(pipe.vae).cpu().eval()
        dummy_latent_vae = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16)
        vae_path = os.path.join(output_dir, "vae_decoder.onnx")
        onnx_export(
            vae, (dummy_latent_vae,), vae_path,
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={
                "latent": {0: "batch", 2: "height", 3: "width"},
                "image":  {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=False,
        )
        fix_fp32_constants(vae_path)
        simplify_with_onnxsim(vae_path)
        free(vae, pipe.vae, pipe)

    write_model_json(output_dir, "sdxl")
    print(f"\n✅ All models exported to {output_dir}  "
          f"(total: {time.time() - t_total:.0f}s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SDXL to ONNX")
    parser.add_argument("model_file", help="Path to the .safetensors checkpoint")
    parser.add_argument("--name", help="Output directory name (default: checkpoint stem)")
    parser.add_argument("--output-dir", help="Override output directory path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.model_file))[0]
    out  = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", name)
    try:
        export_sdxl(args.model_file, out)
    except (FileNotFoundError, ImportError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        sys.exit(1)
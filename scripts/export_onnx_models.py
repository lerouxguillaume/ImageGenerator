"""
Export Stable Diffusion 1.5 models to ONNX for use with ONNX Runtime.

Usage:
    python export_onnx_models.py <model.safetensors> [--name MODEL_NAME]

Outputs (under models/<MODEL_NAME>/):
    text_encoder.onnx   — CLIP-L with clip-skip-2, fp16
    unet.onnx           — UNet, fp16, dynamic H/W
    vae_decoder.onnx    — VAE decoder, fp16, dynamic H/W
"""
import argparse
import os
import sys
import time

import torch
from diffusers import StableDiffusionPipeline

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


# ── Model wrappers ────────────────────────────────────────────────────────────

class CLIPTextEncoderClipSkip2(torch.nn.Module):
    """CLIP-L text encoder with clip-skip-2 (penultimate hidden state).
    Matches the NovelAI / A1111 style: returns the second-to-last hidden state
    passed through the final layer norm."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-2]
        return self.text_encoder.text_model.final_layer_norm(hidden)


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, encoder_hidden_states):
        return self.unet(latent, timestep,
                         encoder_hidden_states=encoder_hidden_states).sample


# ── Export pipeline ───────────────────────────────────────────────────────────

def export_sd15(model_file: str, output_dir: str) -> None:
    check_dependencies(
        required=["torch", "diffusers", "transformers", "onnx"],
        optional=["onnxsim"],
    )
    check_model_file(model_file)
    os.makedirs(output_dir, exist_ok=True)
    patch_clip_for_tracing()

    t_total = time.time()
    print("Loading SD 1.5 pipeline ...")
    pipe = StableDiffusionPipeline.from_single_file(model_file, torch_dtype=torch.float16)
    pipe.enable_attention_slicing()

    # 1. Text encoder ─────────────────────────────────────────────────────────
    with export_step("1/3  Text encoder"):
        clip = CLIPTextEncoderClipSkip2(pipe.text_encoder).eval()
        dummy_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, 77), dtype=torch.int64)
        onnx_export(
            clip, dummy_ids,
            os.path.join(output_dir, "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["text_embeds"],
            dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        )
        free(clip, pipe.text_encoder, pipe.tokenizer)
        pipe.text_encoder = None

    # 2. UNet ─────────────────────────────────────────────────────────────────
    with export_step("2/3  UNet"):
        pipe.unet.to(torch.float16)
        unet = UNetWrapper(pipe.unet).eval()
        dummy_latent   = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        dummy_timestep = torch.tensor([1], dtype=torch.float16)
        dummy_embeds   = torch.randn(1, 77, 768, dtype=torch.float16)
        onnx_export(
            unet, (dummy_latent, dummy_timestep, dummy_embeds),
            os.path.join(output_dir, "unet.onnx"),
            input_names=["latent", "timestep", "encoder_hidden_states"],
            output_names=["latent_out"],
            dynamic_axes={
                "latent":                {0: "batch", 2: "height", 3: "width"},
                "timestep":              {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "latent_out":            {0: "batch", 2: "height", 3: "width"},
            },
        )
        free(unet, pipe.unet)
        pipe.unet = None

    # 3. VAE decoder ──────────────────────────────────────────────────────────
    with export_step("3/3  VAE decoder"):
        pipe.vae.to(torch.float16)
        vae = VAEDecoderWrapper(pipe.vae).eval()
        dummy_latent_vae = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        vae_path = os.path.join(output_dir, "vae_decoder.onnx")
        onnx_export(
            vae, (dummy_latent_vae,), vae_path,
            input_names=["latent"],
            output_names=["image"],
            # Dynamic H/W: VAE loads with CPU session opts (never DML), so the
            # static-shape DML restriction does not apply.
            dynamic_axes={
                "latent": {0: "batch", 2: "height", 3: "width"},
                "image":  {0: "batch", 2: "height", 3: "width"},
            },
        )
        simplify_with_onnxsim(vae_path)
        free(vae, pipe.vae, pipe)

    write_model_json(output_dir, "sd15")
    print(f"\n✅ All models exported to {output_dir}  "
          f"(total: {time.time() - t_total:.0f}s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SD 1.5 to ONNX")
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
        export_sd15(args.model_file, out)
    except (FileNotFoundError, ImportError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        sys.exit(1)
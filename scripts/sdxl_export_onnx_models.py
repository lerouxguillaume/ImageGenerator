"""
Export Stable Diffusion XL models to ONNX for use with ONNX Runtime.

Usage:
    python sdxl_export_onnx_models.py <model.safetensors> [--name MODEL_NAME]

Outputs (under models/<MODEL_NAME>/):
    text_encoder.onnx   — CLIP-L penultimate hidden states, fp32
    text_encoder_2.onnx — OpenCLIP-G hidden states + pooled embeds, fp32
    unet.onnx           — UNet, fp16, batch-dynamic only
    vae_decoder.onnx    — VAE decoder, fp16, static shape
    model.json          — {"type": "sdxl"} for C++ runtime detection

Notes on dtype strategy:
    Text encoders are exported as fp32 for accuracy.
    UNet and VAE are loaded as fp32 then cast to fp16 before export to avoid
    creating a 10 GB fp32 ONNX intermediate.  fix_fp32_constants() is called
    afterwards to cast the few remaining scalar constants the tracer emits as
    fp32 (e.g. time-embedding scalars, GroupNorm eps) — these cause Concat
    type errors in ORT if left as fp32 in an otherwise fp16 graph.

Performance note:
    The default path now prioritizes export speed on machines with enough RAM
    or swap.  The previous low-memory attention slicing behavior is available
    via --optimize-memory. VAE simplification is opt-in via --simplify-vae.
"""
import argparse
import os
import sys
import time

import torch
from diffusers import StableDiffusionXLPipeline

from export_common import (
    ExportComponentSpec,
    SDXLExportPolicy,
    VAEDecoderWrapper,
    assert_no_meta_tensors,
    check_dependencies,
    check_model_file,
    disable_attention_upcasting,
    export_component_to_dir,
    load_single_file_pipeline,
    patch_clip_for_tracing,
    patch_clip_text_model_compat,
    patch_fp32_upcasts_for_tracing,
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

def export_sdxl(model_file: str, output_dir: str, *, optimize_memory: bool = False,
                simplify_vae: bool = False, resume: bool = False,
                validate: bool = False) -> None:
    policy = SDXLExportPolicy()
    check_dependencies(
        required=["torch", "diffusers", "transformers", "onnx"],
        optional=["onnxsim"],
    )
    check_model_file(model_file)
    os.makedirs(output_dir, exist_ok=True)
    patch_clip_text_model_compat()
    patch_clip_for_tracing()

    t_total = time.time()
    patch_fp32_upcasts_for_tracing()
    print("Loading SDXL pipeline (fp32) ...")
    # Load as fp32 — UNet and VAE are cast to fp16 individually before export
    # to avoid a 10 GB fp32 ONNX intermediate.
    pipe = load_single_file_pipeline(
        StableDiffusionXLPipeline,
        model_file,
        torch_dtype=torch.float32,
    )
    assert_no_meta_tensors(pipe.text_encoder, "text_encoder")
    assert_no_meta_tensors(pipe.text_encoder_2, "text_encoder_2")
    assert_no_meta_tensors(pipe.unet, "unet")
    assert_no_meta_tensors(pipe.vae, "vae")
    vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    if optimize_memory:
        print("  Memory-optimized mode enabled: attention slicing ON")
        pipe.enable_attention_slicing()
    else:
        print("  Fast export mode: attention slicing OFF")

    hidden_dim      = pipe.unet.config.cross_attention_dim  # 2048
    pooled_dim      = 1280
    latent_h, latent_w = 128, 128  # SDXL latent for 1024 px

    all_specs = []

    # 1. CLIP-L text encoder ──────────────────────────────────────────────────
    clip_l = CLIPL_Wrapper(pipe.text_encoder).eval()
    dummy_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
    te1_spec = ExportComponentSpec(
        step_name="1/4  Text encoder (CLIP-L)",
        component_name="text_encoder",
        filename="text_encoder.onnx",
        model=clip_l,
        dummy_inputs=(dummy_ids,),
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes=policy.text_encoder_dynamic_axes("input_ids", ["hidden_states"]),
        export_lora_weights=True,
        skip_if_complete=resume,
        validate=validate,
        release_after=(clip_l, pipe.text_encoder, pipe.tokenizer),
    )
    all_specs.append(te1_spec)
    export_component_to_dir(output_dir, te1_spec)
    pipe.text_encoder = None

    # 2. OpenCLIP-G text encoder ──────────────────────────────────────────────
    clip_g = OpenCLIPG_Wrapper(pipe.text_encoder_2).eval()
    dummy_ids_2 = torch.randint(0, pipe.tokenizer_2.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
    te2_spec = ExportComponentSpec(
        step_name="2/4  Text encoder 2 (OpenCLIP-G)",
        component_name="text_encoder_2",
        filename="text_encoder_2.onnx",
        model=clip_g,
        dummy_inputs=(dummy_ids_2,),
        input_names=["input_ids"],
        output_names=["hidden_states", "text_embeds"],
        dynamic_axes=policy.text_encoder_dynamic_axes(
            "input_ids",
            ["hidden_states", "text_embeds"],
        ),
        export_lora_weights=True,
        skip_if_complete=resume,
        validate=validate,
        release_after=(clip_g, pipe.text_encoder_2, pipe.tokenizer_2),
    )
    all_specs.append(te2_spec)
    export_component_to_dir(output_dir, te2_spec)
    pipe.text_encoder_2 = None

    # 3. UNet ─────────────────────────────────────────────────────────────────
    disable_attention_upcasting(pipe.unet)
    pipe.unet.to(torch.float16)
    unet = UNetWrapper(pipe.unet).eval()
    dummy_latent = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16)
    dummy_timestep = torch.tensor([1], dtype=torch.float16)
    dummy_embeds = torch.randn(1, SEQ_LEN, hidden_dim, dtype=torch.float16)
    dummy_pooled = torch.zeros(1, pooled_dim, dtype=torch.float16)
    dummy_time_ids = torch.zeros(1, 6, dtype=torch.float16)
    unet_spec = ExportComponentSpec(
        step_name="3/4  UNet",
        component_name="unet",
        filename="unet.onnx",
        model=unet,
        dummy_inputs=(dummy_latent, dummy_timestep, dummy_embeds, dummy_pooled, dummy_time_ids),
        input_names=["latent", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["latent_out"],
        dynamic_axes=policy.unet_dynamic_axes(),
        exporter=policy.unet_exporter(),
        fix_fp32_constants=policy.should_fix_fp32_constants("unet"),
        fix_attention_sqrt_cast=policy.should_fix_attention_sqrt_cast("unet"),
        fix_resize_fp16=policy.should_fix_resize_fp16("unet"),
        export_lora_weights=True,
        skip_if_complete=resume,
        validate=validate,
        release_after=(unet, pipe.unet),
    )
    all_specs.append(unet_spec)
    export_component_to_dir(output_dir, unet_spec)
    pipe.unet = None

    # 4. VAE decoder ──────────────────────────────────────────────────────────
    pipe.vae.to(torch.float16)
    vae = VAEDecoderWrapper(pipe.vae).eval()
    dummy_latent_vae = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16)
    vae_spec = ExportComponentSpec(
        step_name="4/4  VAE decoder",
        component_name="vae_decoder",
        filename="vae_decoder.onnx",
        model=vae,
        dummy_inputs=(dummy_latent_vae,),
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes=policy.vae_dynamic_axes(),
        exporter=policy.vae_exporter(),
        fix_fp32_constants=policy.should_fix_fp32_constants("vae_decoder"),
        simplify=policy.should_simplify_vae(simplify_vae),
        skip_if_complete=resume,
        validate=validate,
        release_after=(vae, pipe.vae, pipe),
    )
    all_specs.append(vae_spec)
    export_component_to_dir(output_dir, vae_spec)

    write_model_json(output_dir, policy.model_type, all_specs, vae_scaling_factor)
    print(f"\nAll models exported to {output_dir}  "
          f"(total: {time.time() - t_total:.0f}s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SDXL to ONNX")
    parser.add_argument("model_file", help="Path to the .safetensors checkpoint")
    parser.add_argument("--name", help="Output directory name (default: checkpoint stem)")
    parser.add_argument("--output-dir", help="Override output directory path")
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Use lower-memory export settings such as attention slicing (slower)",
    )
    parser.add_argument(
        "--simplify-vae",
        action="store_true",
        help="Run onnxsim on vae_decoder.onnx after export (slower)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip components whose output files already exist (resume an interrupted export)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ORT forward pass after each component export to catch runtime errors (slow)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.model_file))[0]
    out  = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", name)
    try:
        export_sdxl(
            args.model_file,
            out,
            optimize_memory=args.optimize_memory,
            simplify_vae=args.simplify_vae,
            resume=args.resume,
            validate=args.validate,
        )
    except (FileNotFoundError, ImportError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nExport failed: {e}", file=sys.stderr)
        sys.exit(1)

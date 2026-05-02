"""
Export Stable Diffusion 1.5 models to ONNX for use with ONNX Runtime.

Usage:
    python export_onnx_models.py <model.safetensors> [--name MODEL_NAME]

Outputs (under models/<MODEL_NAME>/):
    text_encoder.onnx   — CLIP-L with clip-skip-2, fp16
    unet.onnx           — UNet, fp16, dynamic H/W
    vae_decoder.onnx    — VAE decoder, fp16, dynamic H/W
    vae_encoder.onnx    — VAE encoder, fp16, static shape
"""
import argparse
import gc
import os
import sys
import time

import torch
from diffusers import StableDiffusionPipeline

from export_common import (
    ExportComponentSpec,
    SD15ExportPolicy,
    VAEDecoderWrapper,
    assert_no_meta_tensors,
    check_dependencies,
    check_model_file,
    export_component_to_dir,
    load_single_file_pipeline,
    make_vae_encoder_spec,
    prepare_libraries_for_export,
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

        # transformers >= 4.47 flattened CLIPTextModel; text_model may not exist
        _tm = getattr(self.text_encoder, "text_model", self.text_encoder)
        hidden_norm = _tm.final_layer_norm(hidden)

        # Return both: normalized output (for UNet) and latent (for LoRA)
        return hidden_norm, hidden


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, encoder_hidden_states):
        return self.unet(latent, timestep,
                         encoder_hidden_states=encoder_hidden_states).sample


# ── Export pipeline ───────────────────────────────────────────────────────────

def export_sd15(model_file: str, output_dir: str, *,
                resume: bool = False, validate: bool = False) -> None:
    policy = SD15ExportPolicy()
    check_dependencies(
        required=["torch", "diffusers", "transformers", "onnx", "onnxscript"],
    )
    check_model_file(model_file)
    os.makedirs(output_dir, exist_ok=True)
    t_total = time.time()
    prepare_libraries_for_export("sd15")
    print("Loading SD 1.5 pipeline ...")
    pipe = load_single_file_pipeline(
        StableDiffusionPipeline,
        model_file,
        torch_dtype=torch.float16,
    )
    assert_no_meta_tensors(pipe.text_encoder, "text_encoder")
    assert_no_meta_tensors(pipe.unet, "unet")
    assert_no_meta_tensors(pipe.vae, "vae")
    vae_scaling_factor = float(pipe.vae.config.scaling_factor)

    exported = []

    # 1. Text encoder ─────────────────────────────────────────────────────────
    dummy_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, 77), dtype=torch.int64)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="1/4  Text encoder",
                component_name="text_encoder",
                filename="text_encoder.onnx",
                model=CLIPTextEncoderClipSkip2(pipe.text_encoder).eval(),
                dummy_inputs=dummy_ids,
                input_names=["input_ids"],
                output_names=["text_embeds", "hidden_latent"],
                dynamic_axes=policy.text_encoder_dynamic_axes(
                    "input_ids",
                    ["text_embeds", "hidden_latent"],
                ),
                export_lora_weights=True,
                skip_if_complete=resume,
                validate=validate,
                release_after=(pipe.text_encoder, pipe.tokenizer),
            ),
        ),
    )
    pipe.text_encoder = None
    del dummy_ids
    gc.collect()

    # 2. UNet ─────────────────────────────────────────────────────────────────
    pipe.unet.to(torch.float16)
    dummy_latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)
    dummy_timestep = torch.tensor([1], dtype=torch.float16)
    dummy_embeds = torch.randn(1, 77, 768, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="2/4  UNet",
                component_name="unet",
                filename="unet.onnx",
                model=UNetWrapper(pipe.unet).eval(),
                dummy_inputs=(dummy_latent, dummy_timestep, dummy_embeds),
                input_names=["latent", "timestep", "encoder_hidden_states"],
                output_names=["latent_out"],
                dynamic_axes=policy.unet_dynamic_axes(),
                exporter=policy.unet_exporter(),
                fix_resize_fp16=policy.should_fix_resize_fp16("unet"),
                export_lora_weights=True,
                skip_if_complete=resume,
                validate=validate,
                release_after=(pipe.unet,),
            ),
        )
    )
    pipe.unet = None
    del dummy_latent, dummy_timestep, dummy_embeds
    gc.collect()

    # 3. VAE decoder ──────────────────────────────────────────────────────────
    pipe.vae.to(torch.float16)
    dummy_latent_vae = torch.randn(1, 4, 64, 64, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="3/4  VAE decoder",
                component_name="vae_decoder",
                filename="vae_decoder.onnx",
                model=VAEDecoderWrapper(pipe.vae).eval(),
                dummy_inputs=(dummy_latent_vae,),
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes=policy.vae_dynamic_axes(),
                exporter=policy.vae_exporter(),
                fix_resize_fp16=policy.should_fix_resize_fp16("vae_decoder"),
                simplify=policy.should_simplify_vae(False),
                skip_if_complete=resume,
                validate=validate,
            ),
        )
    )
    del dummy_latent_vae
    gc.collect()

    # 4. VAE encoder ──────────────────────────────────────────────────────────
    dummy_image = torch.randn(1, 3, 512, 512, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            make_vae_encoder_spec(
                step_name="4/4  VAE encoder",
                exporter=policy.vae_exporter(),
                vae=pipe.vae,
                dummy_image=dummy_image,
                fix_resize_fp16=policy.should_fix_resize_fp16("vae_encoder"),
                skip_if_complete=resume,
                validate=validate,
                release_after=(pipe.vae, pipe),
            ),
        )
    )
    del dummy_image
    gc.collect()

    write_model_json(output_dir, policy.model_type, exported, vae_scaling_factor)
    print(f"\nAll models exported to {output_dir}  "
          f"(total: {time.time() - t_total:.0f}s)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SD 1.5 to ONNX")
    parser.add_argument("model_file", help="Path to the .safetensors checkpoint")
    parser.add_argument("--name", help="Output directory name (default: checkpoint stem)")
    parser.add_argument("--output-dir", help="Override output directory path")
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
        export_sd15(args.model_file, out, resume=args.resume, validate=args.validate)
    except (FileNotFoundError, ImportError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nExport failed: {e}", file=sys.stderr)
        sys.exit(1)

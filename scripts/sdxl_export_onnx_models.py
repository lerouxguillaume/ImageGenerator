"""
Export Stable Diffusion XL models to ONNX for use with ONNX Runtime.

Usage:
    python sdxl_export_onnx_models.py <model.safetensors> [--name MODEL_NAME]

Outputs (under models/<MODEL_NAME>/):
    text_encoder.onnx   — CLIP-L penultimate hidden states, fp32
    text_encoder_2.onnx — OpenCLIP-G hidden states + pooled embeds, fp32
    unet.onnx           — UNet, fp16, batch-dynamic only
    vae_decoder.onnx    — VAE decoder, fp16, static shape
    vae_encoder.onnx    — VAE encoder, fp16, static shape
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
import gc
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
    make_vae_encoder_spec,
    prepare_libraries_for_export,
    write_model_json,
)

SEQ_LEN = 77


# ── Model wrappers ────────────────────────────────────────────────────────────

class CLIPL_Wrapper(torch.nn.Module):
    """CLIP-L: returns penultimate hidden state [batch, 77, 768]."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-2]


class OpenCLIPG_Wrapper(torch.nn.Module):
    """OpenCLIP-G: returns penultimate hidden state [batch, 77, 1280]
    and pooled text embeds [batch, 1280]."""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
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
                validate: bool = False, dynamic_spatial: bool = False,
                emit_fp32_hedge: bool = False) -> None:
    policy = SDXLExportPolicy(dynamic_spatial=dynamic_spatial)
    if dynamic_spatial:
        print("  Dynamic-spatial mode: UNet + VAE decoder export with dynamic H/W "
              "axes (hires-capable; also emits an fp32 decoder hedge)")
    check_dependencies(
        required=[
            "torch", "diffusers", "transformers", "onnx", "onnxscript",
            *(["onnxsim"] if simplify_vae else []),
        ],
    )
    check_model_file(model_file)
    os.makedirs(output_dir, exist_ok=True)
    t_total = time.time()
    prepare_libraries_for_export("sdxl")
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
    latent_h, latent_w = 128, 128  # SDXL native latent (1024 px) — encoder + static exports

    # In dynamic-spatial mode the UNet and VAE decoder carry dynamic H/W axes, so
    # the traced spatial size does not change the exported graph — only the legacy
    # tracer's eager forward-pass cost. fp16 Conv2d on CPU has no fast kernel
    # (~1000× slower than fp32), so trace those two at a tiny latent (mirrors the
    # SD1.5 8×8 decoder trace): identical dynamic graph, ~cheap forward. The VAE
    # ENCODER stays static 1024 and is traced at its real size (dynamo, unchanged).
    # 16 keeps the SDXL UNet's two downsamples well-defined (16→8→4 bottleneck)
    # and stays /8-aligned; the decoder mirrors SD1.5's proven 8.
    unet_trace_hw    = 16 if dynamic_spatial else latent_h
    vae_dec_trace_hw = 8  if dynamic_spatial else latent_h

    exported = []

    # 1. CLIP-L text encoder ──────────────────────────────────────────────────
    dummy_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
    dummy_attn_mask = torch.ones(1, SEQ_LEN, dtype=torch.int64)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="1/5  Text encoder (CLIP-L)",
                component_name="text_encoder",
                filename="text_encoder.onnx",
                model=CLIPL_Wrapper(pipe.text_encoder).eval(),
                dummy_inputs=(dummy_ids, dummy_attn_mask),
                input_names=["input_ids", "attention_mask"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids":      {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "hidden_states":  {0: "batch"},
                },
                export_lora_weights=True,
                skip_if_complete=resume,
                validate=validate,
                release_after=(pipe.text_encoder, pipe.tokenizer),
            ),
        )
    )
    pipe.text_encoder = None
    del dummy_ids, dummy_attn_mask
    gc.collect()

    # 2. OpenCLIP-G text encoder ──────────────────────────────────────────────
    dummy_ids_2 = torch.randint(0, pipe.tokenizer_2.vocab_size, (1, SEQ_LEN), dtype=torch.int64)
    dummy_attn_mask_2 = torch.ones(1, SEQ_LEN, dtype=torch.int64)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="2/5  Text encoder 2 (OpenCLIP-G)",
                component_name="text_encoder_2",
                filename="text_encoder_2.onnx",
                model=OpenCLIPG_Wrapper(pipe.text_encoder_2).eval(),
                dummy_inputs=(dummy_ids_2, dummy_attn_mask_2),
                input_names=["input_ids", "attention_mask"],
                output_names=["hidden_states", "text_embeds"],
                dynamic_axes={
                    "input_ids":      {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "hidden_states":  {0: "batch"},
                    "text_embeds":    {0: "batch"},
                },
                export_lora_weights=True,
                skip_if_complete=resume,
                validate=validate,
                release_after=(pipe.text_encoder_2, pipe.tokenizer_2),
            ),
        ),
    )
    pipe.text_encoder_2 = None
    del dummy_ids_2, dummy_attn_mask_2
    gc.collect()

    # 3. UNet ─────────────────────────────────────────────────────────────────
    disable_attention_upcasting(pipe.unet)
    pipe.unet.to(torch.float16)
    dummy_latent = torch.randn(1, 4, unet_trace_hw, unet_trace_hw, dtype=torch.float16)
    dummy_timestep = torch.tensor([1], dtype=torch.float16)
    dummy_embeds = torch.randn(1, SEQ_LEN, hidden_dim, dtype=torch.float16)
    dummy_pooled = torch.zeros(1, pooled_dim, dtype=torch.float16)
    dummy_time_ids = torch.zeros(1, 6, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="3/5  UNet",
                component_name="unet",
                filename="unet.onnx",
                model=UNetWrapper(pipe.unet).eval(),
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
                release_after=(pipe.unet,),
            ),
        )
    )
    pipe.unet = None
    del dummy_latent, dummy_timestep, dummy_embeds, dummy_pooled, dummy_time_ids
    gc.collect()

    # 4. VAE decoder ──────────────────────────────────────────────────────────
    # 4a. fp32 dynamic decoder HEDGE (dynamic-spatial only). Exported FIRST, from
    #     the original fp32 weights — BEFORE the fp16 cast below — so it is a true
    #     fp32 decoder, not a lossy fp16->fp32 roundtrip. Same dynamic H/W axes and
    #     tiny trace, but NO fp16 graph fixes (those are fp16-specific). It is a
    #     spare: NOT appended to `exported`, so it is absent from model.json's
    #     components block and inert to the C++ runtime until swapped in.
    #     Rationale: SDXL fp16-VAE can produce NaN/washed-out output at >native
    #     latents; shipping the fp32 fallback in the SAME run means the fix is a
    #     file swap, never a second ~hour-long export session. Gated on
    #     emit_fp32_hedge (CLI --dynamic-spatial only) — the in-app import ships a
    #     lean fp16-only model, since fp16 was validated stable and there is no
    #     in-app swap mechanism to make the ~200 MB spare worth its disk.
    if dynamic_spatial and emit_fp32_hedge:
        dummy_latent_vae_fp32 = torch.randn(
            1, 4, vae_dec_trace_hw, vae_dec_trace_hw, dtype=torch.float32)
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="4a/5  VAE decoder (fp32 hedge)",
                component_name="vae_decoder_fp32",
                filename="vae_decoder_fp32.onnx",
                model=VAEDecoderWrapper(pipe.vae).eval(),
                dummy_inputs=(dummy_latent_vae_fp32,),
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes=policy.vae_dynamic_axes(),
                exporter=policy.vae_exporter(),   # legacy — dynamo can't do H/W
                fix_fp32_constants=False,          # fp32 graph: no fp16 fixes
                fix_resize_fp16=False,
                simplify=False,
                skip_if_complete=resume,
                validate=validate,
            ),
        )  # return value intentionally discarded (spare, not a registered component)
        del dummy_latent_vae_fp32
        gc.collect()

    # 4b. fp16 dynamic decoder (the default/active decoder).
    pipe.vae.to(torch.float16)
    dummy_latent_vae = torch.randn(1, 4, vae_dec_trace_hw, vae_dec_trace_hw, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            ExportComponentSpec(
                step_name="4/5  VAE decoder",
                component_name="vae_decoder",
                filename="vae_decoder.onnx",
                model=VAEDecoderWrapper(pipe.vae).eval(),
                dummy_inputs=(dummy_latent_vae,),
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes=policy.vae_dynamic_axes(),
                exporter=policy.vae_exporter(),
                fix_fp32_constants=policy.should_fix_fp32_constants("vae_decoder"),
                fix_resize_fp16=policy.should_fix_resize_fp16("vae_decoder"),
                simplify=policy.should_simplify_vae(simplify_vae),
                skip_if_complete=resume,
                validate=validate,
            ),
        )
    )
    del dummy_latent_vae
    gc.collect()

    # 5. VAE encoder ──────────────────────────────────────────────────────────
    # Always traced at native 1024 and kept static. Use vae_encoder_exporter()
    # (not vae_exporter()): in dynamic-spatial mode the decoder switches to the
    # legacy tracer, but the static 1024 encoder must stay on dynamo so it never
    # pays the legacy tracer's ~1000×-slower eager fp16 CPU conv at full size.
    dummy_image = torch.randn(1, 3, latent_h * 8, latent_w * 8, dtype=torch.float16)
    exported.append(
        export_component_to_dir(
            output_dir,
            make_vae_encoder_spec(
                step_name="5/5  VAE encoder",
                exporter=policy.vae_encoder_exporter(),
                vae=pipe.vae,
                dummy_image=dummy_image,
                fix_fp32_constants=policy.should_fix_fp32_constants("vae_encoder"),
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
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help="Export UNet + VAE decoder with dynamic H/W axes so a hires/second pass "
             "can run at >native resolution (and emit an fp32 decoder hedge). Traces "
             "those two at a tiny latent; encoder stays static 1024. This is the "
             "in-app import default; the standalone CLI defaults to the static export.",
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
            dynamic_spatial=args.dynamic_spatial,
            # CLI --dynamic-spatial also emits the fp32 decoder hedge (investigation
            # / manual-swap use); the in-app import passes emit_fp32_hedge=False.
            emit_fp32_hedge=args.dynamic_spatial,
        )
    except (FileNotFoundError, ImportError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nExport failed: {e}", file=sys.stderr)
        sys.exit(1)

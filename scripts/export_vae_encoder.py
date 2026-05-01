"""
Retrofit tool: export vae_encoder.onnx into an already-exported model directory.

Use this to add img2img support to a model directory that was exported before
the VAE encoder was part of the pipeline.  The VAE decoder and all other
components are left untouched.

The output is vae_encoder.onnx (+ vae_encoder.onnx.data) in the same directory.
The C++ runtime loads it automatically when present; existing txt2img runs are
unaffected when it is absent.

Usage:
    python export_vae_encoder.py <model_dir> <model.safetensors>

Example (SD 1.5):
    python export_vae_encoder.py models/my_model checkpoints/my_model.safetensors

Example (SDXL):
    python export_vae_encoder.py models/my_sdxl_model checkpoints/sdxl.safetensors
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

from export_common import (
    assert_no_meta_tensors,
    check_dependencies,
    check_model_file,
    export_component_to_dir,
    load_single_file_pipeline,
    make_vae_encoder_spec,
    patch_clip_text_model_compat,
    patch_fp32_upcasts_for_tracing,
)


def _detect_model_type(model_dir: str) -> str:
    json_path = os.path.join(model_dir, "model.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get("type", "sd15")
    return "sd15"


def _update_model_json(model_dir: str, *, model_type: str, vae_scaling_factor: float) -> None:
    json_path = os.path.join(model_dir, "model.json")
    data: dict = {"type": model_type}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    data["type"] = model_type
    data["vae_scaling_factor"] = float(vae_scaling_factor)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def export_vae_encoder(model_dir: str, model_file: str, *, force: bool = False) -> None:
    check_dependencies(
        required=["torch", "diffusers", "transformers", "onnx", "onnxscript"],
    )
    check_model_file(model_file)

    out_path = os.path.join(model_dir, "vae_encoder.onnx")
    if os.path.exists(out_path) and os.path.exists(out_path + ".data") and not force:
        print(f"vae_encoder.onnx already exists in {model_dir} - use --force to overwrite.")
        return

    model_type = _detect_model_type(model_dir)
    is_sdxl = (model_type == "sdxl")
    print(f"Model type : {model_type}")
    print(f"Output dir : {model_dir}")

    t_total = time.time()
    patch_clip_text_model_compat()

    if is_sdxl:
        patch_fp32_upcasts_for_tracing()
        from diffusers import StableDiffusionXLPipeline
        print("Loading SDXL pipeline (VAE only) ...")
        pipe = load_single_file_pipeline(
            StableDiffusionXLPipeline, model_file, torch_dtype=torch.float32
        )
        latent_h, latent_w = 128, 128
        fix_fp32 = True
        fix_resize = True
    else:
        from diffusers import StableDiffusionPipeline
        print("Loading SD 1.5 pipeline (VAE only) ...")
        pipe = load_single_file_pipeline(
            StableDiffusionPipeline, model_file, torch_dtype=torch.float16
        )
        latent_h, latent_w = 64, 64
        fix_fp32 = False
        fix_resize = False

    assert_no_meta_tensors(pipe.vae, "vae")
    vae_scaling_factor = float(pipe.vae.config.scaling_factor)

    # Extract VAE and free the rest of the pipeline (UNet, text encoders, …)
    # before the tracing step — those components are not needed and keeping them
    # in RAM throughout export is the main cause of OOM on SDXL checkpoints.
    vae = pipe.vae
    if is_sdxl:
        vae.to(torch.float16)
    del pipe
    gc.collect()

    img_h = latent_h * 8
    img_w = latent_w * 8

    dummy_image = torch.randn(1, 3, img_h, img_w, dtype=torch.float16)
    spec = make_vae_encoder_spec(
        step_name="VAE encoder",
        exporter="dynamo",
        vae=vae,
        dummy_image=dummy_image,
        fix_fp32_constants=fix_fp32,
        fix_resize_fp16=fix_resize,
        skip_if_complete=False,
        release_after=(vae,),
    )
    export_component_to_dir(model_dir, spec)
    _update_model_json(model_dir, model_type=model_type, vae_scaling_factor=vae_scaling_factor)

    print(f"\nvae_encoder.onnx exported to {model_dir}  "
          f"(total: {time.time() - t_total:.0f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export vae_encoder.onnx into an existing model directory")
    parser.add_argument("model_dir",  help="Existing model directory (contains vae_decoder.onnx)")
    parser.add_argument("model_file", help="Path to the original .safetensors checkpoint")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite vae_encoder.onnx even if it already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        export_vae_encoder(args.model_dir, args.model_file, force=args.force)
    except (FileNotFoundError, ImportError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nExport failed: {e}", file=sys.stderr)
        sys.exit(1)

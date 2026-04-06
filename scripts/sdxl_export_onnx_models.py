import gc
import os
import json
import numpy as np
import torch
import onnx
from diffusers import StableDiffusionXLPipeline

# ----------------------------
# Paths
# ----------------------------
MODEL_NAME = "ilustmix_v111"
MODEL_FILE = f"./{MODEL_NAME}.safetensors"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load SDXL.
# Text encoders stay float32 for accuracy.
# UNet and VAE are loaded as float32 then converted to float16 in PyTorch
# before export — this avoids creating a 10 GB fp32 ONNX intermediate that
# would require ~15 GB RAM to post-convert with onnxconverter_common.
# After the fp16 export, to_fp16_onnx() is still run to fix the small number
# of fp32 scalar constants that torch.onnx.export leaves in the graph
# (e.g. time-embedding scalars, GroupNorm eps) — these cause Concat type
# errors in ORT if left as fp32 in an otherwise fp16 graph.
# ----------------------------
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_FILE, torch_dtype=torch.float32)
pipe.enable_attention_slicing()
pipe.text_encoder.to(torch.float32).eval()
pipe.text_encoder_2.to(torch.float32).eval()
pipe.unet.to(torch.float32).eval()
pipe.vae.to(torch.float32).eval()

seq_len = 77


def fix_fp32_constants(path: str) -> None:
    """Convert any remaining float32 constants to float16, running in a subprocess.

    Loading a large ONNX proto in the main process doubles peak RAM (protobuf
    expands to ~2× the file size in memory). Running in a subprocess gives the
    loader a clean heap so the main process heap is unaffected.
    """
    import subprocess, sys, textwrap
    script = textwrap.dedent(f"""
        import onnx, numpy as np
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
        onnx.save(model, path)
        print(f"  → fixed {{changed}} fp32 constant(s) to float16")
    """)
    subprocess.run([sys.executable, "-c", script], check=True)


# ----------------------------
# 1. Text Encoder (CLIP-L → hidden states [1, 77, 768])
#    C++ expects: 1 input (input_ids), 1 output (hidden states)
# ----------------------------
class CLIPL_Wrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        # SDXL uses penultimate hidden state ([-2]), shape [batch, 77, 768]
        return out.hidden_states[-2]

clip_l = CLIPL_Wrapper(pipe.text_encoder).cpu().eval()
dummy_input_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, seq_len), dtype=torch.int64)

torch.onnx.export(
    clip_l,
    (dummy_input_ids,),
    os.path.join(OUTPUT_DIR, "text_encoder.onnx"),
    opset_version=18,
    input_names=["input_ids"],
    output_names=["hidden_states"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "hidden_states": {0: "batch"},
    },
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ text_encoder.onnx (CLIP-L) exported successfully")
del clip_l, pipe.text_encoder, pipe.tokenizer
gc.collect()

# ----------------------------
# 2. Text Encoder 2 (OpenCLIP-G → hidden states [1, 77, 1280] + pooled [1, 1280])
#    C++ expects: 1 input (input_ids), 2 outputs (hidden states, pooled)
# ----------------------------
class OpenCLIPG_Wrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        # SDXL uses penultimate hidden state ([-2]), shape [batch, 77, 1280]
        hidden = out.hidden_states[-2]
        # Pooled text embeds from the projection head, shape [batch, 1280]
        pooled = out.text_embeds
        return hidden, pooled

clip_g = OpenCLIPG_Wrapper(pipe.text_encoder_2).cpu().eval()
dummy_input_ids_2 = torch.randint(0, pipe.tokenizer_2.vocab_size, (1, seq_len), dtype=torch.int64)

torch.onnx.export(
    clip_g,
    (dummy_input_ids_2,),
    os.path.join(OUTPUT_DIR, "text_encoder_2.onnx"),
    opset_version=18,
    input_names=["input_ids"],
    output_names=["hidden_states", "text_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "hidden_states": {0: "batch"},
        "text_embeds": {0: "batch"},
    },
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ text_encoder_2.onnx (OpenCLIP-G) exported successfully")
del clip_g, pipe.text_encoder_2, pipe.tokenizer_2
gc.collect()

# ----------------------------
# 3. UNet (exported directly as float16 to avoid a 10 GB fp32 intermediate)
#    to_fp16_onnx() is called afterwards to fix the few remaining fp32 scalar
#    constants emitted by the tracer — a cheap operation on an already-fp16 file.
# ----------------------------
class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        return out.sample

pipe.unet.to(torch.float16)
unet_wrapper = UNetONNXWrapper(pipe.unet).cpu().eval()

batch_size = 1
latent_h, latent_w = 128, 128  # SDXL latent for 1024px
hidden_size = pipe.unet.config.cross_attention_dim  # 2048
pooled_text_dim = 1280
num_time_ids = 6  # [orig_h, orig_w, crop_top, crop_left, target_h, target_w]

latent = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float16)
timestep = torch.tensor([1], dtype=torch.float16)
encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
text_embeds = torch.zeros(batch_size, pooled_text_dim, dtype=torch.float16)
time_ids = torch.zeros(batch_size, num_time_ids, dtype=torch.float16)

unet_path = os.path.join(OUTPUT_DIR, "unet.onnx")
torch.onnx.export(
    unet_wrapper,
    (latent, timestep, encoder_hidden_states, text_embeds, time_ids),
    unet_path,
    opset_version=18,
    input_names=["latent", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
    output_names=["latent_out"],
    dynamic_axes={
        "latent": {0: "batch", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
        "latent_out": {0: "batch", 2: "height", 3: "width"},
    },
    do_constant_folding=False,
    keep_initializers_as_inputs=False,
)
print("✅ unet.onnx exported successfully (float16)")
del unet_wrapper, pipe.unet
gc.collect()

fix_fp32_constants(unet_path)
print("✅ unet.onnx ready")

# ----------------------------
# 4. VAE Decoder (exported directly as float16, same rationale as UNet)
# ----------------------------
class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

pipe.vae.to(torch.float16)
vae_wrapper = VAEWrapper(pipe.vae).cpu().eval()
dummy_latent_vae = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float16)

vae_path = os.path.join(OUTPUT_DIR, "vae_decoder.onnx")
torch.onnx.export(
    vae_wrapper,
    (dummy_latent_vae,),
    vae_path,
    opset_version=18,
    input_names=["latent"],
    output_names=["image"],
    do_constant_folding=False,
    keep_initializers_as_inputs=False,
)
print("✅ vae_decoder.onnx exported successfully (float16)")
del vae_wrapper, pipe.vae, pipe
gc.collect()

fix_fp32_constants(vae_path)

# Optional: simplify VAE ONNX (run after fp16 fixup)
try:
    import onnxsim
    model = onnx.load(vae_path)
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, vae_path)
        print("✅ vae_decoder.onnx simplified with onnxsim")
    else:
        print("⚠  onnxsim simplification check failed, keeping original")
except ImportError:
    pass
print("✅ vae_decoder.onnx ready")

# ----------------------------
# 5. model.json — tells the C++ runtime this is an SDXL model
# ----------------------------
with open(os.path.join(OUTPUT_DIR, "model.json"), "w") as f:
    json.dump({"type": "sdxl"}, f)
print("✅ model.json written")
from diffusers import StableDiffusionPipeline
import torch
import os

# ----------------------------
# Paths
# ----------------------------
MODEL_NAME = "novelaiDiffusionV2_novelaiV2"
MODEL_FILE = f"./{MODEL_NAME}.safetensors"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Stable Diffusion
# ----------------------------
pipe = StableDiffusionPipeline.from_single_file(MODEL_FILE, torch_dtype=torch.float16)
pipe.enable_attention_slicing()

# ----------------------------
# 1. Text Encoder (Clip Skip 2)
# ----------------------------
class CLIPTextEncoderClipSkip2(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-2]
        hidden = self.text_encoder.text_model.final_layer_norm(hidden)
        return hidden

clip_encoder = CLIPTextEncoderClipSkip2(pipe.text_encoder).eval()
dummy_input_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, 77), dtype=torch.int64)

torch.onnx.export(
    clip_encoder,
    dummy_input_ids,
    f"{OUTPUT_DIR}/text_encoder.onnx",
    opset_version=18,
    input_names=["input_ids"],
    output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
    do_constant_folding=True,
    keep_initializers_as_inputs=False,  # ensures single file
)
print("✅ text_encoder.onnx exported as single file")

# ----------------------------
# 2. UNet (FP16)
# ----------------------------
class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, latent, timestep, encoder_hidden_states):
        return self.unet(latent, timestep, encoder_hidden_states=encoder_hidden_states).sample

pipe.unet.to(torch.float16)
unet_wrapper = UNetWrapper(pipe.unet).eval()

latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)
timestep = torch.tensor([1], dtype=torch.float16)
text_embeds = torch.randn(1, 77, 768, dtype=torch.float16)

torch.onnx.export(
    unet_wrapper,
    (latent, timestep, text_embeds),
    f"{OUTPUT_DIR}/unet.onnx",
    opset_version=18,
    input_names=["latent", "timestep", "encoder_hidden_states"],
    output_names=["latent_out"],
    dynamic_axes={
        "latent": {0: "batch"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch"},
        "latent_out": {0: "batch"},
    },
    do_constant_folding=True,
    keep_initializers_as_inputs=False,  # single file
)
print("✅ unet.onnx exported as single file")

# ----------------------------
# 3. VAE Decoder (FP32)
# ----------------------------
class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

pipe.vae.to(torch.float16)
vae_wrapper = VAEWrapper(pipe.vae).eval()

dummy_latent_vae = torch.randn(1, 4, 64, 64, dtype=torch.float16)

vae_path = f"{OUTPUT_DIR}/vae_decoder.onnx"
torch.onnx.export(
    vae_wrapper,
    (dummy_latent_vae,),
    vae_path,
    opset_version=18,
    input_names=["latent"],
    output_names=["image"],
    # No dynamic_axes: fully static graph so constant folding can bake Reshape
    # shape tensors as constants — required for DirectML which rejects runtime-
    # computed shapes in Reshape nodes (E_INVALIDARG on node_view_2).
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ vae_decoder.onnx exported as single file")

# Optional: further simplify with onnxsim (pip install onnxsim) to eliminate
# any remaining dynamic shape computations DML can't handle.
try:
    import onnx, onnxsim
    model = onnx.load(vae_path)
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, vae_path)
        print("✅ vae_decoder.onnx simplified with onnxsim")
    else:
        print("⚠️  onnxsim simplification check failed, keeping original")
except ImportError:
    pass  # onnxsim not installed, skip
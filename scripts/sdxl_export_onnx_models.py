from diffusers import StableDiffusionXLPipeline
import torch
import os

# ----------------------------
# Paths
# ----------------------------
MODEL_NAME = "ilustmix_v111"  # replace with your SDXL model
MODEL_FILE = f"./{MODEL_NAME}.safetensors"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Stable Diffusion XL
# ----------------------------
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_FILE, torch_dtype=torch.float16)
pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()

# ----------------------------
# 1. Text Encoder (only text_encoder_2 for SDXL)
# ----------------------------
class CLIPTextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-2]
        hidden = self.text_encoder.text_model.final_layer_norm(hidden)
        return hidden

clip_encoder = CLIPTextEncoderWrapper(pipe.text_encoder_2).eval()
dummy_input_ids = torch.randint(0, pipe.tokenizer_2.vocab_size, (1, 77), dtype=torch.int64)

torch.onnx.export(
    clip_encoder,
    dummy_input_ids,
    f"{OUTPUT_DIR}/text_encoder2.onnx",
    opset_version=18,
    input_names=["input_ids"],
    output_names=["text_embeds"],
    dynamic_shapes=[{0: "batch"}],  # list, matches input
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ text_encoder2.onnx exported")

# ----------------------------
# 2. UNet (FP16, SDXL fix)
# ----------------------------
class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        batch_size = sample.shape[0]
        # SDXL requires 'text_embeds' in added_cond_kwargs if addition_embed_type='text_time'
        dummy_text_embeds = torch.zeros(
            batch_size,
            self.unet.config.addition_embed_dim,
            dtype=sample.dtype,
            device=sample.device
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": dummy_text_embeds}
        ).sample

pipe.unet.to(torch.float16)
unet_wrapper = UNetWrapper(pipe.unet).eval()

latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)
timestep = torch.tensor([1], dtype=torch.float16)
text_embeds = torch.randn(1, 77, pipe.text_encoder_2.config.hidden_size, dtype=torch.float16)

torch.onnx.export(
    unet_wrapper,
    (latent, timestep, text_embeds),
    f"{OUTPUT_DIR}/unet.onnx",
    opset_version=18,
    input_names=["latent", "timestep", "encoder_hidden_states"],
    output_names=["latent_out"],
    dynamic_shapes=[  # list, matches inputs by position
        {0: "batch"},  # latent
        {0: "batch"},  # timestep
        {0: "batch"},  # encoder_hidden_states
    ],
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ unet.onnx exported")

# ----------------------------
# 3. VAE Decoder (FP16)
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
    dynamic_shapes=[{0: "batch"}],
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ vae_decoder.onnx exported")

# ----------------------------
# Optional: simplify VAE with onnxsim
# ----------------------------
try:
    import onnx, onnxsim
    model = onnx.load(vae_path)
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, vae_path)
        print("✅ vae_decoder.onnx simplified with onnxsim")
    else:
        print("⚠️  onnxsim simplification failed, keeping original")
except ImportError:
    print("ℹ️  onnxsim not installed, skipping simplification")
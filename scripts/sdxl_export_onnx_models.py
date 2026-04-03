import os
import torch
from diffusers import StableDiffusionXLPipeline

# ----------------------------
# Paths
# ----------------------------
MODEL_NAME = "ilustmix_v111"
MODEL_FILE = f"./{MODEL_NAME}.safetensors"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load SDXL
# ----------------------------
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_FILE, torch_dtype=torch.float32)
pipe.enable_attention_slicing()
pipe.unet.to(torch.float32).eval()
pipe.text_encoder.to(torch.float32).eval()
pipe.text_encoder_2.to(torch.float32).eval()
pipe.vae.to(torch.float32).eval()

# ----------------------------
# Helper wrapper for pooled mean
# ----------------------------
class CLIPTextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        pooled = out.last_hidden_state.mean(dim=1)
        return pooled

dummy_input_ids = torch.randint(0, pipe.tokenizer.vocab_size, (1, 77), dtype=torch.int64)

# ----------------------------
# Export both text encoders
# ----------------------------
for idx, encoder in enumerate([pipe.text_encoder, pipe.text_encoder_2], start=1):
    clip_encoder = CLIPTextEncoderWrapper(encoder).cpu().eval()
    path = os.path.join(OUTPUT_DIR, f"text_encoder{'' if idx==1 else '_2'}.onnx")
    torch.onnx.export(
        clip_encoder,
        dummy_input_ids,
        path,
        opset_version=18,
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
    )
    print(f"✅ {os.path.basename(path)} exported successfully")

# ----------------------------
# 2. UNet
# ----------------------------
class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds_main, text_embeds_2, time_ids):
        # Pass both embeddings in added_cond_kwargs if needed
        added_cond_kwargs = {
            "text_embeds": text_embeds_main,  # main prompt
            "text_embeds_2": text_embeds_2,   # secondary/context
            "time_ids": time_ids
        }
        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        return out.sample

unet_wrapper = UNetONNXWrapper(pipe.unet).cpu().eval()

# Dummy inputs
batch_size = 1
seq_len = 77
latent_h, latent_w = 128, 128  # SDXL latent for 1024px
hidden_size = pipe.unet.config.cross_attention_dim
pooled_text_dim = 1280
num_time_ids = 6

latent = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float32)
timestep = torch.tensor([1], dtype=torch.float32)
encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
text_embeds = torch.zeros(batch_size, pooled_text_dim, dtype=torch.float32)
time_ids = torch.zeros(batch_size, num_time_ids, dtype=torch.float32)

torch.onnx.export(
    unet_wrapper,
    (latent, timestep, encoder_hidden_states, text_embeds, time_ids),
    os.path.join(OUTPUT_DIR, "unet.onnx"),
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
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ UNet exported successfully")

# ----------------------------
# 3. VAE Decoder
# ----------------------------
class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

vae_wrapper = VAEWrapper(pipe.vae).cpu().eval()
dummy_latent_vae = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float32)

vae_path = os.path.join(OUTPUT_DIR, "vae_decoder.onnx")
torch.onnx.export(
    vae_wrapper,
    (dummy_latent_vae,),
    vae_path,
    opset_version=18,
    input_names=["latent"],
    output_names=["image"],
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
)
print("✅ VAE decoder exported successfully")

# Optional: simplify VAE ONNX
try:
    import onnx, onnxsim
    model = onnx.load(vae_path)
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, vae_path)
        print("✅ vae_decoder.onnx simplified with onnxsim")
except ImportError:
    pass
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
pipe.text_encoder.to(torch.float32).eval()
pipe.text_encoder_2.to(torch.float32).eval()
pipe.unet.to(torch.float32).eval()
pipe.vae.to(torch.float32).eval()

seq_len = 77

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

# ----------------------------
# 3. UNet
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

unet_wrapper = UNetONNXWrapper(pipe.unet).cpu().eval()

batch_size = 1
latent_h, latent_w = 128, 128  # SDXL latent for 1024px
hidden_size = pipe.unet.config.cross_attention_dim  # 2048
pooled_text_dim = 1280
num_time_ids = 6  # [orig_h, orig_w, crop_top, crop_left, target_h, target_w]

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
print("✅ unet.onnx exported successfully")

# ----------------------------
# 4. VAE Decoder
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
print("✅ vae_decoder.onnx exported successfully")

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
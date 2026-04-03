from diffusers import StableDiffusionXLPipeline
import torch
import os

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
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_FILE, torch_dtype=torch.float16)
pipe.enable_attention_slicing()
pipe.unet.to(torch.float16)

# ----------------------------
# UNet Wrapper for ONNX
# ----------------------------
class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        """
        ONNX export safe: no dicts, all inputs are tensors.
        - sample:                  (B, 4, H/8, W/8)   latent (128x128 for 1024px SDXL)
        - timestep:                (B,)
        - encoder_hidden_states:   (B, 77, 2048)       concat of CLIP-L (768) + OpenCLIP-G (1280)
        - text_embeds:             (B, 1280)            pooled output from text_encoder_2 (2-D)
        - time_ids:                (B, 6)               [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        """
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        return out.sample

unet_wrapper = UNetONNXWrapper(pipe.unet).eval()

# ----------------------------
# Dummy inputs for export
# ----------------------------
batch_size = 1
seq_len    = 77
# SDXL native resolution is 1024x1024; latents are 1/8 the image size.
latent_h, latent_w = 128, 128
# cross_attention_dim for SDXL is 2048 (768 CLIP-L + 1280 OpenCLIP-G concatenated).
hidden_size = pipe.unet.config.cross_attention_dim
# text_embeds is the 2-D pooled output of text_encoder_2, not sequence embeddings.
pooled_text_dim = 1280
# time_ids carries 6 scalars: [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
num_time_ids = 6

latent                = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float16)
timestep              = torch.tensor([1], dtype=torch.float16)
encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
text_embeds           = torch.zeros(batch_size, pooled_text_dim, dtype=torch.float16)
time_ids              = torch.zeros(batch_size, num_time_ids, dtype=torch.float16)

# ----------------------------
# Export UNet to ONNX
# ----------------------------
# dynamo=False uses the TorchScript tracing path (same as SD 1.5 export).
# dynamic_axes is incompatible with the new dynamo exporter and triggers errors.
torch.onnx.export(
    unet_wrapper,
    (latent, timestep, encoder_hidden_states, text_embeds, time_ids),
    os.path.join(OUTPUT_DIR, "unet.onnx"),
    opset_version=18,
    dynamo=False,
    input_names=["latent", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
    output_names=["latent_out"],
    dynamic_axes={
        "latent":                 {0: "batch"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch"},
        "text_embeds":            {0: "batch"},
        "time_ids":               {0: "batch"},
        "latent_out":             {0: "batch"},
    },
    do_constant_folding=True,
)
print("✅ UNet exported to ONNX successfully")
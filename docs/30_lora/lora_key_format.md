# LoRA Key Format (Kohya)

LoRA keys follow naming patterns:

- lora_te_ → text encoder
- lora_unet_ → UNet
- lora_te2_ → SDXL second encoder

---

# Structure

Each layer:
- down
- up
- alpha (optional)

---

# Example

lora_unet_down_blocks_0_attn...
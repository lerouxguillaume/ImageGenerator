# SDXL Specifics

SDXL differs from SD1.5 in architecture and conditioning.

---

# Latent space

- Resolution: 128×128 latent
- (vs 64×64 in SD1.5)

---

# Dual text encoders

SDXL uses:

1. CLIP-L (768)
2. OpenCLIP-G (1280)

Output:
- concatenated embedding: (1, 77, 2048)

---

# Additional UNet inputs

- `text_embeds` → pooled encoder output
- `time_ids` → spatial conditioning vector

Format:
[orig_h, orig_w, crop_top, crop_left, target_h, target_w]


---

# Key implication

SDXL requires:
- different input wiring
- extended conditioning pipeline
- modified UNet signature
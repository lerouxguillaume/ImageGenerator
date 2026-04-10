import re

lines = open("/home/guillaume/Documents/image_generator/scripts/tensor_dump.txt").readlines()

# Collect ONNX names (text encoder + unet sections only, not VAE)
onnx_te, onnx_unet = [], []
section = None
for line in lines:
    line = line.rstrip()
    if "TEXT ENCODER (ONNX)" in line: section = "te"; continue
    if "UNET (ONNX)" in line: section = "unet"; continue
    if "VAE (ONNX)" in line: section = None; continue
    if line.startswith("="): continue
    if section == "te" and line: onnx_te.append(line)
    if section == "unet" and line: onnx_unet.append(line)

# Normalize: . and / -> _
def norm(s): return re.sub(r'[./]', '_', s)

onnx_te_norm  = {norm(n) for n in onnx_te}
onnx_unet_norm = {norm(n) for n in onnx_unet}

# Collect LoRA base names
lora_bases = set()
for line in lines:
    line = line.rstrip()
    for prefix, plen in [("lora_te_", 8), ("lora_te2_", 9), ("lora_unet_", 10)]:
        if line.startswith(prefix):
            body = line[plen:]
            for sfx, slen in [(".lora_down.weight", 17), (".lora_up.weight", 15), (".alpha", 6)]:
                if body.endswith(sfx):
                    lora_bases.add((prefix, body[:-slen]))
                    break

# For each base, try to find the matching ONNX key
matched = []
unmatched = []
for prefix, base in sorted(lora_bases):
    lookup_w = base + "_weight"
    lookup_b = base + "_bias"
    onnx_set = onnx_te_norm if "lora_te" in prefix else onnx_unet_norm
    found = None
    for k in onnx_set:
        if k.endswith(lookup_w) or k.endswith(lookup_b):
            found = k
            break
    if found:
        matched.append((prefix, base, found))
    else:
        unmatched.append((prefix, base))

print(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")
if unmatched:
    print("\nUnmatched LoRA bases:")
    for p, b in unmatched[:30]:
        print(f"  {p}{b}")
 
from safetensors import safe_open

path = r"/media/sf_shared_vm/lora/goblin_v0.1a.safetensors"
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    for k in keys[:10]:
        t = f.get_tensor(k)
        print(f"  {k}: dtype={t.dtype}  shape={list(t.shape)}")
    if len(keys) > 10:
        print(f"  ... ({len(keys) - 10} more)")

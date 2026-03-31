import onnx
import onnxruntime as ort
import numpy as np
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Mapping of model names → expected input dtypes
# Adjust according to your ONNX export
MODEL_DTYPES = {
    "text_encoder": np.int64,       # input_ids
    "unet": np.float16,             # latent, timestep, encoder_hidden_states
    "vae_decoder": np.float32       # latent
}

def check_directml_compatibility(model_path):
    model_name = os.path.basename(model_path).split(".")[0]
    expected_dtype = MODEL_DTYPES.get(model_name, np.float32)

    print(f"\n🔹 Loading model: {model_path}")

    # Load the ONNX model
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("✅ ONNX syntax is valid")
    except Exception as e:
        print(f"❌ Failed to load ONNX: {e}")
        return

    # List all ops
    ops = set(node.op_type for node in model.graph.node)
    print(f"Ops used ({len(ops)} unique): {ops}")

    # Create ONNX Runtime session with DirectML
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        print("✅ DirectML session created successfully")
    except Exception as e:
        print(f"❌ Failed to create DirectML session: {e}")
        return

    # Create dummy inputs
    dummy_feed = {}
    for inp in session.get_inputs():
        shape = [s if isinstance(s, int) else 1 for s in inp.shape]
        # Use model-specific dtype
        dtype = expected_dtype
        # For text_encoder we keep int64
        if model_name == "text_encoder":
            dtype = np.int64
        dummy_feed[inp.name] = np.random.randn(*shape).astype(dtype) if dtype != np.int64 else np.random.randint(0, 1000, size=shape, dtype=dtype)
        print(f"  ➡️ Dummy tensor for '{inp.name}' with shape {shape} and dtype {dtype}")

    # Run dummy inference
    try:
        output = session.run(None, dummy_feed)
        for i, out in enumerate(output):
            print(f"✅ Output[{i}] shape: {out.shape}, dtype: {out.dtype}")
    except Exception as e:
        print(f"❌ Inference failed on DirectML: {e}")
        print("➡️ Model may not be fully compatible with DirectML. Consider CPU or CUDA.")

# Example usage
check_directml_compatibility(f"{DIR}/text_encoder.onnx")
check_directml_compatibility(f"{DIR}/unet.onnx")
check_directml_compatibility(f"{DIR}/vae_decoder.onnx")
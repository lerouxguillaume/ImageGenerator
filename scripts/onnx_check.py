import onnx
from onnx import TensorProto

model = onnx.load(r"/media/sf_shared_vm/models/hassakuSD15_v13/text_encoder.onnx")
inits = model.graph.initializer
print("Initializers:", len(inits))

raw_count = float_count = int32_count = ext_count = other_count = 0
for init in inits:
    if init.HasField("data_location") and init.data_location == TensorProto.EXTERNAL:
        ext_count += 1
    elif len(init.raw_data) > 0:
        raw_count += 1
    elif len(init.float_data) > 0:
        float_count += 1
    elif len(init.int32_data) > 0:
        int32_count += 1
    else:
        other_count += 1

print(f"  raw_data={raw_count}  float_data={float_count}  int32_data={int32_count}  external={ext_count}  other={other_count}")

# Show a sample raw-data initializer to check field numbers
for init in inits:
    if len(init.raw_data) > 0:
        print(f"  Sample raw: '{init.name}' dtype={init.data_type} shape={list(init.dims)} raw_bytes={len(init.raw_data)}")
        break

# Also check model proto structure
print(f"\nModelProto ir_version={model.ir_version}")
print(f"GraphProto name='{model.graph.name}'")
print(f"GraphProto nodes={len(model.graph.node)}")

# Check if model uses external data format
import os
model_path = r"/media/sf_shared_vm/models/hassakuSD15_v13/text_encoder.onnx"
print(f"File size: {os.path.getsize(model_path) / 1e6:.1f} MB")
data_files = [f for f in os.listdir(os.path.dirname(model_path)) if not f.endswith('.onnx') and not f.endswith('.json')]
print(f"Other files in dir: {data_files[:5]}")

// Implementations split into focused translation units:
//   OnnxParser.cpp       — resolveBundle + parseExternalIndex (protobuf parsing)
//   OnnxIndex.cpp        — buildExternalSuffixIndex (O(1) suffix lookup table)
//   LoraParser.cpp       — parseLoraLayers (kohya key grouping)
//   LoraMath.cpp         — computeLoraDelta (matmul + scale)
//   SdLoraApply.cpp      — buildLoraOverrides (reads .onnx.data + applies delta + Ort::Value overrides)
//   SdLoraMatch.hpp/.cpp — matchExternalLoraKey (suffix-index lookup + ambiguity detection)
//
// This file is intentionally empty; it is kept so that the split is
// visible in git history via the file rename / content change.

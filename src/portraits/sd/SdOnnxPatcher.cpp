// Implementations split into focused translation units:
//   OnnxParser.cpp     — parseTensorIndex (inline scan) + parseExternalIndex (external-data scan)
//   OnnxIndex.cpp      — buildSuffixIndex + buildExternalSuffixIndex (O(1) suffix lookup tables)
//   LoraParser.cpp     — parseLoraLayers  (kohya key grouping)
//   LoraMath.cpp       — computeLoraDelta (matmul + scale)
//   OnnxPatcher.cpp    — applyLoraToBytes (legacy inline-buffer patch loop; kept for reference)
//   SdLoraApply.cpp    — buildLoraOverrides (reads .onnx.data + delta + Ort::Value overrides)
//   SdLoraMatch.hpp/.cpp — matchLoraKey + matchExternalLoraKey
//
// This file is intentionally empty; it is kept so that the split is
// visible in git history via the file rename / content change.

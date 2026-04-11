// Implementations split into focused translation units:
//   OnnxParser.cpp  — parseTensorIndex (protobuf parsing + diagnostics)
//   OnnxIndex.cpp   — buildSuffixIndex (O(1) suffix lookup table)
//   LoraParser.cpp  — parseLoraLayers  (kohya key grouping)
//   LoraMath.cpp    — computeLoraDelta (matmul + scale)
//   OnnxPatcher.cpp — applyLoraToBytes (copy + patch loop)
// SdLoraMatch.hpp/.cpp — matchLoraKey (already a separate module)
//
// This file is intentionally empty; it is kept so that the split is
// visible in git history via the file rename / content change.

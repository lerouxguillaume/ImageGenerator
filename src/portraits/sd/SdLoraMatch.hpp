#pragma once
#include "SdOnnxPatcher.hpp"
#include <string>

namespace sd {

// Returns a pointer to the ExternalTensorMeta that best matches the given LoRA
// base name (kohya prefix already stripped, e.g. "down_blocks_0_attn1_to_q").
// Tries "<loraBase>_weight" first, then "<loraBase>_bias".
// Picks the longest suffix match when multiple candidates exist.
// Logs a warning when two candidates share the same suffix length (true ambiguity).
// Returns nullptr if no match is found.
const ExternalTensorMeta* matchExternalLoraKey(const OnnxExternalSuffixIndex& suffixIndex,
                                               const std::string&             loraBase);

} // namespace sd

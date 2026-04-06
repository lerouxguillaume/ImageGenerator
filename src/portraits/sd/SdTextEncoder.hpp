#pragma once
#include "SdTypes.hpp"
#include "../ClipTokenizer.hpp"
#include <vector>
#include <string>

namespace sd {

// SD 1.5: single CLIP-L encoder → fp32 embedding + shape.
std::vector<float> encodeText(const std::string& prompt,
                              ClipTokenizer& tokenizer,
                              GenerationContext& ctx,
                              std::vector<int64_t>& out_shape);

// SDXL: CLIP-L + OpenCLIP-G, hidden states concatenated → (1, 77, 2048).
// out_pooled receives the pooled output of encoder 2: (1, 1280).
std::vector<float> encodeTextSDXL(const std::string& prompt,
                                  ClipTokenizer& tokenizer,
                                  GenerationContext& ctx,
                                  std::vector<int64_t>& out_shape,
                                  std::vector<float>& out_pooled);

} // namespace sd
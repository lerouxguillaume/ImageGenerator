#pragma once
#include "SdTypes.hpp"
#include <vector>

namespace sd {

// Single batch=1 UNet pass. Returns raw eps output.
// pooled_embed: SDXL text_embeds (1,1280); pass empty vector for SD 1.5.
std::vector<float> runUNetSingle(const std::vector<float>& x_t,
                                 int t,
                                 const std::vector<float>& embed,
                                 const std::vector<float>& pooled_embed,
                                 GenerationContext& ctx);

// Two batch=1 passes with CFG blending:
//   eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
// Two separate passes (not batch=2) keep static batch size 1, required for DML.
std::vector<float> runUNetCFG(const std::vector<float>& x_t,
                              float sigma,
                              const std::vector<float>& alphas_cumprod,
                              GenerationContext& ctx);

} // namespace sd
#pragma once
#include "SdTypes.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace sd {

// Decode a denoised latent tensor to an OpenCV BGR Mat.
// The exported VAE handles latent unscaling internally via the model's
// configured VAE scaling factor.
// Uses Ort::RunOptions{nullptr} (not ctx.run_opts) so it is unaffected by
// SetTerminate() from the cancellation watcher.
cv::Mat decodeLatent(const std::vector<float>& x, GenerationContext& ctx);

// Encode a BGR image to a latent vector using the VAE encoder.
// img is resized to (cfg_w × cfg_h), converted BGR→RGB, normalised to [-1,1],
// and transposed to CHW before inference. The encoder output has shape
// [1, 8, H/8, W/8]: the first 4 channels are the mean; the remaining 4 are
// log-variance. With sample=false (default) the posterior mean is returned —
// deterministic, preferred for img2img. With sample=true, z = mean + std*eps.
// Result is scaled by ctx.vaeScalingFactor before return.
// Requires ctx.vaeEncoderAvailable — caller must check before calling.
std::vector<float> encodeImage(const cv::Mat& img,
                                int cfg_w, int cfg_h,
                                GenerationContext& ctx,
                                bool sample = false);

} // namespace sd

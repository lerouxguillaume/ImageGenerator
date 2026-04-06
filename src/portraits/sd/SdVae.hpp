#pragma once
#include "SdTypes.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace sd {

// Decode a denoised latent tensor to an OpenCV BGR Mat.
// The exported VAE handles the 0.18215 unscaling internally.
// Uses Ort::RunOptions{nullptr} (not ctx.run_opts) so it is unaffected by
// SetTerminate() from the cancellation watcher.
cv::Mat decodeLatent(const std::vector<float>& x, GenerationContext& ctx);

} // namespace sd
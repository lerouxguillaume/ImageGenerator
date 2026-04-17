#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "../../managers/Logger.hpp"

namespace sd {

// ── Timing ────────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;

inline std::string fmtMs(Clock::time_point start) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start).count();
    return std::to_string(ms) + " ms";
}

// ── Type conversion ───────────────────────────────────────────────────────────

inline std::vector<Ort::Float16_t> toFp16(const std::vector<float>& src) {
    std::vector<Ort::Float16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = Ort::Float16_t(src[i]);
    return dst;
}

// ── RNG ───────────────────────────────────────────────────────────────────────

// Box-Muller transform — diffusion models require N(0,1) latent initialisation.
inline float randNormal() {
    const float u1 = (static_cast<float>(rand()) + 1.0f) / (static_cast<float>(RAND_MAX) + 2.0f);
    const float u2 = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * static_cast<float>(M_PI) * u2);
}

// seed < 0 → generate randomly; otherwise use the provided value.
inline void seedRng(int64_t seed = -1) {
    uint32_t s = (seed < 0)
        ? std::random_device{}()
        : static_cast<uint32_t>(seed);
    srand(s);
    Logger::info("RNG seed: " + std::to_string(s));
}

// ── Image conversion ──────────────────────────────────────────────────────────

// Convert a CHW float32 buffer (RGB, nominally [-1,1]) to an OpenCV BGR Mat.
inline cv::Mat latentToImage(const float* img_data, int img_w, int img_h) {
    const int plane = img_w * img_h;
    for (int c = 0; c < 3; ++c) {
        float mn = 1e9f, mx = -1e9f;
        for (int j = 0; j < plane; ++j) {
            float v = img_data[c * plane + j];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        Logger::info("VAE ch[" + std::to_string(c) + "] range: ["
                     + std::to_string(mn) + ", " + std::to_string(mx) + "]");
    }
    Logger::info("latentToImage: creating Mat " + std::to_string(img_w) + "x" + std::to_string(img_h));
    cv::Mat img(img_h, img_w, CV_8UC3);
    Logger::info("latentToImage: starting pixel loop");
    for (int y = 0; y < img_h; ++y)
        for (int x = 0; x < img_w; ++x)
            for (int c = 0; c < 3; ++c) {
                float val = img_data[c * plane + y * img_w + x];
                val = std::min(std::max((val + 1.0f) / 2.0f, 0.0f), 1.0f) * 255.0f;
                img.at<cv::Vec3b>(y, x)[c] = static_cast<uint8_t>(val);
            }
    Logger::info("latentToImage: pixel loop done, calling cvtColor");
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    Logger::info("latentToImage: cvtColor done, returning");
    return img;
}

} // namespace sd
#include "SdVae.hpp"
#include "SdUtils.hpp"
#include "../../managers/Logger.hpp"
#include <algorithm>
#include <cmath>

namespace sd {

cv::Mat decodeLatent(const std::vector<float>& x, GenerationContext& ctx) {
    {
        float vMin = 1e9f, vMax = -1e9f, vSum = 0.0f;
        for (float v : x) { vMin = std::min(vMin, v); vMax = std::max(vMax, v); vSum += v; }
        Logger::info("VAE input latent — min: " + std::to_string(vMin)
                     + "  max: " + std::to_string(vMax)
                     + "  mean: " + std::to_string(vSum / static_cast<float>(x.size())));
    }

    // Select dtype based on what the exported VAE expects (do not hard-code fp16).
    std::vector<Ort::Float16_t> vae_latent_fp16;
    std::vector<float>          vae_latent_fp32;
    Ort::Value vae_input{nullptr};
    if (ctx.vaeExpectsFp32) {
        vae_latent_fp32 = x;
        vae_input = Ort::Value::CreateTensor<float>(
            ctx.memory_info, vae_latent_fp32.data(), vae_latent_fp32.size(),
            ctx.latent_shape.data(), ctx.latent_shape.size());
    } else {
        vae_latent_fp16 = toFp16(x);
        vae_input = Ort::Value::CreateTensor<Ort::Float16_t>(
            ctx.memory_info, vae_latent_fp16.data(), vae_latent_fp16.size(),
            ctx.latent_shape.data(), ctx.latent_shape.size());
    }

    const char* vae_in_names[]  = {ctx.vae_in.c_str()};
    const char* vae_out_names[] = {ctx.vae_out.c_str()};
    auto tVae = Clock::now();
    Logger::info("VAE decoding latent → image...");
    // Intentionally use RunOptions{nullptr}, not ctx.run_opts, so that
    // SetTerminate() from the cancel watcher does not abort the VAE.
    auto vae_out = ctx.vae_decoder.Run(Ort::RunOptions{nullptr},
                                       vae_in_names, &vae_input, 1, vae_out_names, 1);
    Logger::info("VAE decode done in " + fmtMs(tVae));

    auto& vae_tensor  = vae_out.front();
    auto  shape_info  = vae_tensor.GetTensorTypeAndShapeInfo();
    auto  shape       = shape_info.GetShape();
    int   img_h       = static_cast<int>(shape[2]);
    int   img_w       = static_cast<int>(shape[3]);
    size_t elem_count = shape_info.GetElementCount();
    auto   elem_type  = shape_info.GetElementType();
    Logger::info("VAE output shape: [1, 3, " + std::to_string(img_h) + ", " + std::to_string(img_w) + "]"
                 + "  dtype=" + ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? "fp16" : "fp32"));

    std::vector<float> img_float(elem_count);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* raw = vae_tensor.GetTensorData<Ort::Float16_t>();
        for (size_t j = 0; j < elem_count; ++j)
            img_float[j] = static_cast<float>(raw[j]);
    } else {
        const auto* raw = vae_tensor.GetTensorData<float>();
        std::copy(raw, raw + elem_count, img_float.begin());
    }

    {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (float v : img_float) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        Logger::info("VAE output stats: min=" + std::to_string(mn)
                     + "  max=" + std::to_string(mx)
                     + "  mean=" + std::to_string(sum / static_cast<float>(elem_count)));
    }

    return latentToImage(img_float.data(), img_w, img_h);
}


std::vector<float> encodeImage(const cv::Mat& img,
                                int cfg_w, int cfg_h,
                                GenerationContext& ctx,
                                bool sample) {
    // Resize and convert BGR→RGB
    cv::Mat resized;
    cv::resize(img, resized, {cfg_w, cfg_h}, 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // HWC uint8 → CHW float32 normalised to [-1, 1]
    const int plane = cfg_w * cfg_h;
    std::vector<float> chw(3 * plane);
    for (int y = 0; y < cfg_h; ++y)
        for (int x = 0; x < cfg_w; ++x)
            for (int c = 0; c < 3; ++c)
                chw[c * plane + y * cfg_w + x] =
                    (static_cast<float>(resized.at<cv::Vec3b>(y, x)[c]) / 127.5f) - 1.0f;

    const std::vector<int64_t> shape = {1, 3, cfg_h, cfg_w};

    std::vector<Ort::Float16_t> chw_fp16;
    Ort::Value enc_input{nullptr};
    if (ctx.vaeEncoderExpectsFp32) {
        enc_input = Ort::Value::CreateTensor<float>(
            ctx.memory_info, chw.data(), chw.size(), shape.data(), shape.size());
    } else {
        chw_fp16 = toFp16(chw);
        enc_input = Ort::Value::CreateTensor<Ort::Float16_t>(
            ctx.memory_info, chw_fp16.data(), chw_fp16.size(), shape.data(), shape.size());
    }

    const char* enc_in_names[]  = {ctx.vae_enc_in.c_str()};
    const char* enc_out_names[] = {ctx.vae_enc_out.c_str()};
    Logger::info("VAE encoding image " + std::to_string(cfg_w) + "x" + std::to_string(cfg_h) + "...");
    auto tEnc = Clock::now();
    auto enc_out = ctx.vae_encoder.Run(Ort::RunOptions{nullptr},
                                        enc_in_names, &enc_input, 1, enc_out_names, 1);
    Logger::info("VAE encode done in " + fmtMs(tEnc));

    auto& enc_tensor = enc_out.front();
    auto  type_info  = enc_tensor.GetTensorTypeAndShapeInfo();
    auto  out_shape  = type_info.GetShape();
    const size_t total   = type_info.GetElementCount();
    const size_t latent_size = total / 2;   // first half = mean, second half = logvar
    const auto   elem_type  = type_info.GetElementType();

    std::vector<float> params_f(total);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* raw = enc_tensor.GetTensorData<Ort::Float16_t>();
        for (size_t j = 0; j < total; ++j)
            params_f[j] = static_cast<float>(raw[j]);
    } else {
        const auto* raw = enc_tensor.GetTensorData<float>();
        std::copy(raw, raw + total, params_f.begin());
    }

    std::vector<float> latent(latent_size);
    if (!sample) {
        // Use posterior mean directly (deterministic).
        std::copy(params_f.begin(), params_f.begin() + static_cast<ptrdiff_t>(latent_size),
                  latent.begin());
    } else {
        // Sample: z = mean + std * eps, std = exp(0.5 * logvar), clamped as in diffusers.
        for (size_t j = 0; j < latent_size; ++j) {
            float mean   = params_f[j];
            float logvar = std::max(-30.0f, std::min(params_f[latent_size + j], 20.0f));
            float std_   = std::exp(0.5f * logvar);
            latent[j]    = mean + std_ * randNormal();
        }
    }

    // Scale by the VAE encoder constant (matches diffusers DiagonalGaussianDistribution).
    for (float& v : latent) v *= 0.18215f;

    {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (float v : latent) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        Logger::info("Encoded latent — min: " + std::to_string(mn)
                     + "  max: " + std::to_string(mx)
                     + "  mean: " + std::to_string(sum / static_cast<float>(latent_size)));
    }
    return latent;
}

} // namespace sd
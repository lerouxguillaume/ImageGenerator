#include "SdVae.hpp"
#include "SdUtils.hpp"
#include "../../managers/Logger.hpp"

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

} // namespace sd
// computeLoraDelta: matmul of LoRA up/down weights scaled to a delta tensor.
#include "SdOnnxPatcher.hpp"

namespace sd {

std::vector<float> computeLoraDelta(const SafeTensor& up,
                                    const SafeTensor& down,
                                    float             effectiveScale) {
    const int64_t rank    = down.shape[0];
    int64_t       in_feat = 1;
    for (size_t d = 1; d < down.shape.size(); ++d) in_feat *= down.shape[d];
    const int64_t out_feat = up.shape[0];

    std::vector<float> delta(static_cast<size_t>(out_feat * in_feat), 0.0f);
    const float* upData   = up.data.data();
    const float* downData = down.data.data();

    for (int64_t o = 0; o < out_feat; ++o)
        for (int64_t r = 0; r < rank; ++r) {
            const float u = upData[o * rank + r];
            if (u == 0.0f) continue;
            for (int64_t i = 0; i < in_feat; ++i)
                delta[static_cast<size_t>(o * in_feat + i)] += u * downData[r * in_feat + i];
        }
    for (float& v : delta) v *= effectiveScale;
    return delta;
}

} // namespace sd

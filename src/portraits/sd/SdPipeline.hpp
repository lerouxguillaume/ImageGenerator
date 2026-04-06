#pragma once
#include "SdTypes.hpp"
#include "../PortraitGeneratorAi.hpp"
#include <atomic>
#include <string>
#include <vector>

namespace sd {

// Full generation pipeline: model load → text encode → denoise → VAE decode → save.
// Called by both PortraitGeneratorAi::generatePortrait and generateFromPrompt.
void runPipeline(const std::string& prompt,
                 const std::string& neg_prompt,
                 const std::string& outputPath,
                 const GenerationParams& params,
                 const std::string& modelDir,
                 std::atomic<int>*  progressStep,
                 std::atomic<int>*  currentImage,
                 std::atomic<bool>* cancelToken);

} // namespace sd
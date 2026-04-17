#pragma once
#include "SdTypes.hpp"
#include "../PortraitGeneratorAi.hpp"
#include <atomic>
#include <stop_token>
#include <string>
#include <vector>

namespace sd {

// Full generation pipeline: model load → text encode → denoise → VAE decode → save.
// Called by both PortraitGeneratorAi::generatePortrait and generateFromPrompt.
// stopToken: each jthread owns its own stop_source; request_stop() fires SetTerminate()
// immediately via std::stop_callback, with no polling delay.
void runPipeline(const std::string& prompt,
                 const std::string& neg_prompt,
                 const std::string& outputPath,
                 const GenerationParams& params,
                 const std::string& modelDir,
                 std::atomic<int>*  progressStep,
                 std::atomic<int>*  currentImage,
                 std::stop_token    stopToken = {});

} // namespace sd
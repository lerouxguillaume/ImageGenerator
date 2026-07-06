#include "PortraitGeneratorAi.hpp"
#include "sd/SdPipeline.hpp"
#include <string>

void PortraitGeneratorAi::generateFromPrompt(const std::string& prompt,
                                             const std::string& negativePrompt,
                                             const std::string& outputPath,
                                             const GenerationParams& params,
                                             const std::string& modelDir,
                                             std::atomic<int>*  progressStep,
                                             std::atomic<int>*  currentImage,
                                             std::stop_token    stopToken,
                                             std::atomic<GenerationStage>* stage) {
    sd::runPipeline(prompt, negativePrompt, outputPath, params, modelDir,
                    progressStep, currentImage, std::move(stopToken), stage);
}
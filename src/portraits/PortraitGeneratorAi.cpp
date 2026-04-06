#include "PortraitGeneratorAi.hpp"
#include "PromptBuilder.hpp"
#include "sd/SdPipeline.hpp"
#include <chrono>
#include <string>

void PortraitGeneratorAi::generatePortrait(const Race race,
                                           const Gender gender,
                                           const GenerationParams& params,
                                           std::atomic<int>* progressStep) {
    const std::string prompt     = buildCharacterPrompt(race, gender).build();
    const std::string neg_prompt = buildNegativePrompt(race, gender).build();
    auto now       = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    const std::string outputPath = "assets/generated/portrait_" + std::to_string(timestamp) + ".png";
    sd::runPipeline(prompt, neg_prompt, outputPath, params, "models", progressStep, nullptr, nullptr);
}

void PortraitGeneratorAi::generateFromPrompt(const std::string& prompt,
                                             const std::string& negativePrompt,
                                             const std::string& outputPath,
                                             const GenerationParams& params,
                                             const std::string& modelDir,
                                             std::atomic<int>*  progressStep,
                                             std::atomic<int>*  currentImage,
                                             std::atomic<bool>* cancelToken) {
    sd::runPipeline(prompt, negativePrompt, outputPath, params, modelDir,
                    progressStep, currentImage, cancelToken);
}
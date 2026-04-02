#pragma once
#include <atomic>
#include <string>

#include "../enum/enums.hpp"

struct GenerationParams {
    int   numSteps      = 20;
    float guidanceScale = 8.0f;
    int   numImages     = 1;
};

class PortraitGeneratorAi {
public:
    static void generatePortrait(const Race race,
                                 const Gender gender,
                                 const GenerationParams& params,
                                 std::atomic<int>* progressStep = nullptr);

    static void generateFromPrompt(const std::string& prompt,
                                   const std::string& negativePrompt,
                                   const std::string& outputPath,
                                   const GenerationParams& params,
                                   const std::string& modelDir = "models",
                                   std::atomic<int>* progressStep = nullptr,
                                   std::atomic<int>* currentImage = nullptr,
                                   std::atomic<bool>* cancelToken = nullptr);
};

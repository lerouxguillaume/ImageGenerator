#pragma once
#include <atomic>
#include <string>
#include "../entities/Character.hpp"

struct GenerationParams {
    int   numSteps      = 20;
    float guidanceScale = 8.0f;
};

class PortraitGeneratorAi {
public:
    static void generatePortrait(const Character& character,
                                 const GenerationParams& params,
                                 std::atomic<int>* progressStep = nullptr);

    static void generateFromPrompt(const std::string& prompt,
                                   const std::string& negativePrompt,
                                   const std::string& outputPath,
                                   const GenerationParams& params,
                                   std::atomic<int>* progressStep = nullptr,
                                   std::atomic<bool>* cancelToken = nullptr);
};

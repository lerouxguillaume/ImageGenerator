#pragma once

#include "../enum/enums.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"

#include <atomic>
#include <functional>
#include <stop_token>
#include <string>
#include <vector>

struct GenerationJob {
    std::string prompt;
    std::string negativePrompt;
    std::string outputPath;
    GenerationParams params;
    std::string modelDir;
    bool vaeEncoderAvailable = true;
    bool loraCompatible      = true;
};

struct GenerationProgress {
    std::atomic<int>*              step         = nullptr;
    std::atomic<int>*              currentImage = nullptr;
    std::atomic<GenerationStage>*  stage        = nullptr;
};

struct GenerationResult {
    std::vector<std::string> rawPaths;
};

struct GenerationCallbacks {
    std::function<void(GenerationResult)> onResult;
    std::function<void(std::string)>      onError;
};

class GenerationService {
public:
    void run(const GenerationJob& job,
             GenerationProgress progress,
             GenerationCallbacks callbacks,
             std::stop_token stopToken) const;
};

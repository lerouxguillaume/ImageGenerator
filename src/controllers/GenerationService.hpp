#pragma once

#include "../projects/Project.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"

#include <atomic>
#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>

struct GenerationReferenceInput {
    bool enabled = false;
    std::string imagePath;
    float structureStrength = 0.0f;
    int canvasWidth = 1;
    int canvasHeight = 1;
    std::filesystem::path cacheDir;
};

struct GenerationPostProcessSpec {
    bool assetMode = false;
    bool requiresTransparency = false;
    std::filesystem::path processedDir;
    AssetExportSpec exportSpec;
    GenerationReferenceInput reference;
};

struct GenerationJob {
    std::string prompt;
    std::string negativePrompt;
    std::string outputPath;
    GenerationParams params;
    std::string modelDir;
    GenerationPostProcessSpec postProcess;
};

struct GenerationResult {
    std::vector<std::string> rawPaths;
    bool referenceUsed = false;
};

class GenerationService {
public:
    GenerationResult run(const GenerationJob& job,
                         std::atomic<int>* progressStep,
                         std::atomic<int>* currentImage,
                         std::stop_token stopToken) const;
};

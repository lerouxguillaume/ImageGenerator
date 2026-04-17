#pragma once
#include <cstdint>
#include <string>

struct Preset {
    std::string id;
    std::string name;
    std::string basePrompt;
    std::string negativePrompt;
    int         steps     = 20;
    float       cfg       = 7.0f;
    std::string modelId;   // matches availableModels entry (model folder name)
    int         width     = 0;
    int         height    = 0;
    uint64_t    createdAt = 0; // Unix seconds
};

// Snapshot of current UI state used as input for preset operations.
// Intentionally decoupled from ImageGeneratorView to avoid circular includes.
struct GenerationSettings {
    std::string prompt;
    std::string negativePrompt;
    std::string modelId;
    int         steps  = 20;
    float       cfg    = 7.0f;
    int         width  = 0;
    int         height = 0;
    std::string presetId; // empty if no preset is active
};

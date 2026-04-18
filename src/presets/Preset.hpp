#pragma once
#include <cstdint>
#include <string>
#include "../prompt/Prompt.hpp"

struct Preset {
    std::string id;
    std::string name;
    Prompt      dsl;       // source of truth for prompts
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
    Prompt      dsl;
    std::string modelId;
    int         steps  = 20;
    float       cfg    = 7.0f;
    int         width  = 0;
    int         height = 0;
    std::string presetId; // empty if no preset is active
};

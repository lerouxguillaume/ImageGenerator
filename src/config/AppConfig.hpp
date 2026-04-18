#pragma once
#include <map>
#include <string>
#include <vector>

struct LoraEntry {
    std::string path;       // path to .safetensors file (relative to working dir)
    float       scale = 1.0f;

    bool operator==(const LoraEntry& o) const { return path == o.path && scale == o.scale; }
};

struct PromptEnhancerConfig {
    bool        enabled  = false;
    std::string modelDir;           // path to the ort-genai model directory
};

// Per-model overrides. Any field left at its zero value falls back to the
// corresponding global default in AppConfig.
struct ModelDefaults {
    std::string positivePrompt;       // empty  → use global default
    std::string negativePrompt;       // empty  → use global default
    int         numSteps      = 0;    // 0      → use global default
    float       guidanceScale = 0.f;  // 0      → use global default
    std::string llmHint;              // plain-English style hint for the LLM enhancer
                                      // empty  → fall back to positivePrompt as style example
    std::vector<LoraEntry>   loras;           // LoRA adapters (empty = none)
    std::vector<std::string> qualityBoosters; // tokens appended as normal DSL tokens at generation time
};

// Persisted application settings. Saved to / loaded from config.json in the
// working directory. All paths are relative to the working directory unless
// the user enters an absolute path.
struct AppConfig {
    std::string modelBaseDir = "models";           // Root directory scanned for model subdirectories
    std::string outputDir    = "assets/generated"; // Directory where generated images are written
    std::string loraBaseDir  = "loras";            // Directory scanned for .safetensors LoRA files

    int   defaultNumSteps      = 25;
    float defaultGuidanceScale = 7.0f;

    // Per-model overrides, keyed by model folder name (e.g. "anything_v5").
    // Missing fields fall back to the global defaults above.
    std::map<std::string, ModelDefaults> modelConfigs;

    // Offline LLM prompt enhancer. Disabled by default; set enabled=true and
    // point modelDir at an ort-genai compatible model (e.g. Phi-3 Mini ONNX).
    PromptEnhancerConfig promptEnhancer;

    // Load from configPath. Returns defaults silently if the file is absent or malformed.
    static AppConfig load(const std::string& configPath = "config.json");

    // Persist current values to configPath.
    void save(const std::string& configPath = "config.json") const;
};

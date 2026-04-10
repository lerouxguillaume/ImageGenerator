#pragma once
#include <map>
#include <string>
#include <vector>

struct LoraEntry {
    std::string path;       // path to .safetensors file (relative to working dir)
    float       scale = 1.0f;
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
    std::vector<LoraEntry> loras;     // LoRA adapters to apply to this model (empty = none)
};

// Persisted application settings. Saved to / loaded from config.json in the
// working directory. All paths are relative to the working directory unless
// the user enters an absolute path.
struct AppConfig {
    std::string modelBaseDir = "models";           // Root directory scanned for model subdirectories
    std::string outputDir    = "assets/generated"; // Directory where generated images are written
    std::string loraBaseDir  = "loras";            // Directory scanned for .safetensors LoRA files

    // Generation defaults — applied when the image generator screen first opens.
    // Edit config.json to change them permanently; the sliders/fields stay editable per-session.
    std::string defaultPositivePrompt =
        "masterpiece, best quality, highly detailed, 1girl, beautiful face, "
        "fantasy character, portrait, upper body, solo, cinematic lighting";

    std::string defaultNegativePrompt =
        "worst quality, low quality, bad anatomy, bad hands, extra fingers, "
        "missing fingers, fused fingers, too many fingers, mutated hands, "
        "poorly drawn hands, extra arms, missing arms, extra limbs, "
        "malformed limbs, disconnected limbs, floating limbs, deformed, "
        "mutation, gross proportions, long neck, ugly, blurry, artifacts, "
        "watermark, signature";

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

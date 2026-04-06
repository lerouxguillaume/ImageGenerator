#pragma once
#include <string>

// Persisted application settings. Saved to / loaded from config.json in the
// working directory. All paths are relative to the working directory unless
// the user enters an absolute path.
struct AppConfig {
    std::string modelBaseDir = "models";          // Root directory scanned for model subdirectories
    std::string outputDir    = "assets/generated"; // Directory where generated images are written

    // Load from configPath. Returns defaults silently if the file is absent or malformed.
    static AppConfig load(const std::string& configPath = "config.json");

    // Persist current values to configPath.
    void save(const std::string& configPath = "config.json") const;
};

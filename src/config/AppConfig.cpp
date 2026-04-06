#include "AppConfig.hpp"
#include "../managers/Logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

AppConfig AppConfig::load(const std::string& configPath) {
    AppConfig cfg;
    try {
        std::ifstream f(configPath);
        if (!f.is_open()) {
            Logger::info("config.json not found — using defaults");
            return cfg;
        }
        const auto j = nlohmann::json::parse(f);
        cfg.modelBaseDir = j.value("modelBaseDir", cfg.modelBaseDir);
        cfg.outputDir    = j.value("outputDir",    cfg.outputDir);
        Logger::info("Config loaded: modelBaseDir=" + cfg.modelBaseDir
                     + "  outputDir=" + cfg.outputDir);
    } catch (const std::exception& e) {
        Logger::info(std::string("Config parse error, using defaults: ") + e.what());
    }
    return cfg;
}

void AppConfig::save(const std::string& configPath) const {
    try {
        nlohmann::json j;
        j["modelBaseDir"] = modelBaseDir;
        j["outputDir"]    = outputDir;
        std::ofstream f(configPath);
        f << j.dump(4);
        Logger::info("Config saved to " + configPath);
    } catch (const std::exception& e) {
        Logger::info(std::string("Config save error: ") + e.what());
    }
}

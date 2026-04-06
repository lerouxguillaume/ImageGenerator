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
        cfg.modelBaseDir            = j.value("modelBaseDir",            cfg.modelBaseDir);
        cfg.outputDir               = j.value("outputDir",               cfg.outputDir);
        cfg.defaultPositivePrompt   = j.value("defaultPositivePrompt",   cfg.defaultPositivePrompt);
        cfg.defaultNegativePrompt   = j.value("defaultNegativePrompt",   cfg.defaultNegativePrompt);
        cfg.defaultNumSteps         = j.value("defaultNumSteps",         cfg.defaultNumSteps);
        cfg.defaultGuidanceScale    = j.value("defaultGuidanceScale",    cfg.defaultGuidanceScale);
        if (j.contains("modelConfigs")) {
            for (const auto& [key, val] : j["modelConfigs"].items()) {
                ModelDefaults md;
                md.positivePrompt = val.value("positivePrompt", "");
                md.negativePrompt = val.value("negativePrompt", "");
                md.numSteps       = val.value("numSteps",       0);
                md.guidanceScale  = val.value("guidanceScale",  0.f);
                cfg.modelConfigs[key] = md;
            }
        }
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
        j["modelBaseDir"]          = modelBaseDir;
        j["outputDir"]             = outputDir;
        j["defaultPositivePrompt"] = defaultPositivePrompt;
        j["defaultNegativePrompt"] = defaultNegativePrompt;
        j["defaultNumSteps"]       = defaultNumSteps;
        j["defaultGuidanceScale"]  = defaultGuidanceScale;
        nlohmann::json mcj = nlohmann::json::object();
        for (const auto& [key, md] : modelConfigs) {
            nlohmann::json entry = nlohmann::json::object();
            if (!md.positivePrompt.empty()) entry["positivePrompt"] = md.positivePrompt;
            if (!md.negativePrompt.empty()) entry["negativePrompt"] = md.negativePrompt;
            if (md.numSteps > 0)            entry["numSteps"]       = md.numSteps;
            if (md.guidanceScale > 0.f)     entry["guidanceScale"]  = md.guidanceScale;
            mcj[key] = entry;
        }
        j["modelConfigs"] = mcj;
        std::ofstream f(configPath);
        f << j.dump(4);
        Logger::info("Config saved to " + configPath);
    } catch (const std::exception& e) {
        Logger::info(std::string("Config save error: ") + e.what());
    }
}

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
        cfg.loraBaseDir             = j.value("loraBaseDir",             cfg.loraBaseDir);
        cfg.defaultNumSteps         = j.value("defaultNumSteps",         cfg.defaultNumSteps);
        cfg.defaultGuidanceScale    = j.value("defaultGuidanceScale",    cfg.defaultGuidanceScale);
        if (j.contains("modelConfigs")) {
            for (const auto& [key, val] : j["modelConfigs"].items()) {
                ModelDefaults md;
                md.positivePrompt = val.value("positivePrompt", "");
                md.negativePrompt = val.value("negativePrompt", "");
                md.numSteps       = val.value("numSteps",       0);
                md.guidanceScale  = val.value("guidanceScale",  0.f);
                md.llmHint        = val.value("llmHint",        "");
                if (val.contains("qualityBoosters") && val["qualityBoosters"].is_array()) {
                    for (const auto& b : val["qualityBoosters"])
                        if (b.is_string()) md.qualityBoosters.push_back(b.get<std::string>());
                }
                if (val.contains("loras") && val["loras"].is_array()) {
                    Logger::info("Model '" + key + "' has " + std::to_string(val["loras"].size()) + " LoRA adapter(s) configured.");
                    for (const auto& lo : val["loras"]) {
                        Logger::info("  LoRA entry: " + lo.dump());
                        LoraEntry entry;
                        entry.path  = lo.value("path",  std::string{});
                        entry.scale = lo.value("scale", 1.0f);
                        if (!entry.path.empty())
                            md.loras.push_back(entry);
                    }
                }
                cfg.modelConfigs[key] = md;
            }
        }
        if (j.contains("promptEnhancer")) {
            const auto& pe      = j["promptEnhancer"];
            cfg.promptEnhancer.enabled  = pe.value("enabled",  false);
            cfg.promptEnhancer.modelDir = pe.value("modelDir", std::string{});
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
        j["loraBaseDir"]           = loraBaseDir;
        j["defaultNumSteps"]       = defaultNumSteps;
        j["defaultGuidanceScale"]  = defaultGuidanceScale;
        nlohmann::json mcj = nlohmann::json::object();
        for (const auto& [key, md] : modelConfigs) {
            nlohmann::json entry = nlohmann::json::object();
            if (!md.positivePrompt.empty()) entry["positivePrompt"] = md.positivePrompt;
            if (!md.negativePrompt.empty()) entry["negativePrompt"] = md.negativePrompt;
            if (md.numSteps > 0)            entry["numSteps"]       = md.numSteps;
            if (md.guidanceScale > 0.f)     entry["guidanceScale"]  = md.guidanceScale;
            if (!md.llmHint.empty())        entry["llmHint"]        = md.llmHint;
            if (!md.qualityBoosters.empty()) {
                nlohmann::json bArr = nlohmann::json::array();
                for (const auto& b : md.qualityBoosters) bArr.push_back(b);
                entry["qualityBoosters"] = bArr;
            }
            if (!md.loras.empty()) {
                nlohmann::json lorasArr = nlohmann::json::array();
                for (const auto& lo : md.loras) {
                    nlohmann::json loEntry;
                    loEntry["path"]  = lo.path;
                    loEntry["scale"] = lo.scale;
                    lorasArr.push_back(loEntry);
                }
                entry["loras"] = lorasArr;
            }
            mcj[key] = entry;
        }
        j["modelConfigs"] = mcj;
        j["promptEnhancer"]["enabled"]  = promptEnhancer.enabled;
        j["promptEnhancer"]["modelDir"] = promptEnhancer.modelDir;
        std::ofstream f(configPath);
        f << j.dump(4);
        Logger::info("Config saved to " + configPath);
    } catch (const std::exception& e) {
        Logger::info(std::string("Config save error: ") + e.what());
    }
}

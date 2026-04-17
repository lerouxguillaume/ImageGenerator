#include "PresetManager.hpp"
#include "../managers/Logger.hpp"
#include "../views/ImageGeneratorView.hpp"
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

// ── ID / timestamp helpers ────────────────────────────────────────────────────

static std::string generateId() {
    const auto now = std::chrono::system_clock::now();
    const uint64_t ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());
    return "preset_" + std::to_string(ms);
}

static uint64_t nowSeconds() {
    const auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());
}

// ── JSON serialisation ────────────────────────────────────────────────────────

static nlohmann::json presetToJson(const Preset& p) {
    nlohmann::json j;
    j["id"]             = p.id;
    j["name"]           = p.name;
    j["basePrompt"]     = p.basePrompt;
    j["negativePrompt"] = p.negativePrompt;
    j["steps"]          = p.steps;
    j["cfg"]            = p.cfg;
    j["modelId"]        = p.modelId;
    j["width"]          = p.width;
    j["height"]         = p.height;
    j["createdAt"]      = p.createdAt;
    return j;
}

static Preset presetFromJson(const nlohmann::json& j) {
    Preset p;
    p.id             = j.value("id",             std::string{});
    p.name           = j.value("name",           std::string{});
    p.basePrompt     = j.value("basePrompt",     std::string{});
    p.negativePrompt = j.value("negativePrompt", std::string{});
    p.steps          = j.value("steps",          20);
    p.cfg            = j.value("cfg",            7.0f);
    p.modelId        = j.value("modelId",        std::string{});
    p.width          = j.value("width",          0);
    p.height         = j.value("height",         0);
    p.createdAt      = j.value("createdAt",      uint64_t{0});
    return p;
}

// ── PresetManager ─────────────────────────────────────────────────────────────

PresetManager::PresetManager(std::string filePath)
    : filePath_(std::move(filePath))
{
    load();
}

void PresetManager::load() {
    try {
        std::ifstream f(filePath_);
        if (!f.is_open()) {
            Logger::info("presets.json not found — starting with empty preset list");
            return;
        }
        const auto j = nlohmann::json::parse(f);
        if (!j.is_array()) {
            Logger::error("presets.json is not a JSON array — starting with empty preset list");
            return;
        }
        for (const auto& item : j)
            presets_.push_back(presetFromJson(item));
        Logger::info("Loaded " + std::to_string(presets_.size()) + " preset(s) from " + filePath_);
    } catch (const std::exception& e) {
        Logger::error(std::string("presets.json parse error, starting empty: ") + e.what());
        presets_.clear();
    }
}

void PresetManager::save() const {
    try {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& p : presets_)
            arr.push_back(presetToJson(p));
        std::ofstream f(filePath_);
        f << arr.dump(4);
        Logger::info("Presets saved to " + filePath_);
    } catch (const std::exception& e) {
        Logger::error(std::string("Preset save error: ") + e.what());
    }
}

Preset PresetManager::createFromGeneration(const Generation& gen, const std::string& name) {
    Preset p;
    p.id             = generateId();
    p.name           = name;
    p.basePrompt     = gen.prompt;
    p.negativePrompt = gen.negativePrompt;
    p.steps          = gen.steps;
    p.cfg            = gen.cfg;
    p.modelId        = gen.modelId;
    p.width          = gen.width;
    p.height         = gen.height;
    p.createdAt      = nowSeconds();
    presets_.push_back(p);
    save();
    return p;
}

void PresetManager::updateFromGeneration(const std::string& presetId, const Generation& gen) {
    for (auto& p : presets_) {
        if (p.id == presetId) {
            p.basePrompt     = gen.prompt;
            p.negativePrompt = gen.negativePrompt;
            p.steps          = gen.steps;
            p.cfg            = gen.cfg;
            p.modelId        = gen.modelId;
            p.width          = gen.width;
            p.height         = gen.height;
            // id, name, createdAt are intentionally preserved
            save();
            return;
        }
    }
    Logger::info("updateFromGeneration: preset '" + presetId + "' not found — no-op");
}

Preset PresetManager::duplicatePreset(const std::string& presetId, const std::string& newName) {
    for (const auto& p : presets_) {
        if (p.id == presetId) {
            Preset copy  = p;
            copy.id       = generateId();
            copy.name     = newName;
            copy.createdAt = nowSeconds();
            presets_.push_back(copy);
            save();
            return copy;
        }
    }
    Logger::info("duplicatePreset: preset '" + presetId + "' not found");
    return {};
}

std::optional<Preset> PresetManager::getPreset(const std::string& id) const {
    for (const auto& p : presets_) {
        if (p.id == id)
            return p;
    }
    return std::nullopt;
}

const std::vector<Preset>& PresetManager::getAllPresets() const {
    return presets_;
}

// ── applyPresetToSettings ─────────────────────────────────────────────────────

void applyPresetToSettings(const Preset& preset, ImageGeneratorView& view) {
    view.positiveArea.setText(preset.basePrompt);
    view.negativeArea.setText(preset.negativePrompt);
    view.generationParams.numSteps      = preset.steps;
    view.generationParams.guidanceScale = preset.cfg;
    view.generationParams.width         = preset.width;
    view.generationParams.height        = preset.height;

    bool modelFound = false;
    for (int i = 0; i < static_cast<int>(view.availableModels.size()); ++i) {
        if (view.availableModels[i] == preset.modelId) {
            view.selectedModelIdx = i;
            modelFound = true;
            break;
        }
    }
    if (!modelFound)
        Logger::info("applyPresetToSettings: model '" + preset.modelId
                     + "' not in availableModels — selectedModelIdx unchanged");

    view.activePresetId = preset.id;
}

#include "ImportedModelRegistry.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

// ── Timestamp helper ──────────────────────────────────────────────────────────

static std::string utcNow() {
    const auto now  = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// ── Constructor ───────────────────────────────────────────────────────────────

ImportedModelRegistry::ImportedModelRegistry(std::filesystem::path registryPath)
    : registryPath_(std::move(registryPath))
{
    load();
}

// ── Public interface ──────────────────────────────────────────────────────────

void ImportedModelRegistry::add(const ImportedModel& model) {
    // Replace existing entry with same id
    for (auto& m : models_) {
        if (m.id == model.id) {
            m = model;
            save();
            return;
        }
    }
    models_.push_back(model);
    save();
}

bool ImportedModelRegistry::exists(const std::string& id) const {
    for (const auto& m : models_)
        if (m.id == id) return true;
    return false;
}

// ── Persistence ───────────────────────────────────────────────────────────────

void ImportedModelRegistry::load() {
    std::ifstream f(registryPath_);
    if (!f) return;
    try {
        const auto j = nlohmann::json::parse(f);
        for (const auto& entry : j.value("models", nlohmann::json::array())) {
            ImportedModel m;
            m.id         = entry.value("id",         std::string{});
            m.name       = entry.value("name",       std::string{});
            m.arch       = entry.value("arch",       std::string{});
            m.onnxPath   = entry.value("onnxPath",   std::string{});
            m.importedAt = entry.value("importedAt", std::string{});
            if (!m.id.empty()) models_.push_back(std::move(m));
        }
    } catch (...) {}
}

void ImportedModelRegistry::save() const {
    std::filesystem::create_directories(registryPath_.parent_path());
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& m : models_) {
        arr.push_back({
            {"id",         m.id},
            {"name",       m.name},
            {"arch",       m.arch},
            {"onnxPath",   m.onnxPath.string()},
            {"importedAt", m.importedAt.empty() ? utcNow() : m.importedAt},
        });
    }
    std::ofstream f(registryPath_);
    f << nlohmann::json{{"models", arr}}.dump(2);
}

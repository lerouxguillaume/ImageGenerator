#pragma once
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "SafetensorsInspector.hpp"

class Subprocess;

class ModelImporter {
public:
    enum class State {
        Idle,
        Analyzing,
        SettingUpPython,
        Exporting,
        Validating,
        Done,
        Failed,
    };

    explicit ModelImporter(std::filesystem::path scriptsDir,
                           std::filesystem::path managedModelsDir,
                           std::filesystem::path venvBaseDir);
    ~ModelImporter();

    ModelImporter(const ModelImporter&)            = delete;
    ModelImporter& operator=(const ModelImporter&) = delete;

    // Begin import on a background thread. No-op if already running.
    void start(const std::filesystem::path& safetensorsPath,
               const std::string& archOverride = "auto");

    // Request cancellation. The background thread will stop at the next safe
    // point (or kill the subprocess immediately).
    void cancel();

    // Reset to Idle after Done/Failed so the modal can start a new import.
    void reset();

    bool isRunning() const;

    // ── Thread-safe accessors (polled by the UI each frame) ─────────────────
    State                    getState()      const;
    std::string              getStatusMsg()  const;
    std::vector<std::string> getLogLines()   const; // up to kMaxLogLines

    // Only valid in State::Done
    std::filesystem::path getOutputDir() const;
    std::string           getModelId()   const;

    // Inspection result (available after Analyzing)
    SafetensorsInfo getInspectionResult() const;

    static constexpr size_t kMaxLogLines = 200;

private:
    void runThread(std::filesystem::path path, std::string archOverride);
    void appendLog(const std::string& line);
    void setStatus(const std::string& msg);

    std::filesystem::path scriptsDir_;
    std::filesystem::path managedModelsDir_;
    std::filesystem::path venvBaseDir_;

    std::thread            thread_;
    std::atomic<bool>      cancelRequested_{false};
    std::atomic<State>     state_{State::Idle};

    mutable std::mutex     dataMutex_;
    std::string            statusMsg_;
    std::vector<std::string> logLines_;
    std::filesystem::path  outputDir_;
    std::string            modelId_;
    SafetensorsInfo        inspectionResult_;

    // Shared with thread for kill support
    std::shared_ptr<Subprocess> activeSubprocess_;
    std::mutex                  subprocessMutex_;
};

#pragma once
#include <atomic>
#include <chrono>
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
        Verifying,
        Done,
        Failed,
    };

    // One line from the Python inference smoke test (VERIFY: protocol).
    struct VerifyCheck {
        enum class Status { Ok, Warn, Fail, Skip };
        std::string name;    // e.g. "text_encoder", "unet", "vae_decoder"
        Status      status = Status::Ok;
        std::string detail; // human-readable reason
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

    // Verification results (populated during/after Verifying)
    std::vector<VerifyCheck> getVerifyChecks() const;

    // Seconds since start() was called; frozen once Done/Failed. 0 while Idle.
    double getElapsedSeconds() const;

    // Sub-progress within the Exporting phase, parsed from "N/M" export lines.
    // total == 0 means "unknown" (bar shows the phase midpoint).
    void getExportProgress(int& step, int& total) const;

    static constexpr size_t kMaxLogLines = 200;

private:
    void runThread(std::filesystem::path path, std::string archOverride);
    void appendLog(const std::string& line);
    void setStatus(const std::string& msg);
    void recordVerify(const std::string& payload); // parse "status:name:detail"
    void parseExportStep(const std::string& line); // parse leading "N/M"

    static int64_t nowNs();

    std::filesystem::path scriptsDir_;
    std::filesystem::path managedModelsDir_;
    std::filesystem::path venvBaseDir_;

    std::thread            thread_;
    std::atomic<bool>      cancelRequested_{false};
    std::atomic<State>     state_{State::Idle};

    // Timing (steady_clock epoch nanoseconds). endNs_ frozen on first terminal poll.
    std::atomic<int64_t>   startNs_{0};
    mutable std::atomic<int64_t> endNs_{0};
    mutable std::atomic<bool>    ended_{false};

    // Export sub-progress ("N/M" from export step lines).
    std::atomic<int>       exportStep_{0};
    std::atomic<int>       exportTotal_{0};

    mutable std::mutex     dataMutex_;
    std::string            statusMsg_;
    std::vector<std::string> logLines_;
    std::filesystem::path  outputDir_;
    std::string            modelId_;
    SafetensorsInfo        inspectionResult_;
    std::vector<VerifyCheck> verifyChecks_;

    // Shared with thread for kill support
    std::shared_ptr<Subprocess> activeSubprocess_;
    std::mutex                  subprocessMutex_;
};

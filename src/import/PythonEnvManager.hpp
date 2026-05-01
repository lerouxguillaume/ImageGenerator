#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

// Manages a dedicated Python virtual environment for model export.
// The venv lives at <baseDir>/python_env/ and is created on first use.
// A sentinel file (.setup_complete) marks a fully installed venv so
// subsequent launches skip the setup step entirely.
class PythonEnvManager {
public:
    using LogFn = std::function<void(const std::string&)>;

    explicit PythonEnvManager(std::filesystem::path baseDir,
                               std::filesystem::path requirementsFile);

    // True iff the venv sentinel exists (setup was previously completed).
    bool isReady() const;

    // Create the venv and pip-install requirements (blocking — call from a
    // background thread). logFn receives status lines for display in the UI.
    // Returns true on success.
    bool setup(LogFn logFn);

    // Path to the venv's python binary (platform-correct).
    std::filesystem::path pythonPath() const;

private:
    // Find a usable system Python to bootstrap the venv.
    static std::string findSystemPython();

    // Run a command, forwarding output to logFn and optionally to a log file.
    static int runCommand(const std::vector<std::string>& args, LogFn logFn,
                          std::ofstream* logFile = nullptr);

    std::filesystem::path venvDir_;
    std::filesystem::path requirementsFile_;
    std::filesystem::path sentinelPath_;
    std::filesystem::path logPath_; // full log written here for post-mortem inspection
};

#include "PythonEnvManager.hpp"
#include "Subprocess.hpp"

#include <array>
#include <fstream>

PythonEnvManager::PythonEnvManager(std::filesystem::path baseDir,
                                   std::filesystem::path requirementsFile)
    : venvDir_(baseDir / "python_env")
    , requirementsFile_(std::move(requirementsFile))
    , sentinelPath_(venvDir_ / ".setup_complete")
    , logPath_(baseDir / "python_env_setup.log")
{}

bool PythonEnvManager::isReady() const {
    return std::filesystem::exists(sentinelPath_);
}

std::filesystem::path PythonEnvManager::pythonPath() const {
#ifdef _WIN32
    return venvDir_ / "Scripts" / "python.exe";
#else
    return venvDir_ / "bin" / "python3";
#endif
}

// ── System Python discovery ───────────────────────────────────────────────────

// Returns both the command name and the resolved full path from `where`/`which`.
static std::string resolveFullPath(const std::string& cmd) {
#ifdef _WIN32
    Subprocess sub;
    if (!sub.start({"where", cmd})) return {};
    std::string line;
    std::string first;
    while (sub.readLine(line)) {
        if (first.empty() && !line.empty()) first = line;
    }
    sub.wait();
    return first;
#else
    Subprocess sub;
    if (!sub.start({"which", cmd})) return {};
    std::string line;
    sub.readLine(line);
    while (sub.readLine(line)) {}
    sub.wait();
    return line.empty() ? std::string{} : line;
#endif
}

std::string PythonEnvManager::findSystemPython() {
#ifdef _WIN32
    static const std::array<const char*, 3> candidates{"python", "python3", "py"};
#else
    static const std::array<const char*, 2> candidates{"python3", "python"};
#endif
    for (const char* cmd : candidates) {
        Subprocess sub;
        if (!sub.start({cmd, "--version"})) continue;
        std::string line, version;
        while (sub.readLine(line)) {
            if (version.empty()) version = line;
        }
        if (sub.wait() == 0)
            return cmd;
    }
    return {};
}

// ── Command runner ────────────────────────────────────────────────────────────

int PythonEnvManager::runCommand(const std::vector<std::string>& args,
                                  LogFn logFn,
                                  std::ofstream* logFile) {
    // Log the command being run
    std::string cmdStr;
    for (const auto& a : args) cmdStr += a + " ";
    if (logFn)   logFn("$ " + cmdStr);
    if (logFile) *logFile << "$ " << cmdStr << "\n";

    Subprocess sub;
    if (!sub.start(args)) {
        const std::string msg = "Failed to launch: " + args[0];
        if (logFn)   logFn(msg);
        if (logFile) *logFile << msg << "\n";
        return -1;
    }
    std::string line;
    while (sub.readLine(line)) {
        if (logFn)   logFn(line);
        if (logFile) *logFile << line << "\n";
    }
    const int code = sub.wait();
    const std::string exitMsg = "Exit code: " + std::to_string(code);
    if (logFn)   logFn(exitMsg);
    if (logFile) *logFile << exitMsg << "\n\n";
    return code;
}

// ── Setup ─────────────────────────────────────────────────────────────────────

bool PythonEnvManager::setup(LogFn logFn) {
    if (isReady()) return true;

    std::filesystem::create_directories(venvDir_.parent_path());
    std::ofstream logFile(logPath_, std::ios::app);

    auto log = [&](const std::string& msg) {
        if (logFn)   logFn(msg);
        if (logFile) logFile << msg << "\n";
    };

    log("=== Python environment setup ===");
    log("Venv dir:     " + venvDir_.string());
    log("Requirements: " + requirementsFile_.string());
    log("Log file:     " + logPath_.string());

    // Find Python
    const std::string sysPython = findSystemPython();
    if (sysPython.empty()) {
        log("ERROR:Python 3 not found. Please install Python 3.10 or later.");
        log("Looked for: python, python3"
#ifdef _WIN32
            ", py"
#endif
        );
        return false;
    }

    // Resolve and log the full path so the user can see exactly which Python is used
    const std::string fullPath = resolveFullPath(sysPython);
    log("Python command: " + sysPython +
        (fullPath.empty() ? "" : " -> " + fullPath));

    // Verify Python version
    runCommand({sysPython, "--version"}, logFn, &logFile);

    // Create venv
    log("Creating virtual environment at: " + venvDir_.string());
    if (runCommand({sysPython, "-m", "venv", venvDir_.string()},
                   logFn, &logFile) != 0) {
        log("ERROR:Failed to create virtual environment. See log: " + logPath_.string());
        return false;
    }

    // Upgrade pip
    const std::string pip = pythonPath().string();
    log("Upgrading pip...");
    runCommand({pip, "-m", "pip", "install", "--upgrade", "pip"},
               logFn, &logFile);

    // Install requirements
    log("Installing packages (first run — may take several minutes)...");
    if (runCommand({pip, "-m", "pip", "install", "-r", requirementsFile_.string()},
                   logFn, &logFile) != 0) {
        log("ERROR:Package installation failed. See log: " + logPath_.string());
        return false;
    }

    std::ofstream{sentinelPath_};
    log("Python environment ready.");
    return true;
}

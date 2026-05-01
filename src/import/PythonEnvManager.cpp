#include "PythonEnvManager.hpp"
#include "Subprocess.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>

PythonEnvManager::PythonEnvManager(std::filesystem::path baseDir,
                                   std::filesystem::path requirementsFile)
    : venvDir_(baseDir / "python_env")
    , requirementsFile_(std::move(requirementsFile))
    , sentinelPath_(venvDir_ / ".setup_complete")
    , logPath_(baseDir / "python_env_setup.log")
{}

bool PythonEnvManager::isReady() const {
    if (!std::filesystem::exists(sentinelPath_)) return false;
    std::ifstream f(sentinelPath_);
    std::string existing;
    std::getline(f, existing);
    return existing == requirementsSignature();
}

std::filesystem::path PythonEnvManager::pythonPath() const {
#ifdef _WIN32
    return venvDir_ / "Scripts" / "python.exe";
#else
    return venvDir_ / "bin" / "python3";
#endif
}

std::string PythonEnvManager::requirementsSignature() const {
    std::ifstream f(requirementsFile_, std::ios::binary);
    if (!f) return {};

    // Stable FNV-1a over requirements_import.txt. This intentionally keys only
    // the dependency contract, not installed package state.
    uint64_t h = 14695981039346656037ull;
    char c;
    while (f.get(c)) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }

    std::ostringstream out;
    out << "requirements-fnv1a:" << std::hex << h;
    return out.str();
}

// ── System Python discovery ───────────────────────────────────────────────────

static std::string commandDisplay(const std::vector<std::string>& args) {
    std::string out;
    for (const auto& a : args) {
        if (!out.empty()) out += " ";
        out += a;
    }
    return out;
}

static bool parsePythonVersion(const std::string& line, int& major, int& minor) {
    const std::string prefix = "Python ";
    const size_t pos = line.find(prefix);
    if (pos == std::string::npos) return false;

    size_t i = pos + prefix.size();
    auto parsePart = [&](int& value) {
        if (i >= line.size() || !std::isdigit(static_cast<unsigned char>(line[i])))
            return false;
        value = 0;
        while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) {
            value = value * 10 + (line[i] - '0');
            ++i;
        }
        return true;
    };

    if (!parsePart(major)) return false;
    if (i >= line.size() || line[i] != '.') return false;
    ++i;
    return parsePart(minor);
}

static bool isSupportedPython(const std::vector<std::string>& command) {
    std::vector<std::string> args = command;
    args.push_back("--version");

    Subprocess sub;
    if (!sub.start(args)) return false;

    std::string line;
    std::string versionLine;
    while (sub.readLine(line)) {
        if (versionLine.empty() && !line.empty()) versionLine = line;
    }
    if (sub.wait() != 0) return false;

    int major = 0;
    int minor = 0;
    if (!parsePythonVersion(versionLine, major, minor)) return false;

    // The SD single-file export stack is validated on these versions. Newer
    // Python releases pull newer transformers/diffusers behavior that leaves
    // CLIP weights on the meta device under Windows.
    return major == 3 && minor >= 10 && minor <= 12;
}

#ifdef _WIN32
static bool containsCommand(const std::vector<std::vector<std::string>>& commands,
                            const std::vector<std::string>& command) {
    return std::find(commands.begin(), commands.end(), command) != commands.end();
}

// Returns true for Windows Store App Execution Aliases.
// These live in %LOCALAPPDATA%\Microsoft\WindowsApps\ and are special stubs
// that only work when launched from Explorer or the shell.  CreateProcess with
// pipe redirection fails with WinError 6 (invalid handle) on them.
static bool isAppExecutionAlias(const std::string& path) {
    return path.find("Microsoft\\WindowsApps") != std::string::npos
        || path.find("Microsoft/WindowsApps")  != std::string::npos;
}

// Returns every resolved path for cmd from `where`, in PATH order.
static std::vector<std::string> whereAll(const std::string& cmd) {
    Subprocess sub;
    if (!sub.start({"where", cmd})) return {};
    std::vector<std::string> results;
    std::string line;
    while (sub.readLine(line)) {
        // strip trailing whitespace / CR
        while (!line.empty() && (line.back() == ' ' || line.back() == '\r'))
            line.pop_back();
        if (!line.empty()) results.push_back(line);
    }
    sub.wait();
    return results;
}
#endif

std::vector<std::string> PythonEnvManager::findSystemPython() {
#ifdef _WIN32
    static const std::array<const char*, 2> directCandidates{"python", "python3"};
    for (const char* cmd : directCandidates) {
        for (const auto& fullPath : whereAll(cmd)) {
            if (isAppExecutionAlias(fullPath)) continue; // skip Store stub
            std::vector<std::string> command{fullPath};
            if (isSupportedPython(command)) return command;
        }
    }

    static const std::array<const char*, 4> pyVersions{"-3.12", "-3.11", "-3.10", ""};
    std::vector<std::vector<std::string>> pyCommands;
    pyCommands.push_back({"py"});
    for (const auto& pyPath : whereAll("py")) {
        if (!isAppExecutionAlias(pyPath))
            pyCommands.push_back({pyPath});
    }
    for (const char* pyPath : {"C:\\Windows\\py.exe", "C:\\Windows\\System32\\py.exe"}) {
        std::vector<std::string> command{pyPath};
        if (!containsCommand(pyCommands, command))
            pyCommands.push_back(command);
    }

    for (const auto& pyCommand : pyCommands) {
        for (const char* version : pyVersions) {
            std::vector<std::string> command = pyCommand;
            if (*version) command.push_back(version);
            if (isSupportedPython(command)) return command;
        }
    }
    return {};
#else
    static const std::array<const char*, 2> candidates{"python3", "python"};
    for (const char* cmd : candidates) {
        std::vector<std::string> command{cmd};
        if (isSupportedPython(command)) return command;
    }
    return {};
#endif
}

// ── Command runner ────────────────────────────────────────────────────────────

int PythonEnvManager::runCommand(const std::vector<std::string>& args,
                                  LogFn logFn,
                                  std::ofstream* logFile) {
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

    const std::vector<std::string> sysPython = findSystemPython();
    if (sysPython.empty()) {
        log("ERROR:Supported Python not found.");
        log("Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/");
        log("Python 3.13+ is not supported by this export environment yet.");
        log("(Do NOT use the Microsoft Store version — it cannot be used from this app.)");
        return false;
    }
    log("Python: " + commandDisplay(sysPython));

    std::vector<std::string> versionArgs = sysPython;
    versionArgs.push_back("--version");
    runCommand(versionArgs, logFn, &logFile);

    log("Creating virtual environment at: " + venvDir_.string());
    std::vector<std::string> venvArgs = sysPython;
    venvArgs.insert(venvArgs.end(), {"-m", "venv", venvDir_.string()});
    if (runCommand(venvArgs, logFn, &logFile) != 0) {
        log("ERROR:Failed to create virtual environment. See log: " + logPath_.string());
        return false;
    }

    const std::string pip = pythonPath().string();
    log("Upgrading pip...");
    runCommand({pip, "-m", "pip", "install", "--upgrade", "pip"}, logFn, &logFile);

    log("Installing packages (first run — may take several minutes)...");
    if (runCommand({pip, "-m", "pip", "install", "-r", requirementsFile_.string()},
                   logFn, &logFile) != 0) {
        log("ERROR:Package installation failed. See log: " + logPath_.string());
        return false;
    }

    std::ofstream sentinel(sentinelPath_);
    sentinel << requirementsSignature() << "\n";
    log("Python environment ready.");
    return true;
}

#pragma once
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Cross-platform subprocess with stdout/stderr capture and kill support.
// Intended to be used from a single background thread: start(), then loop
// readLine() until it returns false, then wait() for the exit code.
class Subprocess {
public:
    Subprocess() = default;
    ~Subprocess();

    Subprocess(const Subprocess&)            = delete;
    Subprocess& operator=(const Subprocess&) = delete;

    // Launch process. args[0] is the executable. Returns false on failure.
    bool start(const std::vector<std::string>& args,
               const std::filesystem::path& workDir = {});

    // Block until a '\n'-terminated line is available.
    // Returns false on EOF or read error (process finished).
    bool readLine(std::string& line);

    // Send SIGTERM / TerminateProcess. Safe to call if not running.
    void kill();

    // Block until the process exits. Returns the exit code (-1 on error).
    int wait();

    bool isStarted() const noexcept { return started_; }

private:
#ifdef _WIN32
    HANDLE hProcess_    = INVALID_HANDLE_VALUE;
    HANDLE hStdoutRead_ = INVALID_HANDLE_VALUE;
#else
    int pid_      = -1;
    int stdoutFd_ = -1;
#endif
    bool started_ = false;
    std::string lineBuf_;
};

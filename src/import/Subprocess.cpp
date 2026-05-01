#include "Subprocess.hpp"

#ifdef _WIN32
// ═══════════════════════════════════════════════════════════════════════════
// Windows implementation
// ═══════════════════════════════════════════════════════════════════════════
#include <sstream>

Subprocess::~Subprocess() {
    kill();
    if (hStdoutRead_ != INVALID_HANDLE_VALUE) CloseHandle(hStdoutRead_);
    if (hProcess_    != INVALID_HANDLE_VALUE) CloseHandle(hProcess_);
}

static std::string buildCommandLine(const std::vector<std::string>& args) {
    // Simple quoting: wrap each arg in double quotes, escape inner quotes.
    std::string cmd;
    for (const auto& arg : args) {
        cmd += '"';
        for (char c : arg) {
            if (c == '"') cmd += '\\';
            cmd += c;
        }
        cmd += "\" ";
    }
    return cmd;
}

bool Subprocess::start(const std::vector<std::string>& args,
                       const std::filesystem::path& workDir) {
    if (args.empty()) return false;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;

    HANDLE hWrite = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&hStdoutRead_, &hWrite, &sa, 0)) return false;
    // Don't inherit the read end in the child
    SetHandleInformation(hStdoutRead_, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = hWrite;
    si.hStdError  = hWrite;
    si.hStdInput  = INVALID_HANDLE_VALUE;

    std::string cmd = buildCommandLine(args);

    PROCESS_INFORMATION pi{};
    const char* wd = workDir.empty() ? nullptr : workDir.string().c_str();
    BOOL ok = CreateProcessA(
        nullptr, cmd.data(), nullptr, nullptr,
        TRUE, CREATE_NO_WINDOW, nullptr, wd, &si, &pi);

    CloseHandle(hWrite);

    if (!ok) {
        CloseHandle(hStdoutRead_);
        hStdoutRead_ = INVALID_HANDLE_VALUE;
        return false;
    }

    hProcess_ = pi.hProcess;
    CloseHandle(pi.hThread);
    started_ = true;
    return true;
}

bool Subprocess::readLine(std::string& line) {
    if (hStdoutRead_ == INVALID_HANDLE_VALUE) return false;

    char c;
    DWORD bytesRead;
    while (true) {
        if (!ReadFile(hStdoutRead_, &c, 1, &bytesRead, nullptr) || bytesRead == 0)
            return false;
        if (c == '\r') continue;
        if (c == '\n') {
            line = std::move(lineBuf_);
            lineBuf_.clear();
            return true;
        }
        lineBuf_ += c;
    }
}

void Subprocess::kill() {
    if (hProcess_ != INVALID_HANDLE_VALUE)
        TerminateProcess(hProcess_, 1);
}

int Subprocess::wait() {
    if (hProcess_ == INVALID_HANDLE_VALUE) return -1;
    WaitForSingleObject(hProcess_, INFINITE);
    DWORD code = static_cast<DWORD>(-1);
    GetExitCodeProcess(hProcess_, &code);
    return static_cast<int>(code);
}

#else
// ═══════════════════════════════════════════════════════════════════════════
// POSIX implementation
// ═══════════════════════════════════════════════════════════════════════════
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

Subprocess::~Subprocess() {
    kill();
    if (stdoutFd_ >= 0) ::close(stdoutFd_);
    if (pid_ > 0)       ::waitpid(pid_, nullptr, WNOHANG);
}

bool Subprocess::start(const std::vector<std::string>& args,
                       const std::filesystem::path& workDir) {
    if (args.empty()) return false;

    int pipefd[2];
    if (::pipe(pipefd) != 0) return false;

    pid_t child = ::fork();
    if (child < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return false;
    }

    if (child == 0) {
        // Child: redirect stdout + stderr to write end of pipe
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        if (!workDir.empty())
            ::chdir(workDir.c_str());

        std::vector<char*> argv;
        std::vector<std::string> copy = args;
        for (auto& a : copy) argv.push_back(a.data());
        argv.push_back(nullptr);

        ::execvp(argv[0], argv.data());
        ::_exit(127);
    }

    // Parent
    ::close(pipefd[1]);
    pid_      = child;
    stdoutFd_ = pipefd[0];
    started_  = true;
    return true;
}

bool Subprocess::readLine(std::string& line) {
    if (stdoutFd_ < 0) return false;

    char c;
    while (true) {
        ssize_t n = ::read(stdoutFd_, &c, 1);
        if (n <= 0) return false;
        if (c == '\n') {
            line = std::move(lineBuf_);
            lineBuf_.clear();
            return true;
        }
        if (c != '\r') lineBuf_ += c;
    }
}

void Subprocess::kill() {
    if (pid_ > 0) ::kill(pid_, SIGTERM);
}

int Subprocess::wait() {
    if (pid_ <= 0) return -1;
    int status = 0;
    ::waitpid(pid_, &status, 0);
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return -1;
}

#endif

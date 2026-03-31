#include "app.hpp"
#include "managers/Logger.hpp"

#include <csignal>
#include <exception>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <limits.h>
#else
#include <unistd.h>
#endif

namespace {

std::filesystem::path executableDir() {
#if defined(_WIN32)
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0) {
        throw std::runtime_error("Cannot get executable path");
    }
    return std::filesystem::path(buffer).parent_path();
#else // Linux/Unix
    char buffer[4096];
    ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
    if (len <= 0) throw std::runtime_error("Cannot get executable path");
    buffer[len] = '\0';
    return std::filesystem::path(buffer).parent_path();
#endif
}

const char* getSignalName(int sig) {
    if (sig == SIGSEGV) return "SIGSEGV";
    if (sig == SIGABRT) return "SIGABRT";
    if (sig == SIGFPE)  return "SIGFPE";
    return "unknown signal";
}

void onSignal(int sig) {
    Logger::error("Crash: " + std::string(getSignalName(sig)));
    std::_Exit(1);
}

void onTerminate() {
    try {
        std::rethrow_exception(std::current_exception());
    } catch (const std::exception& e) {
        Logger::error("Unhandled exception: " + std::string(e.what()));
    } catch (...) {
        Logger::error("Unhandled unknown exception");
    }
    std::_Exit(1);
}

}  // namespace

int main() {
    std::filesystem::current_path(executableDir());
    Logger::init("guild_master.log");

    std::set_terminate(onTerminate);
    std::signal(SIGSEGV, onSignal);
    std::signal(SIGABRT, onSignal);
    std::signal(SIGFPE,  onSignal);

    Logger::info("Main started");

    Logger::info("Config loaded");

    App app;
    Logger::info("Application created");
    app.run();
    return 0;
}

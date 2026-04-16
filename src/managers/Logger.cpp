#include "Logger.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

static std::ofstream s_logFile;

static std::string currentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[20]; // "YYYY-MM-DD HH:MM:SS\0"
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

static void writeLine(const std::string& line) {
    if (s_logFile.is_open()) {
        s_logFile << line << "\n";
        s_logFile.flush();
    }
}

void Logger::init(const std::string& path) {
    s_logFile.open(path, std::ios::out | std::ios::trunc);
    writeLine("=== Image generation log — " + currentTimestamp() + " ===");
}

void Logger::info(const std::string& msg) {
    const std::string line = "[" + currentTimestamp() + "][INFO] " + msg;
    std::cout << line << std::endl;
    writeLine(line);
}

void Logger::error(const std::string& msg) {
    const std::string line = "[" + currentTimestamp() + "][ERROR] " + msg;
    std::cerr << line << std::endl;
    writeLine(line);
}

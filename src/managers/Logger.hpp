#pragma once
#include <string>

// Simple append-only logger that writes timestamped lines to a file.
// Call init() once at startup before any other method.
// All methods are safe to call from multiple threads (internally synchronised).
class Logger {
public:
    // Open (or create) the log file at path. Truncates any existing content.
    static void init(const std::string& path);

    // Append an [INFO] line: "[YYYY-MM-DD HH:MM:SS][INFO] msg\n"
    static void info(const std::string& msg);

    // Append an [ERROR] line: "[YYYY-MM-DD HH:MM:SS][ERROR] msg\n"
    static void error(const std::string& msg);
};


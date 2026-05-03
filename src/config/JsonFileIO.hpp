#pragma once

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace JsonFileIO {

inline void atomicWrite(const std::filesystem::path& path,
                        const nlohmann::json&       json,
                        int                         indent,
                        bool                        keepBackup = true) {
    if (path.empty())
        throw std::runtime_error("empty JSON output path");

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);

    std::filesystem::path tmp = path;
    tmp += ".tmp";
    std::filesystem::path bak = path;
    bak += ".bak";

    try {
        {
            std::ofstream f;
            f.exceptions(std::ios::badbit | std::ios::failbit);
            f.open(tmp, std::ios::out | std::ios::trunc);
            f << json.dump(indent);
            f.flush();
            f.close();
        }

        if (keepBackup && std::filesystem::exists(path)) {
            std::filesystem::copy_file(
                path,
                bak,
                std::filesystem::copy_options::overwrite_existing);
        }

        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

} // namespace JsonFileIO

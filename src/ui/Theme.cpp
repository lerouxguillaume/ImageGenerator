#include "Theme.h"
#include <filesystem>
#include <stdexcept>

Theme& Theme::instance() {
    static Theme instance;
    return instance;
}

Theme::Theme() {
    // Try to load font with extensive fallbacks
    if (!font_.loadFromFile("arial.ttf") &&
        // Linux system fonts - try multiple paths
        !font_.loadFromFile("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/truetype/freefont/FreeSerif.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/TTF/DejaVuSans.ttf") &&
        !font_.loadFromFile("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf") &&
        // Try any .ttf file in common locations
        !tryLoadAnyFont("/usr/share/fonts/truetype") &&
        // Windows system fonts
        !font_.loadFromFile("C:/Windows/Fonts/arial.ttf") &&
        !font_.loadFromFile("C:/Windows/Fonts/segoeui.ttf") &&
        !font_.loadFromFile("C:/Windows/Fonts/tahoma.ttf")) {

        // If still no font loaded, throw an exception to prevent crashes
        throw std::runtime_error("Failed to load any font. Please install fonts on your system.");
    }
}

bool Theme::tryLoadAnyFont(const std::string& directory) {
    // Try to find any .ttf file in the directory
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ttf") {
            if (font_.loadFromFile(entry.path().string())) {
                return true;
            }
        }
    }
    return false;
}

sf::Font& Theme::getFont() {
    return font_;
}

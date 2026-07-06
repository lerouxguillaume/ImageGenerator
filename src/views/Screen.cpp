#include "Screen.hpp"
#include "../ui/Theme.h"

Screen::Screen() {
    // Load from system font paths (a local "arial.ttf" is no longer bundled).
    if (// Linux system fonts
        !font.loadFromFile("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf") &&
        !font.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf") &&
        !font.loadFromFile("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf") &&
        !font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeSerif.ttf") &&
        // Windows system fonts
        !font.loadFromFile("C:/Windows/Fonts/arial.ttf") &&
        !font.loadFromFile("C:/Windows/Fonts/segoeui.ttf") &&
        !font.loadFromFile("C:/Windows/Fonts/tahoma.ttf"))
    {
        // no font found — text will be invisible
    }
    const auto& m = Theme::instance().metrics();
    midRect = {static_cast<float>(m.leftSidebarWidth), static_cast<float>(m.headerHeight),
               static_cast<float>(m.windowWidth - m.leftSidebarWidth),
               static_cast<float>(m.windowHeight - m.headerHeight - m.logHeight)};
}

sf::Vector2f Screen::localPosMid(sf::Vector2f screenPos) const {
    return {screenPos.x - midRect.left, screenPos.y - midRect.top};
}

bool Screen::inMid(sf::Vector2f screenPos) const {
    return midRect.contains(screenPos);
}

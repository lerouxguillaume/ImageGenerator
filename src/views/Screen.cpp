#include "Screen.hpp"
#include "../enum/constants.hpp"

Screen::Screen() {
    if (!font.loadFromFile("arial.ttf") &&
        // Linux system fonts
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
    midRect = {static_cast<float>(MID_X), static_cast<float>(HEADER_H), static_cast<float>(MID_W), static_cast<float>(BODY_H)};
}

sf::Vector2f Screen::localPosMid(sf::Vector2f screenPos) const {
    return {screenPos.x - midRect.left, screenPos.y - midRect.top};
}

bool Screen::inMid(sf::Vector2f screenPos) const {
    return midRect.contains(screenPos);
}

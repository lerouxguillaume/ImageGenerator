#pragma once
#include <SFML/Graphics.hpp>

class Theme {
public:
    static Theme& instance();

    sf::Color colorPanel() const { return {28, 20, 8}; }
    sf::Color colorText() const { return {232, 213, 163}; }
    sf::Color colorPanel2() const { return {38, 28, 12}; }
    sf::Color colorBorder() const { return {80, 55, 22}; }
    sf::Color colorBorderHi() const { return {160, 110, 40}; }
    sf::Color colorGold() const { return {200, 145, 40}; }
    sf::Color colorGoldLt() const { return {240, 192, 80}; }
    sf::Color colorMuted() const { return {130, 112, 80}; }
    sf::Color colorRed() const { return {176, 48, 48}; }
    sf::Color colorGreen() const { return {60, 130, 55}; }
    sf::Color colorBlue() const { return {42, 85, 138}; }
    sf::Color colorOverlay() const { return {0, 0, 0, 190}; }
    sf::Color colorBg() const { return {18, 14, 6}; }

    sf::Font& getFont();

private:
    Theme();
    bool tryLoadAnyFont(const std::string& directory);
    sf::Font font_;
};

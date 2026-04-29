#pragma once
#include <SFML/Graphics.hpp>

struct UiColors {
    sf::Color bg       {13,  17,  24};
    sf::Color panel    {22,  28,  38};
    sf::Color panel2   {29,  36,  48};
    sf::Color overlay  { 5,   8,  12, 210};

    sf::Color border   {47,  58,  74};
    sf::Color borderHi {92, 167, 214};

    sf::Color gold     { 94, 234, 212};
    sf::Color goldLt   {186, 252, 244};

    sf::Color text     {232, 238, 245};
    sf::Color muted    {146, 160, 178};

    sf::Color red      {204,  88, 102};
    sf::Color redLt    {250, 168, 178};
    sf::Color green    { 78, 177, 128};
    sf::Color greenLt  {146, 228, 184};
    sf::Color blue     { 74, 127, 214};
    sf::Color blueLt   {146, 188, 255};
    sf::Color purple   {140, 102, 224};
    sf::Color purpleLt {197, 174, 255};
    sf::Color injury   {224, 121,  74};

    sf::Color buttonDisabledBg {40, 46, 57};
    sf::Color shadow           { 0,  0,  0, 70};
    sf::Color surfaceRaised    {34, 42, 56};
    sf::Color surfaceInset     {17, 22, 31};
    sf::Color accentSoft       { 94, 234, 212, 36};
};

struct UiMetrics {
    int   windowWidth  = 1280;
    int   windowHeight = 800;
    int   headerHeight = 62;
    int   logHeight    = 88;
    int   leftSidebarWidth = 295;
    int   pad = 8;

    float menuBarHeight = 40.f;
    float llmBarHeight = 44.f;
    float llmExpandedExtraHeight = 80.f;
    float generatorLeftPanelWidth = 460.f;

    float borderWidth = 1.f;
    float panelInset = 1.f;
    float shadowOffset = 3.f;

    float spaceXs = 4.f;
    float spaceSm = 8.f;
    float spaceMd = 10.f;
    float spaceLg = 12.f;
    float spaceXl = 14.f;
    float space2xl = 18.f;

    float buttonHeight = 30.f;
    float compactButtonHeight = 28.f;
    float toolbarFieldHeight = 28.f;
    float toolbarLabelGap = 16.f;

    float projectBrowserWidth = 260.f;
    float projectRowHeight = 40.f;
    float assetRowHeight = 34.f;
    float assetRowStep = 38.f;
    float rightRailMinWidth = 320.f;
    float rightRailPreferredWidth = 400.f;
    float rightRailRatio = 0.34f;
};

struct UiTypography {
    unsigned pageTitle = 20;
    unsigned projectTitle = 18;
    unsigned sectionTitle = 13;
    unsigned subsectionTitle = 14;
    unsigned body = 12;
    unsigned compact = 11;
    unsigned helper = 10;
};

class Theme {
public:
    static Theme& instance();

    const UiColors& colors() const { return colors_; }
    const UiMetrics& metrics() const { return metrics_; }
    const UiTypography& typography() const { return typography_; }

    sf::Color colorPanel() const { return colors_.panel; }
    sf::Color colorText() const { return colors_.text; }
    sf::Color colorPanel2() const { return colors_.panel2; }
    sf::Color colorBorder() const { return colors_.border; }
    sf::Color colorBorderHi() const { return colors_.borderHi; }
    sf::Color colorGold() const { return colors_.gold; }
    sf::Color colorGoldLt() const { return colors_.goldLt; }
    sf::Color colorMuted() const { return colors_.muted; }
    sf::Color colorRed() const { return colors_.red; }
    sf::Color colorGreen() const { return colors_.green; }
    sf::Color colorBlue() const { return colors_.blue; }
    sf::Color colorOverlay() const { return colors_.overlay; }
    sf::Color colorBg() const { return colors_.bg; }

    sf::Font& getFont();

private:
    Theme();
    bool tryLoadAnyFont(const std::string& directory);
    UiColors colors_{};
    UiMetrics metrics_{};
    UiTypography typography_{};
    sf::Font font_;
};

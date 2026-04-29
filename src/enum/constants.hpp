#pragma once
#include <SFML/Graphics.hpp>
#include "../ui/Theme.h"

// ─── Colour palette ───────────────────────────────────────────────────────────
// Dark parchment / tavern theme. All values are sRGB (0–255).
// Naming convention: plain = base tone, Lt = lighter/highlight variant.
namespace Col {
    inline const UiColors ThemeColors{};
    // Backgrounds
    inline const sf::Color Bg       = ThemeColors.bg;       // Window background (near-black warm)
    inline const sf::Color Panel    = ThemeColors.panel;    // Primary panel surface
    inline const sf::Color Panel2   = ThemeColors.panel2;   // Raised / secondary panel surface
    inline const sf::Color Overlay  = ThemeColors.overlay;  // Semi-transparent modal overlay

    // Borders
    inline const sf::Color Border   = ThemeColors.border;   // Default border
    inline const sf::Color BorderHi = ThemeColors.borderHi; // Hovered / focused border

    // Gold — primary accent for headings, icons, active elements
    inline const sf::Color Gold     = ThemeColors.gold;
    inline const sf::Color GoldLt   = ThemeColors.goldLt;

    // Text
    inline const sf::Color Text     = ThemeColors.text;  // Body text (warm parchment)
    inline const sf::Color Muted    = ThemeColors.muted; // Disabled / secondary text

    // Status colours
    inline const sf::Color Red      = ThemeColors.red;      // Danger / error
    inline const sf::Color RedLt    = ThemeColors.redLt;    // Danger highlight
    inline const sf::Color Green    = ThemeColors.green;    // Success / healthy
    inline const sf::Color GreenLt  = ThemeColors.greenLt;  // Success highlight
    inline const sf::Color Blue     = ThemeColors.blue;     // Info / selected
    inline const sf::Color BlueLt   = ThemeColors.blueLt;   // Info highlight
    inline const sf::Color Purple   = ThemeColors.purple;   // Magic / special
    inline const sf::Color PurpleLt = ThemeColors.purpleLt; // Magic highlight
    inline const sf::Color Injury   = ThemeColors.injury;   // Injured adventurer badge
}

// ─── Layout constants ─────────────────────────────────────────────────────────
// Fixed window layout — all sizes in pixels.
//
//  ┌─────────────────────────────────────┐  ↑
//  │           HEADER  (HEADER_H)        │  HEADER_H
//  ├──────────┬──────────────────────────┤  ↓
//  │          │                          │  ↑
//  │  LEFT    │        MAIN (MID)        │  BODY_H
//  │ (LEFT_W) │       (MID_W)            │
//  │          │                          │  ↓
//  ├──────────┴──────────────────────────┤  ↑
//  │              LOG (LOG_H)            │  LOG_H
//  └─────────────────────────────────────┘  ↓
//  ←────────────── WIN_W ───────────────→

inline constexpr UiMetrics Ui = {};
constexpr int WIN_W    = Ui.windowWidth; // Total window width
constexpr int WIN_H    = Ui.windowHeight;  // Total window height
constexpr int HEADER_H = Ui.headerHeight;   // Top navigation bar height
constexpr int LOG_H    = Ui.logHeight;   // Bottom log/message strip height
constexpr int LEFT_W   = Ui.leftSidebarWidth;  // Left sidebar width
constexpr int MID_X    = LEFT_W;           // X origin of the main content area
constexpr int MID_W    = WIN_W - LEFT_W;   // Width of the main content area
constexpr int BODY_H   = WIN_H - HEADER_H - LOG_H; // Height of the scrollable content region
constexpr int PAD      = Ui.pad;    // Standard inner padding / margin

// ─── Image generator two-column layout ───────────────────────────────────────
constexpr float MENU_BAR_H   = Ui.menuBarHeight;  // Top menu bar height
constexpr float LLM_BAR_H    = Ui.llmBarHeight;  // Bottom LLM bar height (collapsed)
constexpr float LLM_EXPANDED_H = Ui.llmExpandedExtraHeight; // Extra height added to LLM bar when expanded
constexpr float LEFT_PANEL_W = Ui.generatorLeftPanelWidth; // Settings panel width
constexpr float BODY_Y       = MENU_BAR_H;
constexpr float BODY_H_FULL  = WIN_H - MENU_BAR_H;             // body when no LLM bar
constexpr float BODY_H_LLM   = WIN_H - MENU_BAR_H - LLM_BAR_H; // body when LLM bar visible

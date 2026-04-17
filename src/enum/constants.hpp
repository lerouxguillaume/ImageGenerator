#pragma once
#include <SFML/Graphics.hpp>

// ─── Colour palette ───────────────────────────────────────────────────────────
// Dark parchment / tavern theme. All values are sRGB (0–255).
// Naming convention: plain = base tone, Lt = lighter/highlight variant.
namespace Col {
    // Backgrounds
    inline const sf::Color Bg       {18,  14,   6}; // Window background (near-black warm)
    inline const sf::Color Panel    {28,  20,   8}; // Primary panel surface
    inline const sf::Color Panel2   {38,  28,  12}; // Raised / secondary panel surface
    inline const sf::Color Overlay  {  0,   0,   0, 190}; // Semi-transparent modal overlay

    // Borders
    inline const sf::Color Border   {80,  55,  22}; // Default border
    inline const sf::Color BorderHi {160, 110,  40}; // Hovered / focused border

    // Gold — primary accent for headings, icons, active elements
    inline const sf::Color Gold     {200, 145,  40};
    inline const sf::Color GoldLt   {240, 192,  80};

    // Text
    inline const sf::Color Text     {232, 213, 163}; // Body text (warm parchment)
    inline const sf::Color Muted    {130, 112,  80}; // Disabled / secondary text

    // Status colours
    inline const sf::Color Red      {176,  48,  48}; // Danger / error
    inline const sf::Color RedLt    {220,  80,  80}; // Danger highlight
    inline const sf::Color Green    { 60, 130,  55}; // Success / healthy
    inline const sf::Color GreenLt  { 90, 180,  80}; // Success highlight
    inline const sf::Color Blue     { 42,  85, 138}; // Info / selected
    inline const sf::Color BlueLt   { 80, 140, 200}; // Info highlight
    inline const sf::Color Purple   {100,  55, 155}; // Magic / special
    inline const sf::Color PurpleLt {150,  90, 210}; // Magic highlight
    inline const sf::Color Injury   {200,  80,  40}; // Injured adventurer badge
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

constexpr int WIN_W    = 1280; // Total window width
constexpr int WIN_H    = 800;  // Total window height
constexpr int HEADER_H = 62;   // Top navigation bar height
constexpr int LOG_H    = 88;   // Bottom log/message strip height
constexpr int LEFT_W   = 295;  // Left sidebar width
constexpr int MID_X    = LEFT_W;           // X origin of the main content area
constexpr int MID_W    = WIN_W - LEFT_W;   // Width of the main content area
constexpr int BODY_H   = WIN_H - HEADER_H - LOG_H; // Height of the scrollable content region
constexpr int PAD      = 8;    // Standard inner padding / margin

// ─── Image generator two-column layout ───────────────────────────────────────
constexpr float MENU_BAR_H   = 40.f;  // Top menu bar height
constexpr float LLM_BAR_H    = 44.f;  // Bottom LLM bar height (collapsed)
constexpr float LLM_EXPANDED_H = 80.f; // Extra height added to LLM bar when expanded
constexpr float LEFT_PANEL_W = 460.f; // Settings panel width
constexpr float BODY_Y       = MENU_BAR_H;
constexpr float BODY_H_FULL  = WIN_H - MENU_BAR_H;             // body when no LLM bar
constexpr float BODY_H_LLM   = WIN_H - MENU_BAR_H - LLM_BAR_H; // body when LLM bar visible


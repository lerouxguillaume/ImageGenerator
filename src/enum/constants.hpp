#pragma once
#include <SFML/Graphics.hpp>
#include "../ui/Theme.h"

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

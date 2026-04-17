#pragma once
#include "Screen.hpp"
#include "../ui/widgets/MenuBar.hpp"
#include "../ui/widgets/SettingsPanel.hpp"
#include "../ui/widgets/ResultPanel.hpp"
#include "../ui/widgets/LlmBar.hpp"
#include "../ui/widgets/SettingsModal.hpp"

// Thin composition root for the image generator screen.
// All state and logic lives in the five component panels.
// The controller accesses panels directly via public members.
class ImageGeneratorView : public Screen {
public:
    MenuBar       menuBar;
    SettingsPanel settingsPanel;
    ResultPanel   resultPanel;
    LlmBar        llmBar;
    SettingsModal settingsModal;
    bool          showSettings = false;

    void render(sf::RenderWindow& win) override;
};

#pragma once
#include "Screen.hpp"
#include "../ui/widgets/MenuBar.hpp"
#include "../ui/widgets/SettingsPanel.hpp"
#include "../ui/widgets/ResultPanel.hpp"
#include "../ui/widgets/LlmBar.hpp"
#include "../ui/widgets/SettingsModal.hpp"
#include "../enum/enums.hpp"

// Thin composition root for the image generator screen.
// All state and logic lives in the five component panels.
// The controller accesses panels directly via public members.
class ImageGeneratorView : public Screen {
public:
    explicit ImageGeneratorView(WorkflowMode workflowMode = WorkflowMode::Generate);

    MenuBar       menuBar;
    SettingsPanel settingsPanel;
    ResultPanel   resultPanel;
    LlmBar        llmBar;
    SettingsModal settingsModal;
    bool          showSettings = false;
    WorkflowMode  mode;

    void render(sf::RenderWindow& win) override;
};

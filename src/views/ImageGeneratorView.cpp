#include "ImageGeneratorView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Theme.h"

using namespace Helpers;

ImageGeneratorView::ImageGeneratorView(WorkflowMode workflowMode)
    : mode(workflowMode)
{
    settingsPanel.mode = workflowMode;
    resultPanel.mode   = workflowMode;
}

void ImageGeneratorView::render(sf::RenderWindow& win) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    // Fill window background
    win.clear(colors.bg);

    menuBar.titleOverride = (mode == WorkflowMode::Edit) ? "Edit image" : std::string{};
    menuBar.showPresetControls = (mode == WorkflowMode::Generate);

    const bool  hasLlm = (mode == WorkflowMode::Generate)
        && (llmBar.promptEnhancerAvailable || llmBar.llmLoading);
    const float llmH   = hasLlm ? (llmBar.expanded ? LLM_BAR_H + LLM_EXPANDED_H : LLM_BAR_H) : 0.f;
    const float winW   = static_cast<float>(win.getSize().x);
    const float winH   = static_cast<float>(win.getSize().y);
    const float bodyH  = winH - MENU_BAR_H - llmH;
    const float bodyPad = metrics.spaceLg;
    const float contentY = BODY_Y + bodyPad;
    const float contentH = bodyH - bodyPad * 2.f;
    const float settingsW = LEFT_PANEL_W - bodyPad * 1.5f;
    const float resultX = bodyPad + settingsW + bodyPad;
    const float resultW = winW - resultX - bodyPad;

    menuBar.setRect({0.f, 0.f, winW, MENU_BAR_H});
    settingsPanel.setRect({bodyPad, contentY, settingsW, contentH});
    resultPanel.setRect({resultX, contentY, resultW, contentH});

    settingsPanel.render(win, font);
    resultPanel.render(win, font, settingsPanel.generationParams.numSteps);
    menuBar.render(win, font);

    if (hasLlm) {
        llmBar.setRect({0.f, winH - llmH, winW, llmH});
        llmBar.render(win, font);
    }

    if (showSettings)
        settingsModal.render(win, font);
}

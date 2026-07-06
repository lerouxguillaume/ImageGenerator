#include "ImageGeneratorView.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Theme.h"

using namespace Helpers;

void ImageGeneratorView::render(sf::RenderWindow& win) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    // Fill window background
    win.clear(colors.bg);

    menuBar.showPresetControls = true;

    const bool  hasLlm = llmBar.promptEnhancerAvailable || llmBar.llmLoading;
    const float llmH   = hasLlm ? (llmBar.expanded ? metrics.llmBarHeight + metrics.llmExpandedExtraHeight
                                                   : metrics.llmBarHeight) : 0.f;
    const float winW   = static_cast<float>(win.getSize().x);
    const float winH   = static_cast<float>(win.getSize().y);
    const float bodyH  = winH - metrics.menuBarHeight - llmH;
    const float bodyPad = metrics.spaceLg;
    const float contentY = metrics.menuBarHeight + bodyPad;
    const float contentH = bodyH - bodyPad * 2.f;
    const float settingsW = metrics.generatorLeftPanelWidth - bodyPad * 1.5f;
    const float resultX = bodyPad + settingsW + bodyPad;
    const float resultW = winW - resultX - bodyPad;

    menuBar.setRect({0.f, 0.f, winW, metrics.menuBarHeight});
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

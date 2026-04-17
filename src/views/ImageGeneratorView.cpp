#include "ImageGeneratorView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Helpers.hpp"

using namespace Helpers;

void ImageGeneratorView::render(sf::RenderWindow& win) {
    // Fill window background
    win.clear(Col::Bg);

    const bool  hasLlm = llmBar.promptEnhancerAvailable || llmBar.llmLoading;
    const float llmH   = hasLlm ? (llmBar.expanded ? LLM_BAR_H + LLM_EXPANDED_H : LLM_BAR_H) : 0.f;
    const float bodyH  = static_cast<float>(WIN_H) - MENU_BAR_H - llmH;

    menuBar.setRect({0.f, 0.f, static_cast<float>(WIN_W), MENU_BAR_H});
    settingsPanel.setRect({0.f, BODY_Y, LEFT_PANEL_W, bodyH});
    resultPanel.setRect({LEFT_PANEL_W, BODY_Y, static_cast<float>(WIN_W) - LEFT_PANEL_W, bodyH});

    settingsPanel.render(win, font);
    resultPanel.render(win, font, settingsPanel.generationParams.numSteps);
    menuBar.render(win, font);

    if (hasLlm) {
        llmBar.setRect({0.f, static_cast<float>(WIN_H) - llmH,
                        static_cast<float>(WIN_W), llmH});
        llmBar.render(win, font);
    }

    if (showSettings)
        settingsModal.render(win, font);
}

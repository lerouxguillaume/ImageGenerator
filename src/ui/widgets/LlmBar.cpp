#include "LlmBar.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"

using namespace Helpers;

void LlmBar::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void LlmBar::render(sf::RenderWindow& win, sf::Font& font) {
    const float x = rect_.left;
    const float y = rect_.top;
    const float w = rect_.width;
    constexpr float pad  = static_cast<float>(PAD);
    constexpr float barH = LLM_BAR_H;

    // Bar background + top border
    drawRect(win, rect_, Col::Panel2);
    drawRect(win, {x, y, w, 1.f}, Col::Border);

    // Toggle button (always in the top barH strip)
    const std::string toggleLabel = expanded ? "v LLM" : "> LLM";
    btnToggle_ = {x + pad, y + (barH - 22.f) / 2.f, 70.f, 22.f};
    drawButton(win, btnToggle_, toggleLabel, Col::Panel, expanded ? Col::GoldLt : Col::Muted, false, 11, font);

    if (llmLoading) {
        drawText(win, font, "LLM loading...", Col::Muted, x + pad + 80.f, y + (barH - 12.f) / 2.f, 11);
        btnEnhance_ = {};
    } else if (promptEnhancerAvailable) {
        const std::string enhLabel = enhancing ? "Enhancing..." : "Enhance";
        const sf::Color   enhCol   = enhancing ? Col::Muted : Col::GoldLt;
        btnEnhance_ = {x + w - pad - 100.f, y + (barH - 22.f) / 2.f, 100.f, 22.f};
        drawButton(win, btnEnhance_, enhLabel, Col::Panel, enhCol, false, 11, font);
    } else {
        btnEnhance_ = {};
    }

    // ── Expanded section (below toggle row) ───────────────────────────────────
    if (expanded) {
        const float expandedY = y + barH;
        const float expandedH = rect_.height - barH;
        drawRect(win, {x, expandedY - 1.f, w, 1.f}, Col::Border);

        constexpr float fieldH = 46.f;
        constexpr float labelW = 160.f;
        const float fieldW = w - pad * 2.f - labelW;
        const float fieldY = expandedY + (expandedH - fieldH) / 2.f;

        drawText(win, font, "Instruction (optional):", Col::Muted, x + pad, fieldY + 4.f, 12);
        instructionArea.setRect({x + pad + labelW, fieldY, fieldW, fieldH});
        instructionArea.render(win, font);
    }
}

bool LlmBar::handleEvent(const sf::Event& e) {
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
        const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                               static_cast<float>(e.mouseButton.y)};

        if (btnToggle_.contains(pos)) {
            expanded = !expanded;
            if (!expanded) instructionArea.setActive(false);
            return true;
        }
        if (promptEnhancerAvailable && !enhancing && btnEnhance_.contains(pos)) {
            enhanceRequested = true;
            return true;
        }
        if (expanded && instructionArea.getRect().contains(pos)) {
            instructionArea.handleClick(pos);
            return true;
        }
        if (rect_.contains(pos)) return true;
        return false;
    }

    if (e.type == sf::Event::MouseWheelScrolled && expanded) {
        const sf::Vector2f pos{static_cast<float>(e.mouseWheelScroll.x),
                               static_cast<float>(e.mouseWheelScroll.y)};
        if (instructionArea.getRect().contains(pos)) {
            instructionArea.handleScroll(e.mouseWheelScroll.delta > 0 ? -1.f : 1.f);
            return true;
        }
    }

    if (expanded && instructionArea.isActive()) {
        if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Tab) {
            instructionArea.setActive(false);
            return true;
        }
        if (instructionArea.handleEvent(e)) return true;
    }

    return false;
}

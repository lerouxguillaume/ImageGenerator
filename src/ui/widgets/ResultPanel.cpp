#include "ResultPanel.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"

using namespace Helpers;

void ResultPanel::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void ResultPanel::render(sf::RenderWindow& win, sf::Font& font, int numSteps) {
    const float x  = rect_.left;
    const float y  = rect_.top;
    const float w  = rect_.width;
    const float h  = rect_.height;
    const float cx = x + w / 2.f;

    // Panel background
    drawRect(win, rect_, Col::Bg);

    if (generating) {
        // ── Generating overlay ────────────────────────────────────────────────
        sf::RectangleShape overlay({w, h});
        overlay.setPosition(x, y);
        overlay.setFillColor({0, 0, 0, 140});
        win.draw(overlay);

        constexpr float modalW = 340.f;
        constexpr float modalH = 148.f;
        const float modalX = cx - modalW / 2.f;
        const float modalY = y + h / 2.f - modalH / 2.f;
        const sf::FloatRect modalBox{modalX, modalY, modalW, modalH};
        drawRect(win, modalBox, Col::Panel, Col::Border, 1.f);

        const int imgNum   = generationImageNum.load();
        const int imgTotal = generationTotalImages.load();
        const std::string imgLabel = imgTotal > 1
            ? "Generating image " + std::to_string(imgNum) + " / " + std::to_string(imgTotal) + "..."
            : "Generating image...";
        drawTextC(win, font, imgLabel, Col::GoldLt, cx, modalY + 16.f, 15, true);

        const int   currentStep = generationStep.load();
        const float progress    = numSteps > 0
            ? static_cast<float>(currentStep) / static_cast<float>(numSteps) : 0.f;
        constexpr float barPad = 20.f;
        constexpr float barH   = 10.f;
        const float barW = modalW - barPad * 2.f;
        const float barX = modalX + barPad;
        const float barY = modalY + 48.f;
        drawRect(win, {barX, barY, barW, barH}, Col::Panel2, Col::Border, 1.f);
        if (progress > 0.f)
            drawRect(win, {barX, barY, barW * progress, barH}, Col::Gold);

        const std::string stepLabel = "Step " + std::to_string(currentStep) + " / " + std::to_string(numSteps);
        drawTextC(win, font, stepLabel, Col::Muted, cx, modalY + 72.f, 11);

        btnCancelGenerate_ = {cx - 55.f, modalY + modalH - 34.f, 110.f, 26.f};
        drawButton(win, btnCancelGenerate_, "Cancel", Col::Panel2, Col::RedLt, false, 12, font);

    } else if (resultLoaded) {
        // ── Generated image ───────────────────────────────────────────────────
        const float maxImgW = w - 16.f;
        const float maxImgH = h - 80.f; // leave room for generate button below
        const auto  texSize = resultTexture.getSize();
        const float scale   = std::min(1.f,
                                       std::min(maxImgW / static_cast<float>(texSize.x),
                                                maxImgH / static_cast<float>(texSize.y)));
        const float imgW    = static_cast<float>(texSize.x) * scale;
        const float imgH    = static_cast<float>(texSize.y) * scale;
        const float imgX    = cx - imgW / 2.f;
        const float imgY    = y + 16.f;

        sf::Sprite sprite(resultTexture);
        sprite.setScale(scale, scale);
        sprite.setPosition(imgX, imgY);
        win.draw(sprite);

        // Error banner below image
        if (generationFailed.load()) {
            std::string msg = generationErrorMsg;
            constexpr size_t kMaxLen = 120;
            if (msg.size() > kMaxLen) msg = msg.substr(0, kMaxLen) + "... (see log)";
            constexpr float bannerW = 480.f, bannerH = 54.f;
            const float bannerY = imgY + imgH + 8.f;
            drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, Col::Panel2, Col::RedLt, 1.f);
            drawTextC(win, font, "Generation failed", Col::RedLt, cx, bannerY + 8.f, 12, true);
            drawTextC(win, font, msg, Col::Muted, cx, bannerY + 28.f, 10);
        }

    } else if (generationFailed.load()) {
        // Error banner with no image
        std::string msg = generationErrorMsg;
        constexpr size_t kMaxLen = 120;
        if (msg.size() > kMaxLen) msg = msg.substr(0, kMaxLen) + "... (see log)";
        constexpr float bannerW = 480.f, bannerH = 54.f;
        const float bannerY = y + h / 2.f - bannerH;
        drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, Col::Panel2, Col::RedLt, 1.f);
        drawTextC(win, font, "Generation failed", Col::RedLt, cx, bannerY + 8.f, 12, true);
        drawTextC(win, font, msg, Col::Muted, cx, bannerY + 28.f, 10);
    } else {
        // Placeholder when no image yet
        drawTextC(win, font, "No image generated yet", Col::Border, cx, y + h / 2.f - 20.f, 13);
    }

    // ── Buttons at the bottom of the panel ───────────────────────────────────
    if (!generating) {
        if (resultLoaded) {
            // Two-button layout: "Use as init" left, "Generate" right
            btnUseAsInit_ = {cx - 204.f, y + h - 49.f, 118.f, 30.f};
            drawButton(win, btnUseAsInit_, "Use as init", Col::Panel2, Col::BlueLt, false, 12, font);
            btnGenerate_  = {cx - 78.f, y + h - 52.f, 160.f, 38.f};
        } else {
            btnUseAsInit_ = {};
            btnGenerate_  = {cx - 80.f, y + h - 52.f, 160.f, 38.f};
        }
        drawButton(win, btnGenerate_, "Generate", Col::Panel2, Col::GoldLt, false, 14, font);
    }
}

bool ResultPanel::handleEvent(const sf::Event& e) {
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
        const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                               static_cast<float>(e.mouseButton.y)};
        if (!rect_.contains(pos)) return false;

        if (generating) {
            if (btnCancelGenerate_.contains(pos)) {
                cancelToken.store(true);
                generating = false;
            }
            return true;
        }

        if (btnUseAsInit_.contains(pos)) {
            useAsInitRequested = true;
            return true;
        }
        if (btnGenerate_.contains(pos)) {
            generateRequested = true;
            return true;
        }
    }
    return false;
}

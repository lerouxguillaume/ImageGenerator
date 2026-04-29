#include "ResultPanel.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"
#include <algorithm>

using namespace Helpers;

void ResultPanel::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void ResultPanel::ensureSelectedThumbnailVisible(int visibleCount) {
    if (gallery.empty() || visibleCount <= 0) {
        thumbnailScrollOffset_ = 0;
        return;
    }

    const int maxOffset = std::max(0, static_cast<int>(gallery.size()) - visibleCount);
    thumbnailScrollOffset_ = std::clamp(thumbnailScrollOffset_, 0, maxOffset);
    if (selectedIndex < 0) return;

    if (selectedIndex < thumbnailScrollOffset_) {
        thumbnailScrollOffset_ = selectedIndex;
    } else if (selectedIndex >= thumbnailScrollOffset_ + visibleCount) {
        thumbnailScrollOffset_ = selectedIndex - visibleCount + 1;
    }
    thumbnailScrollOffset_ = std::clamp(thumbnailScrollOffset_, 0, maxOffset);
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
        btnPrevImage_ = {};
        btnNextImage_ = {};
        btnPrevThumbs_ = {};
        btnNextThumbs_ = {};
        thumbnailRects_.clear();
        thumbnailIndices_.clear();
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
        // ── Selected image preview ────────────────────────────────────────────
        const float galleryH = gallery.empty() ? 0.f : 124.f;
        const float previewBottom = y + h - galleryH - 64.f;
        const float maxImgW = w - 16.f;
        const float maxImgH = std::max(80.f, previewBottom - (y + 16.f));
        const auto  texSize = resultTexture.getSize();
        const float scale   = std::min(1.f,
                                       std::min(maxImgW / static_cast<float>(texSize.x),
                                                maxImgH / static_cast<float>(texSize.y)));
        const float imgW    = static_cast<float>(texSize.x) * scale;
        const float imgH    = static_cast<float>(texSize.y) * scale;
        const float imgX    = cx - imgW / 2.f;
        const float imgY    = y + 16.f;
        constexpr float navBtnSize = 34.f;
        constexpr float navBtnGap = 10.f;

        sf::Sprite sprite(resultTexture);
        sprite.setScale(scale, scale);
        sprite.setPosition(imgX, imgY);
        win.draw(sprite);

        if (gallery.size() > 1) {
            const float navY = imgY + imgH / 2.f - navBtnSize / 2.f;
            btnPrevImage_ = {
                std::max(x + 8.f, imgX - navBtnSize - navBtnGap),
                navY,
                navBtnSize,
                navBtnSize
            };
            btnNextImage_ = {
                std::min(x + w - 8.f - navBtnSize, imgX + imgW + navBtnGap),
                navY,
                navBtnSize,
                navBtnSize
            };
            drawButton(win, btnPrevImage_, "<", Col::Panel2, Col::GoldLt, false, 16, font);
            drawButton(win, btnNextImage_, ">", Col::Panel2, Col::GoldLt, false, 16, font);
        } else {
            btnPrevImage_ = {};
            btnNextImage_ = {};
        }

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

        if (!gallery.empty()) {
            const float stripY = y + h - galleryH - 60.f;
            const float stripX = x + 12.f;
            const float stripW = w - 24.f;
            renderThumbnailStrip(win, font, stripX, stripY, stripW);
        } else {
            thumbnailRects_.clear();
        }

    } else if (generationFailed.load()) {
        btnPrevImage_ = {};
        btnNextImage_ = {};
        btnPrevThumbs_ = {};
        btnNextThumbs_ = {};
        thumbnailRects_.clear();
        thumbnailIndices_.clear();
        // Error banner with no image
        std::string msg = generationErrorMsg;
        constexpr size_t kMaxLen = 120;
        if (msg.size() > kMaxLen) msg = msg.substr(0, kMaxLen) + "... (see log)";
        constexpr float bannerW = 480.f, bannerH = 54.f;
        const float bannerY = y + h / 2.f - bannerH;
        drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, Col::Panel2, Col::RedLt, 1.f);
        drawTextC(win, font, "Generation failed", Col::RedLt, cx, bannerY + 8.f, 12, true);
        drawTextC(win, font, msg, Col::Muted, cx, bannerY + 28.f, 10);
    } else if (!gallery.empty()) {
        btnPrevImage_ = {};
        btnNextImage_ = {};
        drawTextC(win, font, "Select an image", Col::Border, cx, y + 32.f, 13);
        const float stripX = x + 12.f;
        const float stripY = y + 56.f;
        const float stripW = w - 24.f;
        renderThumbnailStrip(win, font, stripX, stripY, stripW);
    } else {
        // Placeholder when no image yet
        btnPrevImage_ = {};
        btnNextImage_ = {};
        btnPrevThumbs_ = {};
        btnNextThumbs_ = {};
        thumbnailRects_.clear();
        thumbnailIndices_.clear();
        drawTextC(win, font, "No image generated yet", Col::Border, cx, y + h / 2.f - 20.f, 13);
    }

    // ── Buttons at the bottom of the panel ───────────────────────────────────
    if (!generating) {
        if (resultLoaded) {
            btnUseAsInit_ = {cx - 246.f, y + h - 49.f, 118.f, 30.f};
            drawButton(win, btnUseAsInit_, "Use as init", Col::Panel2, Col::BlueLt, false, 12, font);
            btnImprove_ = {cx - 120.f, y + h - 49.f, 88.f, 30.f};
            drawButton(win, btnImprove_, "Improve", Col::Panel2, Col::GoldLt, false, 12, font);
            btnDelete_ = {cx - 24.f, y + h - 49.f, 88.f, 30.f};
            drawButton(win, btnDelete_, "Delete", Col::Panel2, Col::RedLt, false, 12, font);
            btnGenerate_  = {cx + 74.f, y + h - 52.f, 160.f, 38.f};
        } else {
            btnUseAsInit_ = {};
            btnImprove_   = {};
            btnDelete_    = {};
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
        if (btnImprove_.contains(pos)) {
            improveRequested = true;
            return true;
        }
        if (btnDelete_.contains(pos)) {
            deleteRequested = true;
            return true;
        }
        if (btnPrevImage_.contains(pos) && selectedIndex > 0) {
            selectedIndex -= 1;
            return true;
        }
        if (btnNextImage_.contains(pos) && selectedIndex >= 0
            && selectedIndex + 1 < static_cast<int>(gallery.size())) {
            selectedIndex += 1;
            return true;
        }
        if (btnPrevThumbs_.contains(pos) && !gallery.empty()) {
            const int page = std::max(1, static_cast<int>(thumbnailIndices_.size()));
            thumbnailScrollOffset_ = std::max(0, thumbnailScrollOffset_ - page);
            return true;
        }
        if (btnNextThumbs_.contains(pos) && !gallery.empty()) {
            const int page = std::max(1, static_cast<int>(thumbnailIndices_.size()));
            const int maxOffset = std::max(0, static_cast<int>(gallery.size()) - page);
            thumbnailScrollOffset_ = std::min(maxOffset, thumbnailScrollOffset_ + page);
            return true;
        }
        for (int i = 0; i < static_cast<int>(thumbnailRects_.size()); ++i) {
            if (thumbnailRects_[static_cast<size_t>(i)].contains(pos)) {
                selectedIndex = thumbnailIndices_[static_cast<size_t>(i)];
                return true;
            }
        }
        if (btnGenerate_.contains(pos)) {
            generateRequested = true;
            return true;
        }
    }
    return false;
}

std::string ResultPanel::getSelectedImagePath() const {
    if (selectedIndex < 0 || selectedIndex >= static_cast<int>(gallery.size()))
        return {};
    return gallery[static_cast<size_t>(selectedIndex)].path;
}

void ResultPanel::renderThumbnailStrip(sf::RenderWindow& win, sf::Font& font,
                                        float stripX, float stripY, float stripW) {
    constexpr float stripH = 124.f;
    constexpr float thumbW = 92.f;
    constexpr float thumbH = 92.f;
    constexpr float gap    = 8.f;
    constexpr float navW   = 28.f;

    drawRect(win, {stripX, stripY, stripW, stripH}, Col::Panel, Col::Border, 1.f);
    thumbnailRects_.clear();
    thumbnailIndices_.clear();

    const float thumbY   = stripY + 10.f;
    const bool showThumbNav = static_cast<int>(gallery.size()) > 1;
    const float contentX = stripX + 10.f + (showThumbNav ? navW + gap : 0.f);
    const float contentW = stripW - 20.f - (showThumbNav ? (navW + gap) * 2.f : 0.f);
    const int maxThumbs  = std::max(1, static_cast<int>((contentW + gap) / (thumbW + gap)));
    const int visible    = std::min(maxThumbs, static_cast<int>(gallery.size()));
    ensureSelectedThumbnailVisible(visible);
    float thumbX         = contentX;

    if (showThumbNav) {
        btnPrevThumbs_ = {stripX + 10.f, thumbY + (thumbH - navW) / 2.f, navW, navW};
        btnNextThumbs_ = {stripX + stripW - 10.f - navW, thumbY + (thumbH - navW) / 2.f, navW, navW};
        drawButton(win, btnPrevThumbs_, "<", Col::Panel2, Col::GoldLt,
                   thumbnailScrollOffset_ == 0, 14, font);
        drawButton(win, btnNextThumbs_, ">", Col::Panel2, Col::GoldLt,
                   thumbnailScrollOffset_ + visible >= static_cast<int>(gallery.size()), 14, font);
    } else {
        btnPrevThumbs_ = {};
        btnNextThumbs_ = {};
    }

    for (int i = 0; i < visible; ++i) {
        const int galleryIndex = thumbnailScrollOffset_ + i;
        const auto& item = gallery[static_cast<size_t>(galleryIndex)];
        const sf::FloatRect thumbRect{thumbX, thumbY, thumbW, thumbH};
        thumbnailRects_.push_back(thumbRect);
        thumbnailIndices_.push_back(galleryIndex);
        const bool selected = (galleryIndex == selectedIndex);
        drawRect(win, thumbRect, Col::Panel2, selected ? Col::GoldLt : Col::Border, selected ? 2.f : 1.f);

        if (item.thumbnail && item.thumbnail->getSize().x > 0 && item.thumbnail->getSize().y > 0) {
            const float imgScale = std::min((thumbW - 8.f) / static_cast<float>(item.thumbnail->getSize().x),
                                            (thumbH - 8.f) / static_cast<float>(item.thumbnail->getSize().y));
            sf::Sprite thumb(*item.thumbnail);
            thumb.setScale(imgScale, imgScale);
            const float drawW = static_cast<float>(item.thumbnail->getSize().x) * imgScale;
            const float drawH = static_cast<float>(item.thumbnail->getSize().y) * imgScale;
            thumb.setPosition(thumbX + (thumbW - drawW) / 2.f, thumbY + (thumbH - drawH) / 2.f);
            win.draw(thumb);
        } else {
            drawTextC(win, font, "No preview", Col::Muted,
                      thumbX + thumbW / 2.f, thumbY + thumbH / 2.f - 6.f, 10);
        }
        thumbX += thumbW + gap;
    }

    if (static_cast<int>(gallery.size()) > visible) {
        drawTextR(win, font,
                  std::to_string(thumbnailScrollOffset_ + 1) + "-"
                    + std::to_string(thumbnailScrollOffset_ + visible) + " / "
                    + std::to_string(static_cast<int>(gallery.size())),
                  Col::Muted, stripX + stripW - 10.f, stripY + stripH - 18.f, 10);
    }
}

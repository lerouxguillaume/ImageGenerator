#include "ResultPanel.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"
#include "../../ui/Theme.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace Helpers;

namespace {

static std::string stageLabelText(GenerationStage stage, int step, int numSteps) {
    switch (stage) {
    case GenerationStage::LoadingModel:    return "Loading model...";
    case GenerationStage::EncodingText:    return "Encoding prompt...";
    case GenerationStage::EncodingImage:   return "Encoding image...";
    case GenerationStage::Denoising:       return "Step " + std::to_string(step) + " / " + std::to_string(numSteps);
    case GenerationStage::HiresDenoising:
        // The label flips the instant pass 2 begins — i.e. it is already showing
        // during the one-time per-shape ORT/CUDA setup stall on the first hires
        // Run (no extra warmup session; the status is the mitigation).
        return "Hi-res refine — step " + std::to_string(step) + " / " + std::to_string(numSteps);
    case GenerationStage::DecodingImage:   return "Decoding image...";
    default:
        return numSteps > 0
            ? "Step " + std::to_string(step) + " / " + std::to_string(numSteps)
            : "Generating...";
    }
}

constexpr float kThumb    = 84.f;   // thumbnail edge
constexpr float kThumbGap = 8.f;
constexpr float kHeaderH  = 30.f;   // gallery header row
constexpr float kStripH   = 150.f;  // gallery strip height (header + one row)

}

void ResultPanel::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void ResultPanel::render(sf::RenderWindow& win, sf::Font& font, int numSteps) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const auto& type = theme.typography();
    const float x  = rect_.left;
    const float y  = rect_.top;
    const float w  = rect_.width;
    const float h  = rect_.height;
    const float cx = x + w / 2.f;
    const float actionBarX = x + 16.f;
    const float actionBarY = y + h - 60.f;
    const float actionBarW = w - 32.f;
    const float actionBarH = 44.f;
    const float actionButtonH = 30.f;
    const float actionAuxW = 76.f;
    const float actionGap = 10.f;

    // Panel background
    drawRect(win, rect_, colors.panel2, colors.border, metrics.borderWidth);
    drawRect(win, {x + 1.f, y + 1.f, w - 2.f, h - 2.f}, colors.surfaceInset, sf::Color::Transparent, 0.f);

    if (generating) {
        btnGalleryExpand_ = {};
        thumbnailRects_.clear();
        thumbnailIndices_.clear();
        // ── Generating overlay ────────────────────────────────────────────────
        sf::RectangleShape overlay({w, h});
        overlay.setPosition(x, y);
        overlay.setFillColor(colors.overlay);
        win.draw(overlay);

        constexpr float modalW = 340.f;
        constexpr float modalH = 148.f;
        const float modalX = cx - modalW / 2.f;
        const float modalY = y + h / 2.f - modalH / 2.f;
        const sf::FloatRect modalBox{modalX, modalY, modalW, modalH};
        drawRect(win, modalBox, colors.panel, colors.borderHi, metrics.borderWidth);

        const int imgNum   = generationImageNum.load();
        const int imgTotal = generationTotalImages.load();
        const std::string imgLabel = imgTotal > 1
            ? "Generating image " + std::to_string(imgNum) + " / " + std::to_string(imgTotal) + "..."
            : "Generating image...";
        drawTextC(win, font, imgLabel, colors.goldLt, cx, modalY + 16.f, 15, true);

        const int   currentStep = generationStep.load();
        const float progress    = numSteps > 0
            ? static_cast<float>(currentStep) / static_cast<float>(numSteps) : 0.f;
        constexpr float barPad = 20.f;
        constexpr float barH   = 10.f;
        const float barW = modalW - barPad * 2.f;
        const float barX = modalX + barPad;
        const float barY = modalY + 48.f;
        drawRect(win, {barX, barY, barW, barH}, colors.surfaceInset, colors.border, metrics.borderWidth);
        if (progress > 0.f)
            drawRect(win, {barX, barY, barW * progress, barH}, colors.gold);

        const std::string stepLabel = stageLabelText(generationStage.load(), currentStep, numSteps);
        drawTextC(win, font, stepLabel, colors.muted, cx, modalY + 72.f, type.compact);

        btnCancelGenerate_ = {cx - 55.f, modalY + modalH - 34.f, 110.f, 26.f};
        drawButton(win, btnCancelGenerate_, "Cancel", colors.panel2, colors.redLt, false, type.body, font);
        return;
    }

    const bool haveGallery = !gallery.empty();
    btnDeselect_ = {};   // only shown over the preview (set below when resultLoaded)

    // ── Expanded grid mode: the gallery takes over the whole panel body ───────
    if (galleryExpanded_ && haveGallery) {
        const sf::FloatRect area{x + 16.f, y + 16.f, w - 32.f, actionBarY - (y + 16.f) - 8.f};
        renderGallery(win, font, area);
    } else {
        // ── Preview + caption + strip ─────────────────────────────────────────
        const float stripH   = haveGallery ? kStripH : 0.f;
        const float stripTop = actionBarY - stripH - (haveGallery ? 8.f : 0.f);

        if (resultLoaded) {
            const float infoH = generationFailed.load() ? 62.f
                              : (!validationChips.empty() ? 30.f : 0.f);
            const float captionH  = 22.f;
            const float frameX = x + 16.f;
            const float frameY = y + 16.f;
            const float frameW = w - 32.f;
            const float frameH = std::max(120.f, stripTop - frameY - captionH - infoH);
            drawRect(win, {frameX, frameY, frameW, frameH}, colors.panel, colors.border, metrics.borderWidth);

            const float maxImgW = frameW - 24.f;
            const float maxImgH = std::max(80.f, frameH - 24.f);
            const auto  texSize = resultTexture.getSize();
            const float scale   = std::min(1.f,
                                           std::min(maxImgW / static_cast<float>(texSize.x),
                                                    maxImgH / static_cast<float>(texSize.y)));
            const float imgW    = static_cast<float>(texSize.x) * scale;
            const float imgH    = static_cast<float>(texSize.y) * scale;
            const float imgX    = frameX + (frameW - imgW) / 2.f;
            const float imgY    = frameY + (frameH - imgH) / 2.f;

            if (showCheckerboard) {
                constexpr float sqSz = 16.f;
                const int cols = static_cast<int>(std::ceil(imgW / sqSz));
                const int rows = static_cast<int>(std::ceil(imgH / sqSz));
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        const float rx = imgX + static_cast<float>(col) * sqSz;
                        const float ry = imgY + static_cast<float>(row) * sqSz;
                        const float rw = std::min(sqSz, imgX + imgW - rx);
                        const float rh = std::min(sqSz, imgY + imgH - ry);
                        const sf::Color chCol = ((row + col) % 2 == 0)
                            ? sf::Color(200, 200, 200) : sf::Color(155, 155, 155);
                        drawRect(win, {rx, ry, rw, rh}, chCol, sf::Color::Transparent, 0.f);
                    }
                }
            }

            sf::Sprite sprite(resultTexture);
            sprite.setScale(scale, scale);
            sprite.setPosition(imgX, imgY);
            win.draw(sprite);

            // Deselect (×) — unselect this image and drop out of edit mode.
            btnDeselect_ = {frameX + frameW - 30.f, frameY + 8.f, 22.f, 22.f};
            drawButton(win, btnDeselect_, "\xc3\x97", colors.panel2, colors.muted, false, 14, font);

            // ── Caption: filename · dimensions · position ─────────────────────
            std::string fname;
            if (selectedIndex >= 0 && selectedIndex < static_cast<int>(gallery.size()))
                fname = gallery[static_cast<size_t>(selectedIndex)].filename;
            char dims[48];
            std::snprintf(dims, sizeof(dims), "%u \xc3\x97 %u", texSize.x, texSize.y);
            std::string caption = fname.empty() ? std::string(dims) : fname + "   \xc2\xb7   " + dims;
            if (haveGallery && selectedIndex >= 0)
                caption += "   \xc2\xb7   " + std::to_string(selectedIndex + 1) + " of "
                         + std::to_string(static_cast<int>(gallery.size()));
            const float captionY = frameY + frameH + 5.f;
            drawText(win, font, caption, colors.muted, frameX, captionY, type.compact);

            // ── Validation chips (below caption) ──────────────────────────────
            if (!validationChips.empty()) {
                constexpr float chipH   = 20.f;
                constexpr float chipGap = 6.f;
                const float chipsY = captionY + captionH;
                const int   n      = static_cast<int>(validationChips.size());
                const float totalGaps = static_cast<float>(n - 1) * chipGap;
                const float chipW = (frameW - totalGaps) / static_cast<float>(n);
                for (int ci = 0; ci < n; ++ci) {
                    const auto& chip = validationChips[static_cast<size_t>(ci)];
                    const float cx2  = frameX + static_cast<float>(ci) * (chipW + chipGap);
                    sf::Color borderCol = (chip.status == 0) ? colors.green
                                        : (chip.status == 1) ? colors.goldLt
                                                             : colors.redLt;
                    drawRect(win, {cx2, chipsY, chipW, chipH}, colors.panel2, borderCol, 1.f);
                    const std::string label = chip.name + ": " + chip.detail;
                    drawTextC(win, font, label, borderCol, cx2 + chipW / 2.f, chipsY + 3.f, type.helper);
                }
            }

            // Error banner below caption
            if (generationFailed.load()) {
                std::string msg = generationErrorMsg;
                constexpr size_t kMaxLen = 120;
                if (msg.size() > kMaxLen) msg = msg.substr(0, kMaxLen) + "... (see log)";
                constexpr float bannerW = 480.f, bannerH = 54.f;
                const float bannerY = captionY + captionH;
                drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, colors.panel2, colors.redLt, metrics.borderWidth);
                drawTextC(win, font, "Generation failed", colors.redLt, cx, bannerY + 8.f, type.body, true);
                drawTextC(win, font, msg, colors.muted, cx, bannerY + 28.f, type.helper);
            }
        } else if (generationFailed.load()) {
            // Error banner with no image
            std::string msg = generationErrorMsg;
            constexpr size_t kMaxLen = 120;
            if (msg.size() > kMaxLen) msg = msg.substr(0, kMaxLen) + "... (see log)";
            constexpr float bannerW = 480.f, bannerH = 54.f;
            const float bannerY = y + h / 2.f - bannerH;
            drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, colors.panel2, colors.redLt, metrics.borderWidth);
            drawTextC(win, font, "Generation failed", colors.redLt, cx, bannerY + 8.f, type.body, true);
            drawTextC(win, font, msg, colors.muted, cx, bannerY + 28.f, type.helper);
        } else if (haveGallery) {
            drawTextC(win, font, "Select an image", colors.borderHi, cx, y + h / 2.f - 30.f, type.sectionTitle);
        } else {
            drawTextC(win, font, "No image generated yet", colors.borderHi, cx, y + h / 2.f - 20.f, type.sectionTitle);
        }

        if (haveGallery)
            renderGallery(win, font, {x + 12.f, stripTop, w - 24.f, stripH});
        else {
            btnGalleryExpand_ = {};
            thumbnailRects_.clear();
            thumbnailIndices_.clear();
            galleryRegion_ = {};
        }
    }

    // ── Action bar: Edit · Delete act on the selected result ──────────────────
    // (Generate lives at the bottom of the settings rail, not here.)
    if (resultLoaded) {
        drawRect(win, {actionBarX, actionBarY, actionBarW, actionBarH},
                 colors.panel, colors.border, metrics.borderWidth);
        float lx = actionBarX + 8.f;
        if (showImproveButton) {
            btnImprove_ = {lx, y + h - 49.f, actionAuxW, actionButtonH};
            drawButton(win, btnImprove_, "Edit", colors.panel2, colors.goldLt, false, type.body, font);
            lx += actionAuxW + actionGap;
        } else {
            btnImprove_ = {};
        }
        btnDelete_ = {lx, y + h - 49.f, actionAuxW, actionButtonH};
        drawButton(win, btnDelete_, "Delete", colors.panel2, colors.redLt, false, type.body, font);
    } else {
        btnImprove_ = {};
        btnDelete_  = {};
    }
}

void ResultPanel::renderGallery(sf::RenderWindow& win, sf::Font& font, const sf::FloatRect& area) {
    const auto& theme  = Theme::instance();
    const auto& colors = theme.colors();

    drawRect(win, area, colors.panel2, colors.border, 1.f);

    // Header row: label · count · expand/collapse toggle
    drawText(win, font, "GALLERY", colors.muted, area.left + 12.f, area.top + 10.f, 10);
    const std::string count = (selectedIndex >= 0)
        ? std::to_string(selectedIndex + 1) + " / " + std::to_string(static_cast<int>(gallery.size()))
        : std::to_string(static_cast<int>(gallery.size())) + " images";
    btnGalleryExpand_ = {area.left + area.width - 96.f, area.top + 5.f, 86.f, 20.f};
    drawButton(win, btnGalleryExpand_, galleryExpanded_ ? "Collapse" : "Grid view",
               colors.surfaceInset, colors.text, false, 11, font);
    drawTextR(win, font, count, colors.gold, btnGalleryExpand_.left - 12.f, area.top + 10.f, 10);

    const sf::FloatRect content{area.left + 10.f, area.top + kHeaderH,
                                area.width - 20.f, area.height - kHeaderH - 8.f};
    thumbnailRects_.clear();
    thumbnailIndices_.clear();
    galleryRegion_ = content;

    const int n = static_cast<int>(gallery.size());
    const float step = kThumb + kThumbGap;

    // Layout metrics differ per mode; both scroll along one axis via galleryScroll_.
    int cols = 1;
    if (galleryExpanded_) {
        cols = std::max(1, static_cast<int>((content.width + kThumbGap) / step));
        const int rows = (n + cols - 1) / cols;
        const float totalH = static_cast<float>(rows) * step - kThumbGap;
        galleryScrollMax_ = std::max(0.f, totalH - content.height);
    } else {
        const float totalW = static_cast<float>(n) * step - kThumbGap;
        galleryScrollMax_ = std::max(0.f, totalW - content.width);
    }

    // Bring the selection into view once, when it changes (don't fight user scroll).
    if (selectedIndex >= 0 && selectedIndex != lastSelectedIndex_) {
        if (galleryExpanded_) {
            const float selY = static_cast<float>(selectedIndex / cols) * step;
            if (selY < galleryScroll_) galleryScroll_ = selY;
            else if (selY + kThumb > galleryScroll_ + content.height)
                galleryScroll_ = selY + kThumb - content.height;
        } else {
            const float selX = static_cast<float>(selectedIndex) * step;
            if (selX < galleryScroll_) galleryScroll_ = selX;
            else if (selX + kThumb > galleryScroll_ + content.width)
                galleryScroll_ = selX + kThumb - content.width;
        }
    }
    lastSelectedIndex_ = selectedIndex;
    galleryScroll_ = std::clamp(galleryScroll_, 0.f, galleryScrollMax_);

    // Clip drawing to the content region so scrolled thumbnails don't overdraw.
    const sf::View prevView = win.getView();
    const sf::Vector2f wsz(static_cast<float>(win.getSize().x), static_cast<float>(win.getSize().y));
    sf::View clip(sf::FloatRect(content.left, content.top, content.width, content.height));
    clip.setViewport(sf::FloatRect(content.left / wsz.x, content.top / wsz.y,
                                   content.width / wsz.x, content.height / wsz.y));
    win.setView(clip);

    const float rowY0 = content.top + (galleryExpanded_ ? 0.f : (content.height - kThumb) / 2.f);
    for (int i = 0; i < n; ++i) {
        float tx, ty;
        if (galleryExpanded_) {
            tx = content.left + static_cast<float>(i % cols) * step;
            ty = rowY0 + static_cast<float>(i / cols) * step - galleryScroll_;
            if (ty + kThumb < content.top || ty > content.top + content.height) continue;
        } else {
            tx = content.left + static_cast<float>(i) * step - galleryScroll_;
            ty = rowY0;
            if (tx + kThumb < content.left || tx > content.left + content.width) continue;
        }

        const sf::FloatRect thumbRect{tx, ty, kThumb, kThumb};
        thumbnailRects_.push_back(thumbRect);
        thumbnailIndices_.push_back(i);

        const bool selected = (i == selectedIndex);
        const auto& item = gallery[static_cast<size_t>(i)];
        drawRect(win, thumbRect, colors.panel, selected ? colors.gold : colors.border, selected ? 2.f : 1.f);

        if (item.thumbnail && item.thumbnail->getSize().x > 0 && item.thumbnail->getSize().y > 0) {
            const float sc = std::min((kThumb - 8.f) / static_cast<float>(item.thumbnail->getSize().x),
                                      (kThumb - 8.f) / static_cast<float>(item.thumbnail->getSize().y));
            sf::Sprite thumb(*item.thumbnail);
            thumb.setScale(sc, sc);
            const float dw = static_cast<float>(item.thumbnail->getSize().x) * sc;
            const float dh = static_cast<float>(item.thumbnail->getSize().y) * sc;
            thumb.setPosition(tx + (kThumb - dw) / 2.f, ty + (kThumb - dh) / 2.f);
            win.draw(thumb);
        } else {
            drawTextC(win, font, "No preview", colors.muted, tx + kThumb / 2.f, ty + kThumb / 2.f - 6.f, 10);
        }

        if (selected) {
            const sf::FloatRect tick{tx + kThumb - 20.f, ty + kThumb - 20.f, 15.f, 15.f};
            drawRect(win, tick, colors.gold, sf::Color::Transparent, 0.f);
            drawTextC(win, font, "\xe2\x9c\x93", colors.surfaceInset,
                      tick.left + tick.width / 2.f, tick.top + 1.f, 11, true);
        }
    }

    win.setView(prevView);

    // Scrollbar hint when content overflows.
    if (galleryScrollMax_ > 0.f) {
        if (galleryExpanded_) {
            const float trackH = content.height;
            const float thumbH = std::max(24.f, trackH * (content.height / (content.height + galleryScrollMax_)));
            const float t = galleryScroll_ / galleryScrollMax_;
            const float sy = content.top + t * (trackH - thumbH);
            drawRect(win, {area.left + area.width - 5.f, sy, 3.f, thumbH}, colors.border, sf::Color::Transparent, 0.f);
        } else {
            const float trackW = content.width;
            const float thumbW = std::max(24.f, trackW * (content.width / (content.width + galleryScrollMax_)));
            const float t = galleryScroll_ / galleryScrollMax_;
            const float sx = content.left + t * (trackW - thumbW);
            drawRect(win, {sx, content.top + content.height + 2.f, thumbW, 3.f}, colors.border, sf::Color::Transparent, 0.f);
        }
    }
}

bool ResultPanel::handleEvent(const sf::Event& e) {
    // Wheel over the gallery region scrolls it (horizontal strip / vertical grid).
    if (e.type == sf::Event::MouseWheelScrolled) {
        const sf::Vector2f pos{static_cast<float>(e.mouseWheelScroll.x),
                               static_cast<float>(e.mouseWheelScroll.y)};
        if (!generating && !gallery.empty() && galleryRegion_.contains(pos) && galleryScrollMax_ > 0.f) {
            galleryScroll_ = std::clamp(galleryScroll_ - e.mouseWheelScroll.delta * 48.f,
                                        0.f, galleryScrollMax_);
            return true;
        }
        return false;
    }

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

        if (btnGalleryExpand_.contains(pos)) {
            galleryExpanded_ = !galleryExpanded_;
            galleryScroll_ = 0.f;
            lastSelectedIndex_ = -1;   // re-scroll selection into view in the new mode
            return true;
        }
        if (btnDeselect_.contains(pos)) {
            selectedIndex = -1;
            deselectRequested = true;
            return true;
        }
        if (btnImprove_.contains(pos)) { improveRequested = true; return true; }
        if (btnDelete_.contains(pos))  { deleteRequested  = true; return true; }

        for (size_t i = 0; i < thumbnailRects_.size(); ++i) {
            if (thumbnailRects_[i].contains(pos)) {
                // Clicking the already-selected thumbnail deselects it (exits edit mode).
                if (thumbnailIndices_[i] == selectedIndex) {
                    selectedIndex = -1;
                    deselectRequested = true;
                } else {
                    selectedIndex = thumbnailIndices_[i];
                }
                return true;
            }
        }
    }
    return false;
}

std::string ResultPanel::getSelectedImagePath() const {
    if (selectedIndex < 0 || selectedIndex >= static_cast<int>(gallery.size()))
        return {};
    return gallery[static_cast<size_t>(selectedIndex)].path;
}

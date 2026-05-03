#include "ResultPanel.hpp"
#include "../../enum/constants.hpp"
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
    case GenerationStage::DecodingImage:   return "Decoding image...";
    case GenerationStage::PostProcessing:  return "Post-processing...";
    case GenerationStage::Exploring:       return "Exploring...";
    case GenerationStage::Scoring:         return "Scoring candidates...";
    case GenerationStage::Refining:        return "Refining...";
    case GenerationStage::WritingManifest: return "Writing manifest...";
    default:
        return numSteps > 0
            ? "Step " + std::to_string(step) + " / " + std::to_string(numSteps)
            : "Generating...";
    }
}
void drawContractOverlay(sf::RenderWindow& win, const ResultPanel& panel,
                         const sf::FloatRect& imageRect, const Theme& theme) {
    if (!panel.showContractOverlay) return;
    const auto& spec = panel.activeSpec;
    const float canvasW = spec.canvasWidth > 0 ? static_cast<float>(spec.canvasWidth) : imageRect.width;
    const float canvasH = spec.canvasHeight > 0 ? static_cast<float>(spec.canvasHeight) : imageRect.height;
    if (canvasW <= 0.f || canvasH <= 0.f) return;

    const float sx = imageRect.width / canvasW;
    const float sy = imageRect.height / canvasH;

    if (spec.expectedBounds.w > 0 && spec.expectedBounds.h > 0) {
        const sf::FloatRect boundsRect{
            imageRect.left + static_cast<float>(spec.expectedBounds.x) * sx,
            imageRect.top + static_cast<float>(spec.expectedBounds.y) * sy,
            static_cast<float>(spec.expectedBounds.w) * sx,
            static_cast<float>(spec.expectedBounds.h) * sy
        };
        drawRect(win, boundsRect, sf::Color(0, 0, 0, 0), theme.colors().borderHi, 2.f);
    }

    if (spec.anchor.x != 0 || spec.anchor.y != 0) {
        const float ax = imageRect.left + static_cast<float>(spec.anchor.x) * sx;
        const float ay = imageRect.top + static_cast<float>(spec.anchor.y) * sy;
        constexpr float cross = 8.f;
        sf::Vertex horiz[] = {
            sf::Vertex({ax - cross, ay}, theme.colors().goldLt),
            sf::Vertex({ax + cross, ay}, theme.colors().goldLt)
        };
        sf::Vertex vert[] = {
            sf::Vertex({ax, ay - cross}, theme.colors().goldLt),
            sf::Vertex({ax, ay + cross}, theme.colors().goldLt)
        };
        win.draw(horiz, 2, sf::Lines);
        win.draw(vert, 2, sf::Lines);
    }
}
}

void ResultPanel::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void ResultPanel::ensureSelectedThumbnailVisible(int visibleCount) {
    if (gallery.empty() || visibleCount <= 0) {
        thumbnailScrollOffset_ = 0;
        lastVisibleSelectedIndex_ = -1;
        return;
    }

    const int maxOffset = std::max(0, static_cast<int>(gallery.size()) - visibleCount);
    thumbnailScrollOffset_ = std::clamp(thumbnailScrollOffset_, 0, maxOffset);
    if (selectedIndex < 0) {
        lastVisibleSelectedIndex_ = -1;
        return;
    }

    if (selectedIndex == lastVisibleSelectedIndex_) return;

    if (selectedIndex < thumbnailScrollOffset_) {
        thumbnailScrollOffset_ = selectedIndex;
    } else if (selectedIndex >= thumbnailScrollOffset_ + visibleCount) {
        thumbnailScrollOffset_ = selectedIndex - visibleCount + 1;
    }
    thumbnailScrollOffset_ = std::clamp(thumbnailScrollOffset_, 0, maxOffset);
    lastVisibleSelectedIndex_ = selectedIndex;
}

static constexpr float TAB_H = 28.f;

void ResultPanel::renderTabBar(sf::RenderWindow& win, sf::Font& font,
                                float barX, float barY, float barW) {
    tabRects_.clear();
    if (tabs.empty()) return;

    drawRect(win, {barX, barY, barW, TAB_H}, Col::Panel, Col::Border, 1.f);

    constexpr float tabPadX = 12.f;
    constexpr float tabGap  =  4.f;
    float tx = barX + tabGap;
    for (int i = 0; i < static_cast<int>(tabs.size()); ++i) {
        const float tw = tabPadX * 2.f
            + static_cast<float>(tabs[static_cast<size_t>(i)].name.size()) * 7.2f;
        const sf::FloatRect r{tx, barY + 3.f, tw, TAB_H - 6.f};
        const bool active = (i == activeTabIndex);
        drawRect(win, r, active ? Col::Blue : Col::Panel2,
                 active ? Col::GoldLt : Col::Border, 1.f);
        drawTextC(win, font, tabs[static_cast<size_t>(i)].name,
                  active ? Col::GoldLt : Col::Muted,
                  tx + tw / 2.f, barY + 7.f, 11, active);
        tabRects_.push_back(r);
        tx += tw + tabGap;
    }
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
    const float actionGenerateW = 136.f;
    const float actionAuxW = 76.f;
    const float actionGap = 10.f;

    const float tabBarH = (showTabs && !tabs.empty()) ? TAB_H : 0.f;

    // Panel background
    drawRect(win, rect_, colors.panel2, colors.border, metrics.borderWidth);
    drawRect(win, {x + 1.f, y + 1.f, w - 2.f, h - 2.f}, colors.surfaceInset, sf::Color::Transparent, 0.f);
    processedToggleRect_ = {};
    rawToggleRect_ = {};

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

    } else if (resultLoaded) {
        // ── Selected image preview ────────────────────────────────────────────
        const float galleryH = gallery.empty() ? 0.f : 132.f;
        const bool hasOutputMode = showOutputModeToggle;
        const bool hasReferenceInfo = showOutputModeToggle || selectedReferenceUsed || !selectedReferenceImage.empty();
        const float infoH = generationFailed.load() ? 62.f
                          : ((hasOutputMode ? 28.f : 0.f)
                             + (hasReferenceInfo ? 22.f : 0.f)
                             + (!validationChips.empty() ? 30.f : 0.f));
        const float previewBottom = y + h - galleryH - tabBarH - 72.f - infoH;
        const float frameX = x + 16.f;
        const float frameY = y + 16.f;
        const float frameW = w - 32.f;
        const float frameH = std::max(120.f, previewBottom - frameY);
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
        constexpr float navBtnSize = 34.f;
        constexpr float navBtnGap = 10.f;

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
        drawContractOverlay(win, *this, {imgX, imgY, imgW, imgH}, theme);

        float infoY = frameY + frameH + 5.f;
        if (showOutputModeToggle) {
            constexpr float toggleW = 88.f;
            constexpr float toggleH = 22.f;
            constexpr float toggleGap = 8.f;
            processedToggleRect_ = {frameX, infoY, toggleW, toggleH};
            rawToggleRect_ = {frameX + toggleW + toggleGap, infoY, toggleW, toggleH};
            drawButton(win, processedToggleRect_, "Processed",
                       showProcessedOutput ? colors.blue : colors.panel2,
                       showProcessedOutput ? colors.goldLt : colors.muted,
                       false, type.compact, font);
            drawButton(win, rawToggleRect_, "Raw",
                       showProcessedOutput ? colors.panel2 : colors.blue,
                       showProcessedOutput ? colors.muted : colors.goldLt,
                       false, type.compact, font);
            infoY += 28.f;
        }

        if (selectedReferenceUsed || !selectedReferenceImage.empty()) {
            std::string refLabel = selectedReferenceUsed ? "Reference shape" : "Prompt only";
            if (selectedReferenceUsed && selectedStructureStrength > 0.f) {
                char strengthBuf[16];
                std::snprintf(strengthBuf, sizeof(strengthBuf), "%.2f", selectedStructureStrength);
                refLabel += "  strength " + std::string(strengthBuf);
            }
            drawText(win, font, refLabel,
                     selectedReferenceUsed ? colors.blueLt : colors.muted,
                     frameX, infoY, type.compact, false);
            infoY += 22.f;
        }

        // ── Validation chips (below image frame, above gallery/action bar) ────
        if (!validationChips.empty()) {
            constexpr float chipH   = 20.f;
            constexpr float chipGap = 6.f;
            const float chipsY = infoY;
            const int   n      = static_cast<int>(validationChips.size());
            const float totalGaps = static_cast<float>(n - 1) * chipGap;
            const float chipW = (frameW - totalGaps) / static_cast<float>(n);
            for (int ci = 0; ci < n; ++ci) {
                const auto& chip = validationChips[static_cast<size_t>(ci)];
                const float cx2  = frameX + static_cast<float>(ci) * (chipW + chipGap);
                sf::Color borderCol = (chip.status == 0) ? sf::Color(60, 180, 80)
                                    : (chip.status == 1) ? colors.goldLt
                                                         : colors.redLt;
                drawRect(win, {cx2, chipsY, chipW, chipH}, colors.panel2, borderCol, 1.f);
                const std::string label = chip.name + ": " + chip.detail;
                drawTextC(win, font, label, borderCol,
                          cx2 + chipW / 2.f, chipsY + 3.f, type.helper);
            }
        }

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
            drawButton(win, btnPrevImage_, "<", colors.panel2, colors.goldLt, false, 16, font);
            drawButton(win, btnNextImage_, ">", colors.panel2, colors.goldLt, false, 16, font);
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
            drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, colors.panel2, colors.redLt, metrics.borderWidth);
            drawTextC(win, font, "Generation failed", colors.redLt, cx, bannerY + 8.f, type.body, true);
            drawTextC(win, font, msg, colors.muted, cx, bannerY + 28.f, type.helper);
        }

        if (!gallery.empty()) {
            const float stripX = x + 12.f;
            const float stripW = w - 24.f;
            const float stripY = y + h - galleryH - 60.f;
            if (showTabs && !tabs.empty())
                renderTabBar(win, font, stripX, stripY - tabBarH, stripW);
            renderThumbnailStrip(win, font, stripX, stripY, stripW);
        } else {
            tabRects_.clear();
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
        drawRect(win, {cx - bannerW / 2.f, bannerY, bannerW, bannerH}, colors.panel2, colors.redLt, metrics.borderWidth);
        drawTextC(win, font, "Generation failed", colors.redLt, cx, bannerY + 8.f, type.body, true);
        drawTextC(win, font, msg, colors.muted, cx, bannerY + 28.f, type.helper);
    } else if (!gallery.empty()) {
        btnPrevImage_ = {};
        btnNextImage_ = {};
        drawTextC(win, font, "Select an image", colors.borderHi, cx, y + 32.f, type.sectionTitle);
        const float stripX = x + 12.f;
        const float stripW = w - 24.f;
        const float stripY = y + 56.f + tabBarH;
        if (showTabs && !tabs.empty())
            renderTabBar(win, font, stripX, stripY - tabBarH, stripW);
        renderThumbnailStrip(win, font, stripX, stripY, stripW);
    } else {
        // Placeholder when no image yet
        btnPrevImage_ = {};
        btnNextImage_ = {};
        btnPrevThumbs_ = {};
        btnNextThumbs_ = {};
        thumbnailRects_.clear();
        thumbnailIndices_.clear();
        drawTextC(win, font, "No image generated yet", colors.borderHi, cx, y + h / 2.f - 20.f, type.sectionTitle);
    }

    // ── Buttons at the bottom of the panel ───────────────────────────────────
    if (!generating) {
            drawRect(win, {actionBarX, actionBarY, actionBarW, actionBarH},
                     colors.panel, colors.border, metrics.borderWidth);

            if (bestWallCandidateScore >= 0.f) {
                char scoreBuf[32];
                std::snprintf(scoreBuf, sizeof(scoreBuf), "best: %.0f", bestWallCandidateScore);
                const sf::Color scoreCol = (bestWallCandidateScore < 150.f) ? sf::Color(60, 180, 80)
                                         : (bestWallCandidateScore < 500.f) ? colors.goldLt
                                                                             : colors.redLt;
                drawTextC(win, font, scoreBuf, scoreCol, cx, actionBarY + 4.f, type.helper, false);
            }

            if (resultLoaded) {
                auto placeButtons = [&](bool withImprove) {
                    const float totalW = (withImprove ? actionAuxW + actionGap : 0.f)
                                       + actionAuxW + actionGap
                                       + actionGenerateW;
                    float curX = cx - totalW / 2.f;
                    if (withImprove) {
                        btnImprove_ = {curX, y + h - 49.f, actionAuxW, actionButtonH};
                        drawButton(win, btnImprove_, "Edit", colors.panel2, colors.goldLt, false, type.body, font);
                        curX += actionAuxW + actionGap;
                    } else {
                        btnImprove_ = {};
                    }
                    btnDelete_ = {curX, y + h - 49.f, actionAuxW, actionButtonH};
                    curX += actionAuxW + actionGap;
                    btnGenerate_ = {curX, y + h - 52.f, actionGenerateW, 38.f};
                };

                if (mode == WorkflowMode::Generate && showImproveButton)
                    placeButtons(true);
                else if (mode == WorkflowMode::Generate)
                    placeButtons(false);
                else {
                    btnImprove_          = {};
                    const float totalW   = actionAuxW + actionGap + actionGenerateW;
                    btnDelete_   = {cx - totalW / 2.f, y + h - 49.f, actionAuxW, actionButtonH};
                    btnGenerate_ = {btnDelete_.left + actionAuxW + actionGap, y + h - 52.f, actionGenerateW, 38.f};
                }
                drawButton(win, btnDelete_, "Delete", colors.panel2, colors.redLt, false, type.body, font);
            } else {
                btnImprove_          = {};
                btnDelete_           = {};
                btnGenerate_ = {cx - actionGenerateW / 2.f, y + h - 52.f, actionGenerateW, 38.f};
            }
            drawButton(win, btnGenerate_, generateButtonLabel, colors.blue, colors.goldLt, false, 13, font);
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

        if (btnImprove_.contains(pos)) {
            improveRequested = true;
            return true;
        }
        if (btnDelete_.contains(pos)) {
            deleteRequested = true;
            return true;
        }
        if (processedToggleRect_.contains(pos) && !showProcessedOutput) {
            showProcessedOutput = true;
            outputModeChanged = true;
            return true;
        }
        if (rawToggleRect_.contains(pos) && showProcessedOutput) {
            showProcessedOutput = false;
            outputModeChanged = true;
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
            lastVisibleSelectedIndex_ = selectedIndex;
            return true;
        }
        if (btnNextThumbs_.contains(pos) && !gallery.empty()) {
            const int page = std::max(1, static_cast<int>(thumbnailIndices_.size()));
            const int maxOffset = std::max(0, static_cast<int>(gallery.size()) - page);
            thumbnailScrollOffset_ = std::min(maxOffset, thumbnailScrollOffset_ + page);
            lastVisibleSelectedIndex_ = selectedIndex;
            return true;
        }
        for (int i = 0; i < static_cast<int>(thumbnailRects_.size()); ++i) {
            if (thumbnailRects_[static_cast<size_t>(i)].contains(pos)) {
                selectedIndex = thumbnailIndices_[static_cast<size_t>(i)];
                return true;
            }
        }
        if (showTabs) {
            for (int i = 0; i < static_cast<int>(tabRects_.size()); ++i) {
                if (tabRects_[static_cast<size_t>(i)].contains(pos) && i != activeTabIndex) {
                    activeTabIndex = i;
                    tabChanged     = true;
                    return true;
                }
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
        if (item.recommended || item.usable || item.near) {
            const std::string badge = item.recommended ? "BEST" : (item.usable ? "OK" : "NEAR");
            const sf::Color badgeFill = item.recommended ? Col::GoldLt
                                      : item.usable ? sf::Color(60, 180, 80)
                                                    : sf::Color(180, 140, 60);
            const sf::FloatRect badgeRect{thumbX + 5.f, thumbY + 5.f,
                                          item.recommended ? 34.f : (item.usable ? 24.f : 34.f), 15.f};
            drawRect(win, badgeRect, badgeFill, sf::Color::Transparent, 0.f);
            drawTextC(win, font, badge, sf::Color(24, 26, 30),
                      badgeRect.left + badgeRect.width / 2.f, badgeRect.top + 2.f, 9, true);
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

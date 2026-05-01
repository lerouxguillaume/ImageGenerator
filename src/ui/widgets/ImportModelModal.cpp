#include "ImportModelModal.hpp"

#include "../Buttons.hpp"
#include "../Helpers.hpp"
#include "../Theme.h"

// ── Arch cycle ────────────────────────────────────────────────────────────────

static const char* kArchLabels[] = {"Auto (detect)", "SD 1.5", "SDXL"};
static const char* kArchArgs[]   = {"auto",          "sd15",   "sdxl"};

void ImportModelModal::cycleArch() {
    archIndex_ = (archIndex_ + 1) % 3;
}

std::string ImportModelModal::archArg() const {
    return kArchArgs[archIndex_];
}

// ── Sync from importer ────────────────────────────────────────────────────────

void ImportModelModal::syncFrom(const ModelImporter& importer) {
    importerState_ = importer.getState();
    statusMsg_     = importer.getStatusMsg();

    const auto all = importer.getLogLines();
    if (all.size() > static_cast<size_t>(kVisibleLogLines))
        logLines_ = std::vector<std::string>(all.end() - kVisibleLogLines, all.end());
    else
        logLines_ = all;
}

// ── Log area renderer ─────────────────────────────────────────────────────────

void ImportModelModal::drawLogArea(sf::RenderWindow& win, float x, float y, float w, float h) {
    auto& theme        = Theme::instance();
    const auto& colors = theme.colors();
    sf::Font& font     = theme.getFont();
    const unsigned sz  = theme.typography().compact;
    const float lineH  = static_cast<float>(sz) + 4.f;

    Helpers::drawRect(win, {x, y, w, h}, colors.surfaceInset, colors.border, 1.f);

    // Clip lines to the box via a scissor view
    const sf::View origView = win.getView();
    const sf::FloatRect vp  = origView.getViewport();
    const auto winSz        = win.getSize();
    const float scaleX      = vp.width  / static_cast<float>(winSz.x);
    const float scaleY      = vp.height / static_cast<float>(winSz.y);

    sf::View clipView(sf::FloatRect(x, y, w, h));
    clipView.setViewport({
        (x * scaleX) / static_cast<float>(winSz.x),
        (y * scaleY) / static_cast<float>(winSz.y),
        (w * scaleX) / static_cast<float>(winSz.x),
        (h * scaleY) / static_cast<float>(winSz.y),
    });
    win.setView(clipView);

    const float pad = 6.f;
    for (size_t i = 0; i < logLines_.size(); ++i) {
        const float ly = y + pad + static_cast<float>(i) * lineH;
        Helpers::drawText(win, font, logLines_[i], colors.muted,
                          x + pad, ly, sz, false);
    }

    win.setView(origView);
}

// ── Render ────────────────────────────────────────────────────────────────────

void ImportModelModal::render(sf::RenderWindow& win) {
    auto& theme        = Theme::instance();
    const auto& colors = theme.colors();
    const auto& type   = theme.typography();
    sf::Font& font     = theme.getFont();

    const float winW = static_cast<float>(win.getSize().x);
    const float winH = static_cast<float>(win.getSize().y);
    const float cx   = winW / 2.f;
    const float cy   = winH / 2.f;

    // Dim overlay
    sf::RectangleShape dim({winW, winH});
    dim.setFillColor(colors.overlay);
    win.draw(dim);

    // Modal box
    const float mW = 680.f;
    const float mH = 420.f;
    const float mX = cx - mW / 2.f;
    const float mY = cy - mH / 2.f;
    modalRect_ = {mX, mY, mW, mH};
    Helpers::drawRect(win, modalRect_, colors.panel, colors.border, 1.f);
    Helpers::drawText(win, font, "Import Model", colors.gold,
                      mX + 20.f, mY + 18.f, type.subsectionTitle, true);

    const float pad  = 20.f;
    const float rowH = 32.f;
    const float labelW = 70.f;
    float curY = mY + 54.f;

    // ── File row ─────────────────────────────────────────────────────────────
    Helpers::drawText(win, font, "File", colors.muted,
                      mX + pad, curY + 8.f, type.sectionTitle, false);

    const float fieldW  = mW - pad * 2.f - labelW - 80.f;
    const float fieldX  = mX + pad + labelW;
    fileFieldRect_ = {fieldX, curY, fieldW, rowH};
    Helpers::drawRect(win, fileFieldRect_, colors.surfaceInset, colors.border, 1.f);

    // Clip the file path text if too long (show tail)
    const std::string displayPath = filePath.empty() ? "No file selected…" : filePath;
    Helpers::drawText(win, font,
                      displayPath.size() > 60
                          ? "…" + displayPath.substr(displayPath.size() - 57)
                          : displayPath,
                      filePath.empty() ? colors.muted : colors.text,
                      fieldX + 6.f, curY + 9.f, type.sectionTitle, false);

    btnBrowse_ = {fieldX + fieldW + 8.f, curY, 68.f, rowH};
    drawButton(win, btnBrowse_, "Browse", colors.panel2, colors.text, false, 12, font);

    curY += rowH + 10.f;

    // ── Arch row ──────────────────────────────────────────────────────────────
    Helpers::drawText(win, font, "Arch", colors.muted,
                      mX + pad, curY + 8.f, type.sectionTitle, false);

    btnArch_ = {mX + pad + labelW, curY, 160.f, rowH};
    drawButton(win, btnArch_, kArchLabels[archIndex_], colors.panel2, colors.text, false, 12, font);
    Helpers::drawText(win, font, "▶", colors.muted,
                      btnArch_.left + btnArch_.width - 18.f, curY + 9.f, 10, false);

    curY += rowH + 14.f;

    // ── Log area ──────────────────────────────────────────────────────────────
    const float logH = static_cast<float>(kVisibleLogLines) * 16.f + 12.f;
    drawLogArea(win, mX + pad, curY, mW - pad * 2.f, logH);
    curY += logH + 12.f;

    // ── Status label ──────────────────────────────────────────────────────────
    const sf::Color statusColor = [&] {
        switch (importerState_) {
            case ModelImporter::State::Done:   return colors.green;
            case ModelImporter::State::Failed: return colors.red;
            default:                           return colors.muted;
        }
    }();
    if (!statusMsg_.empty())
        Helpers::drawText(win, font, statusMsg_, statusColor,
                          mX + pad, curY, type.sectionTitle, false);

    // ── Bottom buttons ────────────────────────────────────────────────────────
    const float btnW = 100.f;
    const float btnH = 34.f;
    const float btnY = mY + mH - btnH - 16.f;
    btnClose_  = {mX + mW - btnW - pad, btnY, btnW, btnH};
    btnAction_ = {mX + mW - btnW * 2.f - pad - 10.f, btnY, btnW, btnH};

    const bool importing = importerState_ != ModelImporter::State::Idle
                        && importerState_ != ModelImporter::State::Done
                        && importerState_ != ModelImporter::State::Failed;

    const bool canImport = !filePath.empty()
                        && importerState_ == ModelImporter::State::Idle;

    drawButton(win, btnAction_,
               importing ? "Cancel" : "Import",
               importing ? colors.red : colors.blue,
               colors.text,
               !importing && !canImport,
               13, font);

    drawButton(win, btnClose_, "Close", colors.panel2, colors.text, false, 13, font);
}

// ── Event handling ────────────────────────────────────────────────────────────

bool ImportModelModal::handleEvent(const sf::Event& e, sf::RenderWindow& win) {
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        closeRequested = true;
        return true;
    }

    if (e.type != sf::Event::MouseButtonPressed
            || e.mouseButton.button != sf::Mouse::Left)
        return modalRect_.contains(
            win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}));

    const sf::Vector2f pos = win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y});

    if (!modalRect_.contains(pos)) return false;

    if (btnBrowse_.contains(pos))  { browseRequested = true; return true; }
    if (btnArch_.contains(pos))    { cycleArch(); return true; }
    if (btnClose_.contains(pos))   { closeRequested = true; return true; }

    if (btnAction_.contains(pos)) {
        const bool importing = importerState_ != ModelImporter::State::Idle
                            && importerState_ != ModelImporter::State::Done
                            && importerState_ != ModelImporter::State::Failed;
        if (importing)
            cancelRequested = true;
        else if (!filePath.empty() && importerState_ == ModelImporter::State::Idle)
            importRequested = true;
        return true;
    }

    return true; // consume all clicks inside modal
}

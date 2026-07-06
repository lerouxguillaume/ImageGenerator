#include "ImportModelModal.hpp"

#include <algorithm>
#include <cmath>

#include "../Buttons.hpp"
#include "../Helpers.hpp"
#include "../Theme.h"

// m:ss formatter for elapsed / ETA captions.
static std::string fmtClock(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const int total = static_cast<int>(seconds + 0.5);
    const int m = total / 60;
    const int s = total % 60;
    return std::to_string(m) + ":" + (s < 10 ? "0" : "") + std::to_string(s);
}

// ── Sync from importer ────────────────────────────────────────────────────────

void ImportModelModal::syncFrom(const ModelImporter& importer) {
    importerState_ = importer.getState();
    statusMsg_     = importer.getStatusMsg();
    inspection_    = importer.getInspectionResult();
    verifyChecks_  = importer.getVerifyChecks();
    elapsed_       = importer.getElapsedSeconds();
    importer.getExportProgress(exportStep_, exportTotal_);

    const auto all = importer.getLogLines();
    latestLog_ = all.empty() ? std::string() : all.back();

    updateEta();
}

// ── ETA: countdown re-based at each completed unit (never climbs) ──────────────

// Cumulative relative cost of the first `k` steps of a phase. Export is heavily
// dominated by the UNet step (~minutes vs seconds for the text-encoder/VAE
// steps), so a step count alone is a poor time proxy; verify checks are roughly
// uniform. `total` distinguishes the 5-step SDXL export from the 4-step SD1.5
// export. Weights are relative — only their ratios matter, since the actual
// seconds-per-weight rate is learned at runtime.
static double phaseCumWeight(int phase, int k, int total) {
    if (k <= 0) return 0.0;
    if (phase == 2) { // Exporting
        static const double sdxl[5] = {1.0, 1.0, 90.0, 5.0, 5.0}; // TE, TE2, UNet, VAEdec, VAEenc
        static const double sd15[4] = {1.0, 60.0, 4.0, 4.0};      // TE, UNet, VAEdec, VAEenc
        const double* w = total >= 5 ? sdxl : sd15;
        const int     n = total >= 5 ? 5 : 4;
        double sum = 0.0;
        for (int i = 0; i < k && i < n; ++i) sum += w[i];
        return sum;
    }
    return static_cast<double>(k); // uniform (verify)
}

void ImportModelModal::updateEta() {
    // Pick the countable unit for the current phase. `done` counts the in-progress
    // step for export (the log announces a step as it *starts*), but completed
    // checks for verify — so completed-unit count differs by phase.
    int  done = 0, total = 0, phase = -1;
    bool doneIsInProgress = false;
    if (importerState_ == ModelImporter::State::Exporting) {
        done = exportStep_; total = exportTotal_; phase = 2; doneIsInProgress = true;
    } else if (importerState_ == ModelImporter::State::Verifying) {
        done  = static_cast<int>(verifyChecks_.size());
        total = inspection_.architecture == SafetensorsInfo::Architecture::SDXL ? 4 : 3;
        phase = 4;
    }

    // Reset tracking on a new import or when the phase changes.
    if (importerState_ == ModelImporter::State::Idle || phase != etaPhase_) {
        etaPhase_      = phase;
        etaUnitDone_   = -1;
        etaPhaseStart_ = elapsed_;
        etaValid_      = false;
    }

    if (total <= 0 || done <= 0 || done > total) return;

    // Re-base only when a unit boundary is crossed; between boundaries the
    // displayed remaining = etaTargetSec_ - elapsed_ ticks down on its own.
    if (done != etaUnitDone_) {
        etaUnitDone_ = done;

        // Steps whose duration is now known, and their cost weight.
        const int    completedSteps = doneIsInProgress ? done - 1 : done;
        const double completedW     = phaseCumWeight(phase, completedSteps, total);
        const double totalW         = phaseCumWeight(phase, total, total);
        const double timeInPhase    = elapsed_ - etaPhaseStart_;

        // Need at least one completed unit to learn the seconds-per-weight rate;
        // until then (e.g. still inside the first, fast text-encoder step) we show
        // no ETA rather than a bogus one.
        if (completedW > 0.0 && timeInPhase > 0.0 && totalW > completedW) {
            const double rate = timeInPhase / completedW; // seconds per weight-unit
            etaTargetSec_ = elapsed_ + rate * (totalW - completedW);
            etaValid_     = true;
        } else {
            etaValid_ = false;
        }
    }
}

// ── Overall progress fraction (0..1) across all phases ─────────────────────────

float ImportModelModal::overallProgress() const {
    const int expectedVerify =
        inspection_.architecture == SafetensorsInfo::Architecture::SDXL ? 4 : 3;
    switch (importerState_) {
        case ModelImporter::State::Idle:            return 0.0f;
        case ModelImporter::State::Analyzing:       return 0.02f;
        case ModelImporter::State::SettingUpPython: return 0.06f;
        case ModelImporter::State::Exporting: {
            float f = exportTotal_ > 0
                          ? static_cast<float>(exportStep_) / static_cast<float>(exportTotal_)
                          : 0.25f;
            return 0.10f + 0.72f * std::min(1.0f, std::max(0.0f, f));
        }
        case ModelImporter::State::Validating:      return 0.85f;
        case ModelImporter::State::Verifying: {
            float f = static_cast<float>(verifyChecks_.size())
                    / static_cast<float>(expectedVerify);
            return 0.88f + 0.11f * std::min(1.0f, f);
        }
        case ModelImporter::State::Done:            return 1.0f;
        case ModelImporter::State::Failed: {
            // Freeze roughly where it broke: use whatever the last phase implies.
            if (!verifyChecks_.empty()) return 0.90f;
            return 0.45f;
        }
    }
    return 0.0f;
}

// ── Progress bar (with elapsed / ETA and an indeterminate marquee for setup) ──

void ImportModelModal::drawProgressBar(sf::RenderWindow& win, float x, float y, float w) {
    auto& theme        = Theme::instance();
    const auto& colors = theme.colors();
    sf::Font& font     = theme.getFont();

    const bool running = importerState_ != ModelImporter::State::Idle
                      && importerState_ != ModelImporter::State::Done
                      && importerState_ != ModelImporter::State::Failed;
    const bool failed  = importerState_ == ModelImporter::State::Failed;
    const bool done    = importerState_ == ModelImporter::State::Done;

    // Phase caption (left) + elapsed / ETA (right).
    const char* phaseName = "Ready to import";
    switch (importerState_) {
        case ModelImporter::State::Analyzing:       phaseName = "Analyzing file";        break;
        case ModelImporter::State::SettingUpPython: phaseName = "Setting up Python";     break;
        case ModelImporter::State::Exporting:       phaseName = "Exporting ONNX";        break;
        case ModelImporter::State::Validating:      phaseName = "Validating files";      break;
        case ModelImporter::State::Verifying:       phaseName = "Verifying inference";   break;
        case ModelImporter::State::Done:            phaseName = "Done";                  break;
        case ModelImporter::State::Failed:          phaseName = "Failed";                break;
        default: break;
    }
    std::string leftCap = phaseName;
    if (importerState_ == ModelImporter::State::Exporting && exportTotal_ > 0)
        leftCap += "  (" + std::to_string(exportStep_) + "/" + std::to_string(exportTotal_) + ")";

    const float frac = overallProgress();
    std::string rightCap;
    if (running || done || failed)
        rightCap = fmtClock(elapsed_) + " elapsed";
    // Countdown ETA — rebased at each completed unit, so it ticks down.
    if (etaValid_) {
        const double remaining = etaTargetSec_ - elapsed_;
        if (remaining > 0.5)
            rightCap += "   ~" + fmtClock(remaining) + " left";
    }

    Helpers::drawText(win, font, leftCap,
                      failed ? colors.red : (done ? colors.green : colors.text),
                      x, y, 13, true);
    if (!rightCap.empty()) {
        const float approxW = static_cast<float>(rightCap.size()) * 6.2f;
        Helpers::drawText(win, font, rightCap, colors.muted,
                          x + w - approxW, y + 1.f, 12, false);
    }

    // Bar track + fill.
    const float barY = y + 22.f;
    const float barH = 12.f;
    Helpers::drawRect(win, {x, barY, w, barH}, colors.surfaceInset, colors.border, 1.f);

    const sf::Color fillCol = failed ? colors.red : (done ? colors.green : colors.blue);
    if (importerState_ == ModelImporter::State::SettingUpPython) {
        // Unknown duration → indeterminate marquee so it still feels alive.
        const float segW = w * 0.28f;
        const float span = w - segW;
        float t = std::fmod(static_cast<float>(elapsed_) * 90.f, 2.f * span);
        if (t > span) t = 2.f * span - t;  // ping-pong
        Helpers::drawRect(win, {x + t + 1.f, barY + 1.f, segW - 2.f, barH - 2.f},
                          fillCol, fillCol, 0.f);
    } else if (frac > 0.0f) {
        const float fw = std::max(2.f, (w - 2.f) * frac);
        Helpers::drawRect(win, {x + 1.f, barY + 1.f, fw, barH - 2.f}, fillCol, fillCol, 0.f);
    }
}

// ── Checklist (green ✓ done · blue ▶ active · red ✗ failed · • pending) ───────

void ImportModelModal::drawChecklist(sf::RenderWindow& win, float x, float y, float w) {
    auto& theme        = Theme::instance();
    const auto& colors = theme.colors();
    sf::Font& font     = theme.getFont();

    static const char* kSteps[] = {
        "Analyze file", "Set up Python", "Export ONNX", "Validate files", "Verify inference"};
    constexpr int kN = 5;

    // Current step index and whether we failed.
    int cur = -1;
    bool failed = importerState_ == ModelImporter::State::Failed;
    switch (importerState_) {
        case ModelImporter::State::Idle:            cur = -1; break;
        case ModelImporter::State::Analyzing:       cur = 0;  break;
        case ModelImporter::State::SettingUpPython: cur = 1;  break;
        case ModelImporter::State::Exporting:       cur = 2;  break;
        case ModelImporter::State::Validating:      cur = 3;  break;
        case ModelImporter::State::Verifying:       cur = 4;  break;
        case ModelImporter::State::Done:            cur = 5;  break;
        case ModelImporter::State::Failed:
            // Approximate the failed step: if we produced any verify checks the
            // break was in Verify, otherwise treat Export as the failing stage.
            cur = verifyChecks_.empty() ? 2 : 4;
            break;
    }

    // Animated ellipsis on the active step for a sense of life.
    const int dots = static_cast<int>(elapsed_ * 2.0) % 4;
    const std::string ell(static_cast<size_t>(dots), '.');

    const float rowH = 26.f;
    float ly = y;
    for (int i = 0; i < kN; ++i) {
        const bool isDone   = i < cur;
        const bool isActive = i == cur && !failed;
        const bool isFail   = i == cur && failed;

        const char* icon = "•";
        sf::Color   col  = colors.muted;
        if (isDone)        { icon = "✓"; col = colors.green; }
        else if (isFail)   { icon = "✗"; col = colors.red;   }
        else if (isActive) { icon = "▶"; col = colors.blue;  }

        Helpers::drawText(win, font, icon, col, x, ly, 15, true);
        std::string label = kSteps[i];
        if (isActive) label += ell;
        Helpers::drawText(win, font, label,
                          (isDone || isActive || isFail) ? colors.text : colors.muted,
                          x + 24.f, ly + 1.f, 14, isActive || isFail);

        ly += rowH;

        // Verification sub-items nest under "Verify inference".
        if (i == 4 && !verifyChecks_.empty()) {
            for (const auto& c : verifyChecks_) {
                const char* sicon = "✓";
                sf::Color   scol  = colors.green;
                if (c.status == ModelImporter::VerifyCheck::Status::Warn) { sicon = "!"; scol = colors.injury; }
                if (c.status == ModelImporter::VerifyCheck::Status::Fail) { sicon = "✗"; scol = colors.red;    }
                if (c.status == ModelImporter::VerifyCheck::Status::Skip) { sicon = "–"; scol = colors.muted;  }

                Helpers::drawText(win, font, sicon, scol, x + 26.f, ly, 13, true);
                Helpers::drawText(win, font, c.name, scol, x + 42.f, ly + 1.f, 12, false);

                // Reason, clipped to the remaining width.
                const float detailX = x + 42.f + 118.f;
                const size_t maxCh   = static_cast<size_t>((x + w - detailX) / 6.0f);
                std::string detail   = c.detail;
                if (maxCh > 1 && detail.size() > maxCh)
                    detail = detail.substr(0, maxCh - 1) + "…";
                Helpers::drawText(win, font, detail, colors.muted, detailX, ly + 1.f, 12, false);
                ly += 20.f;
            }
        }
    }
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
    const float mW = 720.f;
    const float mH = 540.f;
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

    // ── Arch row (read-only — architecture is always inferred) ────────────────
    Helpers::drawText(win, font, "Arch", colors.muted,
                      mX + pad, curY + 8.f, type.sectionTitle, false);

    const sf::FloatRect archField = {mX + pad + labelW, curY, fieldW + 76.f, rowH};
    Helpers::drawRect(win, archField, colors.surfaceInset, colors.border, 1.f);

    const bool archDetected =
        inspection_.valid
        && inspection_.architecture != SafetensorsInfo::Architecture::Unknown;
    const std::string archText =
        archDetected ? (std::string(inspection_.architectureName())
                        + "  ·  " + inspection_.dtype + "   (auto-detected)")
                     : "Auto-detected from file on import";
    Helpers::drawText(win, font, archText,
                      archDetected ? colors.goldLt : colors.muted,
                      archField.left + 8.f, curY + 9.f, type.sectionTitle, false);

    curY += rowH + 16.f;

    // ── Progress bar (phase caption + elapsed/ETA + fill) ─────────────────────
    drawProgressBar(win, mX + pad, curY, mW - pad * 2.f);
    curY += 46.f;

    // ── Checklist of pipeline steps ───────────────────────────────────────────
    drawChecklist(win, mX + pad, curY, mW - pad * 2.f);

    // ── Bottom buttons ────────────────────────────────────────────────────────
    const float btnW = 100.f;
    const float btnH = 34.f;
    const float btnY = mY + mH - btnH - 16.f;

    // ── Live caption above the buttons: status on Done/Failed, else last log ──
    const sf::Color statusColor = [&] {
        switch (importerState_) {
            case ModelImporter::State::Done:   return colors.green;
            case ModelImporter::State::Failed: return colors.red;
            default:                           return colors.muted;
        }
    }();
    const bool terminal = importerState_ == ModelImporter::State::Done
                       || importerState_ == ModelImporter::State::Failed;
    std::string caption = terminal ? statusMsg_ : latestLog_;
    if (caption.size() > 92) caption = caption.substr(0, 91) + "…";
    if (!caption.empty())
        Helpers::drawText(win, font, caption, statusColor,
                          mX + pad, btnY - 26.f, type.compact, false);
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

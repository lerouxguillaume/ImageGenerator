#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include <SFML/Graphics.hpp>

#include "../../import/ModelImporter.hpp"

class ImportModelModal {
public:
    // ── Action flags (read by controller each frame) ─────────────────────────
    bool                  browseRequested  = false;
    bool                  importRequested  = false;  // read filePath + archArg()
    bool                  cancelRequested  = false;
    bool                  closeRequested   = false;

    // ── Input state (written by controller) ──────────────────────────────────
    std::string           filePath;         // set by controller after file-picker

    // ── Render / event ───────────────────────────────────────────────────────

    // Call once per frame before render() to pull latest importer state.
    void syncFrom(const ModelImporter& importer);

    void         render(sf::RenderWindow& win);
    // Returns true if the event was consumed (prevents menu from handling it).
    bool         handleEvent(const sf::Event& e, sf::RenderWindow& win);

    // Arch is always inferred from the file — the import script re-detects when
    // passed "auto", so the UI never lets the user override it.
    std::string archArg() const { return "auto"; }

private:
    void  drawProgressBar(sf::RenderWindow& win, float x, float y, float w);
    void  drawChecklist(sf::RenderWindow& win, float x, float y, float w);
    float overallProgress() const;   // 0..1 across all phases

    // Modal geometry (computed in render, reused in handleEvent)
    sf::FloatRect modalRect_{};
    sf::FloatRect btnBrowse_{};
    sf::FloatRect btnAction_{}; // Import or Cancel
    sf::FloatRect btnClose_{};
    sf::FloatRect fileFieldRect_{};

    // Display state synced from ModelImporter
    ModelImporter::State  importerState_  = ModelImporter::State::Idle;
    std::string           statusMsg_;
    std::string           latestLog_;       // last log line, shown as live caption
    SafetensorsInfo       inspection_;      // detected arch/dtype (after Analyzing)
    std::vector<ModelImporter::VerifyCheck> verifyChecks_;
    double                elapsed_    = 0.0; // seconds since start
    int                   exportStep_ = 0;
    int                   exportTotal_= 0;

    // ETA estimate — a countdown re-based each time a discrete unit (export step
    // or verify check) completes, so it ticks down instead of climbing.
    void   updateEta();
    int    etaPhase_        = -1;   // which phase the unit tracking belongs to
    int    etaUnitDone_     = -1;
    double etaUnitTime_     = 0.0;  // EMA seconds per unit
    double etaElapsedAtUnit_= 0.0;
    double etaTargetSec_    = 0.0;  // predicted total elapsed at completion
    bool   etaValid_        = false;
};

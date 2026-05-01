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

    // Returns the arch argument string for the import script ("auto"/"sd15"/"sdxl")
    std::string archArg() const;

private:
    void drawLogArea(sf::RenderWindow& win, float x, float y, float w, float h);
    void cycleArch();

    // Modal geometry (computed in render, reused in handleEvent)
    sf::FloatRect modalRect_{};
    sf::FloatRect btnBrowse_{};
    sf::FloatRect btnArch_{};
    sf::FloatRect btnAction_{}; // Import or Cancel
    sf::FloatRect btnClose_{};
    sf::FloatRect fileFieldRect_{};

    // Display state synced from ModelImporter
    ModelImporter::State  importerState_  = ModelImporter::State::Idle;
    std::string           statusMsg_;
    std::vector<std::string> logLines_;

    // Arch selection: 0=Auto, 1=SD 1.5, 2=SDXL
    int archIndex_ = 0;

    static constexpr int kVisibleLogLines = 14;
};

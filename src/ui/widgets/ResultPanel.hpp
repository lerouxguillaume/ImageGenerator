#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../enum/enums.hpp"

// Right panel: displays the generated image, Generate/Cancel buttons,
// progress bar during generation, and error banner on failure.
class ResultPanel {
public:
    struct GalleryItem {
        std::string                 path;
        std::string                 filename;
        std::shared_ptr<sf::Texture> thumbnail;
        float                       score = -1.f;
    };

    // ── Display ───────────────────────────────────────────────────────────────
    sf::Texture resultTexture;
    bool        resultLoaded = false;
    bool        showCheckerboard = false; // render checkerboard behind transparent images
    std::vector<GalleryItem> gallery;
    int                      selectedIndex = -1;
    bool                     showImproveButton = true;

    // ── Generation state (shared with background thread via atomics) ──────────
    bool                         generating          = false;
    std::atomic<bool>            generationDone{false};
    std::atomic<bool>            cancelToken{false};
    std::atomic<int>             generationStep{0};
    std::atomic<int>             generationImageNum{0};
    std::atomic<int>             generationTotalImages{1};
    std::atomic<bool>            generationFailed{false};
    std::atomic<int>             generationId{0};
    std::atomic<GenerationStage> generationStage{GenerationStage::Idle};
    std::string       generationErrorMsg;
    std::string       lastImagePath;
    std::string       displayedImagePath;

    // ── Validation chips (set by controller after selectGalleryImage) ─────────
    struct ValidationChip {
        std::string name;
        int         status = 0; // 0=pass 1=warning 2=fail
        std::string detail;
    };
    std::vector<ValidationChip> validationChips;

    // Action flags — set by handleEvent, cleared by controller
    bool improveRequested  = false;
    bool deleteRequested   = false;
    bool deselectRequested = false;  // unselect the current image → exit edit mode

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font,
                int numSteps); // numSteps needed for progress bar denominator

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);
    std::string getSelectedImagePath() const;
    sf::FloatRect getRect() const { return rect_; }

private:
    sf::FloatRect rect_;
    sf::FloatRect btnCancelGenerate_;
    sf::FloatRect btnImprove_;
    sf::FloatRect btnDelete_;
    sf::FloatRect btnDeselect_;                   // × on the preview → unselect
    sf::FloatRect btnGalleryExpand_;              // strip ⇄ grid toggle

    // Single interaction model: click a thumbnail to select; wheel to scroll.
    std::vector<sf::FloatRect> thumbnailRects_;   // on-screen rects of visible thumbs
    std::vector<int> thumbnailIndices_;           // gallery index for each visible thumb
    sf::FloatRect galleryRegion_;                 // scrollable area (for wheel hit-test)
    bool  galleryExpanded_ = false;              // false = bottom strip, true = grid wall
    float galleryScroll_   = 0.f;                 // px; horizontal in strip, vertical in grid
    float galleryScrollMax_ = 0.f;                // clamp bound computed during render
    int   lastSelectedIndex_ = -1;                // to auto-scroll selection into view once

    void renderGallery(sf::RenderWindow& win, sf::Font& font, const sf::FloatRect& area);
};

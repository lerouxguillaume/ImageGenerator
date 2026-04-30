#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../enum/enums.hpp"
#include "../../projects/Project.hpp"

// Right panel: displays the generated image, Generate/Cancel buttons,
// progress bar during generation, and error banner on failure.
class ResultPanel {
public:
    WorkflowMode mode = WorkflowMode::Generate;

    struct GalleryItem {
        std::string                 path;
        std::string                 filename;
        std::shared_ptr<sf::Texture> thumbnail;
        float                       score = -1.f;
        bool                        recommended = false;
        bool                        usable = false;
        bool                        near = false;
    };

    struct GalleryTab {
        std::string name;
        std::string assetTypeId;
        std::string outputSubpath;
    };

    // ── Display ───────────────────────────────────────────────────────────────
    sf::Texture resultTexture;
    bool        resultLoaded = false;
    bool        showCheckerboard = false; // render checkerboard behind transparent images
    std::vector<GalleryItem> gallery;
    int                      selectedIndex = -1;
    bool                     showImproveButton = true;
    bool                     showTabs = true;

    // ── Gallery tabs (asset types, populated by controller) ──────────────────────
    std::vector<GalleryTab> tabs;
    int                     activeTabIndex = 0;
    bool                    tabChanged     = false;
    std::string             generateButtonLabel = "Generate";

    // ── Phase tabs (PhasedRefinement assets only) ─────────────────────────────
    struct PhaseTab { int phase; std::string label; };
    std::vector<PhaseTab> phaseTabs;
    int                   activePhaseTabIndex = 0;
    bool                  phaseTabChanged     = false;
    bool                  showPhaseTabs       = false;

    // ── Generation state (shared with background thread via atomics) ──────────
    bool              generating          = false;
    std::atomic<bool> generationDone{false};
    std::atomic<bool> cancelToken{false};
    std::atomic<int>  generationStep{0};
    std::atomic<int>  generationImageNum{0};
    std::atomic<int>  generationTotalImages{1};
    std::atomic<bool> generationFailed{false};
    std::atomic<int>  generationId{0};
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
    AssetSpec                   activeSpec;
    bool                        showContractOverlay = false;
    bool                        showOutputModeToggle = false;
    bool                        showProcessedOutput = true;
    bool                        outputModeChanged = false;
    bool                        selectedReferenceUsed = false;
    std::string                 selectedReferenceImage;
    float                       selectedStructureStrength = 0.0f;

    // Action flags — set by handleEvent, cleared by controller
    bool generateRequested = false;
    bool improveRequested  = false;
    bool deleteRequested   = false;

    // Phased refinement actions
    bool refineRequested    = false;  // controller checks refineUsesSelected for label
    bool refineUsesSelected = false;  // true = refine from selected image; false = best scored
    bool showRefineButton   = false;
    bool refineEnabled      = true;

    // Auto-refine toggle
    bool autoRefineEnabled  = false;
    bool autoRefineToggled  = false;  // set when user clicks toggle
    bool showAutoRefineToggle = false;

    // Phase replace confirm dialog
    bool showPhaseReplaceConfirm = false;
    int  phaseReplaceConfirmPhase = 0;
    bool phaseReplaceConfirmed   = false;
    bool phaseReplaceCancelled   = false;

    // Phase indicator and best score display
    int   phaseIndicatorCurrent = 0;
    int   phaseIndicatorMax     = 0;
    float bestWallCandidateScore = -1.f; // < 0 = not computed

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
    sf::FloatRect btnGenerate_;
    sf::FloatRect btnCancelGenerate_;
    sf::FloatRect btnImprove_;
    sf::FloatRect btnDelete_;
    sf::FloatRect btnRefine_;
    sf::FloatRect btnAutoRefineToggle_;
    sf::FloatRect btnPhaseReplaceYes_;
    sf::FloatRect btnPhaseReplaceNo_;
    sf::FloatRect btnPrevImage_;
    sf::FloatRect btnNextImage_;
    sf::FloatRect btnPrevThumbs_;
    sf::FloatRect btnNextThumbs_;
    std::vector<sf::FloatRect> thumbnailRects_;
    std::vector<int> thumbnailIndices_;
    std::vector<sf::FloatRect> tabRects_;
    std::vector<sf::FloatRect> phaseTabRects_;
    sf::FloatRect processedToggleRect_;
    sf::FloatRect rawToggleRect_;
    int thumbnailScrollOffset_ = 0;
    int lastVisibleSelectedIndex_ = -1;

    void renderTabBar(sf::RenderWindow& win, sf::Font& font,
                      float barX, float barY, float barW);
    void renderPhaseTabBar(sf::RenderWindow& win, sf::Font& font,
                           float barX, float barY, float barW);
    void renderThumbnailStrip(sf::RenderWindow& win, sf::Font& font,
                               float stripX, float stripY, float stripW);
    void ensureSelectedThumbnailVisible(int visibleCount);
};

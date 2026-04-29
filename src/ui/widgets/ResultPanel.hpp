#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>

// Right panel: displays the generated image, Generate/Cancel buttons,
// progress bar during generation, and error banner on failure.
class ResultPanel {
public:
    struct GalleryItem {
        std::string                 path;
        std::string                 filename;
        std::shared_ptr<sf::Texture> thumbnail;
    };

    // ── Display ───────────────────────────────────────────────────────────────
    sf::Texture resultTexture;
    bool        resultLoaded = false;
    std::vector<GalleryItem> gallery;
    int                      selectedIndex = -1;

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

    // Action flag: set by handleEvent on Generate click; cleared by controller after launching
    bool generateRequested  = false;
    // Action flag: set when user clicks "Use as init"; controller copies lastImagePath → initImagePath
    bool useAsInitRequested = false;
    bool improveRequested   = false;
    bool deleteRequested    = false;

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font,
                int numSteps); // numSteps needed for progress bar denominator

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);
    std::string getSelectedImagePath() const;

private:
    sf::FloatRect rect_;
    sf::FloatRect btnGenerate_;
    sf::FloatRect btnCancelGenerate_;
    sf::FloatRect btnUseAsInit_;
    sf::FloatRect btnImprove_;
    sf::FloatRect btnDelete_;
    sf::FloatRect btnPrevImage_;
    sf::FloatRect btnNextImage_;
    sf::FloatRect btnPrevThumbs_;
    sf::FloatRect btnNextThumbs_;
    std::vector<sf::FloatRect> thumbnailRects_;
    std::vector<int> thumbnailIndices_;
    int thumbnailScrollOffset_ = 0;

    void renderThumbnailStrip(sf::RenderWindow& win, sf::Font& font,
                               float stripX, float stripY, float stripW);
    void ensureSelectedThumbnailVisible(int visibleCount);
};

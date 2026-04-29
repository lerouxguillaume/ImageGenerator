#pragma once
#include <atomic>
#include <string>
#include <SFML/Graphics.hpp>

// Right panel: displays the generated image, Generate/Cancel buttons,
// progress bar during generation, and error banner on failure.
class ResultPanel {
public:
    // ── Display ───────────────────────────────────────────────────────────────
    sf::Texture resultTexture;
    bool        resultLoaded = false;

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

    // Action flag: set by handleEvent on Generate click; cleared by controller after launching
    bool generateRequested  = false;
    // Action flag: set when user clicks "Use as init"; controller copies lastImagePath → initImagePath
    bool useAsInitRequested = false;

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font,
                int numSteps); // numSteps needed for progress bar denominator

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);

private:
    sf::FloatRect rect_;
    sf::FloatRect btnGenerate_;
    sf::FloatRect btnCancelGenerate_;
    sf::FloatRect btnUseAsInit_;
};

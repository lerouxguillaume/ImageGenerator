#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <SFML/Graphics.hpp>
#include "Screen.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../ui/SliderTypes.hpp"

// All mutable state for the image generator screen.
// The controller reads and writes this struct; the render() method only reads it.
// Atomics are written by the background generation thread and read on the render thread.
class ImageGeneratorView : public Screen {
public:
    // ── Prompt text fields ────────────────────────────────────────────────────
    std::string positivePrompt;
    std::string negativePrompt;
    int  positiveCursor = 0;  // Byte offset of the text cursor in positivePrompt
    int  negativeCursor = 0;  // Byte offset of the text cursor in negativePrompt
    bool positiveActive = true;  // Whether the positive field has keyboard focus
    bool negativeActive = false;

    // Soft-wrapped line layout cache, rebuilt every render frame from the prompt
    // string and the current field width. Used to map cursor positions to screen
    // coordinates and to implement vertical scrolling.
    struct VisualLine { int start, end; }; // [start, end) byte range in the prompt string
    std::vector<VisualLine> positiveLines;
    std::vector<VisualLine> negativeLines;
    int  positiveScrollLine  = 0; // Index of the first visible wrapped line
    int  negativeScrollLine  = 0;
    bool positiveAllSelected = false; // Ctrl+A selection state
    bool negativeAllSelected = false;

    // ── Generation parameters ─────────────────────────────────────────────────
    GenerationParams generationParams;
    bool showAdvancedParams = false; // Whether the steps/CFG/count sliders are visible

    // ── Seed input (advanced section) ─────────────────────────────────────────
    std::string seedInput;          // Raw text; empty → seed = -1 (random)
    int  seedInputCursor = 0;
    bool seedInputActive = false;

    // ── Generation state (shared with background thread via atomics) ──────────
    bool              generating    = false;     // Set on the UI thread; guards against double-starts
    std::atomic<bool> generationDone{false};     // Set by the pipeline thread when finished
    std::atomic<bool> cancelToken{false};        // Set to true to request cancellation
    std::atomic<int>  generationStep{0};         // Steps completed so far (for progress bar)
    std::atomic<int>  generationImageNum{0};     // 1-based index of the image currently being generated
    std::atomic<int>  generationTotalImages{1};  // Total images requested in this run
    std::string       lastImagePath;             // Output path written by the pipeline; read after done

    // Incremented each time a new generation starts. The update() loop uses this
    // to discard stale completion signals from a previous run.
    std::atomic<int> generationId{0};

    // Set by the background thread (before generationDone) when the pipeline throws.
    // Safe to read after acquiring on generationDone (seq_cst store ordering).
    std::atomic<bool> generationFailed{false};
    std::string       generationErrorMsg;

    // ── Prompt enhancement ────────────────────────────────────────────────────
    bool              promptEnhancerAvailable = false; // set by controller after loading
    bool              llmLoading    = false;            // set while the LLM model is loading async
    bool              enhancing     = false;            // set on UI thread while running
    std::atomic<bool> enhanceDone{false};               // set by enhancer thread when done
    std::string       enhancedPositive;                 // written by enhancer thread, read after enhanceDone
    std::string       enhancedNegative;
    sf::FloatRect     btnEnhance;

    // ── Result display ────────────────────────────────────────────────────────
    sf::Texture resultTexture;
    bool        resultLoaded = false; // True once resultTexture holds a valid image

    // ── Model selection ───────────────────────────────────────────────────────
    std::vector<std::string> availableModels; // Populated at screen init from the models/ directory
    int selectedModelIdx = 0;

    // ── LoRA selection ────────────────────────────────────────────────────────
    std::vector<std::string> availableLoras;    // Full paths from loraBaseDir (*.safetensors)
    std::vector<bool>        loraSelected;      // Per-lora enabled toggle
    std::vector<float>       loraScales;        // Per-lora scale (default 1.0f)
    std::vector<std::string> loraScaleInputs;   // Raw text for the scale field
    bool showLoraPanel      = false;
    int  activeLoraScaleIdx = -1;               // Which scale field has keyboard focus (-1 = none)

    // ── Hit rects (written during render, read by controller) ─────────────────
    // Laid out by render(); the controller checks mouse positions against these.
    sf::FloatRect positiveField;
    sf::FloatRect negativeField;
    sf::FloatRect btnGenerate;
    sf::FloatRect btnBack;
    sf::FloatRect btnAdvanced;
    sf::FloatRect btnCancelGenerate;
    sf::FloatRect btnModelDropdown;
    std::vector<sf::FloatRect> modelDropdownItems; // one per availableModels entry
    bool          showModelDropdown = false;
    sf::FloatRect              btnLoraPanel;
    std::vector<sf::FloatRect> loraRowToggleRects; // per-lora: checkbox+name area
    std::vector<sf::FloatRect> loraScaleRects;     // per-lora: scale text field
    sf::FloatRect stepsSliderTrack;
    sf::FloatRect cfgSliderTrack;
    sf::FloatRect imagesSliderTrack;
    sf::FloatRect seedField;
    DraggingSlider draggingSlider = DraggingSlider::None; // Which slider (if any) is being dragged

    // ── Settings modal ────────────────────────────────────────────────────────
    bool        showSettings              = false;
    std::string settingsModelDir;           // Editable copy while the modal is open
    std::string settingsOutputDir;
    std::string settingsLlmModelDir;        // Path to the ort-genai LLM model directory
    std::string settingsLoraDir;            // Path to the LoRA directory
    int         settingsModelDirCursor    = 0;
    int         settingsOutputDirCursor   = 0;
    int         settingsLlmModelDirCursor = 0;
    int         settingsLoraDirCursor     = 0;
    bool        settingsModelDirActive    = true;
    bool        settingsOutputDirActive   = false;
    bool        settingsLlmModelDirActive = false;
    bool        settingsLoraDirActive     = false;

    // Hit rects for the settings button and modal controls
    sf::FloatRect btnSettings;
    sf::FloatRect settingsModelDirField;
    sf::FloatRect settingsOutputDirField;
    sf::FloatRect settingsLlmModelDirField;
    sf::FloatRect settingsLoraDirField;
    sf::FloatRect settingsBtnBrowseModel;
    sf::FloatRect settingsBtnBrowseOutput;
    sf::FloatRect settingsBtnBrowseLlm;
    sf::FloatRect settingsBtnBrowseLora;
    sf::FloatRect settingsBtnSave;
    sf::FloatRect settingsBtnCancel;

    void render(sf::RenderWindow& win) override;

private:
    void drawPromptField(sf::RenderWindow& win,
                         const sf::FloatRect& field,
                         const std::string& text,
                         int cursor, bool active, bool allSelected,
                         sf::Color textColor,
                         std::vector<VisualLine>& outLines,
                         int& scrollLine);

    void drawSlider(sf::RenderWindow& win,
                    const sf::FloatRect& track, float normalised,
                    const std::string& label, const std::string& valueStr);

    void drawGeneratingOverlay(sf::RenderWindow& win);

    // Single-line editable text field (no word-wrap) used inside the settings modal.
    void drawSingleLineField(sf::RenderWindow& win,
                             const sf::FloatRect& field,
                             const std::string& text,
                             int cursor, bool active);

    // Settings modal overlay.
    void drawSettingsModal(sf::RenderWindow& win);
};

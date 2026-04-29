#include "ImageGeneratorController.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../managers/Logger.hpp"
#include "../enum/enums.hpp"
#include "../presets/PresetManager.hpp"
#include "../prompt/PromptParser.hpp"
#include "../prompt/PromptCompiler.hpp"
#include "../prompt/PromptMerge.hpp"
#include <SFML/Window/Clipboard.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>

// ── Folder browser ────────────────────────────────────────────────────────────

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>

static int CALLBACK browseCb(HWND hwnd, UINT msg, LPARAM, LPARAM data) {
    if (msg == BFFM_INITIALIZED && data)
        SendMessageW(hwnd, BFFM_SETSELECTIONW, TRUE, data);
    return 0;
}

static std::string browseForFolder(const std::string& startPath) {
    CoInitialize(nullptr);
    std::wstring wStart;
    if (!startPath.empty()) {
        int n = MultiByteToWideChar(CP_UTF8, 0, startPath.c_str(), -1, nullptr, 0);
        wStart.resize(n - 1);
        MultiByteToWideChar(CP_UTF8, 0, startPath.c_str(), -1, &wStart[0], n);
    }
    BROWSEINFOW bi  = {};
    bi.ulFlags      = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    bi.lpszTitle    = L"Select folder";
    bi.lpfn         = browseCb;
    bi.lParam       = wStart.empty() ? 0 : reinterpret_cast<LPARAM>(wStart.c_str());
    std::string result;
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (pidl) {
        wchar_t path[MAX_PATH] = {};
        if (SHGetPathFromIDListW(pidl, path)) {
            int len = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
            if (len > 1) { result.resize(len - 1); WideCharToMultiByte(CP_UTF8, 0, path, -1, &result[0], len, nullptr, nullptr); }
        }
        CoTaskMemFree(pidl);
    }
    CoUninitialize();
    return result;
}
#else
static std::string browseForFolder(const std::string& startPath) {
    std::string cmd = "zenity --file-selection --directory --title='Select folder'";
    if (!startPath.empty()) cmd += " --filename='" + startPath + "/'";
    cmd += " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    char buf[4096] = {};
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
}
#endif

// ── Helpers ───────────────────────────────────────────────────────────────────

static ModelType inferModelType(const std::string& modelDir) {
    if (modelDir.empty()) return ModelType::SD15;
    std::ifstream f(modelDir + "/model.json");
    if (!f.is_open()) return ModelType::SD15;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return (content.find("\"sdxl\"") != std::string::npos) ? ModelType::SDXL : ModelType::SD15;
}

static GenerationSettings buildGenerationSettings(const ImageGeneratorView& view) {
    const auto& sp = view.settingsPanel;
    GenerationSettings gs;
    gs.dsl     = PromptParser::parse(sp.positiveArea.getText(), sp.negativeArea.getText());
    gs.modelId = sp.getSelectedModelDir().empty()
        ? std::string{}
        : std::filesystem::path(sp.getSelectedModelDir()).filename().string();
    gs.steps   = sp.generationParams.numSteps;
    gs.cfg     = sp.generationParams.guidanceScale;
    gs.width   = sp.generationParams.width;
    gs.height  = sp.generationParams.height;
    gs.presetId = sp.activePresetId;
    return gs;
}

// ── Model defaults ────────────────────────────────────────────────────────────

void ImageGeneratorController::applyModelDefaults(ImageGeneratorView& view) {
    auto& sp = view.settingsPanel;
    const ModelDefaults* md = nullptr;
    if (!sp.availableModels.empty()) {
        const std::string name = std::filesystem::path(
            sp.availableModels[static_cast<size_t>(sp.selectedModelIdx)]).filename().string();
        const auto it = config.modelConfigs.find(name);
        if (it != config.modelConfigs.end()) md = &it->second;
    }

    sp.positiveArea.setText((md && !md->positivePrompt.empty()) ? md->positivePrompt : std::string{});
    sp.positiveArea.setActive(true);

    sp.negativeArea.setText((md && !md->negativePrompt.empty()) ? md->negativePrompt : std::string{});
    sp.negativeArea.setActive(false);

    sp.generationParams.numSteps = (md && md->numSteps > 0)
        ? md->numSteps : config.defaultNumSteps;

    sp.generationParams.guidanceScale = (md && md->guidanceScale > 0.f)
        ? md->guidanceScale : config.defaultGuidanceScale;
}

// ── LLM async load ────────────────────────────────────────────────────────────

void ImageGeneratorController::startLlmLoad(const std::string& modelDir) {
    llmLoadFuture = std::async(std::launch::async,
        [modelDir]() -> std::unique_ptr<IPromptEnhancer> {
            return PromptEnhancerFactory::create(true, modelDir);
        });
}

// ── Settings helpers ──────────────────────────────────────────────────────────

void ImageGeneratorController::openSettings(ImageGeneratorView& view) {
    auto& m = view.settingsModal;
    m.settingsModelDir            = config.modelBaseDir;
    m.settingsOutputDir           = config.outputDir;
    m.settingsLlmModelDir         = config.promptEnhancer.modelDir;
    m.settingsLoraDir             = config.loraBaseDir;
    m.settingsModelDirCursor      = static_cast<int>(config.modelBaseDir.size());
    m.settingsOutputDirCursor     = static_cast<int>(config.outputDir.size());
    m.settingsLlmModelDirCursor   = static_cast<int>(config.promptEnhancer.modelDir.size());
    m.settingsLoraDirCursor       = static_cast<int>(config.loraBaseDir.size());
    m.settingsModelDirActive      = true;
    m.settingsOutputDirActive     = false;
    m.settingsLlmModelDirActive   = false;
    m.settingsLoraDirActive       = false;
    m.saveRequested               = false;
    m.cancelRequested             = false;
    m.browseTarget                = SettingsModal::BrowseTarget::None;
    m.llmLoading                  = view.llmBar.llmLoading;
    view.showSettings             = true;
}

void ImageGeneratorController::saveSettings(ImageGeneratorView& view) {
    auto& m = view.settingsModal;
    config.modelBaseDir             = m.settingsModelDir;
    config.outputDir                = m.settingsOutputDir;
    config.loraBaseDir              = m.settingsLoraDir;

    const std::string newLlmDir  = m.settingsLlmModelDir;
    const bool llmDirChanged     = (newLlmDir != config.promptEnhancer.modelDir);
    config.promptEnhancer.modelDir  = newLlmDir;
    config.promptEnhancer.enabled   = !newLlmDir.empty();
    config.save();

    view.showSettings = false;

    view.settingsPanel.availableModels.clear();
    view.settingsPanel.selectedModelIdx = 0;
    modelsDirty = true;
    lorasDirty  = true;

    if (llmDirChanged) {
        enhancer = std::make_shared<NullPromptEnhancer>();
        view.llmBar.promptEnhancerAvailable = false;
        if (!newLlmDir.empty()) startLlmLoad(newLlmDir);
    }
}

// ── Prompt helpers ────────────────────────────────────────────────────────────

static void injectBoosters(Prompt& dsl, const ModelDefaults& md) {
    for (const auto& booster : md.qualityBoosters) {
        const bool inSubject  = dsl.subject && dsl.subject->value == booster;
        const bool inPositive = std::any_of(dsl.positive.begin(), dsl.positive.end(),
                                            [&](const Token& t){ return t.value == booster; });
        if (!inSubject && !inPositive)
            dsl.positive.push_back({booster, 1.0f});
    }
}

// ── Generation ────────────────────────────────────────────────────────────────

void ImageGeneratorController::launchGeneration(ImageGeneratorView& view) {
    auto& sp = view.settingsPanel;
    auto& rp = view.resultPanel;

    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    rp.lastImagePath = config.outputDir + "/img_" + std::to_string(now) + ".png";

    rp.generating = true;
    rp.generationDone.store(false);
    rp.cancelToken.store(false);
    rp.generationStep.store(0);
    rp.resultLoaded = false;
    rp.generationFailed.store(false);
    rp.generationErrorMsg.clear();

    const int myId = ++rp.generationId;
    rp.generationTotalImages.store(sp.generationParams.numImages);

    const std::string modelDir  = sp.getSelectedModelDir();
    Prompt            dsl       = PromptParser::parse(sp.positiveArea.getText(),
                                                      sp.negativeArea.getText());
    const ModelType   modelType = inferModelType(modelDir);

    const std::string modelKey = std::filesystem::path(modelDir).filename().string();
    if (const auto it = config.modelConfigs.find(modelKey); it != config.modelConfigs.end())
        injectBoosters(dsl, it->second);

    const std::string prompt    = PromptCompiler::compile(dsl, modelType);
    const std::string negPrompt = PromptCompiler::compileNegative(dsl);
    const std::string outPath   = rp.lastImagePath;
    GenerationParams params = sp.generationParams;
    if (!sp.seedInput.empty()) {
        try { params.seed = std::stoll(sp.seedInput); }
        catch (const std::exception&) { params.seed = -1; }
    }

    for (size_t i = 0; i < sp.availableLoras.size(); ++i) {
        if (i < sp.loraSelected.size() && sp.loraSelected[i])
            params.loras.push_back({sp.availableLoras[i],
                                    i < sp.loraScales.size() ? sp.loraScales[i] : 1.0f});
    }

    std::atomic<bool>* done     = &rp.generationDone;
    std::atomic<int>*  step     = &rp.generationStep;
    std::atomic<int>*  idPtr    = &rp.generationId;
    std::atomic<int>*  imgNum   = &rp.generationImageNum;
    std::atomic<bool>* failed   = &rp.generationFailed;
    std::string*       errorMsg = &rp.generationErrorMsg;

    // Assigning a new jthread implicitly request_stop() + join()s the previous one,
    // ensuring only one pipeline runs at a time and no thread is ever abandoned.
    generationThread_ = std::jthread(
        [prompt, negPrompt, outPath, params, modelDir,
         done, step, idPtr, imgNum, myId, failed, errorMsg]
        (std::stop_token st) {
            try {
                PortraitGeneratorAi::generateFromPrompt(
                    prompt, negPrompt, outPath, params, modelDir, step, imgNum, std::move(st));
            } catch (const std::exception& e) {
                Logger::error("Generation failed: " + std::string(e.what()));
                *errorMsg = e.what();
                failed->store(true);
            } catch (...) {
                Logger::error("Generation failed: unknown error");
                *errorMsg = "Unknown error during generation. See log for details.";
                failed->store(true);
            }
            if (idPtr->load() == myId) done->store(true);
        });
}

void ImageGeneratorController::launchEnhancement(ImageGeneratorView& view) {
    auto& sp  = view.settingsPanel;
    auto& llm = view.llmBar;

    llm.enhancing = true;
    llm.originalPositive = sp.positiveArea.getText();
    llm.originalNegative = sp.negativeArea.getText();

    const std::string posCapture  = sp.positiveArea.getText();
    const std::string instruction = llm.instructionArea.getText();
    const std::string modelDir    = sp.getSelectedModelDir();
    const std::string modelName   = modelDir.empty()
        ? std::string{} : std::filesystem::path(modelDir).filename().string();

    std::string effectiveInstruction = instruction;
    if (effectiveInstruction.empty()) {
        const auto it = config.modelConfigs.find(modelName);
        if (it != config.modelConfigs.end() && !it->second.llmHint.empty())
            effectiveInstruction = it->second.llmHint;
    }

    const ModelType modelType = inferModelType(modelDir);

    std::shared_ptr<IPromptEnhancer> enhCopy = enhancer;

    enhancementFuture_ = std::async(std::launch::async,
        [posCapture, effectiveInstruction, modelType, enhCopy]() -> LLMResponse {
            LLMRequest req;
            req.prompt      = posCapture;
            req.instruction = effectiveInstruction;
            req.model       = modelType;
            req.strength    = 0.5f;
            return enhCopy->transform(req);
        });
}

// ── Event handling ────────────────────────────────────────────────────────────

void ImageGeneratorController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                            ImageGeneratorView& view, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) { win.close(); return; }

    // Escape: close modals/dropdowns in priority order, or navigate to menu
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        if (view.menuBar.showSaveModal)       { view.menuBar.showSaveModal = false; return; }
        if (view.showSettings)                { view.showSettings = false;          return; }
        if (view.menuBar.showPresetDropdown)  { view.menuBar.showPresetDropdown = false; return; }
        if (view.settingsPanel.showModelDropdown) { view.settingsPanel.showModelDropdown = false; return; }
        appScreen = AppScreen::MENU;
        return;
    }

    // Settings modal intercepts all input while open
    if (view.showSettings) {
        if (view.settingsModal.handleEvent(e)) {
            if (view.settingsModal.saveRequested) {
                view.settingsModal.saveRequested = false;
                saveSettings(view);
            }
            if (view.settingsModal.cancelRequested) {
                view.settingsModal.cancelRequested = false;
                view.showSettings = false;
            }
            if (view.settingsModal.browseTarget != SettingsModal::BrowseTarget::None) {
                auto& m = view.settingsModal;
                std::string startPath;
                switch (m.browseTarget) {
                    case SettingsModal::BrowseTarget::ModelDir:  browseTarget = BrowseTarget::ModelDir;  startPath = m.settingsModelDir;    break;
                    case SettingsModal::BrowseTarget::OutputDir: browseTarget = BrowseTarget::OutputDir; startPath = m.settingsOutputDir;   break;
                    case SettingsModal::BrowseTarget::LlmDir:    browseTarget = BrowseTarget::LlmDir;    startPath = m.settingsLlmModelDir; break;
                    case SettingsModal::BrowseTarget::LoraDir:   browseTarget = BrowseTarget::LoraDir;   startPath = m.settingsLoraDir;     break;
                    default: break;
                }
                m.browseTarget = SettingsModal::BrowseTarget::None;
                if (!browseFuture.valid())
                    browseFuture = std::async(std::launch::async, browseForFolder, startPath);
            }
        }
        return;
    }

    // MenuBar (handles save modal, preset dropdown, Back, Settings buttons)
    if (view.menuBar.handleEvent(e)) {
        if (view.menuBar.backRequested) {
            view.menuBar.backRequested = false;
            appScreen = AppScreen::MENU;
        }
        if (view.menuBar.settingsRequested) {
            view.menuBar.settingsRequested = false;
            openSettings(view);
        }
        if (!view.menuBar.selectedPresetId.empty()) {
            const auto p = presetManager.getPreset(view.menuBar.selectedPresetId);
            if (p) applyPresetToSettings(*p, view.settingsPanel);
            view.menuBar.selectedPresetId.clear();
        }
        if (view.menuBar.saveCurrentRequested) {
            view.menuBar.saveCurrentRequested = false;
            const std::string pid = view.settingsPanel.activePresetId;
            if (!pid.empty()) {
                presetManager.updateFromGeneration(pid, buildGenerationSettings(view));
                view.menuBar.setPresets(presetManager.getAllPresets(), pid);
            }
        }
        if (view.menuBar.saveConfirmed) {
            const std::string name = view.menuBar.saveNameInput;
            view.menuBar.saveNameInput.clear();
            view.menuBar.saveConfirmed = false;
            const auto preset = presetManager.createFromGeneration(
                buildGenerationSettings(view), name);
            view.settingsPanel.activePresetId = preset.id;
            view.menuBar.setPresets(presetManager.getAllPresets(), preset.id);
        }
        return;
    }

    // Panels handle their own events
    if (view.settingsPanel.handleEvent(e)) {
        if (view.settingsPanel.positiveArea.isActive() ||
            view.settingsPanel.negativeArea.isActive() ||
            view.settingsPanel.seedInputActive) {
            view.llmBar.instructionArea.setActive(false);
        }
        return;
    }

    if (view.resultPanel.handleEvent(e)) {
        if (view.resultPanel.generateRequested) {
            view.resultPanel.generateRequested = false;
            launchGeneration(view);
        }
        if (view.resultPanel.useAsInitRequested) {
            view.resultPanel.useAsInitRequested = false;
            view.settingsPanel.generationParams.initImagePath = view.resultPanel.lastImagePath;
        }
        if (view.resultPanel.cancelToken.exchange(false))
            generationThread_.request_stop();
        return;
    }

    if (view.llmBar.handleEvent(e)) {
        if (view.llmBar.instructionArea.isActive()) {
            view.settingsPanel.positiveArea.setActive(false);
            view.settingsPanel.negativeArea.setActive(false);
            view.settingsPanel.seedInputActive = false;
        }
        if (view.llmBar.enhanceRequested && !view.llmBar.enhancing) {
            view.llmBar.enhanceRequested = false;
            launchEnhancement(view);
        }
        return;
    }
}

// ── Update (async polling) ────────────────────────────────────────────────────

void ImageGeneratorController::update(ImageGeneratorView& view) {
    auto& sp = view.settingsPanel;
    auto& rp = view.resultPanel;

    // Apply model defaults on first open or model change
    if (!viewInitialized || sp.selectedModelIdx != lastModelIdx) {
        applyModelDefaults(view);
        lastModelIdx     = sp.selectedModelIdx;
        viewInitialized  = true;
        cachedModelType_ = inferModelType(sp.getSelectedModelDir());
        dslDirty_        = true;
    }

    // Poll async LLM load
    if (llmLoadFuture.valid()) {
        view.llmBar.llmLoading = true;
        if (llmLoadFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            enhancer = llmLoadFuture.get();
            view.llmBar.llmLoading = false;
        }
    }
    view.llmBar.promptEnhancerAvailable = !view.llmBar.llmLoading && enhancer->isAvailable();
    view.settingsModal.llmLoading       = view.llmBar.llmLoading;

    // Poll async folder browse
    if (browseFuture.valid() &&
        browseFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::string picked = browseFuture.get();
        if (!picked.empty()) {
            auto& m = view.settingsModal;
            switch (browseTarget) {
                case BrowseTarget::ModelDir:  m.settingsModelDir    = picked; m.settingsModelDirCursor    = static_cast<int>(picked.size()); break;
                case BrowseTarget::OutputDir: m.settingsOutputDir   = picked; m.settingsOutputDirCursor   = static_cast<int>(picked.size()); break;
                case BrowseTarget::LlmDir:    m.settingsLlmModelDir = picked; m.settingsLlmModelDirCursor = static_cast<int>(picked.size()); break;
                case BrowseTarget::LoraDir:   m.settingsLoraDir     = picked; m.settingsLoraDirCursor     = static_cast<int>(picked.size()); break;
            }
        }
    }

    // Scan models directory
    if (modelsDirty) {
        sp.availableModels.clear();
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(config.modelBaseDir, ec)) {
            if (!entry.is_directory()) continue;
            if (std::filesystem::exists(entry.path() / "unet.onnx"))
                sp.availableModels.push_back(entry.path().string());
        }
        std::sort(sp.availableModels.begin(), sp.availableModels.end());
        sp.selectedModelIdx = 0;
        modelsDirty = false;
    }

    // Scan LoRA directory
    if (lorasDirty) {
        sp.availableLoras.clear();
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(config.loraBaseDir, ec)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".safetensors")
                sp.availableLoras.push_back(entry.path().string());
        }
        std::sort(sp.availableLoras.begin(), sp.availableLoras.end());
        sp.loraSelected.assign(sp.availableLoras.size(), false);
        sp.loraScales.assign(sp.availableLoras.size(), 1.0f);
        sp.loraScaleInputs.assign(sp.availableLoras.size(), "1");
        sp.activeLoraScaleIdx = -1;
        lorasDirty = false;
    }

    // Collect enhancement result — merge LLM patch into original DSL
    if (view.llmBar.enhancing && enhancementFuture_.valid() &&
        enhancementFuture_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const LLMResponse result = enhancementFuture_.get();
        const Prompt base   = PromptParser::parse(view.llmBar.originalPositive,
                                                   view.llmBar.originalNegative);
        const Prompt patch  = PromptParser::parse(result.prompt, result.negative_prompt);
        const Prompt merged = PromptMerge::merge(base, patch);
        sp.positiveArea.setText(PromptCompiler::compile(merged, ModelType::SDXL));
        sp.negativeArea.setText(PromptCompiler::compileNegative(merged));
        view.llmBar.enhancing = false;
    }

    // Collect generation result — also joins cancelled threads once they finish
    if (generationThread_.joinable() && rp.generationDone.load()) {
        generationThread_.join(); // instant: thread already set done before returning
        rp.generationDone.store(false);

        if (rp.generating) { // false when user cancelled
            rp.generating = false;
            if (!rp.generationFailed.load()) {
                // Multi-image: load the last image (has _N suffix)
                std::string pathToLoad = rp.lastImagePath;
                const int n = sp.generationParams.numImages;
                if (n > 1) {
                    const auto dot = rp.lastImagePath.rfind('.');
                    const std::string idx = std::to_string(n);
                    pathToLoad = (dot == std::string::npos)
                        ? rp.lastImagePath + "_" + idx
                        : rp.lastImagePath.substr(0, dot) + "_" + idx + rp.lastImagePath.substr(dot);
                }
                if (rp.resultTexture.loadFromFile(pathToLoad))
                    rp.resultLoaded = true;
            }
        }
    }

    // Sync preset list in menu bar (cheap — only name/id comparison needed)
    view.menuBar.setPresets(presetManager.getAllPresets(), sp.activePresetId);

    // Update DSL-derived display state (chips + compiled preview) every frame
    // Re-parse only when text or model actually changed
    const std::string& posText = sp.positiveArea.getText();
    const std::string& negText = sp.negativeArea.getText();
    if (dslDirty_ || posText != lastPositiveText_ || negText != lastNegativeText_) {
        lastPositiveText_ = posText;
        lastNegativeText_ = negText;
        dslDirty_         = false;

        sp.currentDsl = PromptParser::parse(posText, negText);
        if (cachedModelType_ == ModelType::SD15) {
            Prompt previewDsl = sp.currentDsl;
            const std::string previewKey = std::filesystem::path(sp.getSelectedModelDir()).filename().string();
            if (const auto it = config.modelConfigs.find(previewKey); it != config.modelConfigs.end())
                injectBoosters(previewDsl, it->second);
            sp.compiledPreview = PromptCompiler::compile(previewDsl, ModelType::SD15);
        } else {
            sp.compiledPreview = std::string{};
        }
    }
}

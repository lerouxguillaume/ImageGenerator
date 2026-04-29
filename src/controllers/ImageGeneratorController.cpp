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
#include <cctype>
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

static std::string sanitiseName(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' ||
            c == '?' || c == '"' || c == '<' || c == '>' || c == '|')
            r += '_';
        else
            r += c;
    }
    return r;
}

static std::string trimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string buildEditPrompt(const std::string& basePrompt, const ImageGeneratorView& view) {
    const auto& params = view.settingsPanel.generationParams;
    if (params.initImagePath.empty()) return basePrompt;

    const std::string instruction = trimCopy(view.settingsPanel.editInstructionArea.getText());
    if (instruction.empty()) return basePrompt;

    static const std::string preserveClause =
        "same character identity, same facial features, same pose, same framing, same lighting";

    if (basePrompt.empty())
        return instruction + ", " + preserveClause;
    return basePrompt + ", " + instruction + ", " + preserveClause;
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

static bool isGalleryImageFile(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp";
}

static sf::Image resizeImage(const sf::Image& src, unsigned dstW, unsigned dstH) {
    sf::Image dst;
    dst.create(dstW, dstH);
    const unsigned srcW = src.getSize().x;
    const unsigned srcH = src.getSize().y;
    const float scaleX = static_cast<float>(srcW) / static_cast<float>(dstW);
    const float scaleY = static_cast<float>(srcH) / static_cast<float>(dstH);
    for (unsigned dy = 0; dy < dstH; ++dy) {
        for (unsigned dx = 0; dx < dstW; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * scaleX - 0.5f;
            const float sy = (static_cast<float>(dy) + 0.5f) * scaleY - 0.5f;
            const int x0 = std::max(0, static_cast<int>(sx));
            const int y0 = std::max(0, static_cast<int>(sy));
            const int x1 = std::min(static_cast<int>(srcW) - 1, x0 + 1);
            const int y1 = std::min(static_cast<int>(srcH) - 1, y0 + 1);
            const float fx = sx - std::floor(sx);
            const float fy = sy - std::floor(sy);
            const auto c00 = src.getPixel(static_cast<unsigned>(x0), static_cast<unsigned>(y0));
            const auto c10 = src.getPixel(static_cast<unsigned>(x1), static_cast<unsigned>(y0));
            const auto c01 = src.getPixel(static_cast<unsigned>(x0), static_cast<unsigned>(y1));
            const auto c11 = src.getPixel(static_cast<unsigned>(x1), static_cast<unsigned>(y1));
            auto blerp = [&](sf::Uint8 a, sf::Uint8 b, sf::Uint8 c, sf::Uint8 d) -> sf::Uint8 {
                const float top = static_cast<float>(a) + fx * (static_cast<float>(b) - static_cast<float>(a));
                const float bot = static_cast<float>(c) + fx * (static_cast<float>(d) - static_cast<float>(c));
                return static_cast<sf::Uint8>(top + fy * (bot - top));
            };
            dst.setPixel(dx, dy, {blerp(c00.r,c10.r,c01.r,c11.r),
                                   blerp(c00.g,c10.g,c01.g,c11.g),
                                   blerp(c00.b,c10.b,c01.b,c11.b),
                                   blerp(c00.a,c10.a,c01.a,c11.a)});
        }
    }
    return dst;
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
    sp.positiveArea.setActive(mode_ == WorkflowMode::Generate);

    sp.negativeArea.setText((md && !md->negativePrompt.empty()) ? md->negativePrompt : std::string{});
    sp.negativeArea.setActive(false);
    sp.editInstructionArea.setActive(mode_ == WorkflowMode::Edit);

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
    const std::string previousOutputDir = config.outputDir;
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
    if (config.outputDir != previousOutputDir)
        refreshGallery(view);
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

    if (mode_ == WorkflowMode::Edit && sp.generationParams.initImagePath.empty()) {
        rp.generating = false;
        rp.generationFailed.store(true);
        rp.generationErrorMsg = "Select an image to edit first.";
        return;
    }

    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::string outDir = config.outputDir;
    if (!projectContext_.outputSubpath.empty()) {
        outDir = config.outputDir + "/" + projectContext_.outputSubpath;
        std::filesystem::create_directories(outDir);
    }
    rp.lastImagePath = outDir + "/img_" + std::to_string(now) + ".png";

    rp.generating = true;
    rp.generationDone.store(false);
    rp.cancelToken.store(false);
    rp.generationStep.store(0);
    rp.resultLoaded = false;
    rp.displayedImagePath.clear();
    rp.generationFailed.store(false);
    rp.generationErrorMsg.clear();

    const int myId = ++rp.generationId;
    rp.generationTotalImages.store(sp.generationParams.numImages);

    const std::string modelDir  = sp.getSelectedModelDir();
    Prompt            userDsl   = PromptParser::parse(sp.positiveArea.getText(),
                                                      sp.negativeArea.getText());
    // Merge project style → constraints → asset type tokens → user input
    Prompt dsl = userDsl;
    if (!projectContext_.empty()) {
        Prompt base = projectContext_.stylePrompt;
        if (!projectContext_.constraintTokens.positive.empty()
            || !projectContext_.constraintTokens.negative.empty())
            base = PromptMerge::merge(base, projectContext_.constraintTokens);
        if (!projectContext_.assetTypeTokens.positive.empty()
            || projectContext_.assetTypeTokens.subject.has_value())
            base = PromptMerge::merge(base, projectContext_.assetTypeTokens);
        dsl = PromptMerge::merge(base, userDsl);
    }
    const ModelType   modelType = inferModelType(modelDir);

    const std::string modelKey = std::filesystem::path(modelDir).filename().string();
    if (const auto it = config.modelConfigs.find(modelKey); it != config.modelConfigs.end())
        injectBoosters(dsl, it->second);

    const std::string prompt    = buildEditPrompt(PromptCompiler::compile(dsl, modelType), view);
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
        [posCapture, effectiveInstruction, modelType, enhCopy, strength = sp.generationParams.strength]() -> LLMResponse {
            LLMRequest req;
            req.prompt      = posCapture;
            req.instruction = effectiveInstruction;
            req.model       = modelType;
            req.strength    = strength;
            return enhCopy->transform(req);
        });
}

void ImageGeneratorController::selectGalleryImage(ImageGeneratorView& view, int index) {
    auto& rp = view.resultPanel;
    if (index < 0 || index >= static_cast<int>(rp.gallery.size())) {
        rp.selectedIndex = -1;
        rp.resultLoaded = false;
        rp.displayedImagePath.clear();
        return;
    }

    rp.selectedIndex = index;
    const auto& item = rp.gallery[static_cast<size_t>(index)];
    if (rp.resultTexture.loadFromFile(item.path)) {
        rp.resultLoaded = true;
        rp.displayedImagePath = item.path;
    } else {
        rp.resultLoaded = false;
        rp.displayedImagePath.clear();
    }
}

void ImageGeneratorController::refreshGallery(ImageGeneratorView& view, const std::string& preferredSelection) {
    auto& rp = view.resultPanel;
    struct ImageEntry {
        std::string path;
        std::string filename;
        std::filesystem::file_time_type modified;
    };

    std::vector<ImageEntry> entries;
    std::error_code ec;
    const std::filesystem::path galleryDir =
        projectContext_.outputSubpath.empty()
            ? std::filesystem::path(config.outputDir)
            : std::filesystem::path(config.outputDir) / projectContext_.outputSubpath;
    if (std::filesystem::exists(galleryDir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(galleryDir, ec)) {
            if (!entry.is_regular_file()) continue;
            if (!isGalleryImageFile(entry.path())) continue;
            entries.push_back({
                entry.path().string(),
                entry.path().filename().string(),
                entry.last_write_time(ec)
            });
        }
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.modified > b.modified;
    });

    pendingThumbs_.clear();
    std::vector<ResultPanel::GalleryItem> gallery;
    gallery.reserve(entries.size());
    for (const auto& entry : entries) {
        gallery.push_back({entry.path, entry.filename, nullptr});
        pendingThumbs_.push_back({
            entry.path,
            std::async(std::launch::async, [path = entry.path]() -> sf::Image {
                sf::Image img;
                if (!img.loadFromFile(path)) return {};
                const unsigned w = img.getSize().x;
                const unsigned h = img.getSize().y;
                if (w == 0 || h == 0) return {};
                constexpr unsigned kThumbSz = 92;
                const float scale = std::min(static_cast<float>(kThumbSz) / w,
                                             static_cast<float>(kThumbSz) / h);
                return resizeImage(img,
                    static_cast<unsigned>(w * scale),
                    static_cast<unsigned>(h * scale));
            })
        });
    }
    rp.gallery = std::move(gallery);

    std::string target = preferredSelection.empty() ? rp.displayedImagePath : preferredSelection;
    int selected = -1;
    for (int i = 0; i < static_cast<int>(rp.gallery.size()); ++i) {
        if (rp.gallery[static_cast<size_t>(i)].path == target) {
            selected = i;
            break;
        }
    }
    if (selected < 0 && !rp.gallery.empty())
        selected = 0;

    selectGalleryImage(view, selected);
}

void ImageGeneratorController::flushPendingThumbs(ImageGeneratorView& view) {
    auto& gallery = view.resultPanel.gallery;
    for (auto it = pendingThumbs_.begin(); it != pendingThumbs_.end();) {
        if (it->imageFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        sf::Image img = it->imageFuture.get();
        if (img.getSize().x > 0) {
            for (auto& item : gallery) {
                if (item.path == it->path) {
                    auto tex = std::make_shared<sf::Texture>();
                    if (tex->loadFromImage(img))
                        item.thumbnail = std::move(tex);
                    break;
                }
            }
        }
        it = pendingThumbs_.erase(it);
    }
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
        appScreen = (mode_ == WorkflowMode::Edit) ? backScreen_ : AppScreen::MENU;
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
            appScreen = (mode_ == WorkflowMode::Edit) ? backScreen_ : AppScreen::MENU;
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
            view.settingsPanel.editInstructionArea.isActive() ||
            view.settingsPanel.seedInputActive) {
            view.llmBar.instructionArea.setActive(false);
        }
        return;
    }

    if (view.resultPanel.handleEvent(e)) {
        const std::string selectedPath = view.resultPanel.getSelectedImagePath();
        if (!selectedPath.empty() && selectedPath != view.resultPanel.displayedImagePath) {
            selectGalleryImage(view, view.resultPanel.selectedIndex);
            if (mode_ == WorkflowMode::Edit)
                view.settingsPanel.generationParams.initImagePath = selectedPath;
        }
        if (view.resultPanel.generateRequested) {
            view.resultPanel.generateRequested = false;
            launchGeneration(view);
        }
        if (view.resultPanel.improveRequested) {
            view.resultPanel.improveRequested = false;
            if (mode_ == WorkflowMode::Generate) {
                pendingEditNavigationPath_ = view.resultPanel.displayedImagePath;
            }
        }
        if (view.resultPanel.deleteRequested) {
            view.resultPanel.deleteRequested = false;
            const std::filesystem::path selected = view.resultPanel.getSelectedImagePath();
            std::string nextSelection;
            if (view.resultPanel.selectedIndex > 0
                && view.resultPanel.selectedIndex - 1 < static_cast<int>(view.resultPanel.gallery.size())) {
                nextSelection = view.resultPanel.gallery[static_cast<size_t>(view.resultPanel.selectedIndex - 1)].path;
            } else if (view.resultPanel.selectedIndex >= 0
                       && view.resultPanel.selectedIndex + 1 < static_cast<int>(view.resultPanel.gallery.size())) {
                nextSelection = view.resultPanel.gallery[static_cast<size_t>(view.resultPanel.selectedIndex + 1)].path;
            }
            std::error_code ec2;
            const std::filesystem::path outputDir = std::filesystem::weakly_canonical(config.outputDir, ec2);
            const std::filesystem::path canonicalSelected = std::filesystem::weakly_canonical(selected, ec2);
            // Allow deletion from any subdirectory under outputDir (e.g. project/assettype/)
            const auto rel = std::filesystem::relative(canonicalSelected, outputDir, ec2);
            bool hasParentRef = false;
            for (const auto& part : rel)
                if (part.string() == "..") { hasParentRef = true; break; }
            const bool underOutput = !ec2 && !rel.empty() && !hasParentRef;
            if (underOutput && !selected.empty()) {
                std::filesystem::remove(canonicalSelected, ec2);
                refreshGallery(view, nextSelection);
            }
        }
        if (view.resultPanel.cancelToken.exchange(false))
            generationThread_.request_stop();
        return;
    }

    if (mode_ == WorkflowMode::Generate && view.llmBar.handleEvent(e)) {
        if (view.llmBar.instructionArea.isActive()) {
            view.settingsPanel.positiveArea.setActive(false);
            view.settingsPanel.negativeArea.setActive(false);
            view.settingsPanel.editInstructionArea.setActive(false);
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
        refreshGallery(view);
    }

    flushPendingThumbs(view);

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
                std::string preferred = rp.lastImagePath;
                const int n = sp.generationParams.numImages;
                if (n > 1) {
                    const auto dot = rp.lastImagePath.rfind('.');
                    const std::string idx = std::to_string(n);
                    preferred = (dot == std::string::npos)
                        ? rp.lastImagePath + "_" + idx
                        : rp.lastImagePath.substr(0, dot) + "_" + idx + rp.lastImagePath.substr(dot);
                }
                refreshGallery(view, preferred);
            }
        }
    }

    // Sync gallery tabs from active project context
    if (projectContext_.empty()) {
        rp.tabs.clear();
    } else {
        if (rp.tabs.size() != projectContext_.allAssetTypes.size()) {
            rp.tabs.clear();
            for (const auto& at : projectContext_.allAssetTypes) {
                rp.tabs.push_back({
                    at.name, at.id,
                    sanitiseName(projectContext_.projectName) + "/" + sanitiseName(at.name)
                });
            }
            for (int i = 0; i < static_cast<int>(rp.tabs.size()); ++i) {
                if (rp.tabs[static_cast<size_t>(i)].assetTypeId == projectContext_.assetTypeId) {
                    rp.activeTabIndex = i;
                    break;
                }
            }
        }
    }

    // Handle user switching gallery tab
        if (rp.tabChanged) {
            rp.tabChanged = false;
            const int idx = rp.activeTabIndex;
            if (!rp.tabs.empty() && idx >= 0 && idx < static_cast<int>(rp.tabs.size())) {
                const auto& tab = rp.tabs[static_cast<size_t>(idx)];
            projectContext_.assetTypeId     = tab.assetTypeId;
            projectContext_.assetTypeName   = tab.name;
            projectContext_.outputSubpath   = tab.outputSubpath;
                for (const auto& at : projectContext_.allAssetTypes) {
                    if (at.id == tab.assetTypeId) {
                        projectContext_.assetTypeTokens = at.promptTokens;
                        view.settingsPanel.positiveArea.setText(PromptCompiler::compile(at.promptTokens, ModelType::SDXL));
                        view.settingsPanel.negativeArea.setText(PromptCompiler::compileNegative(at.promptTokens));
                        dslDirty_ = true;
                        break;
                    }
                }
                refreshGallery(view);
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

void ImageGeneratorController::prepareEditSession(ImageGeneratorView& view, const std::string& imagePath) {
    view.settingsPanel.generationParams.initImagePath = imagePath;
    view.settingsPanel.generationParams.strength = 0.5f;
    view.settingsPanel.editInstructionArea.setActive(true);
    view.settingsPanel.positiveArea.setActive(false);
    view.settingsPanel.negativeArea.setActive(false);
    view.settingsPanel.seedInputActive = false;
    refreshGallery(view, imagePath);
}

std::string ImageGeneratorController::consumePendingEditNavigation() {
    std::string path = pendingEditNavigationPath_;
    pendingEditNavigationPath_.clear();
    return path;
}

void ImageGeneratorController::setBackScreen(AppScreen screen) {
    backScreen_ = screen;
}

void ImageGeneratorController::activateProjectSession(ImageGeneratorView& view, const ResolvedProjectContext& ctx) {
    setProjectContext(ctx);
    view.settingsPanel.positiveArea.setText(PromptCompiler::compile(ctx.assetTypeTokens, ModelType::SDXL));
    view.settingsPanel.negativeArea.setText(PromptCompiler::compileNegative(ctx.assetTypeTokens));
    dslDirty_ = true;
}

ResolvedProjectContext ImageGeneratorController::getProjectContext() const {
    return projectContext_;
}

void ImageGeneratorController::triggerGeneration(ImageGeneratorView& view) {
    launchGeneration(view);
}

void ImageGeneratorController::openSettingsDialog(ImageGeneratorView& view) {
    openSettings(view);
}

void ImageGeneratorController::setProjectContext(const ResolvedProjectContext& ctx) {
    projectContext_  = ctx;
    viewInitialized  = false; // forces gallery refresh on next update()
    Logger::info("ImageGeneratorController: project context set — '"
                 + ctx.projectName + " / " + ctx.assetTypeName + "'");
}

void ImageGeneratorController::clearProjectContext() {
    projectContext_ = {};
    viewInitialized = false;
}

#include "ImageGeneratorController.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include <SFML/Window/Clipboard.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <future>
#include <string>
#include <thread>

// ── Folder browser ───────────────────────────────────────────────────────────

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>

// Callback that pre-selects the initial folder in the dialog.
static int CALLBACK browseCb(HWND hwnd, UINT msg, LPARAM, LPARAM data) {
    if (msg == BFFM_INITIALIZED && data)
        SendMessageW(hwnd, BFFM_SETSELECTIONW, TRUE, data);
    return 0;
}

// Opens the native Windows folder-picker dialog.
// Runs on a background thread — CoInitialize is called per-thread.
static std::string browseForFolder(const std::string& startPath) {
    CoInitialize(nullptr);

    // Convert startPath (UTF-8) to wide string for the callback.
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
            if (len > 1) {
                result.resize(len - 1);
                WideCharToMultiByte(CP_UTF8, 0, path, -1, &result[0], len, nullptr, nullptr);
            }
        }
        CoTaskMemFree(pidl);
    }

    CoUninitialize();
    return result;
}

#else

// Opens a folder-picker dialog via zenity (Linux/macOS).
static std::string browseForFolder(const std::string& startPath) {
    std::string cmd = "zenity --file-selection --directory --title='Select folder'";
    if (!startPath.empty())
        cmd += " --filename='" + startPath + "/'";
    cmd += " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    char buf[4096] = {};
    std::string result;
    while (fgets(buf, sizeof(buf), pipe))
        result += buf;
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

#endif

// ── Model defaults ────────────────────────────────────────────────────────────

void ImageGeneratorController::applyModelDefaults(ImageGeneratorView& view) {
    // Resolve per-model overrides, falling back to global config defaults.
    const ModelDefaults* md = nullptr;
    if (!view.availableModels.empty()) {
        const std::string name = std::filesystem::path(
            view.availableModels[static_cast<size_t>(view.selectedModelIdx)]).filename().string();
        const auto it = config.modelConfigs.find(name);
        if (it != config.modelConfigs.end())
            md = &it->second;
    }

    view.positivePrompt = (md && !md->positivePrompt.empty())
        ? md->positivePrompt : config.defaultPositivePrompt;
    view.positiveCursor = static_cast<int>(view.positivePrompt.size());

    view.negativePrompt = (md && !md->negativePrompt.empty())
        ? md->negativePrompt : config.defaultNegativePrompt;
    view.negativeCursor = static_cast<int>(view.negativePrompt.size());

    view.generationParams.numSteps = (md && md->numSteps > 0)
        ? md->numSteps : config.defaultNumSteps;

    view.generationParams.guidanceScale = (md && md->guidanceScale > 0.f)
        ? md->guidanceScale : config.defaultGuidanceScale;
}

// ── Settings helpers ──────────────────────────────────────────────────────────

void ImageGeneratorController::openSettings(ImageGeneratorView& view) {
    view.settingsModelDir         = config.modelBaseDir;
    view.settingsOutputDir        = config.outputDir;
    view.settingsModelDirCursor   = static_cast<int>(config.modelBaseDir.size());
    view.settingsOutputDirCursor  = static_cast<int>(config.outputDir.size());
    view.settingsModelDirActive   = true;
    view.settingsOutputDirActive  = false;
    view.showSettings             = true;
}

void ImageGeneratorController::saveSettings(ImageGeneratorView& view) {
    config.modelBaseDir = view.settingsModelDir;
    config.outputDir    = view.settingsOutputDir;
    config.save();
    view.showSettings = false;
    // Trigger a model rescan with the new base directory.
    view.availableModels.clear();
    view.selectedModelIdx = 0;
    modelsDirty = true;
}

// ── Event handling ────────────────────────────────────────────────────────────

void ImageGeneratorController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                            ImageGeneratorView& view, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) win.close();

    // Escape: close dropdowns/modals in order, or navigate to menu.
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        if (view.showSettings)     { view.showSettings = false;     return; }
        if (view.showModelDropdown){ view.showModelDropdown = false; return; }
        appScreen = AppScreen::MENU;
    }

    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), win, view, appScreen);

    if (e.type == sf::Event::MouseButtonReleased && e.mouseButton.button == sf::Mouse::Left)
        view.draggingSlider = DraggingSlider::None;

    if (e.type == sf::Event::MouseMoved && view.draggingSlider != DraggingSlider::None) {
        const sf::Vector2f mousePos = win.mapPixelToCoords({e.mouseMove.x, e.mouseMove.y});
        if (view.draggingSlider == DraggingSlider::Steps) {
            const sf::FloatRect& track = view.stepsSliderTrack;
            const float t = std::clamp((mousePos.x - track.left) / track.width, 0.f, 1.f);
            view.generationParams.numSteps = static_cast<int>(std::round(5.f + t * 45.f));
        } else if (view.draggingSlider == DraggingSlider::Cfg) {
            const sf::FloatRect& track = view.cfgSliderTrack;
            const float t   = std::clamp((mousePos.x - track.left) / track.width, 0.f, 1.f);
            const float raw = 1.0f + t * 19.0f;
            view.generationParams.guidanceScale = std::round(raw * 2.f) / 2.f;
        } else if (view.draggingSlider == DraggingSlider::Images) {
            const sf::FloatRect& track = view.imagesSliderTrack;
            const float t = std::clamp((mousePos.x - track.left) / track.width, 0.f, 1.f);
            view.generationParams.numImages = static_cast<int>(std::round(1.f + t * 19.f));
        }
    }

    // Mouse-wheel scrolling on text areas
    if (e.type == sf::Event::MouseWheelScrolled && !view.generating) {
        const sf::Vector2f pos = win.mapPixelToCoords({e.mouseWheelScroll.x, e.mouseWheelScroll.y});
        const int delta = (e.mouseWheelScroll.delta > 0) ? -1 : 1;
        if (view.positiveField.contains(pos))
            view.positiveScrollLine = std::max(0, view.positiveScrollLine + delta);
        else if (view.negativeField.contains(pos))
            view.negativeScrollLine = std::max(0, view.negativeScrollLine + delta);
    }

    // Settings modal keyboard input
    if (view.showSettings) {
        std::string& activeField = view.settingsModelDirActive
                                    ? view.settingsModelDir
                                    : view.settingsOutputDir;
        int& cursor = view.settingsModelDirActive
                       ? view.settingsModelDirCursor
                       : view.settingsOutputDirCursor;

        if (e.type == sf::Event::KeyPressed) {
            if (e.key.code == sf::Keyboard::Tab) {
                view.settingsModelDirActive  = !view.settingsModelDirActive;
                view.settingsOutputDirActive = !view.settingsOutputDirActive;
            } else if (e.key.code == sf::Keyboard::Enter) {
                saveSettings(view);
            } else if (e.key.code == sf::Keyboard::Left && cursor > 0) {
                --cursor;
            } else if (e.key.code == sf::Keyboard::Right
                       && cursor < static_cast<int>(activeField.size())) {
                ++cursor;
            } else if (e.key.code == sf::Keyboard::Home) {
                cursor = 0;
            } else if (e.key.code == sf::Keyboard::End) {
                cursor = static_cast<int>(activeField.size());
            } else if (e.key.code == sf::Keyboard::BackSpace && cursor > 0) {
                activeField.erase(static_cast<size_t>(cursor - 1), 1);
                --cursor;
            } else if (e.key.code == sf::Keyboard::Delete
                       && cursor < static_cast<int>(activeField.size())) {
                activeField.erase(static_cast<size_t>(cursor), 1);
            }
        }
        if (e.type == sf::Event::TextEntered) {
            const auto c = e.text.unicode;
            // Accept printable ASCII; paths use /, \, :, space, etc. — all < 127.
            if (c >= 32 && c < 127) {
                activeField.insert(static_cast<size_t>(cursor), 1, static_cast<char>(c));
                ++cursor;
            }
        }
        return; // block all other input while settings is open
    }

    // Text input for active field
    if (!view.generating) {
        // Seed field keyboard handling
        if (view.seedInputActive) {
            if (e.type == sf::Event::KeyPressed) {
                auto& s = view.seedInput;
                auto& c = view.seedInputCursor;
                if (e.key.code == sf::Keyboard::Left  && c > 0) { --c; }
                else if (e.key.code == sf::Keyboard::Right && c < static_cast<int>(s.size())) { ++c; }
                else if (e.key.code == sf::Keyboard::Home) { c = 0; }
                else if (e.key.code == sf::Keyboard::End)  { c = static_cast<int>(s.size()); }
                else if (e.key.code == sf::Keyboard::BackSpace && c > 0) { s.erase(static_cast<size_t>(--c), 1); }
                else if (e.key.code == sf::Keyboard::Delete && c < static_cast<int>(s.size())) { s.erase(static_cast<size_t>(c), 1); }
                else if (e.key.code == sf::Keyboard::Escape || e.key.code == sf::Keyboard::Tab) {
                    view.seedInputActive = false;
                } else if (e.key.control && e.key.code == sf::Keyboard::C) {
                    sf::Clipboard::setString(s);
                } else if (e.key.control && e.key.code == sf::Keyboard::V) {
                    const std::string clip = sf::Clipboard::getString().toAnsiString();
                    for (std::size_t i = 0; i < clip.size() && s.size() < 20; ++i) {
                        const char ch = clip[i];
                        const bool isDigit = (ch >= '0' && ch <= '9');
                        const bool isMinus = (ch == '-' && c == 0 && s.empty());
                        if (isDigit || isMinus) {
                            s.insert(static_cast<size_t>(c), 1, ch);
                            ++c;
                        }
                    }
                } else if (e.key.control && e.key.code == sf::Keyboard::A) {
                    // Select all: move cursor to end (no visual selection, but prepares for overwrite)
                    c = static_cast<int>(s.size());
                }
            }
            if (e.type == sf::Event::TextEntered) {
                const auto ch = e.text.unicode;
                // Allow digits and a leading minus sign
                const bool isDigit = (ch >= '0' && ch <= '9');
                const bool isMinus = (ch == '-' && view.seedInputCursor == 0 && view.seedInput.empty());
                if ((isDigit || isMinus) && view.seedInput.size() < 20) {
                    view.seedInput.insert(static_cast<size_t>(view.seedInputCursor), 1, static_cast<char>(ch));
                    ++view.seedInputCursor;
                }
            }
            return; // don't forward to prompt fields
        }

        if (e.type == sf::Event::KeyPressed) {
            std::string& activeField = view.positiveActive ? view.positivePrompt : view.negativePrompt;
            int& cursor      = view.positiveActive ? view.positiveCursor      : view.negativeCursor;
            bool& allSel     = view.positiveActive ? view.positiveAllSelected : view.negativeAllSelected;
            const bool anyActive = view.positiveActive || view.negativeActive;

            // Helper: find cursor's visual line index
            auto findLine = [](const std::vector<ImageGeneratorView::VisualLine>& lines, int cur) {
                int l = static_cast<int>(lines.size()) - 1;
                for (int i = 0; i + 1 < static_cast<int>(lines.size()); ++i)
                    if (cur < lines[i + 1].start) { l = i; break; }
                return l;
            };

            if (!anyActive) {
                // nothing
            } else if (e.key.code == sf::Keyboard::Left) {
                allSel = false;
                if (cursor > 0) --cursor;
            } else if (e.key.code == sf::Keyboard::Right) {
                allSel = false;
                if (cursor < static_cast<int>(activeField.size())) ++cursor;
            } else if (e.key.code == sf::Keyboard::Up) {
                allSel = false;
                auto& lines = view.positiveActive ? view.positiveLines : view.negativeLines;
                if (!lines.empty()) {
                    const int l = findLine(lines, cursor);
                    if (l > 0) {
                        const int col     = cursor - lines[l].start;
                        const int prevLen = lines[l - 1].end - lines[l - 1].start;
                        cursor = lines[l - 1].start + std::min(col, prevLen);
                    }
                }
            } else if (e.key.code == sf::Keyboard::Down) {
                allSel = false;
                auto& lines = view.positiveActive ? view.positiveLines : view.negativeLines;
                if (!lines.empty()) {
                    const int l = findLine(lines, cursor);
                    if (l + 1 < static_cast<int>(lines.size())) {
                        const int col     = cursor - lines[l].start;
                        const int nextLen = lines[l + 1].end - lines[l + 1].start;
                        cursor = lines[l + 1].start + std::min(col, nextLen);
                    }
                }
            } else if (e.key.code == sf::Keyboard::Home) {
                allSel = false; cursor = 0;
            } else if (e.key.code == sf::Keyboard::End) {
                allSel = false; cursor = static_cast<int>(activeField.size());
            } else if (e.key.code == sf::Keyboard::BackSpace) {
                if (allSel) {
                    activeField.clear(); cursor = 0; allSel = false;
                } else if (!activeField.empty() && cursor > 0) {
                    --cursor;
                    activeField.erase(static_cast<size_t>(cursor), 1);
                }
            } else if (e.key.code == sf::Keyboard::Delete) {
                if (allSel) {
                    activeField.clear(); cursor = 0; allSel = false;
                } else if (cursor < static_cast<int>(activeField.size())) {
                    activeField.erase(static_cast<size_t>(cursor), 1);
                }
            } else if (e.key.control && e.key.code == sf::Keyboard::V) {
                if (allSel) { activeField.clear(); cursor = 0; allSel = false; }
                const std::string clip = sf::Clipboard::getString().toAnsiString();
                for (char c : clip) {
                    if (static_cast<unsigned char>(c) >= 32 && activeField.size() < 2000) {
                        activeField.insert(static_cast<size_t>(cursor), 1, c);
                        ++cursor;
                    }
                }
            } else if (e.key.control && e.key.code == sf::Keyboard::C) {
                sf::Clipboard::setString(activeField);
            } else if (e.key.control && e.key.code == sf::Keyboard::A) {
                allSel = true;
            }
        }
        if (e.type == sf::Event::TextEntered) {
            const auto c = e.text.unicode;
            if (c >= 32 && c < 127) {
                std::string& activeField = view.positiveActive ? view.positivePrompt : view.negativePrompt;
                int& cursor  = view.positiveActive ? view.positiveCursor      : view.negativeCursor;
                bool& allSel = view.positiveActive ? view.positiveAllSelected : view.negativeAllSelected;
                if ((view.positiveActive || view.negativeActive) && activeField.size() < 2000) {
                    if (allSel) { activeField.clear(); cursor = 0; allSel = false; }
                    activeField.insert(static_cast<size_t>(cursor), 1, static_cast<char>(c));
                    ++cursor;
                }
            }
        }
    }
}

void ImageGeneratorController::update(ImageGeneratorView& view) {
    // Apply defaults on first open, and whenever the selected model changes.
    if (!viewInitialized || view.selectedModelIdx != lastModelIdx) {
        applyModelDefaults(view);
        lastModelIdx    = view.selectedModelIdx;
        viewInitialized = true;
    }

    // Apply async browse result when zenity finishes.
    if (browseFuture.valid() &&
        browseFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::string picked = browseFuture.get();
        if (!picked.empty()) {
            if (browsingForModel) {
                view.settingsModelDir       = picked;
                view.settingsModelDirCursor = static_cast<int>(picked.size());
            } else {
                view.settingsOutputDir       = picked;
                view.settingsOutputDirCursor = static_cast<int>(picked.size());
            }
        }
    }

    // Scan modelBaseDir for subdirectories containing unet.onnx.
    // Runs on first update and whenever modelsDirty is set (e.g. after settings save).
    if (modelsDirty) {
        view.availableModels.clear();
        std::error_code ec;
        for (const auto& entry :
             std::filesystem::directory_iterator(config.modelBaseDir, ec)) {
            if (!entry.is_directory()) continue;
            if (std::filesystem::exists(entry.path() / "unet.onnx"))
                view.availableModels.push_back(entry.path().string());
        }
        std::sort(view.availableModels.begin(), view.availableModels.end());
        view.selectedModelIdx = 0;
        modelsDirty = false;
    }

    // Sync enhancer availability flag to the view.
    view.promptEnhancerAvailable = enhancer->isAvailable();

    // Collect enhancement result when the background thread finishes.
    if (view.enhancing && view.enhanceDone.load()) {
        view.positivePrompt = view.enhancedPositive;
        view.positiveCursor = static_cast<int>(view.positivePrompt.size());
        view.negativePrompt = view.enhancedNegative;
        view.negativeCursor = static_cast<int>(view.negativePrompt.size());
        view.positiveScrollLine = 0;
        view.negativeScrollLine = 0;
        view.enhancing = false;
        view.enhanceDone.store(false);
    }

    if (view.generating && view.generationDone.load()) {
        presenter.finishGeneration(view);
        // For multi-image runs the last image has an _N suffix; single image has no suffix.
        std::string pathToLoad = view.lastImagePath;
        const int n = view.generationParams.numImages;
        if (n > 1) {
            const auto dot = view.lastImagePath.rfind('.');
            const std::string idx = std::to_string(n);
            pathToLoad = (dot == std::string::npos)
                ? view.lastImagePath + "_" + idx
                : view.lastImagePath.substr(0, dot) + "_" + idx + view.lastImagePath.substr(dot);
        }
        if (view.resultTexture.loadFromFile(pathToLoad))
            view.resultLoaded = true;
    }
}

void ImageGeneratorController::handleClick(sf::Vector2f pos, sf::RenderWindow&,
                                            ImageGeneratorView& view, AppScreen& appScreen) {
    // Settings modal intercepts all clicks while open.
    if (view.showSettings) {
        if (view.settingsBtnSave.contains(pos))   { saveSettings(view); return; }
        if (view.settingsBtnCancel.contains(pos))  { view.showSettings = false; return; }
        if (view.settingsBtnBrowseModel.contains(pos)) {
            browsingForModel = true;
            const std::string start = view.settingsModelDir;
            browseFuture = std::async(std::launch::async, browseForFolder, start);
            return;
        }
        if (view.settingsBtnBrowseOutput.contains(pos)) {
            browsingForModel = false;
            const std::string start = view.settingsOutputDir;
            browseFuture = std::async(std::launch::async, browseForFolder, start);
            return;
        }
        if (view.settingsModelDirField.contains(pos)) {
            view.settingsModelDirActive  = true;
            view.settingsOutputDirActive = false;
            return;
        }
        if (view.settingsOutputDirField.contains(pos)) {
            view.settingsModelDirActive  = false;
            view.settingsOutputDirActive = true;
            return;
        }
        return; // absorb clicks outside modal controls
    }

    if (view.generating || view.enhancing) {
        if (!view.enhancing && view.btnCancelGenerate.contains(pos)) {
            ++view.generationId;
            view.cancelToken.store(true);
            view.generating = false;
        }
        return;
    }

    // Model dropdown must be checked before all other hit rects because the
    // open list renders on top and may overlap elements further down the page.
    if (view.btnModelDropdown.contains(pos)) {
        view.showModelDropdown = !view.showModelDropdown;
        return;
    }
    if (view.showModelDropdown) {
        for (int i = 0; i < static_cast<int>(view.modelDropdownItems.size()); ++i) {
            if (view.modelDropdownItems[static_cast<size_t>(i)].contains(pos)) {
                view.selectedModelIdx  = i;
                view.showModelDropdown = false;
                return;
            }
        }
        view.showModelDropdown = false; // click outside closes it
        return;
    }

    if (view.btnSettings.contains(pos)) {
        openSettings(view);
        return;
    }

    if (view.btnBack.contains(pos)) {
        appScreen = AppScreen::MENU;
        return;
    }

    if (view.positiveField.contains(pos)) {
        presenter.activatePositive(view);
        view.seedInputActive = false;
        return;
    }

    if (view.negativeField.contains(pos)) {
        presenter.activateNegative(view);
        view.seedInputActive = false;
        return;
    }

    if (view.showAdvancedParams && view.seedField.contains(pos)) {
        view.positiveActive  = false;
        view.negativeActive  = false;
        view.seedInputActive = true;
        return;
    }

    if (view.btnAdvanced.contains(pos)) {
        presenter.toggleAdvanced(view);
        return;
    }

    if (view.promptEnhancerAvailable && view.btnEnhance.contains(pos)
        && !view.enhancing && !view.generating) {
        view.enhancing = true;
        view.enhanceDone.store(false);

        const std::string posCapture  = view.positivePrompt;
        const std::string negCapture  = view.negativePrompt;
        const std::string modelName   = view.availableModels.empty() ? std::string{}
            : std::filesystem::path(view.availableModels[
                static_cast<size_t>(view.selectedModelIdx)]).filename().string();

        // Build style context: prefer explicit llmHint, fall back to the
        // model's default positive prompt as a style example.
        std::string styleContext;
        const auto it = config.modelConfigs.find(modelName);
        if (it != config.modelConfigs.end()) {
            styleContext = !it->second.llmHint.empty()
                ? it->second.llmHint
                : it->second.positivePrompt;
        }

        std::atomic<bool>* done   = &view.enhanceDone;
        std::string*       outPos = &view.enhancedPositive;
        std::string*       outNeg = &view.enhancedNegative;
        IPromptEnhancer*   enh   = enhancer.get();

        std::thread([posCapture, negCapture, modelName, styleContext, done, outPos, outNeg, enh]() {
            auto result = enh->enhance(posCapture, negCapture, modelName, styleContext);
            *outPos = result.positive;
            *outNeg = result.negative;
            done->store(true);
        }).detach();
        return;
    }

    if (view.btnResolutionPrev.contains(pos)) {
        view.selectedResolutionIdx = (view.selectedResolutionIdx - 1 + ImageGeneratorView::kNumResolutions)
                                     % ImageGeneratorView::kNumResolutions;
        return;
    }
    if (view.btnResolutionNext.contains(pos)) {
        view.selectedResolutionIdx = (view.selectedResolutionIdx + 1) % ImageGeneratorView::kNumResolutions;
        return;
    }

    if (view.showAdvancedParams) {
        if (view.stepsSliderTrack.contains(pos)) {
            view.draggingSlider = DraggingSlider::Steps;
            return;
        }
        if (view.cfgSliderTrack.contains(pos)) {
            view.draggingSlider = DraggingSlider::Cfg;
            return;
        }
        if (view.imagesSliderTrack.contains(pos)) {
            view.draggingSlider = DraggingSlider::Images;
            return;
        }
    }

    if (view.btnGenerate.contains(pos)) {
        const auto now = std::chrono::system_clock::now().time_since_epoch().count();
        view.lastImagePath = config.outputDir + "/img_" + std::to_string(now) + ".png";

        presenter.beginGeneration(view);

        const int myId = ++view.generationId;

        const std::string prompt      = view.positivePrompt;
        const std::string negPrompt   = view.negativePrompt;
        const std::string outPathBase = view.lastImagePath; // used for single image; multi uses indexed paths
        GenerationParams params = view.generationParams;
        params.seed = view.seedInput.empty() ? -1 : std::stoll(view.seedInput);
        const auto [rw, rh] = ImageGeneratorView::kResolutions[view.selectedResolutionIdx];
        params.width  = rw;
        params.height = rh;
        const std::string modelDir    = view.availableModels.empty() ? "models" : view.availableModels[view.selectedModelIdx];
        std::atomic<bool>* done       = &view.generationDone;
        std::atomic<int>*  step       = &view.generationStep;
        std::atomic<bool>* cancel     = &view.cancelToken;
        std::atomic<int>*  idPtr      = &view.generationId;
        std::atomic<int>*  imgNum     = &view.generationImageNum;

        view.generationTotalImages.store(params.numImages);

        std::thread generationThread([prompt, negPrompt, outPathBase, params, modelDir,
                                      done, step, cancel, idPtr, imgNum, myId]() {
            // Models and text encoding happen once inside generateFromPrompt;
            // the multi-image loop runs there too, so no reload per image.
            PortraitGeneratorAi::generateFromPrompt(
                prompt, negPrompt, outPathBase, params, modelDir, step, imgNum, cancel);

            if (idPtr->load() == myId)
                done->store(true);
        });

        generationThread.detach();
    }
}

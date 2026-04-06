#include "ImageGeneratorController.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include <SFML/Window/Clipboard.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>

void ImageGeneratorController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                            ImageGeneratorView& view, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) win.close();
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape)
        appScreen = AppScreen::MENU;

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

    // Text input for active field
    if (!view.generating) {
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
    // Scan models/ subdirectories once on first update
    if (view.availableModels.empty()) {
        for (const auto& entry : std::filesystem::directory_iterator("models")) {
            if (!entry.is_directory()) continue;
            const auto dir = entry.path();
            if (std::filesystem::exists(dir / "unet.onnx"))
                view.availableModels.push_back(dir.string());
        }
        std::sort(view.availableModels.begin(), view.availableModels.end());
        view.selectedModelIdx = 0;
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
    if (view.generating) {
        if (view.btnCancelGenerate.contains(pos)) {
            ++view.generationId;          // invalidate the thread's epoch
            view.cancelToken.store(true); // signal ORT watcher to abort
            view.generating = false;      // UI responds immediately
        }
        return;
    }

    if (view.btnBack.contains(pos)) {
        appScreen = AppScreen::MENU;
        return;
    }

    if (view.positiveField.contains(pos)) {
        presenter.activatePositive(view);
        return;
    }

    if (view.negativeField.contains(pos)) {
        presenter.activateNegative(view);
        return;
    }

    if (view.btnAdvanced.contains(pos)) {
        presenter.toggleAdvanced(view);
        return;
    }

    if (!view.availableModels.empty()) {
        if (view.btnModelPrev.contains(pos)) {
            view.selectedModelIdx = (view.selectedModelIdx - 1 + static_cast<int>(view.availableModels.size()))
                                    % static_cast<int>(view.availableModels.size());
            return;
        }
        if (view.btnModelNext.contains(pos)) {
            view.selectedModelIdx = (view.selectedModelIdx + 1) % static_cast<int>(view.availableModels.size());
            return;
        }
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
        view.lastImagePath = "assets/generated/img_" + std::to_string(now) + ".png";

        presenter.beginGeneration(view);

        const int myId = ++view.generationId;

        const std::string prompt      = view.positivePrompt;
        const std::string negPrompt   = view.negativePrompt;
        const std::string outPathBase = view.lastImagePath; // used for single image; multi uses indexed paths
        GenerationParams params = view.generationParams;
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

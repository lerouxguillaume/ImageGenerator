#include "ImageGeneratorController.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"

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
            view.generationParams.numSteps = static_cast<int>(std::round(5.f + t * 25.f));
        } else if (view.draggingSlider == DraggingSlider::Cfg) {
            const sf::FloatRect& track = view.cfgSliderTrack;
            const float t   = std::clamp((mousePos.x - track.left) / track.width, 0.f, 1.f);
            const float raw = 1.0f + t * 14.0f;
            view.generationParams.guidanceScale = std::round(raw * 2.f) / 2.f;
        } else if (view.draggingSlider == DraggingSlider::Images) {
            const sf::FloatRect& track = view.imagesSliderTrack;
            const float t = std::clamp((mousePos.x - track.left) / track.width, 0.f, 1.f);
            view.generationParams.numImages = static_cast<int>(std::round(1.f + t * 9.f));
        }
    }

    // Text input for active field
    if (!view.generating) {
        if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::BackSpace) {
            if (view.positiveActive && !view.positivePrompt.empty())
                view.positivePrompt.pop_back();
            else if (view.negativeActive && !view.negativePrompt.empty())
                view.negativePrompt.pop_back();
        }
        if (e.type == sf::Event::TextEntered) {
            const auto c = e.text.unicode;
            if (c >= 32 && c < 127) {
                if (view.positiveActive && view.positivePrompt.size() < 300)
                    view.positivePrompt += static_cast<char>(c);
                else if (view.negativeActive && view.negativePrompt.size() < 300)
                    view.negativePrompt += static_cast<char>(c);
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
        const GenerationParams params = view.generationParams;
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

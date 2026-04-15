#pragma once

#include "../views/ImageGeneratorView.hpp"

// Stateless presenter: translates user intent into view state mutations.
// Each method takes the view by reference and updates only the fields it owns,
// keeping controller logic out of the view and UI logic out of the controller.
class ImageGeneratorPresenter {
public:
    // Give keyboard focus to the positive prompt field.
    void activatePositive(ImageGeneratorView& view) const;

    // Give keyboard focus to the negative prompt field.
    void activateNegative(ImageGeneratorView& view) const;

    // Show or hide the advanced parameters panel (steps, CFG, image count).
    void toggleAdvanced(ImageGeneratorView& view) const;

    // Switch the view into generating state (shows progress overlay, disables Generate button).
    void beginGeneration(ImageGeneratorView& view) const;

    // Switch the view back to idle state and trigger a result image reload.
    void finishGeneration(ImageGeneratorView& view) const;

    // Switch the view back to idle state after a pipeline error.
    // Leaves generationFailed=true so the view renders the error banner.
    void failGeneration(ImageGeneratorView& view) const;
};

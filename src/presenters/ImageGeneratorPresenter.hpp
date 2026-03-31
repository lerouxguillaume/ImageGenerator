#pragma once

#include "../views/ImageGeneratorView.hpp"

class ImageGeneratorPresenter {
public:
    void activatePositive(ImageGeneratorView& view) const;
    void activateNegative(ImageGeneratorView& view) const;
    void toggleAdvanced(ImageGeneratorView& view) const;
    void beginGeneration(ImageGeneratorView& view) const;
    void finishGeneration(ImageGeneratorView& view) const;
};

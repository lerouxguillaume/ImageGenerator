#include "ImageGeneratorPresenter.hpp"

void ImageGeneratorPresenter::activatePositive(ImageGeneratorView& view) const
{
    view.positiveActive = true;
    view.negativeActive = false;
    view.positiveCursor = static_cast<int>(view.positivePrompt.size());
}

void ImageGeneratorPresenter::activateNegative(ImageGeneratorView& view) const
{
    view.negativeActive = true;
    view.positiveActive = false;
    view.negativeCursor = static_cast<int>(view.negativePrompt.size());
}

void ImageGeneratorPresenter::toggleAdvanced(ImageGeneratorView& view) const
{
    view.showAdvancedParams = !view.showAdvancedParams;
}

void ImageGeneratorPresenter::beginGeneration(ImageGeneratorView& view) const
{
    view.generating = true;
    view.generationDone.store(false);
    view.cancelToken.store(false);
    view.generationStep.store(0);
    view.resultLoaded = false;
    view.generationFailed.store(false);
    view.generationErrorMsg.clear();
}

void ImageGeneratorPresenter::finishGeneration(ImageGeneratorView& view) const
{
    view.generating = false;
    view.generationDone.store(false);
}

void ImageGeneratorPresenter::failGeneration(ImageGeneratorView& view) const
{
    view.generating = false;
    view.generationDone.store(false);
    // generationFailed stays true — the view renders the error banner until the next run
}

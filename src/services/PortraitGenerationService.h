#pragma once
#include <atomic>
#include <thread>

#include "../enum/enums.hpp"

// Manages the background generation thread for character portraits.
// Wraps PortraitGeneratorAi so the UI can start, poll, and cancel a generation
// without blocking the render loop.
class PortraitGenerationService {
public:
    struct GenerationParams {
        int   numSteps      = 20;
        float guidanceScale = 7.5f;
    };

    PortraitGenerationService();
    ~PortraitGenerationService(); // joins any running thread

    // Spawn a background thread that runs the full SD pipeline for the given character.
    // No-op if a generation is already in progress.
    void startGeneration(Race race, Gender gender, const GenerationParams& params);

    // True while the background thread is alive (between startGeneration and completion/cancel).
    bool isGenerating() const;

    // Number of denoising steps completed so far in the current run.
    int getCurrentStep() const;

    // True once the pipeline finishes (success or error). Reset on next startGeneration.
    bool isGenerationComplete() const;

    // Request cancellation. The pipeline will stop before or during the next ORT Run().
    void cancelGeneration();

private:
    std::atomic<bool> generationDone_{false};
    std::atomic<int>  generationStep_{0};
    std::thread       generationThread_;

    void generationThreadFunc(Race race, Gender gender, const GenerationParams& params);
    void cleanupGeneration();
    void emitProgressEvent(int currentStep, int totalSteps);
};

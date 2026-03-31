#pragma once

struct WidgetEvent {
    enum Type {
        None, RowClicked, PortraitClicked, GenerateButtonClicked,
        SliderChanged, ModalClosed, GenerationStarted, GenerationComplete,
        BackButtonClicked
    };
    Type type = None;
    int data = -1;
};

#pragma once

#include "../views/MenuView.hpp"

class MenuPresenter {
public:
    void openNewGameModal(MenuView& view) const;
    void closeNewGameModal(MenuView& view) const;
    void openLoadModal(MenuView& view) const;
    void closeLoadModal(MenuView& view) const;
    void openDeleteConfirm(MenuView& view, int idx) const;
    void closeDeleteConfirm(MenuView& view) const;
};

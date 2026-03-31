#include "MenuPresenter.hpp"

void MenuPresenter::openNewGameModal(MenuView& view) const
{
    view.guildName = "My Guild";
    view.showModal = true;
}

void MenuPresenter::closeNewGameModal(MenuView& view) const
{
    view.showModal = false;
}

void MenuPresenter::openLoadModal(MenuView& view) const
{
    view.showLoadModal = true;
}

void MenuPresenter::closeLoadModal(MenuView& view) const
{
    view.showLoadModal = false;
}

void MenuPresenter::openDeleteConfirm(MenuView& view, int idx) const
{
    view.pendingDeleteIdx = idx;
    view.showDeleteConfirm = true;
}

void MenuPresenter::closeDeleteConfirm(MenuView& view) const
{
    view.showDeleteConfirm = false;
    view.pendingDeleteIdx = -1;
}

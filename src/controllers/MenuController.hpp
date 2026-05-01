#pragma once
#include <filesystem>
#include <optional>
#include <cstdint>
#include <SFML/Graphics.hpp>

#include "../presenters/MenuPresenter.hpp"
#include "../views/MenuView.hpp"
#include "../import/ModelImporter.hpp"
#include "../import/ImportedModelRegistry.hpp"
#include "../ui/widgets/ImportModelModal.hpp"

enum class AppScreen : std::uint8_t { MENU, Playing, ImageGenerator, ImageEditor, Projects };

class MenuController {
public:
    MenuController();

    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     MenuView& screen, AppScreen& appScreen);

    // Called by App::run() after menuScreen.render() to draw modal overlays.
    void renderOverlay(sf::RenderWindow& win);

private:
    void handleClick(sf::Vector2f pos, MenuView& screen, AppScreen& appScreen);
    void openFilePicker();
    void handleModalActions();

    static std::filesystem::path executableDir();
    static std::filesystem::path managedModelsDir();
    static std::filesystem::path localDataDir(); // safe local path for venv (not on shared drives)

    MenuPresenter         presenter_;
    ModelImporter         importer_;
    ImportedModelRegistry registry_;
    ImportModelModal      modal_;
    bool                  showModal_ = false;
};

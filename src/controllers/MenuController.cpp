#include "MenuController.hpp"
#include "../import/ImportedModelRegistry.hpp"
#include "../managers/Logger.hpp"

#include <filesystem>

// ── File picker (single file, .safetensors) ───────────────────────────────────

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>

static std::string browseForFile(const std::string& startDir) {
    char buf[MAX_PATH] = {};
    OPENFILENAMEA ofn{};
    ofn.lStructSize  = sizeof(ofn);
    ofn.lpstrFilter  = "Safetensors files\0*.safetensors\0All files\0*.*\0\0";
    ofn.lpstrFile    = buf;
    ofn.nMaxFile     = MAX_PATH;
    ofn.lpstrTitle   = "Select safetensors model";
    ofn.Flags        = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!startDir.empty()) ofn.lpstrInitialDir = startDir.c_str();
    std::string result;
    if (GetOpenFileNameA(&ofn)) result = buf;
    return result;
}
#else
static std::string browseForFile(const std::string& startDir) {
    std::string cmd = "zenity --file-selection"
                      " --title='Select safetensors model'"
                      " --file-filter='Safetensors files (*.safetensors) | *.safetensors'";
    if (!startDir.empty()) cmd += " --filename='" + startDir + "/'";
    cmd += " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    char tmp[4096] = {};
    std::string result;
    while (fgets(tmp, sizeof(tmp), pipe)) result += tmp;
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}
#endif

// ── Executable directory ──────────────────────────────────────────────────────

std::filesystem::path MenuController::executableDir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
#else
    return std::filesystem::read_symlink("/proc/self/exe").parent_path();
#endif
}

std::filesystem::path MenuController::managedModelsDir() {
    return executableDir() / "models" / "imported";
}

std::filesystem::path MenuController::localDataDir() {
    // On Windows the app may run from a shared/network drive (e.g. VirtualBox
    // sf_shared_vm). Python venv creation fails on such paths because they
    // don't support symlinks. Use %LOCALAPPDATA% instead which is always local.
#ifdef _WIN32
    const char* base = std::getenv("LOCALAPPDATA");
    if (base && *base)
        return std::filesystem::path(base) / "ImageGenerator";
#endif
    return executableDir();
}

// ── Constructor ───────────────────────────────────────────────────────────────

MenuController::MenuController()
    : importer_(executableDir() / "scripts", managedModelsDir(), localDataDir())
    , registry_(managedModelsDir() / "registry.json")
{}

// ── Event handling ────────────────────────────────────────────────────────────

void MenuController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                 MenuView& screen, AppScreen& appScreen) {
    if (showModal_) {
        const bool consumed = modal_.handleEvent(e, win);
        handleModalActions();
        if (consumed) return;
    }

    if (e.type == sf::Event::Closed) win.close();
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        if (showModal_) showModal_ = false;
        else win.close();
        return;
    }
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), screen, appScreen);
}

void MenuController::handleClick(sf::Vector2f pos, MenuView& screen, AppScreen& appScreen) {
    if (screen.btnImageGen.contains(pos)) {
        appScreen = AppScreen::ImageGenerator;
    } else if (screen.btnImageEdit.contains(pos)) {
        appScreen = AppScreen::ImageEditor;
    } else if (screen.btnProjects.contains(pos)) {
        appScreen = AppScreen::Projects;
    } else if (screen.btnImportModel.contains(pos)) {
        showModal_ = true;
        importer_.reset();
    }
}

// ── Overlay render + modal action dispatch ────────────────────────────────────

void MenuController::renderOverlay(sf::RenderWindow& win) {
    if (!showModal_) return;

    modal_.syncFrom(importer_);
    modal_.render(win);
    handleModalActions();
}

void MenuController::handleModalActions() {
    if (modal_.closeRequested) {
        modal_.closeRequested = false;
        if (!importer_.isRunning()) {
            showModal_ = false;
            importer_.reset();
        }
        return;
    }

    if (modal_.browseRequested) {
        modal_.browseRequested = false;
        const std::string picked = browseForFile(modal_.filePath.empty()
                                                 ? std::string{}
                                                 : std::filesystem::path(modal_.filePath)
                                                       .parent_path().string());
        if (!picked.empty()) modal_.filePath = picked;
        return;
    }

    if (modal_.importRequested) {
        modal_.importRequested = false;
        importer_.reset();
        importer_.start(modal_.filePath, modal_.archArg());
        return;
    }

    if (modal_.cancelRequested) {
        modal_.cancelRequested = false;
        importer_.cancel();
        return;
    }

    // Register on completion (once)
    if (importer_.getState() == ModelImporter::State::Done) {
        const std::string id = importer_.getModelId();
        if (!id.empty() && !registry_.exists(id)) {
            const SafetensorsInfo info = importer_.getInspectionResult();
            ImportedModel m;
            m.id       = id;
            m.name     = id;
            m.arch     = info.archArg();
            m.onnxPath = importer_.getOutputDir();
            registry_.add(m);
            Logger::info("Registered imported model: " + id);
        }
    }
}

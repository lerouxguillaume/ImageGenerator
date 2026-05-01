#include "ModelImporter.hpp"

#include "PythonEnvManager.hpp"
#include "SafetensorsInspector.hpp"
#include "Subprocess.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string sanitizeId(const std::string& stem) {
    std::string id;
    id.reserve(stem.size());
    for (char c : stem)
        id += (std::isalnum(static_cast<unsigned char>(c)) || c == '-') ? c : '_';
    return id;
}

// ── Constructor / destructor ──────────────────────────────────────────────────

ModelImporter::ModelImporter(std::filesystem::path scriptsDir,
                             std::filesystem::path managedModelsDir,
                             std::filesystem::path venvBaseDir)
    : scriptsDir_(std::move(scriptsDir))
    , managedModelsDir_(std::move(managedModelsDir))
    , venvBaseDir_(std::move(venvBaseDir))
{}

ModelImporter::~ModelImporter() {
    cancel();
    if (thread_.joinable()) thread_.join();
}

// ── Public control ────────────────────────────────────────────────────────────

void ModelImporter::start(const std::filesystem::path& path,
                          const std::string& archOverride) {
    if (isRunning()) return;

    cancelRequested_ = false;
    state_           = State::Idle;
    {
        std::lock_guard lk(dataMutex_);
        statusMsg_.clear();
        logLines_.clear();
        outputDir_.clear();
        modelId_.clear();
    }

    thread_ = std::thread(&ModelImporter::runThread, this, path, archOverride);
}

void ModelImporter::cancel() {
    cancelRequested_ = true;
    std::lock_guard lk(subprocessMutex_);
    if (activeSubprocess_) activeSubprocess_->kill();
}

void ModelImporter::reset() {
    if (isRunning()) return;
    if (thread_.joinable()) thread_.join();
    state_ = State::Idle;
    std::lock_guard lk(dataMutex_);
    statusMsg_.clear();
    logLines_.clear();
}

bool ModelImporter::isRunning() const {
    const auto s = state_.load();
    return s != State::Idle && s != State::Done && s != State::Failed;
}

// ── Thread-safe accessors ─────────────────────────────────────────────────────

ModelImporter::State ModelImporter::getState() const {
    return state_.load();
}

std::string ModelImporter::getStatusMsg() const {
    std::lock_guard lk(dataMutex_);
    return statusMsg_;
}

std::vector<std::string> ModelImporter::getLogLines() const {
    std::lock_guard lk(dataMutex_);
    return logLines_;
}

std::filesystem::path ModelImporter::getOutputDir() const {
    std::lock_guard lk(dataMutex_);
    return outputDir_;
}

std::string ModelImporter::getModelId() const {
    std::lock_guard lk(dataMutex_);
    return modelId_;
}

SafetensorsInfo ModelImporter::getInspectionResult() const {
    std::lock_guard lk(dataMutex_);
    return inspectionResult_;
}

// ── Private helpers ───────────────────────────────────────────────────────────

void ModelImporter::appendLog(const std::string& line) {
    std::lock_guard lk(dataMutex_);
    if (logLines_.size() >= kMaxLogLines)
        logLines_.erase(logLines_.begin());
    logLines_.push_back(line);
}

void ModelImporter::setStatus(const std::string& msg) {
    std::lock_guard lk(dataMutex_);
    statusMsg_ = msg;
}

// ── Import thread ─────────────────────────────────────────────────────────────

void ModelImporter::runThread(std::filesystem::path path, std::string archOverride) {
    // 1. Inspect safetensors header ───────────────────────────────────────────
    state_ = State::Analyzing;
    setStatus("Analyzing file…");
    appendLog("Inspecting: " + path.filename().string());

    SafetensorsInfo info = SafetensorsInspector::inspect(path);
    {
        std::lock_guard lk(dataMutex_);
        inspectionResult_ = info;
    }

    if (!info.valid) {
        setStatus("Invalid file: " + info.error);
        state_ = State::Failed;
        return;
    }
    if (info.modelType == SafetensorsInfo::ModelType::LoRA) {
        setStatus("LoRA adapters are not supported in this import flow.");
        state_ = State::Failed;
        return;
    }

    appendLog(std::string("Type: ") + info.modelTypeName()
              + "  Architecture: " + info.architectureName()
              + "  Dtype: " + info.dtype
              + "  Tensors: " + std::to_string(info.tensorCount));

    if (archOverride == "auto")
        archOverride = info.archArg(); // use detected arch if possible

    if (cancelRequested_) { state_ = State::Failed; return; }

    // 2. Determine model ID and output dir ────────────────────────────────────
    const std::string stem = path.stem().string();
    const std::string id   = sanitizeId(stem);
    const std::filesystem::path outDir = managedModelsDir_ / id;
    {
        std::lock_guard lk(dataMutex_);
        modelId_   = id;
        outputDir_ = outDir;
    }
    appendLog("Output: " + outDir.string());

    // 3. Python environment ───────────────────────────────────────────────────
    state_ = State::SettingUpPython;
    setStatus("Setting up Python environment…");

    const std::filesystem::path requirementsPath =
        scriptsDir_ / "requirements_import.txt";
    PythonEnvManager pyEnv(venvBaseDir_, requirementsPath);

    if (!pyEnv.isReady()) {
        appendLog("First run: installing Python packages…");
        bool ok = pyEnv.setup([this](const std::string& line) {
            appendLog(line);
            // Surface errors from setup immediately
            if (line.rfind("ERROR:", 0) == 0)
                setStatus(line.substr(6));
        });
        if (!ok || cancelRequested_) {
            state_ = State::Failed;
            return;
        }
    }

    if (cancelRequested_) { state_ = State::Failed; return; }

    // 4. Run export script ────────────────────────────────────────────────────
    state_ = State::Exporting;
    setStatus("Exporting…");

    const std::string pythonExe  = pyEnv.pythonPath().string();
    const std::string scriptPath = (scriptsDir_ / "import_model.py").string();

    const std::vector<std::string> args = {
        pythonExe, scriptPath,
        "--input",  path.string(),
        "--output", outDir.string(),
        "--arch",   archOverride,
    };

    auto sub = std::make_shared<Subprocess>();
    {
        std::lock_guard lk(subprocessMutex_);
        activeSubprocess_ = sub;
    }

    if (!sub->start(args, scriptsDir_)) {
        setStatus("Failed to launch Python export script.");
        state_ = State::Failed;
        return;
    }

    std::string line;
    while (sub->readLine(line)) {
        if (cancelRequested_) {
            sub->kill();
            state_ = State::Failed;
            setStatus("Cancelled.");
            return;
        }

        // PROGRESS: protocol
        if (line.rfind("PROGRESS:", 0) == 0) {
            const std::string tag = line.substr(9);
            if (tag == "analyzing")
                setStatus("Analyzing…");
            else if (tag.rfind("exporting", 0) == 0)
                setStatus("Exporting…");
            else if (tag == "validating") {
                state_ = State::Validating;
                setStatus("Validating output…");
            } else if (tag == "done") {
                // handled after loop
            }
            continue;
        }

        // ERROR: protocol
        if (line.rfind("ERROR:", 0) == 0) {
            setStatus(line.substr(6));
            appendLog(line);
            continue;
        }

        appendLog(line);
    }

    const int exitCode = sub->wait();
    {
        std::lock_guard lk(subprocessMutex_);
        activeSubprocess_.reset();
    }

    if (cancelRequested_ || exitCode != 0) {
        if (!cancelRequested_)
            setStatus("Export failed (exit " + std::to_string(exitCode) + "). See log.");
        state_ = State::Failed;
        return;
    }

    state_ = State::Done;
    setStatus("Import complete: " + id);
    appendLog("Model ready at: " + outDir.string());
}

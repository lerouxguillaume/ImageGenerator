#!/usr/bin/env bash
# Cross-compile for Windows from Linux using MinGW-w64.
# Usage:
#   bash scripts/build-windows.sh                    # DirectML (default)
#   bash scripts/build-windows.sh --cuda             # CUDA GPU
#   bash scripts/build-windows.sh --genai            # DirectML + LLM prompt enhancer
#   bash scripts/build-windows.sh --cuda --genai     # CUDA + LLM prompt enhancer
set -e

# ── Mode selection ─────────────────────────────────────────────────────────────
USE_CUDA_BUILD=false
USE_GENAI_BUILD=false
for arg in "$@"; do
    [ "$arg" = "--cuda"  ] && USE_CUDA_BUILD=true
    [ "$arg" = "--genai" ] && USE_GENAI_BUILD=true
done

DEPS_DIR="$(pwd)/deps/windows"
BUILD_DIR="$(pwd)/build-windows"
INSTALL_DIR="$(pwd)/dist-windows"

SFML_VERSION="2.6.1"
ONNX_VERSION="1.24.4"
OPENCV_VERSION="4.10.0"
GENAI_VERSION="0.12.0"

mkdir -p "$DEPS_DIR" "$BUILD_DIR" "$INSTALL_DIR"

# ── Prerequisites ──────────────────────────────────────────────────────────────
echo "Checking prerequisites..."
for tool in x86_64-w64-mingw32-g++ cmake wget unzip gendef; do
    command -v "$tool" >/dev/null || { echo "Missing: $tool. Install with: sudo apt install mingw-w64 cmake wget unzip mingw-w64-tools"; exit 1; }
done

# ── SFML ──────────────────────────────────────────────────────────────────────
SFML_DIR="$DEPS_DIR/SFML-$SFML_VERSION"
if [ ! -d "$SFML_DIR" ]; then
    echo "Downloading SFML $SFML_VERSION for Windows..."
    wget -q "https://www.sfml-dev.org/files/SFML-${SFML_VERSION}-windows-gcc-13.1.0-mingw-64-bit.zip" -O "$DEPS_DIR/sfml.zip"
    unzip -q "$DEPS_DIR/sfml.zip" -d "$DEPS_DIR"
    rm "$DEPS_DIR/sfml.zip"
fi

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Helper: generate a MinGW-compatible import library from a DLL.
gen_import_lib() {
    local dll="$1"          # e.g. onnxruntime.dll
    local lib="$2"          # e.g. onnxruntime.lib
    gendef "$dll"
    local def="${dll%.dll}.def"
    [ -f "$def" ] || { echo "gendef failed to produce $def"; exit 1; }
    x86_64-w64-mingw32-dlltool -d "$def" -l "$lib" --as-flags="--64"
    rm -f "$def"
}

if $USE_CUDA_BUILD; then
    # ── ORT CUDA (GitHub release) ──────────────────────────────────────────────
    # The CUDA package includes onnxruntime.dll, onnxruntime_providers_cuda.dll,
    # and onnxruntime_providers_shared.dll. It does NOT bundle the CUDA runtime
    # (cudart64_*.dll etc.) — those must come from the CUDA toolkit on the target
    # machine (CUDA 12.x recommended).
    ONNX_DIR="$DEPS_DIR/onnxruntime-cuda-$ONNX_VERSION"
    if [ ! -d "$ONNX_DIR" ]; then
        echo "Downloading ONNX Runtime $ONNX_VERSION (CUDA) from GitHub..."
        wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-win-x64-gpu-${ONNX_VERSION}.zip" \
             -O "$DEPS_DIR/onnx-cuda.zip"
        TMP="$DEPS_DIR/onnx-cuda-tmp"
        unzip -q "$DEPS_DIR/onnx-cuda.zip" -d "$TMP"
        rm "$DEPS_DIR/onnx-cuda.zip"

        # GitHub release layout: onnxruntime-win-x64-gpu-X.Y.Z/{include,lib}/
        EXTRACTED=$(find "$TMP" -maxdepth 1 -mindepth 1 -type d | head -1)
        mkdir -p "$ONNX_DIR"
        cp -r "$EXTRACTED/." "$ONNX_DIR/"
        rm -rf "$TMP"

        # Regenerate MinGW-compatible import libraries from each DLL
        echo "Generating MinGW import libraries for CUDA ORT DLLs..."
        pushd "$ONNX_DIR/lib" > /dev/null
        gen_import_lib onnxruntime.dll onnxruntime.lib
        popd > /dev/null
    fi
    ONNX_CMAKE_FLAG="-DUSE_CUDA=ON -DUSE_DML=OFF"
else
    # ── ORT DirectML (NuGet) ───────────────────────────────────────────────────
    # The DirectML provider lives in the NuGet package, not the GitHub GPU release.
    # The .nupkg is a zip; extract and normalize into a layout CMake expects:
    #   include/  ← build/native/include/
    #   lib/      ← onnxruntime.lib + all DLLs
    ONNX_DIR="$DEPS_DIR/onnxruntime-dml-$ONNX_VERSION"
    if [ ! -d "$ONNX_DIR" ]; then
        echo "Downloading ONNX Runtime $ONNX_VERSION (DirectML) from NuGet..."
        wget -q "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ONNX_VERSION}" \
             -O "$DEPS_DIR/onnx-dml.nupkg"
        TMP="$DEPS_DIR/onnx-dml-tmp"
        unzip -q "$DEPS_DIR/onnx-dml.nupkg" -d "$TMP"
        rm "$DEPS_DIR/onnx-dml.nupkg"

        mkdir -p "$ONNX_DIR/include" "$ONNX_DIR/lib"
        cp -r "$TMP/build/native/include/." "$ONNX_DIR/include/"
        cp -r "$TMP/runtimes/win-x64/native/." "$ONNX_DIR/lib/"
        rm -rf "$TMP"

        echo "Generating MinGW import library for onnxruntime.dll..."
        pushd "$ONNX_DIR/lib" > /dev/null
        gen_import_lib onnxruntime.dll onnxruntime.lib
        popd > /dev/null
    fi
    ONNX_CMAKE_FLAG="-DUSE_DML=ON -DUSE_CUDA=OFF"
fi

# ── nlohmann/json (header-only) ───────────────────────────────────────────────
JSON_DIR="$DEPS_DIR/nlohmann_json"
if [ ! -f "$JSON_DIR/include/nlohmann/json.hpp" ]; then
    echo "Downloading nlohmann/json..."
    mkdir -p "$JSON_DIR/include/nlohmann"
    wget -q "https://github.com/nlohmann/json/releases/latest/download/json.hpp" -O "$JSON_DIR/include/nlohmann/json.hpp"
fi
if [ ! -f "$JSON_DIR/nlohmann_jsonConfig.cmake" ]; then
    cat > "$JSON_DIR/nlohmann_jsonConfig.cmake" <<'EOF'
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
EOF
fi

# ── SQLite3 (amalgamation) ────────────────────────────────────────────────────
SQLITE_DIR="$DEPS_DIR/sqlite3"
if [ ! -f "$SQLITE_DIR/sqlite3.h" ]; then
    echo "Downloading SQLite3..."
    mkdir -p "$SQLITE_DIR"
    wget -q "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip" -O "$DEPS_DIR/sqlite.zip"
    unzip -q "$DEPS_DIR/sqlite.zip" -d "$DEPS_DIR/sqlite_tmp"
    cp "$DEPS_DIR"/sqlite_tmp/*/sqlite3.{c,h} "$SQLITE_DIR/"
    rm -rf "$DEPS_DIR/sqlite_tmp" "$DEPS_DIR/sqlite.zip"
    x86_64-w64-mingw32-gcc -O2 -c "$SQLITE_DIR/sqlite3.c" -o "$SQLITE_DIR/sqlite3.o"
    x86_64-w64-mingw32-ar rcs "$SQLITE_DIR/libsqlite3.a" "$SQLITE_DIR/sqlite3.o"
fi

# ── OpenCV ────────────────────────────────────────────────────────────────────
OPENCV_DIR="$DEPS_DIR/opencv-$OPENCV_VERSION"
OPENCV_SRC="$DEPS_DIR/opencv-src-$OPENCV_VERSION"
if [ ! -f "$OPENCV_DIR/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    echo "Building OpenCV $OPENCV_VERSION for Windows (this takes a while)..."
    rm -rf "$OPENCV_DIR" "$DEPS_DIR/opencv-build" "$OPENCV_SRC"

    wget -q "https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz" -O "$DEPS_DIR/opencv.tar.gz"
    tar -xzf "$DEPS_DIR/opencv.tar.gz" -C "$DEPS_DIR"
    mv "$DEPS_DIR/opencv-$OPENCV_VERSION" "$OPENCV_SRC"
    rm "$DEPS_DIR/opencv.tar.gz"

    cmake -S "$OPENCV_SRC" -B "$DEPS_DIR/opencv-build" \
        -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-mingw64.cmake" \
        -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        -DWITH_IPP=OFF -DWITH_ITT=OFF \
        -DBUILD_LIST="core,imgproc,imgcodecs" \
        -DCMAKE_BUILD_TYPE=Release || { echo "OpenCV cmake configure failed"; exit 1; }

    cmake --build "$DEPS_DIR/opencv-build" --target install \
        || { echo "OpenCV build/install failed"; exit 1; }

    rm -rf "$DEPS_DIR/opencv-build" "$OPENCV_SRC"
fi

OPENCV_CMAKE_DIR=$(find "$OPENCV_DIR" -name "OpenCVConfig.cmake" -printf "%h\n" 2>/dev/null | head -1)
if [ -z "$OPENCV_CMAKE_DIR" ]; then
    echo "ERROR: OpenCVConfig.cmake not found under $OPENCV_DIR"
    exit 1
fi

# ── ORT GenAI (optional, for LLM prompt enhancement) ─────────────────────────
GENAI_CMAKE_FLAG="-DUSE_GENAI=OFF"
if $USE_GENAI_BUILD; then
    GENAI_DIR="$DEPS_DIR/ort-genai-$GENAI_VERSION"
    if [ ! -d "$GENAI_DIR" ]; then
        if $USE_CUDA_BUILD; then
            GENAI_URL="https://github.com/microsoft/onnxruntime-genai/releases/download/v${GENAI_VERSION}/onnxruntime-genai-${GENAI_VERSION}-win-x64-cuda.zip"
        else
            GENAI_URL="https://github.com/microsoft/onnxruntime-genai/releases/download/v${GENAI_VERSION}/onnxruntime-genai-${GENAI_VERSION}-win-x64.zip"
        fi
        echo "Downloading ORT GenAI $GENAI_VERSION..."
        echo $GENAI_URL
        wget -q "$GENAI_URL" -O "$DEPS_DIR/ort-genai.zip" || {
            echo "ERROR: Could not download ORT GenAI $GENAI_VERSION."
            echo "  Check available releases at: https://github.com/microsoft/onnxruntime-genai/releases"
            echo "  Then set GENAI_VERSION at the top of this script."
            exit 1
        }
        TMP="$DEPS_DIR/ort-genai-tmp"
        unzip -q "$DEPS_DIR/ort-genai.zip" -d "$TMP"
        rm "$DEPS_DIR/ort-genai.zip"

        # Normalise layout: the zip may or may not have a top-level directory.
        EXTRACTED=$(find "$TMP" -maxdepth 1 -mindepth 1 -type d | head -1)
        if [ -d "$EXTRACTED/lib" ]; then
            mkdir -p "$GENAI_DIR"
            cp -r "$EXTRACTED/." "$GENAI_DIR/"
        else
            # Files are directly at the zip root — wrap them.
            mkdir -p "$GENAI_DIR"
            cp -r "$TMP/." "$GENAI_DIR/"
        fi
        rm -rf "$TMP"

        echo "Generating MinGW import library for onnxruntime-genai.dll..."
        pushd "$GENAI_DIR/lib" > /dev/null
        gen_import_lib onnxruntime-genai.dll onnxruntime-genai.lib
        popd > /dev/null
    fi
    GENAI_CMAKE_FLAG="-DUSE_GENAI=ON -DORT_GENAI_ROOT=$GENAI_DIR"
    echo "ORT GenAI: $GENAI_DIR"
fi

# ── Configure & Build ─────────────────────────────────────────────────────────
LABEL="DirectML"
$USE_CUDA_BUILD  && LABEL="CUDA"
$USE_GENAI_BUILD && LABEL="$LABEL + GenAI"
echo "Configuring ($LABEL)..."

# Remove stale cache so switching between DML and CUDA doesn't bleed over.
rm -f "$BUILD_DIR/CMakeCache.txt"

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-mingw64.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DSFML_DIR="$SFML_DIR/lib/cmake/SFML" \
    -DONNXRUNTIME_ROOT="$ONNX_DIR" \
    -DOpenCV_DIR="$OPENCV_CMAKE_DIR" \
    -Dnlohmann_json_DIR="$JSON_DIR" \
    -DSQLITE3_ROOT="$SQLITE_DIR" \
    $ONNX_CMAKE_FLAG \
    $GENAI_CMAKE_FLAG

echo "Building..."
cmake --build "$BUILD_DIR"

ZIP_SUFFIX=""
$USE_CUDA_BUILD  && ZIP_SUFFIX="${ZIP_SUFFIX}-cuda"
$USE_GENAI_BUILD && ZIP_SUFFIX="${ZIP_SUFFIX}-genai"
ZIP_OUT="$(pwd)/image-generation-windows${ZIP_SUFFIX}.zip"
echo "Packaging..."
cd "$BUILD_DIR"
zip -r "$ZIP_OUT" . -x "*.o" "*.a" "CMakeFiles/*" "cmake_install.cmake" "Makefile" "CMakeCache.txt"
cd - > /dev/null

echo ""
echo "Done."
echo "  Build output : $BUILD_DIR"
echo "  Release zip  : $ZIP_OUT"

if $USE_CUDA_BUILD; then
    echo ""
    echo "NOTE: CUDA build requires CUDA 12.x and cuDNN 9.x on the target machine."
    echo "      1. CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads"
    echo "      2. cuDNN 9.x for CUDA 12: https://developer.nvidia.com/cudnn"
    echo "         Required DLLs (must be in PATH or next to the .exe):"
    echo "           cudnn64_9.dll, cudnn_ops64_9.dll, cudnn_cnn64_9.dll,"
    echo "           cudnn_adv64_9.dll, cudnn_graph64_9.dll"
    echo "      CUDNN_STATUS_NOT_INITIALIZED usually means cuDNN 9 DLLs are missing."
fi

if $USE_GENAI_BUILD; then
    echo ""
    echo "NOTE: GenAI build — place the Phi-3 model next to the .exe:"
    echo "      models/phi3-mini-onnx/  (contents from HuggingFace)"
    if $USE_CUDA_BUILD; then
        echo "      Use the cuda-int4-rtn-block-32 model variant."
    else
        echo "      Use the directml-int4-awq-block-128 model variant for DML,"
        echo "      or cpu-int4-rtn-block-32 if you want CPU-only inference."
    fi
    echo "      Set promptEnhancer.enabled=true in config.json to activate."
fi

#!/usr/bin/env bash
# Cross-compile GuildMaster for Windows from Linux using MinGW-w64
# Usage: bash scripts/build-windows.sh
set -e

DEPS_DIR="$(pwd)/deps/windows"
BUILD_DIR="$(pwd)/build-windows"
INSTALL_DIR="$(pwd)/dist-windows"

SFML_VERSION="2.6.1"
ONNX_VERSION="1.20.1"
OPENCV_VERSION="4.10.0"

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

# ── ONNX Runtime (DirectML via NuGet) ────────────────────────────────────────
# The DirectML provider lives in the NuGet package, not the GitHub GPU release.
# The .nupkg is a zip; we extract and normalize it into a layout CMake expects:
#   include/  ← build/native/include/
#   lib/      ← onnxruntime.lib + all DLLs (onnxruntime, providers_dml,
#                providers_shared, DirectML)
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

    # The NuGet .lib is MSVC-format; MinGW can't link it.
    # Regenerate a compatible import library from the DLL using gendef + dlltool.
    # The NuGet .lib is MSVC-format; regenerate a MinGW-compatible one.
    echo "Generating MinGW import library for onnxruntime.dll..."
    pushd "$ONNX_DIR/lib" > /dev/null
    gendef onnxruntime.dll
    [ -f onnxruntime.def ] || { echo "gendef failed to produce onnxruntime.def"; exit 1; }
    x86_64-w64-mingw32-dlltool -d onnxruntime.def -l onnxruntime.lib --as-flags="--64"
    rm -f onnxruntime.def
    popd > /dev/null
fi

# ── nlohmann/json (header-only) ───────────────────────────────────────────────
JSON_DIR="$DEPS_DIR/nlohmann_json"
if [ ! -f "$JSON_DIR/include/nlohmann/json.hpp" ]; then
    echo "Downloading nlohmann/json..."
    mkdir -p "$JSON_DIR/include/nlohmann"
    wget -q "https://github.com/nlohmann/json/releases/latest/download/json.hpp" -O "$JSON_DIR/include/nlohmann/json.hpp"
fi
# Generate a minimal CMake config so find_package(nlohmann_json) works
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
    # Cross-compile sqlite3 as a static library
    x86_64-w64-mingw32-gcc -O2 -c "$SQLITE_DIR/sqlite3.c" -o "$SQLITE_DIR/sqlite3.o"
    x86_64-w64-mingw32-ar rcs "$SQLITE_DIR/libsqlite3.a" "$SQLITE_DIR/sqlite3.o"
fi

# ── OpenCV ────────────────────────────────────────────────────────────────────
OPENCV_DIR="$DEPS_DIR/opencv-$OPENCV_VERSION"
OPENCV_SRC="$DEPS_DIR/opencv-src-$OPENCV_VERSION"
# Guard on the cmake config file, not the directory — a partial install won't fool this
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

# ── Locate OpenCVConfig.cmake ─────────────────────────────────────────────────
OPENCV_CMAKE_DIR=$(find "$OPENCV_DIR" -name "OpenCVConfig.cmake" -printf "%h\n" 2>/dev/null | head -1)
if [ -z "$OPENCV_CMAKE_DIR" ]; then
    echo "ERROR: OpenCVConfig.cmake not found under $OPENCV_DIR"
    exit 1
fi
echo "Found OpenCVConfig.cmake at: $OPENCV_CMAKE_DIR"

# ── Configure & Build ─────────────────────────────────────────────────────────
echo "Configuring..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-mingw64.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DSFML_DIR="$SFML_DIR/lib/cmake/SFML" \
    -DONNXRUNTIME_ROOT="$ONNX_DIR" \
    -DOpenCV_DIR="$OPENCV_CMAKE_DIR" \
    -Dnlohmann_json_DIR="$JSON_DIR" \
    -DSQLITE3_ROOT="$SQLITE_DIR" \
    -DUSE_DML=ON

echo "Building..."
cmake --build "$BUILD_DIR"

ZIP_OUT="$(pwd)/guild_master-windows.zip"
echo "Packaging..."
cd "$BUILD_DIR"
zip -r "$ZIP_OUT" . -x "*.o" "*.a" "CMakeFiles/*" "cmake_install.cmake" "Makefile" "CMakeCache.txt"
cd - > /dev/null

echo ""
echo "Done."
echo "  Build output : $BUILD_DIR"
echo "  Release zip  : $ZIP_OUT"
cmake_minimum_required(VERSION 3.14)
project(ImageGenerator CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA  "Enable CUDA GPU acceleration via ONNX Runtime" OFF)
option(USE_DML   "Enable DirectML GPU acceleration via ONNX Runtime (Windows)" OFF)
option(USE_GENAI "Enable ONNX Runtime GenAI for offline LLM prompt enhancement" OFF)

# ── Find SFML ─────────────────────────────────────────────────────────────────
# If SFML is not in a standard location, set SFML_DIR:
#   cmake -DSFML_DIR=/path/to/sfml/lib/cmake/SFML ..
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(WIN32)
        set(ONNXRUNTIME_ROOT "C:/onnxruntime" CACHE PATH "ONNX Runtime root directory")
    elseif(APPLE)
        set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime root directory")
    elseif(USE_CUDA)
        set(ONNXRUNTIME_ROOT "/home/guillaume/libs/onnxruntime/onnxruntime-linux-x64-gpu-1.24.4" CACHE PATH "ONNX Runtime root directory")
    else()
        set(ONNXRUNTIME_ROOT "/home/guillaume/libs/onnxruntime/onnxruntime-linux-x64-1.24.4" CACHE PATH "ONNX Runtime root directory")
    endif()
endif()
# On Windows the import library is onnxruntime.lib; set it directly to avoid
# find_library suffix mismatches during Linux→Windows cross-compilation.
if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    if(NOT EXISTS "${ONNXRUNTIME_LIB}")
        message(FATAL_ERROR "ONNX Runtime import library not found: ${ONNXRUNTIME_LIB}")
    endif()
else()
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
endif()

file(GLOB_RECURSE SOURCES
        "src/*.cpp"
        "src/*.h"
)

# Embed the application icon.
# Generate all icon assets first with:  python scripts/generate_icon.py
if(WIN32)
    # Embeds icon into the .exe via windres.
    list(APPEND SOURCES resources/app.rc)
    set_source_files_properties(resources/app.rc PROPERTIES
        COMPILE_FLAGS "--include-dir \"${CMAKE_SOURCE_DIR}/assets\"")
elseif(APPLE)
    # Bundle the .icns and declare it as the app icon.
    set(MACOSX_BUNDLE_ICON_FILE icon.icns)
    set_source_files_properties("${CMAKE_SOURCE_DIR}/assets/icon.icns" PROPERTIES
        MACOSX_PACKAGE_LOCATION Resources)
    list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/assets/icon.icns")
endif()

if(APPLE)
    add_executable(image_generator MACOSX_BUNDLE ${SOURCES})
    set_target_properties(image_generator PROPERTIES
        MACOSX_BUNDLE_BUNDLE_NAME      "ImageGenerator"
        MACOSX_BUNDLE_GUI_IDENTIFIER   "com.imageGenerator.app"
        MACOSX_BUNDLE_ICON_FILE        "icon.icns")
else()
    add_executable(image_generator ${SOURCES})
endif()

if(USE_CUDA)
    target_compile_definitions(image_generator PRIVATE USE_CUDA)
    message(STATUS "CUDA GPU acceleration enabled")
endif()
if(USE_DML)
    target_compile_definitions(image_generator PRIVATE USE_DML)
    message(STATUS "DirectML GPU acceleration enabled")
endif()

if(USE_GENAI)
    if(NOT DEFINED ORT_GENAI_ROOT)
        if(WIN32)
            set(ORT_GENAI_ROOT "C:/ort-genai" CACHE PATH "ORT GenAI root directory")
        else()
            set(ORT_GENAI_ROOT "/home/guillaume/libs/ort-genai" CACHE PATH "ORT GenAI root directory")
        endif()
    endif()
    # On Windows use the import library directly (same pattern as ONNXRUNTIME_LIB).
    if(WIN32)
        set(ORT_GENAI_LIB "${ORT_GENAI_ROOT}/lib/onnxruntime-genai.lib")
        if(NOT EXISTS "${ORT_GENAI_LIB}")
            message(FATAL_ERROR "ORT GenAI import library not found: ${ORT_GENAI_LIB}")
        endif()
    else()
        find_library(ORT_GENAI_LIB onnxruntime-genai PATHS ${ORT_GENAI_ROOT}/lib REQUIRED)
    endif()
    target_compile_definitions(image_generator PRIVATE USE_GENAI)
    target_include_directories(image_generator PRIVATE ${ORT_GENAI_ROOT}/include)
    target_link_libraries(image_generator ${ORT_GENAI_LIB})
    # Copy the genai DLL next to the binary on Windows.
    if(WIN32)
        file(GLOB _genai_dlls "${ORT_GENAI_ROOT}/lib/*.dll")
        foreach(_dll ${_genai_dlls})
            add_custom_command(TARGET image_generator POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:image_generator>")
        endforeach()
    endif()
    message(STATUS "ORT GenAI prompt enhancement enabled (${ORT_GENAI_ROOT})")
endif()

if(UNIX AND NOT APPLE)
    target_compile_options(image_generator PRIVATE -g -fno-omit-frame-pointer -fno-pie)
    target_link_options(image_generator PRIVATE -rdynamic -no-pie)
elseif(APPLE)
    target_compile_options(image_generator PRIVATE -g -fno-omit-frame-pointer)
else()
    # Statically link GCC/C++ runtimes so libgcc_s_seh-1.dll / libstdc++-6.dll
    # don't need to be shipped alongside the executable.
    target_link_options(image_generator PRIVATE -static-libgcc -static-libstdc++)

    # ONNX Runtime Windows headers use MSVC SAL annotations not available in MinGW
    target_compile_definitions(image_generator PRIVATE
        _Frees_ptr_opt_=
        _In_=
        _Out_=
        _Inout_=
        _In_opt_=
        _Out_opt_=
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(image_generator PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_include_directories(image_generator PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image_generator
    sfml-graphics
    sfml-window
    sfml-system
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
)

if(WIN32)
    # shell32: SHBrowseForFolder / SHGetPathFromIDList
    # ole32:   CoInitialize / CoUninitialize / CoTaskMemFree
    target_link_libraries(image_generator shell32 ole32)
endif()

# Copy default config.json next to the binary only when none exists there yet,
# so user-edited configs are never overwritten by a rebuild.
add_custom_command(TARGET image_generator POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DSRC="${CMAKE_SOURCE_DIR}/config.json"
        -DDST="$<TARGET_FILE_DIR:image_generator>/config.json"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_default_config.cmake")

# Copy portrait models (ONNX + tokenizer files) next to the binary
add_custom_command(TARGET image_generator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:image_generator>/models")

# Create generated portraits directory next to the binary and drop a placeholder
# so the empty directory survives zip packaging (zip tools drop empty dirs).
add_custom_command(TARGET image_generator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:image_generator>/assets/generated"
    COMMAND ${CMAKE_COMMAND} -E touch
        "$<TARGET_FILE_DIR:image_generator>/assets/generated/.gitkeep")

# On Windows copy all required DLLs next to the binary
if(WIN32)
    # Ask the cross-compiler where each runtime DLL lives.
    # libgcc_s_seh-1.dll and libwinpthread-1.dll can be in different directories
    # (e.g. gcc/x86_64-w64-mingw32/<ver>/ vs x86_64-w64-mingw32/bin/).
    foreach(_rt_dll libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_rt_dll}
            OUTPUT_VARIABLE _RT_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(IS_ABSOLUTE "${_RT_PATH}")
            list(APPEND _MINGW_RUNTIME_DLLS "${_RT_PATH}")
        endif()
    endforeach()

    set(_WIN_DLL_DIRS
        "${ONNXRUNTIME_ROOT}/lib"
        "${SFML_DIR}/../../../bin"
        "${OpenCV_DIR}/../../../bin"  # OpenCV_DIR is lib/cmake/opencv4; go up 3 levels to install root
    )
    # Copy MinGW runtime DLLs individually (they may come from different dirs)
    foreach(_dll ${_MINGW_RUNTIME_DLLS})
        add_custom_command(TARGET image_generator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:image_generator>")
    endforeach()
    foreach(_dir ${_WIN_DLL_DIRS})
        file(GLOB _dlls "${_dir}/*.dll")
        foreach(_dll ${_dlls})
            add_custom_command(TARGET image_generator POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:image_generator>")
        endforeach()
    endforeach()
endif()

# Copy a bundled font next to the binary so "arial.ttf" resolves on all platforms.
# On Linux this is a fallback; on Windows it avoids depending on system fonts entirely.
if(WIN32)
    # Use a font bundled in the repo; copy it as arial.ttf so Screen.cpp finds it.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/fonts/arial.ttf")
        add_custom_command(TARGET image_generator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/assets/fonts/arial.ttf"
            "$<TARGET_FILE_DIR:image_generator>/arial.ttf")
    endif()
endif()

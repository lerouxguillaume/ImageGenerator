cmake_minimum_required(VERSION 3.14)
project(GuildMaster CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA GPU acceleration via ONNX Runtime" OFF)
option(USE_DML  "Enable DirectML GPU acceleration via ONNX Runtime (Windows)" OFF)

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

add_executable(guild_master ${SOURCES})

if(USE_CUDA)
    target_compile_definitions(guild_master PRIVATE USE_CUDA)
    message(STATUS "CUDA GPU acceleration enabled")
endif()
if(USE_DML)
    target_compile_definitions(guild_master PRIVATE USE_DML)
    message(STATUS "DirectML GPU acceleration enabled")
endif()

if(UNIX AND NOT APPLE)
    target_compile_options(guild_master PRIVATE -g -fno-omit-frame-pointer -fno-pie)
    target_link_options(guild_master PRIVATE -rdynamic -no-pie)
elseif(APPLE)
    target_compile_options(guild_master PRIVATE -g -fno-omit-frame-pointer)
else()
    # Statically link GCC/C++ runtimes so libgcc_s_seh-1.dll / libstdc++-6.dll
    # don't need to be shipped alongside the executable.
    target_link_options(guild_master PRIVATE -static-libgcc -static-libstdc++)

    # ONNX Runtime Windows headers use MSVC SAL annotations not available in MinGW
    target_compile_definitions(guild_master PRIVATE
        _Frees_ptr_opt_=
        _In_=
        _Out_=
        _Inout_=
        _In_opt_=
        _Out_opt_=
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(guild_master PRIVATE -Wall -Wextra -Wpedantic)
endif()

# On Windows the SQLite3 amalgamation is in the custom deps dir;
# on Linux it is provided by the system package.
if(WIN32)
    if(NOT DEFINED SQLITE3_ROOT)
        message(FATAL_ERROR "Set SQLITE3_ROOT to the SQLite3 amalgamation directory")
    endif()
    set(SQLITE3_INCLUDE_DIR "${SQLITE3_ROOT}")
    set(SQLITE3_LIB         "${SQLITE3_ROOT}/libsqlite3.a")
else()
    set(SQLITE3_INCLUDE_DIR "")   # system header, no extra path needed
    set(SQLITE3_LIB         sqlite3)
endif()

target_include_directories(guild_master PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${SQLITE3_INCLUDE_DIR}
)

target_link_libraries(guild_master
    sfml-graphics
    sfml-window
    sfml-system
    nlohmann_json::nlohmann_json
    ${SQLITE3_LIB}
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
)

# Copy config.json next to the binary so the game can find it at runtime
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/config.json"
    "$<TARGET_FILE_DIR:guild_master>/config.json")

# Copy schema.sql next to the binary so Connection::init() can find it
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/db/schema.sql"
    "$<TARGET_FILE_DIR:guild_master>/schema.sql")

# Copy lang/ directory next to the binary
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/lang"
    "$<TARGET_FILE_DIR:guild_master>/lang")

# Copy portrait models (ONNX + tokenizer files) next to the binary
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:guild_master>/models")

# Create saves directory next to the binary
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "$<TARGET_FILE_DIR:guild_master>/saves")

# Create generated portraits directory next to the binary and drop a placeholder
# so the empty directory survives zip packaging (zip tools drop empty dirs).
add_custom_command(TARGET guild_master POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:guild_master>/assets/generated"
    COMMAND ${CMAKE_COMMAND} -E touch
        "$<TARGET_FILE_DIR:guild_master>/assets/generated/.gitkeep")

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
        add_custom_command(TARGET guild_master POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:guild_master>")
    endforeach()
    foreach(_dir ${_WIN_DLL_DIRS})
        file(GLOB _dlls "${_dir}/*.dll")
        foreach(_dll ${_dlls})
            add_custom_command(TARGET guild_master POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:guild_master>")
        endforeach()
    endforeach()
endif()

# Copy a bundled font next to the binary so "arial.ttf" resolves on all platforms.
# On Linux this is a fallback; on Windows it avoids depending on system fonts entirely.
if(WIN32)
    # Use a font bundled in the repo; copy it as arial.ttf so Screen.cpp finds it.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/fonts/arial.ttf")
        add_custom_command(TARGET guild_master POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/assets/fonts/arial.ttf"
            "$<TARGET_FILE_DIR:guild_master>/arial.ttf")
    endif()
endif()

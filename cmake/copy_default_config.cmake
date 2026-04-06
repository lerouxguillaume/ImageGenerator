# Copies the default config.json to the binary directory only when no config
# exists there yet, so user-edited configs are never overwritten by a rebuild.
if(NOT EXISTS "${DST}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${SRC}" "${DST}")
    message(STATUS "Installed default config.json to ${DST}")
endif()

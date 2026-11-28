aux_source_directory(. SRCS)
message(STATUS "onnx_plugin SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    message(WARNING "onnx_plugin: no source files found, skip")
    return()
endif()

set(TARGET_NAME ${ONNX_PLUGIN_TARGET})
set(LIBRARY_OUTPUT_PATH ${ONNX_PLUGIN_TARGET_OUT_DIR})

add_library(${TARGET_NAME} SHARED ${SRCS})

target_compile_definitions(${TARGET_NAME} PRIVATE
    google=ascend_private
)

find_package(nlohmann_json QUIET)
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)
else()
    message(STATUS "nlohmann_json target not found; relying on system include path for <nlohmann/json.hpp>")
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_INC}/../../opp
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${ASCEND_INC}/../lib64/libgraph.so
)

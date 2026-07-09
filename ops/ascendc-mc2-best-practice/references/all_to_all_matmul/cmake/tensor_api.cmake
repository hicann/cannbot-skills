
if(TARGET cann_samples_tensor_api)
    return()
endif()

find_package(Git QUIET)

set(TENSOR_API_PATH "${PROJECT_SOURCE_DIR}/third_party/tensor_api")

if(NOT EXISTS "${TENSOR_API_PATH}/CMakeLists.txt" AND NOT GIT_FOUND)
    message(FATAL_ERROR
        "Git is required to initialize third_party/tensor_api automatically."
    )
endif()

# 剥离工程适配：third_party/tensor_api 已 symlink 到源工程（CMakeLists.txt 必然存在），
# 只有在 CMakeLists.txt 不存在时才需要 git submodule update（本工程永远不会触发）
set(TENSOR_API_PREPARE_COMMANDS)
if(GIT_FOUND AND NOT EXISTS "${TENSOR_API_PATH}/CMakeLists.txt")
    list(APPEND TENSOR_API_PREPARE_COMMANDS
        COMMAND ${GIT_EXECUTABLE} submodule update --init --no-fetch third_party/tensor_api
    )
endif()

add_custom_target(cann_samples_tensor_api_dependencies
    ${TENSOR_API_PREPARE_COMMANDS}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Initializing third_party/tensor_api"
    VERBATIM
)

add_library(cann_samples_tensor_api INTERFACE)
add_library(cann_samples::tensor_api ALIAS cann_samples_tensor_api)
add_dependencies(cann_samples_tensor_api cann_samples_tensor_api_dependencies)

target_include_directories(cann_samples_tensor_api INTERFACE
    "${TENSOR_API_PATH}/include/tensor_api"
    "${ASCEND_DIR}/asc"
)

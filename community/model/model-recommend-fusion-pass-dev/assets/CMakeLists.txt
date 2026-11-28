cmake_minimum_required(VERSION 3.16)
project(sample_pass)

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_PATH)
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/include/graph
    ${ASCEND_PATH}/include/ge
    ${ASCEND_PATH}/include/register
    ${ASCEND_PATH}/include/platform
    ${ASCEND_PATH}/opp/built-in/op_graph/inc
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
)

add_library(sample_pass SHARED src/sample_pass.cpp)

target_link_directories(sample_pass PRIVATE ${ASCEND_PATH}/lib64)
target_link_libraries(sample_pass PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)

target_compile_options(sample_pass PRIVATE
    -fPIC
    -Wall
    -D_GLIBCXX_USE_CXX11_ABI=0
)

cmake_minimum_required(VERSION 3.16.0)
project(sample_op_prj)
find_package(ASC REQUIRED)

set(ARCH22_COMPUTE_UNITS ascend910b ascend910_93)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH22_COMPUTE_UNITS})
endif()

# package_name 即 vendor 名，安装后位于 opp/vendors/<package_name>
set(package_name sample_op_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/framework")
    add_subdirectory(framework)
endif()

# kernel 二进制不会在默认 all target 里编出，必须用 build.sh --soc=<soc>
message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")

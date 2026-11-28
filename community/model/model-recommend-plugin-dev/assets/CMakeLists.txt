cmake_minimum_required(VERSION 3.5.1)
project(cust_framework_plugins LANGUAGES CXX)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# CANN SDK include 路径：优先 ASCEND_TENSOR_COMPILER_INCLUDE，其次 ASCEND_HOME_PATH
if(NOT DEFINED ASCEND_INC)
    if(DEFINED ENV{ASCEND_TENSOR_COMPILER_INCLUDE})
        set(ASCEND_INC $ENV{ASCEND_TENSOR_COMPILER_INCLUDE})
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_INC $ENV{ASCEND_HOME_PATH}/include)
    else()
        message(FATAL_ERROR "set ASCEND_TENSOR_COMPILER_INCLUDE or pass -DASCEND_INC=...")
    endif()
endif()
message(STATUS "ASCEND_INC=${ASCEND_INC}")
include_directories(${ASCEND_INC})

# <vendor>: 团队统一的 vendor 名，与自定义算子 package_name 保持一致，避免与已装 vendor 重名
set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/vendors/<vendor>)
set(ONNX_PLUGIN_TARGET          cust_onnx_parsers)
set(ONNX_PLUGIN_TARGET_OUT_DIR  ${OUT_DIR}/framework/onnx)

add_subdirectory(onnx_plugin)

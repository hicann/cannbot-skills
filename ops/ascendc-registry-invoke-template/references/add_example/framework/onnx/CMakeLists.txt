# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

file(GLOB ONNX_PLUGIN_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*_onnx_plugin.cpp")

if(NOT ONNX_PLUGIN_SRCS)
    message(WARNING "framework/onnx: no *_onnx_plugin.cpp found, skip")
    return()
endif()

# Host architecture directory under ${ASCEND_HOME_PATH}, e.g. aarch64-linux.
# Accept ASCEND_HOST_LINUX when it is already provided (with or without the -linux suffix).
if(DEFINED ASCEND_HOST_LINUX AND NOT "${ASCEND_HOST_LINUX}" STREQUAL "")
    set(ASCEND_HOST_ARCH "${ASCEND_HOST_LINUX}")
else()
    set(ASCEND_HOST_ARCH "${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
if(NOT "${ASCEND_HOST_ARCH}" MATCHES "-linux$")
    set(ASCEND_HOST_ARCH "${ASCEND_HOST_ARCH}-linux")
endif()

set(ONNX_PLUGIN_NAME oponnx_plugin_add_example)

add_library(${ONNX_PLUGIN_NAME} SHARED ${ONNX_PLUGIN_SRCS})

target_compile_options(${ONNX_PLUGIN_NAME} PRIVATE
    -fPIC
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Dgoogle=ascend_private
    -fvisibility=hidden
)

target_include_directories(${ONNX_PLUGIN_NAME} PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}/include
)

target_link_directories(${ONNX_PLUGIN_NAME} PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}/lib64
)

target_link_libraries(${ONNX_PLUGIN_NAME} PRIVATE
    -Wl,--no-as-needed
    register
    -Wl,--as-needed
    -Wl,--whole-archive
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}/lib64/librt2_registry.a
    -Wl,--no-whole-archive
    -Wl,-Bsymbolic
    unified_dlog
    c_sec
)

install(TARGETS ${ONNX_PLUGIN_NAME}
    LIBRARY DESTINATION packages/vendors/${package_name}/framework/onnx
)

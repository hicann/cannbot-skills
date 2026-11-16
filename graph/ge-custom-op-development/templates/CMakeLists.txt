# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# 纯 CXX 交付件骨架；只选一个最终实现，原型与实现统一 op type。
cmake_minimum_required(VERSION 3.16)
project(my_custom_op LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}/include")
    message(FATAL_ERROR "ASCEND_HOME_PATH must point to the target CANN installation")
endif()
set(CUSTOM_OP_IMPLEMENTATION "src/my_eager_execute_op.cpp" CACHE FILEPATH "Selected implementation")
foreach(required_file "${CUSTOM_OP_IMPLEMENTATION}" "src/proto_reg.cpp" "src/proto_reg.h")
    get_filename_component(required_path "${required_file}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Missing ${required_file}; copy and adapt the selected templates into src/")
    endif()
endforeach()
add_library(cust_opapi SHARED "${CUSTOM_OP_IMPLEMENTATION}" src/proto_reg.cpp)
target_include_directories(cust_opapi PRIVATE
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/external"
    "${ASCEND_HOME_PATH}/include/graph_metadef/external"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_directories(cust_opapi PRIVATE "${ASCEND_HOME_PATH}/lib64")
# ABI 和链接集合按目标安装包核对；这里沿用官方 Eager 样例的 Linux 基线。
target_compile_definitions(cust_opapi PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
# custom_op_registry_static 是静态库 libcustom_op_registry_static.a（无同名 .so），提供离线 OM
# 加载必需的 pull creator 符号（GetRegisteredCustomOpCreator*）；删除后 OM 加载整体失败。
target_link_libraries(cust_opapi PRIVATE
    -Wl,--no-as-needed ascendcl acl_rt ge_common ge_compiler ge_executor runtime
    register gert custom_op_registry_static lowering dl -Wl,--as-needed)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" CUSTOM_OP_OS_DIR)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CUSTOM_OP_ARCH_DIR)
if(CUSTOM_OP_ARCH_DIR STREQUAL "arm64")
    set(CUSTOM_OP_ARCH_DIR "aarch64")
elseif(CUSTOM_OP_ARCH_DIR STREQUAL "amd64")
    set(CUSTOM_OP_ARCH_DIR "x86_64")
endif()
set(OPP_OUTPUT_PATH "${CMAKE_BINARY_DIR}/opp")
set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME "cust_opapi"
    LIBRARY_OUTPUT_DIRECTORY "${OPP_OUTPUT_PATH}/op_graph/lib/${CUSTOM_OP_OS_DIR}/${CUSTOM_OP_ARCH_DIR}")
add_custom_command(TARGET cust_opapi POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OPP_OUTPUT_PATH}/op_graph/include"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/proto_reg.h"
            "${OPP_OUTPUT_PATH}/op_graph/include/proto_reg.h")

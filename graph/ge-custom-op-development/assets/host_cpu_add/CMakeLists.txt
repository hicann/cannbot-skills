# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(host_cpu_add_custom_constant_folding_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ES_OUTPUT_PATH "${CMAKE_BINARY_DIR}/es_output")
set(OPP_OUTPUT_PATH "${CMAKE_BINARY_DIR}/opp")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first.")
endif()

list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0 google=ascend_private)

add_library(add_custom_op_proto SHARED proto/add_custom.cc)
target_compile_definitions(add_custom_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(add_custom_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(add_custom_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
)
target_link_libraries(add_custom_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/lib64/libopp_registry.so"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OPP_OS_TYPE "windows")
else()
  set(OPP_OS_TYPE "linux")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PROCESSOR_LOWER)
if(PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
  set(OPP_CPU_TYPE "aarch64")
elseif(PROCESSOR_LOWER MATCHES "^(x86_64|amd64)$")
  set(OPP_CPU_TYPE "x86_64")
else()
  set(OPP_CPU_TYPE "${PROCESSOR_LOWER}")
endif()
set(OPP_PROTO_OUTPUT_PATH "${OPP_OUTPUT_PATH}/op_proto/custom")
set(CUSTOM_OP_OUTPUT_PATH "${OPP_OUTPUT_PATH}/op_graph/lib/${OPP_OS_TYPE}/${OPP_CPU_TYPE}")
set_target_properties(add_custom_op_proto PROPERTIES
  OUTPUT_NAME cust_opapi
  LIBRARY_OUTPUT_DIRECTORY "${OPP_PROTO_OUTPUT_PATH}"
  RUNTIME_OUTPUT_DIRECTORY "${OPP_PROTO_OUTPUT_PATH}"
  ARCHIVE_OUTPUT_DIRECTORY "${OPP_PROTO_OUTPUT_PATH}"
)
add_custom_command(TARGET add_custom_op_proto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSTOM_OP_OUTPUT_PATH}"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:add_custom_op_proto>" "${CUSTOM_OP_OUTPUT_PATH}"
)

add_es_library_and_whl(
  ES_LINKABLE_AND_ALL_TARGET es_custom
  OPP_PROTO_TARGET add_custom_op_proto
  OUTPUT_PATH ${ES_OUTPUT_PATH}
)

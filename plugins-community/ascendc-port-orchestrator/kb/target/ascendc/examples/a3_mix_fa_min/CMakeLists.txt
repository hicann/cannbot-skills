# Self-contained standalone build for the a3 MIX FA-core minimal worked-example.
# Uses the CANN ascendc_kernel_cmake helper directly (no op-gen harness needed).
#   Invoke with (from a CANN 9.0.0 arch22 / Ascend910_9382 environment):
#     cmake -B build -DSOC_VERSION=Ascend910_9382 \
#           -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann
#     cmake --build build -j
# See README.md for the full device recipe.
cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mix_fa_min_kernel.cpp
)

ascendc_include_directories(kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

add_library(pybind11_lib SHARED "${CMAKE_CURRENT_SOURCE_DIR}/mix_fa_min_pybind.cpp")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")
set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")
target_link_directories(pybind11_lib PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(pybind11_lib PRIVATE
  ${PYBIND11_INC}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME _a3_mix_fa_min_example
  PREFIX ""
  SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)

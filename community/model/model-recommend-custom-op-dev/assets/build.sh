#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

SUPPORT_COMPUTE_UNITS=("ascend910b" "ascend910_93")

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if [ ${CORE_NUMS} -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Usage: bash build.sh [OPTIONS]"
  echo "  --soc=soc_version   required: ascend910b | ascend910_93"
  echo "  --make_clean        clean build artifacts"
  echo "  -j[n]               compile threads (default ${CORE_NUMS})"
  echo "  --no-opapi          exclude ACLNN opapi library"
  echo "  -h, --help          print help"
}

check_compute_unit() {
  local unit="$1"
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "$unit" == "$support_unit" ]]; then
      return 0
    fi
  done
  return 1
}

THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
ENABLE_OPAPI=TRUE

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -j*) THREAD_NUM="${1:2}"; [ -z "$THREAD_NUM" ] && THREAD_NUM=${CORE_NUMS}; shift ;;
    --no-opapi) ENABLE_OPAPI=FALSE; shift ;;
    --make_clean) rm -rf ${BUILD_PATH}; exit 0 ;;
    --soc=*) COMPUTE_UNIT="${1#*=}"; shift ;;
    *) echo "[ERROR] Invalid option: $1"; usage; exit 1 ;;
  esac
done

if [ -z "$COMPUTE_UNIT" ]; then
  echo "[ERROR] --soc parameter is required"
  usage
  exit 1
fi

COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | tr '[:upper:]' '[:lower:]')
if ! check_compute_unit "$COMPUTE_UNIT"; then
  echo "[ERROR] Invalid SoC: $COMPUTE_UNIT (supported: ${SUPPORT_COMPUTE_UNITS[@]})"
  exit 1
fi
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"

CMAKE_ARGS="-DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT"
if [ "$ENABLE_OPAPI" = "FALSE" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_OPAPI=OFF"
fi

mkdir -p "${BUILD_PATH}"
[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt

echo "[INFO] Configuring project..."
cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

echo "[INFO] Building with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j ${THREAD_NUM}

KERNEL_O=$(find ${BUILD_PATH}/op_kernel/ascendc_kernels/binary -name "*.o" 2>/dev/null | head -1)
if [ -z "$KERNEL_O" ]; then
  echo "[ERROR] Kernel binary not found"
  exit 1
fi

PKG_PATH=$(ls ${BUILD_PATH}/custom_opp_*.run 2>/dev/null | head -1)
if [ -z "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
  echo "[ERROR] Package not found or empty"
  exit 1
fi

echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project_path=$(cd "$(dirname "$0")"; pwd)

if [ "x$1" == "xclean" ] 2>/dev/null; then
  rm -rf $project_path/build_out
  echo "[INFO] Clean successfully."
  exit 0
fi

log() {
  cur_date=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$cur_date] "$1
}

if [[ -n "${ASCEND_TENSOR_COMPILER_INCLUDE}" ]]; then
    if [[ ! -d "${ASCEND_TENSOR_COMPILER_INCLUDE}" ]]; then
        log "[ERROR] ENV ASCEND_TENSOR_COMPILER_INCLUDE=${ASCEND_TENSOR_COMPILER_INCLUDE} dir is not exist"
        exit 1
    fi
else
    find_ascend_include() {
        local cand
        for cand in \
            "${ASCEND_HOME_PATH:+${ASCEND_HOME_PATH}/include}" \
            "/usr/local/Ascend/ascend-toolkit/latest/include" \
            "/usr/local/Ascend/latest/include" \
            "${HOME}/Ascend/ascend-toolkit/latest/include" \
            "${HOME}/Ascend/latest/include"; do
            if [[ -n "${cand}" && -d "${cand}" ]]; then
                printf "%s" "${cand}"
                return 0
            fi
        done
        return 1
    }
    if ! ascend_inc=$(find_ascend_include); then
        log "[ERROR] Cannot find CANN include path. Set ASCEND_TENSOR_COMPILER_INCLUDE or ASCEND_HOME_PATH."
        exit 1
    fi
    export ASCEND_TENSOR_COMPILER_INCLUDE="${ascend_inc}"
    log "[INFO] Auto-detect ASCEND_TENSOR_COMPILER_INCLUDE=${ASCEND_TENSOR_COMPILER_INCLUDE}"
fi

log "[INFO] Cmake begin."
rm -rf $project_path/build_out
mkdir -p $project_path/build_out
cd $project_path/build_out
cmake ..
if [ $? -ne 0 ]; then
  log "[ERROR] Please check cmake result."
  exit 1
fi

log "[INFO] Make begin."
make -j
if [ $? -ne 0 ]; then
  log "[ERROR] Please check make result."
  exit 1
fi

cat > "$project_path/build_out/makepkg/set_env.bash" << 'EOF'
#!/bin/bash
vendor_path=$(cd "$(dirname "${BASH_SOURCE[0]}")/packages/vendors/<vendor>"; pwd)
export ASCEND_CUSTOM_OPP_PATH=${vendor_path}:${ASCEND_CUSTOM_OPP_PATH}
EOF
log "[INFO] Generated set_env.bash for deployment."

log "[INFO] Build successfully, output in $project_path/build_out/makepkg/."

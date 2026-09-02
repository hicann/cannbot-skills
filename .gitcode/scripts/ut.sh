#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e

REPOSITORY_NAME="cannbot-skills"
sudo update-alternatives --set gcc /usr/bin/gcc-14
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
gcc --version
rm -rf /home/jenkins/opensource/json

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi


if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi

LOG_HEAD()
{
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

LOG_DO()
{
    echo "[LOG_DO] $*"
    "$@"
}

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

cd "${WORKSPACE}/tests" || exit 1
source ~/.bashrc
echo $PATH
export PATH=/root/.nvm/versions/node/v24.14.0/bin:$PATH
LOG_DO bash run-tests.sh
BUILD_EXIT_CODE=$?
DP_ASSERT_EQUAL "$BUILD_EXIT_CODE" "0" "Run UT testcase" "true"

echo "ut_process=coverage" >> "${ATOMGIT_OUTPUT}"

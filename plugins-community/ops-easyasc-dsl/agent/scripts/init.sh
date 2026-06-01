#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Restore the archived runtime/docs payload and example payload into the repository.
# Safe to run multiple times; only missing trees are restored.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ARCHIVE="${REPO_ROOT}/agent/assets/ops-easyasc-dsl-runtime.tar.gz"
EXAMPLE_ARCHIVE="${REPO_ROOT}/agent/assets/ops-easyasc-dsl-example.tar.gz"

restore_runtime() {
    if [ ! -f "${RUNTIME_ARCHIVE}" ]; then
        echo "ERROR: runtime archive not found at ${RUNTIME_ARCHIVE}" >&2
        return 1
    fi

    local need_extract=0
    for tree in easyasc doc doc_cn; do
        if [ ! -d "${REPO_ROOT}/${tree}" ]; then
            need_extract=1
            break
        fi
    done

    if [ "${need_extract}" -eq 0 ]; then
        echo "[init] runtime trees already present (easyasc/, doc/, doc_cn/) — skipping"
        return 0
    fi

    echo "[init] restoring runtime/docs trees from ${RUNTIME_ARCHIVE}"
    tar -xzf "${RUNTIME_ARCHIVE}" -C "${REPO_ROOT}"
}

restore_examples() {
    if [ ! -f "${EXAMPLE_ARCHIVE}" ]; then
        echo "ERROR: example archive not found at ${EXAMPLE_ARCHIVE}" >&2
        return 1
    fi

    if [ -d "${REPO_ROOT}/agent/example" ]; then
        echo "[init] agent/example/ already present — skipping"
        return 0
    fi

    echo "[init] restoring agent/example/ from ${EXAMPLE_ARCHIVE}"
    tar -xzf "${EXAMPLE_ARCHIVE}" -C "${REPO_ROOT}"
}

restore_runtime
restore_examples

echo "[init] done"

#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# post_migration_check.sh <operator_dir> <soc> [target_repo] [source_optype]
# Exit 0 = no FAIL; 1 = failed checks; 2 = invalid input.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ] || [ -z "$2" ]; then
    echo "usage: $0 <operator_dir> <soc> [target_repo] [source_optype]" >&2
    exit 2
fi
args=(check "$1" --soc "$2")
if [ -n "${3:-}" ]; then
    args+=(--repo "$3")
fi
if [ -n "${4:-}" ]; then
    args+=(--op-type "$4")
fi
python3 "$SCRIPT_DIR/migration_contract.py" "${args[@]}"

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

# Wrapper script to run evaluation with proper environment setup

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <op_name> [--work-dir <dir>] [other options...]"
    exit 1
fi

OP_NAME="$1"
shift

# Parse work-dir argument
WORK_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Default work directory if not specified
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="output/$OP_NAME"
fi

# Setup environment variables BEFORE running Python
CUSTOM_OPP_PATH="$WORK_DIR/vendors/customize"
if [ ! -d "$CUSTOM_OPP_PATH" ]; then
    echo "Error: vendors/customize not found in $WORK_DIR"
    exit 1
fi

export ASCEND_CUSTOM_OPP_PATH="$(realpath $CUSTOM_OPP_PATH)"
LIB_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib"
if [ -d "$LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"
fi

echo "Environment setup:"
echo "  ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"

# Run Python evaluation script
SCRIPT_DIR="$(dirname "$0")"
python3 "$SCRIPT_DIR/evaluate.py" "$OP_NAME" --work-dir "$WORK_DIR" "${ARGS[@]}"

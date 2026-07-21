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
# Unified build & evaluate script for AscendC custom operators.
#
# Usage:
#   ./eval_op.sh <op_name>                              # base version
#   ./eval_op.sh <op_name> <evo_date> <round> <parallel> # evo variant
#
# Examples:
#   ./eval_op.sh causal_conv1d_fn
#   ./eval_op.sh causal_conv1d_fn 20260211 1 0
#
# The script runs 4 phases:
#   1. Compile   — build.sh in the AscendC project
#   2. Install   — install the .run package into the base output dir
#   3. PyBind    — generate and install the pybind wheel
#   4. Evaluate  — run correctness + performance evaluation
#
# To add skip flags later, wrap each phase in an if-block, e.g.:
#   if [ "$SKIP_BUILD" != "1" ]; then ... fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Directory containing evaluation skills (generate_pybind, evaluate, etc.)
EVAL_SKILL_DIR="${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Convert underscore_name to PascalCase (e.g. causal_conv1d_fn_custom -> CausalConv1dFnCustom)
to_pascal_case() {
    echo "$1" | sed -r 's/(^|_)([a-z0-9])/\U\2/g'
}

usage() {
    echo "Usage:"
    echo "  $0 <op_name>                                # base version"
    echo "  $0 <op_name> <evo_date> <round> <parallel>  # evo variant"
    echo ""
    echo "Examples:"
    echo "  $0 causal_conv1d_fn"
    echo "  $0 causal_conv1d_fn 20260211 1 0"
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    usage
fi

OP_NAME="$1"
EVO_MODE=0

if [ $# -eq 4 ]; then
    EVO_DATE="$2"
    ROUND="$3"
    PARALLEL="$4"
    EVO_MODE=1
elif [ $# -ne 1 ]; then
    echo "Error: expected 1 or 4 arguments, got $#"
    usage
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

# PascalCase project directory name: <OpName>Custom -> e.g. CausalConv1dFnCustom
PASCAL_NAME="$(to_pascal_case "${OP_NAME}_custom")"

# Base install directory (evaluate.py always resolves runtime paths here)
INSTALL_DIR="$(realpath output/${OP_NAME})"

if [ "$EVO_MODE" -eq 1 ]; then
    VARIANT_DIR="output/${OP_NAME}_evo_${EVO_DATE}/round_${ROUND}/parallel_${PARALLEL}"
    ASCENDC_DIR="${VARIANT_DIR}/${PASCAL_NAME}"
    RESULT_DIR="$(realpath "$VARIANT_DIR")"
    echo "=== Evo variant: ${VARIANT_DIR} ==="
else
    ASCENDC_DIR="output/${OP_NAME}/${PASCAL_NAME}"
    RESULT_DIR=""
    echo "=== Base version: output/${OP_NAME} ==="
fi

echo "  AscendC project : ${ASCENDC_DIR}"
echo "  Install target  : ${INSTALL_DIR}"
echo ""

# Validate
if [ ! -d "$ASCENDC_DIR" ]; then
    echo "Error: AscendC project directory not found: ${ASCENDC_DIR}"
    exit 1
fi

if [ ! -f "${ASCENDC_DIR}/build.sh" ]; then
    echo "Error: build.sh not found in ${ASCENDC_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Compile
# ---------------------------------------------------------------------------
echo "==== Phase 1/4: Compile ===="
(cd "$ASCENDC_DIR" && bash build.sh)
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Install .run package
# ---------------------------------------------------------------------------
echo "==== Phase 2/4: Install .run ===="

RUN_FILE="${ASCENDC_DIR}/build_out/custom_opp_ubuntu_aarch64.run"
if [ ! -f "$RUN_FILE" ]; then
    echo "Error: .run file not found at ${RUN_FILE}"
    exit 1
fi

bash "$RUN_FILE" --install-path="$INSTALL_DIR"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Generate PyBind
# ---------------------------------------------------------------------------
echo "==== Phase 3/4: Generate PyBind ===="
python3 "${EVAL_SKILL_DIR}/generate_pybind.py" "$OP_NAME"
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Evaluate
# ---------------------------------------------------------------------------
echo "==== Phase 4/4: Evaluate ===="

# Locate test_cases.py: prefer variant's own, fall back to base's
if [ "$EVO_MODE" -eq 1 ] && [ -f "${VARIANT_DIR}/test_cases.py" ]; then
    TEST_CASES_FILE="$(realpath "${VARIANT_DIR}/test_cases.py")"
elif [ -f "output/${OP_NAME}/test_cases.py" ]; then
    TEST_CASES_FILE="$(realpath "output/${OP_NAME}/test_cases.py")"
else
    TEST_CASES_FILE=""
fi

# Build evaluate command
EVAL_CMD=(python3 "${EVAL_SKILL_DIR}/evaluate.py" "$OP_NAME")

if [ -n "$TEST_CASES_FILE" ]; then
    EVAL_CMD+=(--test-cases "$TEST_CASES_FILE")
    echo "  test_cases: ${TEST_CASES_FILE}"
else
    echo "  test_cases: (none, single-test mode)"
fi

if [ "$EVO_MODE" -eq 1 ]; then
    OUTPUT_FILE="${RESULT_DIR}/evaluation_result.txt"
    EVAL_CMD+=(-o "$OUTPUT_FILE")
    echo "  output: ${OUTPUT_FILE}"
fi

echo ""
"${EVAL_CMD[@]}"

echo ""
echo "==== Done ===="

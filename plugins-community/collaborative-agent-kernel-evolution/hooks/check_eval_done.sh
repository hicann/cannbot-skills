#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Stop hook: block Claude from exiting if evaluation hasn't passed yet.
#
# Relies on CAKE_OUTPUT_DIR being set by the caller (e.g. an external batch runner).
# If not set (e.g. manual runs), always allow stop.

# No output dir configured — not a batch run, allow stop
if [ -z "$CAKE_OUTPUT_DIR" ]; then
    echo '{"decision":"approve"}'
    exit 0
fi

# Look for evaluation result files
RESULT_FILE=""
for name in evaluation_results.json results.json eval_result.json result.json; do
    if [ -f "$CAKE_OUTPUT_DIR/$name" ]; then
        RESULT_FILE="$CAKE_OUTPUT_DIR/$name"
        break
    fi
done

# No result file yet — evaluation hasn't run
if [ -z "$RESULT_FILE" ]; then
    echo '{"decision":"block","reason":"🚫 评测结果文件尚未生成，请先完成代码生成并运行评测。"}'
    exit 0
fi

# Parse passed/total from result file
PASSED=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('passed_cases',0))" "$RESULT_FILE" 2>/dev/null)
TOTAL=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total_cases',0))" "$RESULT_FILE" 2>/dev/null)

# Parse failure — allow stop to avoid deadlock
if [ -z "$PASSED" ] || [ -z "$TOTAL" ] || [ "$TOTAL" = "0" ]; then
    echo '{"decision":"approve"}'
    exit 0
fi

if [ "$PASSED" = "$TOTAL" ]; then
    echo '{"decision":"approve"}'
else
    echo "{\"decision\":\"block\",\"reason\":\"🚫 评测未全部通过 ($PASSED/$TOTAL)，请继续修复。\"}"
fi

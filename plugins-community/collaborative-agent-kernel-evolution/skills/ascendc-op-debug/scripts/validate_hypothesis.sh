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

# validate_hypothesis.sh — 校验 hypothesis 文件格式和 taxonomy 合法性
# 用法: bash validate_hypothesis.sh <hypothesis_file>
# 返回: 0=通过, 1=失败

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TAXONOMY="$SKILL_DIR/TAXONOMY.md"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hypothesis_file>"
    exit 1
fi

FILE="$1"
ERRORS=()

# ── 1. 文件存在检查 ─────────────────────────────────────────
if [[ ! -f "$FILE" ]]; then
    echo "ERROR: File not found: $FILE"
    exit 1
fi

# ── 2. 必填字段存在检查（仅在 frontmatter 区域内检查，避免 body 误匹配）────
# 提取 frontmatter：第一个 --- 之后、第二个 --- 之前的内容
FRONTMATTER=$(awk 'NR==1{if(/^---/)fm=1;next} fm&&/^---/{exit} fm{print}' "$FILE")
if [[ -z "$FRONTMATTER" ]]; then
    ERRORS+=("YAML frontmatter not found (file must start with ---)")
fi
for field in id title symptom when root_cause evidence escalate_to source; do
    if ! echo "$FRONTMATTER" | grep -qE "^${field}:"; then
        ERRORS+=("Missing frontmatter field: ${field}")
    fi
done

# ── 3. 必填节（## headers）存在检查 ────────────────────────
for section in triggers read_target code_pattern fix_template verify_cmd; do
    if ! grep -qE "^## ${section}" "$FILE"; then
        ERRORS+=("Missing required section: ## ${section}")
    fi
done

# ── 4. code_pattern 和 fix_template 必须包含代码块 ──────────
# 注意：awk range /start/,/end/ 中起始行同时匹配结束条件会立即关闭 range，
# 改用状态机：匹配到 section header 后跳过该行，遇到下一个 ## 退出。
if ! awk '/^## code_pattern/ { s=1; next } s && /^## / { exit } s' "$FILE" | grep -q '```'; then
    ERRORS+=('## code_pattern must contain a code block (```)')
fi
if ! awk '/^## fix_template/ { s=1; next } s && /^## / { exit } s' "$FILE" | grep -q '```'; then
    ERRORS+=('## fix_template must contain a code block (```)')
fi

# ── 5. Taxonomy 合法值校验 ──────────────────────────────────
extract_value() {
    echo "$FRONTMATTER" | grep -E "^${1}:" | head -1 | sed 's/^[^:]*: *//' | tr -d '"' | xargs
}

check_taxonomy() {
    local dimension="$1"
    local value="$2"
    # 在 TAXONOMY.md 中查找对应维度的表格，判断 value 是否在合法值列表中
    local in_section=0
    local found=0
    while IFS= read -r line; do
        if echo "$line" | grep -qE "^## 维度.*${dimension}"; then
            in_section=1
        elif [[ $in_section -eq 1 ]] && echo "$line" | grep -qE "^## 维度"; then
            break
        elif [[ $in_section -eq 1 ]] && echo "$line" | grep -qE "^\| \`${value}\`"; then
            found=1
            break
        fi
    done < "$TAXONOMY"
    echo $found
}

for dim in symptom when root_cause evidence; do
    val=$(extract_value "$dim")
    if [[ -n "$val" ]]; then
        found=$(check_taxonomy "$dim" "$val")
        if [[ "$found" -ne 1 ]]; then
            # 提取该维度的合法值列表供提示
            valid_vals=$(awk "/^## 维度.*${dim}/,/^## 维度/" "$TAXONOMY" \
                         | grep -oE '`[a-z_]+`' | tr -d '`' | tr '\n' ', ' | sed 's/,$//')
            ERRORS+=("Invalid ${dim} value: '${val}'. Valid: [${valid_vals}]")
        fi
    fi
done

# ── 6. id 格式检查 ──────────────────────────────────────────
id_val=$(extract_value "id")
if [[ ! "$id_val" =~ ^H[0-9]+$ ]] && [[ "$id_val" != "H_NEW" ]]; then
    ERRORS+=("id must be H{number} or H_NEW, got: '${id_val}'")
fi

# ── 7. 输出结果 ─────────────────────────────────────────────
if [[ ${#ERRORS[@]} -eq 0 ]]; then
    echo "✓ PASS: $FILE"
    exit 0
else
    echo "✗ FAIL: $FILE"
    for err in "${ERRORS[@]}"; do
        echo "  - $err"
    done
    exit 1
fi

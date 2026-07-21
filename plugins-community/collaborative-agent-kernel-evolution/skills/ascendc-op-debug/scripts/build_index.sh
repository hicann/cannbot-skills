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

# build_index.sh — 从 hypotheses/ 中的 frontmatter 自动生成 INDEX.md
# 用法: bash build_index.sh
# 在 retro/ 中新 hypothesis 通过 validate 后执行此脚本重建索引

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HYP_DIR="$SKILL_DIR/hypotheses"
INDEX="$SKILL_DIR/INDEX.md"

# 解析单个 hypothesis 文件的 frontmatter 字段
extract_field() {
    local file="$1" field="$2"
    grep -E "^${field}:" "$file" | head -1 | sed 's/^[^:]*: *//' | tr -d '"' | xargs
}

# 收集所有 hypothesis 的元数据
declare -A HYP_TITLE HYP_SYMPTOM HYP_WHEN HYP_RC HYP_EV HYP_ESC
IDS=()

for f in "$HYP_DIR"/H*.md; do
    [[ -f "$f" ]] || continue
    id=$(extract_field "$f" "id")
    [[ -z "$id" || "$id" == "H_NEW" ]] && continue
    IDS+=("$id")
    HYP_TITLE[$id]=$(extract_field "$f" "title")
    HYP_SYMPTOM[$id]=$(extract_field "$f" "symptom")
    HYP_WHEN[$id]=$(extract_field "$f" "when")
    HYP_RC[$id]=$(extract_field "$f" "root_cause")
    HYP_EV[$id]=$(extract_field "$f" "evidence")
    HYP_ESC[$id]=$(extract_field "$f" "escalate_to")
done

# 按 id 数字排序
IFS=$'\n' SORTED=($(for id in "${IDS[@]}"; do echo "$id"; done | sort -t'H' -k2 -n))
unset IFS

# ── 生成 INDEX.md ────────────────────────────────────────────
cat > "$INDEX" << 'HEADER'
# INDEX.md — 自动生成，勿手动编辑
# 由 scripts/build_index.sh 从 hypotheses/ frontmatter 生成
# 更新方法：新增/修改 hypothesis 后执行 bash scripts/build_index.sh

HEADER

echo "## 全量假设列表" >> "$INDEX"
echo "" >> "$INDEX"
echo "| ID | Title | symptom | when | root_cause | evidence | escalate_to |" >> "$INDEX"
echo "|---|---|---|---|---|---|---|" >> "$INDEX"
for id in "${SORTED[@]}"; do
    fname=$(ls "$HYP_DIR"/${id}_*.md 2>/dev/null | head -1)
    fname=$(basename "$fname")
    echo "| [$id](hypotheses/$fname) | ${HYP_TITLE[$id]} | \`${HYP_SYMPTOM[$id]}\` | \`${HYP_WHEN[$id]}\` | \`${HYP_RC[$id]}\` | \`${HYP_EV[$id]}\` | ${HYP_ESC[$id]} |" >> "$INDEX"
done

echo "" >> "$INDEX"
echo "---" >> "$INDEX"
echo "" >> "$INDEX"

# ── 按 symptom 分组 ──────────────────────────────────────────
echo "## 按症状索引（symptom）" >> "$INDEX"
echo "" >> "$INDEX"

declare -A SEEN_SYMPTOMS
for id in "${SORTED[@]}"; do
    s="${HYP_SYMPTOM[$id]}"
    SEEN_SYMPTOMS[$s]=1
done

for sym in $(echo "${!SEEN_SYMPTOMS[@]}" | tr ' ' '\n' | sort); do
    echo "### \`$sym\`" >> "$INDEX"
    for id in "${SORTED[@]}"; do
        [[ "${HYP_SYMPTOM[$id]}" == "$sym" ]] || continue
        echo "- **$id** [when=\`${HYP_WHEN[$id]}\`] ${HYP_TITLE[$id]} → ev=\`${HYP_EV[$id]}\`" >> "$INDEX"
    done
    echo "" >> "$INDEX"
done

# ── 按 root_cause 分组 ───────────────────────────────────────
echo "---" >> "$INDEX"
echo "" >> "$INDEX"
echo "## 按根因索引（root_cause）" >> "$INDEX"
echo "" >> "$INDEX"

declare -A SEEN_RC
for id in "${SORTED[@]}"; do
    rc="${HYP_RC[$id]}"
    SEEN_RC[$rc]=1
done

for rc in $(echo "${!SEEN_RC[@]}" | tr ' ' '\n' | sort); do
    echo "### \`$rc\`" >> "$INDEX"
    for id in "${SORTED[@]}"; do
        [[ "${HYP_RC[$id]}" == "$rc" ]] || continue
        echo "- **$id** ${HYP_TITLE[$id]}" >> "$INDEX"
    done
    echo "" >> "$INDEX"
done

echo "INDEX.md rebuilt: ${#SORTED[@]} hypotheses indexed."

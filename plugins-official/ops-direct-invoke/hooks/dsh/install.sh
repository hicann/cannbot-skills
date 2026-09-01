#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# DSH（DeepSeek Harness）部署级权限守卫安装器
#
# 背景：dsh 无「项目级、文件驱动的 hook」机制（opencode .opencode/plugin/、
# claude .claude/hooks/ 在 dsh 中不存在），init.sh 无法在项目层挂载
# permission-guard。但 dsh 有**部署级** Cordis 插件机制：本脚本把
# permission-guard.js 复制到 $DSH_HOME/plugins/，并在 $DSH_HOME/cordis.patch.yml
# （home 级 patch 层，对所有 profile 生效，watchUserPatches 支持热加载）
# 幂等注册插件条目。安装后：
#   - 角色写权限隔离恢复机制保证（tools/pre-execute allow/deny 门）
#   - 静默模式问卷拦截恢复机制兜底（.cannbot/settings.json mode=silent）
#   - 仅在 ops-direct-invoke 初始化的工作区（cwd 下有 .cannbot/permissions/）生效，
#     其它项目不受影响
#
# 用法:
#   install.sh                 # 安装（交互确认）
#   install.sh --yes           # 安装（跳过确认）
#   install.sh --remove        # 卸载（移除 patch 条目，保留插件文件）
#   install.sh --remove --yes  # 卸载（跳过确认）
#   install.sh --dsh-home <dir> # 指定 DSH_HOME（默认 ${DSH_HOME:-~/.dsh}）
#
# 生效方式：patch 文件变更后 dsh 会热加载（watchUserPatches/HMR）；未生效时
# 重启 dsh 会话即可。可用 `dsh --dump-config` 校验条目已入组合树。
#
# patch 条目格式说明（与 dsh 的 applyEntryPatches 语义一致）：
#   dsh 的用户 patch 层是对空 entry list 应用补丁——**新增条目必须用
#   `insert:` 包装**，裸 `id: xxx / name: xxx` 条目会因 id 不存在被丢弃。
#   模块名用**绝对路径**（相对路径按 include 的 baseUrl=profile 目录解析，
#   与 $DSH_HOME 不一致，会导致加载失败）。

set -e

show_help() {
    sed -n '24,38p' "$0" | sed 's/^# \{0,1\}//'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="${SCRIPT_DIR}/permission-guard.js"
ENTRY_ID="cannbot-permission-guard"

DO_REMOVE=false
ASSUME_YES=false
DSH_HOME_DIR="${DSH_HOME:-${HOME}/.dsh}"

while [ $# -gt 0 ]; do
    case "$1" in
        --remove)      DO_REMOVE=true; shift ;;
        --yes|-y)      ASSUME_YES=true; shift ;;
        --dsh-home)    DSH_HOME_DIR="$2"; shift 2 ;;
        --dsh-home=*)  DSH_HOME_DIR="${1#*=}"; shift ;;
        --help|-h)     show_help; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "${PLUGIN_SRC}" ]; then
    echo "✗ plugin source not found: ${PLUGIN_SRC}" >&2
    exit 1
fi

PATCH_FILE="${DSH_HOME_DIR}/cordis.patch.yml"
PLUGIN_DIR="${DSH_HOME_DIR}/plugins"
PLUGIN_DST="${PLUGIN_DIR}/cannbot-permission-guard.js"
# 模块名用绝对路径：dsh 的相对 name 按 include 的 baseUrl（profile 目录）解析，
# 与 $DSH_HOME 不一致；绝对路径不依赖解析基准（移动 DSH_HOME 后重装即可）。
ENTRY_NAME="${PLUGIN_DST}"

echo "DSH home:     ${DSH_HOME_DIR}"
echo "Patch file:   ${PATCH_FILE}"
echo "Plugin dst:   ${PLUGIN_DST}"

confirm() {
    if [ "${ASSUME_YES}" = true ]; then
        return 0
    fi
    printf "继续? [Y/n] "
    read -r ans
    case "${ans}" in
        [Nn]*) return 1 ;;
        *)     return 0 ;;
    esac
}

# 整块删除 insert 条目：`- insert:` 起始、缩进的子条目，直到下一个顶格行。
# 仅当块内含目标 ENTRY_ID 时删除整块（含 insert 包装），避免留下 YAML 残渣。
remove_entry_block() {
    awk -v id="${ENTRY_ID}" '
        /^[[:space:]]*-[[:space:]]*insert:[[:space:]]*$/ {
            block = $0 "\n"
            in_block = 1
            has_id = 0
            next
        }
        in_block && /^[[:space:]]/ {
            block = block $0 "\n"
            if ($0 ~ ("id:[[:space:]]*" id "$")) has_id = 1
            next
        }
        in_block {
            if (has_id) skip = 1
            in_block = 0
            if (!skip) printf "%s", block
            skip = 0
            block = ""
        }
        { print }
        END {
            if (in_block) {
                if (!has_id) printf "%s", block
            }
        }
    ' "${1}" > "${1}.tmp" && mv "${1}.tmp" "${1}"
}

# ------------------------------------------------------------
# 卸载
# ------------------------------------------------------------
if [ "${DO_REMOVE}" = true ]; then
    if [ ! -f "${PATCH_FILE}" ]; then
        echo "✓ 未安装（patch 文件不存在）"
        exit 0
    fi
    if ! grep -q "id: ${ENTRY_ID}" "${PATCH_FILE}" 2>/dev/null; then
        echo "✓ 未安装（patch 中无 ${ENTRY_ID} 条目）"
        exit 0
    fi
    if ! confirm; then
        echo "已取消"
        exit 0
    fi
    cp -a "${PATCH_FILE}" "${PATCH_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    remove_entry_block "${PATCH_FILE}"
    # 空文件/纯注释收尾为合法空 patch 层（[]）
    if ! grep -qE "^- |^[^#[:space:]]" "${PATCH_FILE}" 2>/dev/null; then
        printf '[]\n' > "${PATCH_FILE}"
    fi
    echo "✓ 已从 ${PATCH_FILE} 移除 ${ENTRY_ID}（原文件已备份）"
    echo "  插件文件仍保留在 ${PLUGIN_DST}；如需彻底删除请手动执行 rm"
    echo "  建议用 \`dsh --dump-config\` 确认条目已从组合树消失"
    exit 0
fi

# ------------------------------------------------------------
# 安装
# ------------------------------------------------------------
if [ -f "${PATCH_FILE}" ] && grep -q "id: ${ENTRY_ID}" "${PATCH_FILE}" 2>/dev/null; then
    echo "✓ 已安装（patch 中已有 ${ENTRY_ID} 条目，跳过）"
    if [ ! -f "${PLUGIN_DST}" ]; then
        echo "  ⚠ 但插件文件缺失，重新复制…"
        mkdir -p "${PLUGIN_DIR}"
        cp "${PLUGIN_SRC}" "${PLUGIN_DST}"
        echo "  ✓ ${PLUGIN_DST}"
    fi
    exit 0
fi

if ! confirm; then
    echo "已取消"
    exit 0
fi

mkdir -p "${PLUGIN_DIR}"
cp "${PLUGIN_SRC}" "${PLUGIN_DST}"
echo "✓ 插件已复制: ${PLUGIN_DST}"

# insert 条目块（dsh patch 层新增条目的唯一合法格式）
entry_block() {
    cat << EOF
- insert:
    - id: ${ENTRY_ID}
      name: ${ENTRY_NAME}
EOF
}

if [ ! -f "${PATCH_FILE}" ]; then
    # 新建 home 级 patch 层（顶层 YAML 数组，含一个 insert 条目）
    mkdir -p "$(dirname "${PATCH_FILE}")"
    {
        cat << 'EOF'
# DeepSeek Harness user patch layer (home level, applied after every profile's
# bundle layers). Generated by cannbot ops-direct-invoke hooks/dsh/install.sh.
EOF
        entry_block
    } > "${PATCH_FILE}"
    echo "✓ 已创建 ${PATCH_FILE} 并注册 ${ENTRY_ID}"
else
    # 备份后追加 insert 条目块（幂等；保留既有内容）
    cp -a "${PATCH_FILE}" "${PATCH_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    if [ -n "$(tail -c 1 "${PATCH_FILE}")" ]; then
        echo "" >> "${PATCH_FILE}"
    fi
    entry_block >> "${PATCH_FILE}"
    echo "✓ 已追加 ${ENTRY_ID} 到 ${PATCH_FILE}（原文件已备份）"
fi

echo ""
echo "生效方式：dsh 会热加载用户 patch 层（watchUserPatches）；若当前会话未生效，"
echo "重启 dsh 会话即可。可用 \`dsh --dump-config\` 校验条目已入组合树。"
echo "卸载: ${0} --remove"

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
# V3 Anti-Cheat: 还原整个内置 kernel 树
#
# 职责单一：把备份目录中的 kernel 树移回 ${ASCEND_OPP_PATH}/.../tbe/kernel。
# 权限：默认不需要 sudo（同 disable 脚本，属主临时加写位、mv、恢复）。
#
# 用法：
#   bash restore_builtin_kernels_v3.sh [--backup-dir=<dir>] [--dry-run]
#
# --backup-dir: 备份目录（默认 ~/.cann_bench_kernel_backup）
# --dry-run:    仅打印将要执行的操作
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    *) echo "Unknown arg: $arg"; exit 1;;
  esac
done

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set (source the CANN set_env.sh)}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel"
KPARENT="$(dirname "$KROOT")"
BK="${BACKUP_DIR}/v3_disabled_kernels/kernel"

echo "======================================================================"
echo "V3 Anti-Cheat: 还原内置 kernel 树"
echo "======================================================================"
echo "Backup from:  ${BK}"
echo "Restore to:   ${KROOT}"
echo "Dry run:      $([[ $DRY_RUN -eq 1 ]] && echo YES || echo NO)"
echo "======================================================================"

# --- 前置校验 ---------------------------------------------------------------

if [[ ! -d "$BK" ]]; then
  echo "[ERROR] Backup not found: ${BK} (nothing to restore)"
  exit 1
fi

# 目标已存在：若为空目录可清掉，非空则拒绝（避免覆盖真实内容）
if [[ -e "$KROOT" ]]; then
  if [[ -n "$(ls -A "$KROOT" 2>/dev/null)" ]]; then
    echo "[ERROR] Target is non-empty: ${KROOT}"
    echo "        Already restored? Or remove it manually before restoring."
    exit 1
  fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "[DRY RUN] Would execute:"
  echo "  chmod u+w ${KPARENT} ${BK}   # 临时加写位（目标父目录 + 备份 kernel 自身）"
  echo "  [ -e ${KROOT} ] && rmdir ${KROOT}   # 清理空目标"
  echo "  mv ${BK} ${KROOT}"
  echo "  chmod u-w ${KPARENT} ${BK}   # 恢复权限"
  exit 0
fi

# --- 权限预检：优先无 sudo ---------------------------------------------------
# 跨目录 mv 需要对目标父目录(KPARENT)和被移动目录(BK 自身)都有写权限。
RESTORE_PERMS=()
cleanup() {
  # mv 成功后被移动目录从 BK 变为 KROOT 路径，对两者都尝试恢复权限。
  for p in "${RESTORE_PERMS[@]:-}" "$KROOT"; do chmod u-w "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

add_writable() {
  local p="$1"
  [[ -w "$p" ]] && return 0
  if chmod u+w "$p" 2>/dev/null; then
    RESTORE_PERMS+=("$p")
  else
    echo "[ERROR] 无法写入且无法 chmod: ${p}"
    echo "        请改用 sudo 执行："
    echo "          sudo env ASCEND_OPP_PATH=\"\$ASCEND_OPP_PATH\" bash $0 $*"
    exit 1
  fi
}
add_writable "$KPARENT"
add_writable "$BK"

# --- 执行 -------------------------------------------------------------------

[[ -e "$KROOT" ]] && rmdir "$KROOT" 2>/dev/null || true
mv "$BK" "$KROOT"

echo ""
echo "✓ Kernel tree restored to: ${KROOT}"
echo ""
echo "======================================================================"
echo "V3 Anti-Cheat: Kernel 树还原完成"
echo "======================================================================"
echo "内置算子现已可用。再次禁用: bash ${SCRIPT_DIR}/disable_builtin_kernels_v3.sh"

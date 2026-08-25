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
# V3 Anti-Cheat: 禁用整个内置 kernel 树
#
# 职责：把 ${ASCEND_OPP_PATH}/.../tbe/kernel 移到备份目录（移走后提交方无法
#       调用内置 aclnn 算子，框架 warmup 由 cann_bench_utils 提供）。
#       禁用前自动确保 cann_bench_utils 可用（未装则编译+安装+验证）。
#
# 权限：默认不需要 sudo。CANN 若装在当前用户目录下（属主 = 当前用户），
#       脚本会临时给 kernel 的父目录加写位、mv、再恢复权限，全程无需提权。
#       仅当当前用户既非属主、又无写权限时，才提示改用 sudo。
#
# 用法：
#   bash disable_builtin_kernels_v3.sh [--backup-dir=<dir>] [--dry-run] [--yes|-y]
#
# --backup-dir: 备份目录（默认 ~/.cann_bench_kernel_backup）
# --dry-run:    仅打印将要执行的操作，不实际移动、不触发 cann_bench_utils 安装
# --yes, -y:    非交互模式，跳过确认提示
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0
ASSUME_YES=0

for arg in "$@"; do
  case "$arg" in
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    --yes|-y)       ASSUME_YES=1;;
    *) echo "Unknown arg: $arg"; exit 1;;
  esac
done

# --- 确保 cann_bench_utils 可用 ------------------------------------------------
# 禁用内置 kernel 树后，框架的升频/L2 清 cache warmup 不再有内置 MatMul/ReduceMax
# 可用，必须由 cann_bench_utils 顶上。故禁用前先确保它可导入；未安装则自动
# 编译+安装+验证。逻辑与 run_evaluation.sh 的 ensure_cann_bench_utils 对齐。
# dry-run 模式跳过（仅预览 mv，不应触发编译）。
ensure_cann_bench_utils() {
  [[ $DRY_RUN -eq 1 ]] && { echo "[INFO] dry-run: 跳过 cann_bench_utils 检查"; return 0; }
  local py="${PYTHON:-python3}"
  if "$py" -c "from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean" 2>/dev/null; then
    echo "✓ cann_bench_utils 已安装"
    return 0
  fi
  echo "[INFO] cann_bench_utils 未安装，开始自动编译安装..."
  local UTILS_DIR="${PROJECT_ROOT}/src/cann_bench_utils"
  if [[ ! -d "${UTILS_DIR}" ]]; then
    echo "[ERROR] cann_bench_utils 源码目录不存在: ${UTILS_DIR}"
    echo "        V3 Anti-Cheat 需要 cann_bench_utils，请检查代码库完整性"
    exit 1
  fi
  echo "[INFO] 编译 cann_bench_utils..."
  ( cd "${UTILS_DIR}" && PYTHON="$py" bash build.sh --clean ) &> /tmp/cann_bench_utils_build.log || {
    echo "[ERROR] cann_bench_utils 编译失败，日志: /tmp/cann_bench_utils_build.log"
    tail -15 /tmp/cann_bench_utils_build.log
    exit 1
  }
  echo "[INFO] 安装 cann_bench_utils..."
  local WHEEL
  WHEEL=$(ls -t "${UTILS_DIR}"/dist/cann_bench_utils-*.whl 2>/dev/null | head -1)
  if [[ -z "${WHEEL}" ]]; then
    echo "[ERROR] 未找到编译的 wheel 包"
    exit 1
  fi
  "$py" -m pip install "${WHEEL}" --force-reinstall --no-deps &> /tmp/cann_bench_utils_install.log || {
    echo "[ERROR] cann_bench_utils 安装失败，日志: /tmp/cann_bench_utils_install.log"
    exit 1
  }
  if "$py" -c "from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean" 2>/dev/null; then
    echo "✓ cann_bench_utils 安装成功"
  else
    echo "[ERROR] cann_bench_utils 安装验证失败"
    exit 1
  fi
}

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set (source the CANN set_env.sh)}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel"
KPARENT="$(dirname "$KROOT")"
BK="${BACKUP_DIR}/v3_disabled_kernels/kernel"

# --- 前置校验 ---------------------------------------------------------------

# kernel 树必须存在且非空（空 = 可能已禁用，避免误覆盖备份）
if [[ ! -d "$KROOT" ]]; then
  echo "[ERROR] Kernel tree not found: ${KROOT}"
  echo "        Check ASCEND_OPP_PATH (already disabled?)"
  exit 1
fi
if [[ -z "$(ls -A "$KROOT" 2>/dev/null)" ]]; then
  echo "[ERROR] Kernel tree is empty: ${KROOT} (already disabled?)"
  exit 1
fi

# 备份不能已存在（否则覆盖会永久丢失原始 kernel —— 唯一可能丢数据的场景）
if [[ -e "$BK" ]]; then
  echo "[ERROR] Backup already exists: ${BK}"
  echo "        Restore first: bash restore_builtin_kernels_v3.sh --backup-dir=${BACKUP_DIR}"
  exit 1
fi

echo "======================================================================"
echo "V3 Anti-Cheat: 禁用整个内置 kernel 树"
echo "======================================================================"
echo "Kernel tree:  ${KROOT}"
echo "Backup to:    ${BK}"
echo "Dry run:      $([[ $DRY_RUN -eq 1 ]] && echo YES || echo NO)"
echo "======================================================================"

cat <<WARN

⚠️  移走整个内置 kernel 树后，本机所有依赖该 CANN 安装的进程都将无法调用内置算子。
    仅建议在专用评测环境 / 容器中执行。操作可逆（mv，非 rm），用 restore 脚本还原。
WARN

# 确保 cann_bench_utils 可用（禁用后框架 warmup 依赖它；未装则自动编译安装）
ensure_cann_bench_utils

# --- 确认 -------------------------------------------------------------------

if [[ $ASSUME_YES -eq 0 ]]; then
  if [[ -t 0 ]]; then
    echo "Type 'MOVE' (uppercase) to confirm, or Ctrl-C to abort:"
    read -r confirm
    [[ "$confirm" == "MOVE" ]] || { echo "Aborted."; exit 0; }
  else
    echo "[ERROR] Non-interactive mode requires --yes flag"
    exit 1
  fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "[DRY RUN] Would execute:"
  echo "  chmod u+w ${KPARENT} ${KROOT}   # 临时加写位（父目录 + kernel 自身）"
  echo "  mkdir -p $(dirname "$BK")"
  echo "  mv ${KROOT} ${BK}"
  echo "  chmod u-w ${KPARENT} ${KROOT}   # 恢复权限"
  exit 0
fi

# --- 权限预检：优先无 sudo，属主可自行加写位 --------------------------------

# CANN 目录常见权限为 r-xr-x---（属主也无写位），但属主可自行 chmod。
# 跨目录 mv 一个目录需要同时有写权限于：
#   1) 父目录 KPARENT （要删除其下的 kernel 条目）
#   2) 被移动目录 KROOT 自身（要更新其 .. 硬链接指向新父目录）
# 因此对两者都临时加写位，操作完恢复。
RESTORE_PERMS=()

# 无论成功失败都恢复权限
# 注意：被移动目录在 mv 成功后从 KROOT 变为 BK 路径，故对两者都尝试恢复，
# 存在的那个会被命中。
cleanup() {
  for p in "${RESTORE_PERMS[@]:-}" "$BK"; do chmod u-w "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

add_writable() {  # <path>
  local p="$1"
  [[ -w "$p" ]] && return 0
  if chmod u+w "$p" 2>/dev/null; then
    RESTORE_PERMS+=("$p")
  else
    echo "[ERROR] 无法写入且无法 chmod: ${p}"
    echo "        当前用户既非属主、又无写权限。请改用 sudo 执行："
    echo "          sudo env ASCEND_OPP_PATH=\"\$ASCEND_OPP_PATH\" bash $0 $*"
    exit 1
  fi
}
add_writable "$KPARENT"
add_writable "$KROOT"

# --- 执行 -------------------------------------------------------------------

mkdir -p "$(dirname "$BK")"
mv "$KROOT" "$BK"

echo ""
echo "✓ Kernel tree moved to backup: ${BK}"
echo "  Restore: bash ${SCRIPT_DIR}/restore_builtin_kernels_v3.sh --backup-dir=${BACKUP_DIR}"
echo ""
echo "======================================================================"
echo "V3 Anti-Cheat: Kernel 树禁用完成"
echo "======================================================================"

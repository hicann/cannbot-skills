#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# start.sh — one-time setup. Symlinks the cpx wrapper into a PATH dir so
# `cpx claude ...` works from any directory. Run once: ./proxy/start.sh
# (re-run safe). Command name on PATH is `cpx`; the wrapper file stays `cpx-cli`.
set -e
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WRAPPER="$SCRIPT_DIR/cpx-cli"
if [ ! -x "$WRAPPER" ]; then
  echo "error: $WRAPPER not found or not executable (run: chmod +x $WRAPPER)" >&2
  exit 1
fi

# Pick a writable dir already on PATH; fall back to ~/.local/bin (no sudo).
BINDIR=""
for d in /usr/local/bin "$HOME/.local/bin" "$HOME/bin"; do
  [ -d "$d" ] && [ -w "$d" ] && BINDIR="$d" && break
done
if [ -z "$BINDIR" ]; then
  BINDIR="$HOME/.local/bin"
  mkdir -p "$BINDIR"
fi

# If target not on PATH, add it to ~/.bashrc for future shells.
case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *)
    if [ -f "$HOME/.bashrc" ]; then
      grep -q "$BINDIR" "$HOME/.bashrc" 2>/dev/null || echo "export PATH=\"$BINDIR:\$PATH\"" >> "$HOME/.bashrc"
      echo "[install] added $BINDIR to PATH via ~/.bashrc — open a new terminal or: source ~/.bashrc"
    fi
    ;;
esac

ln -sfn "$WRAPPER" "$BINDIR/cpx"

cat <<EOF

✓ cpx 已安装，可在任意目录使用：

  cpx claude                        # 正式交互式（/exit 退出后自动导入并开浏览器）
  cpx opencode                      # opencode

只要在原有 claude / opencode 命令前面加上 cpx，其他使用完全无变化。
重跑本脚本安全（仅更新软链）。
EOF

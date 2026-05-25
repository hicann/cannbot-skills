# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#!/bin/bash
# verify_catlass_ready.sh - 校验工作区根 ./catlass/ 源码就绪
# 使用：bash verify_catlass_ready.sh

set -e

CATLASS_DIR="./catlass"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "Catlass 源码就绪校验"
echo "================================================================"
echo ""

if [ ! -d "$CATLASS_DIR" ]; then
    echo -e "  ${YELLOW}⚠${NC} ./catlass/ 不存在，正在自动克隆..."
    if git clone --quiet https://gitcode.com/cann/catlass.git "$CATLASS_DIR" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} catlass 已克隆到 ./catlass/"
    else
        echo -e "${RED}❌ 错误：catlass 克隆失败${NC}"
        echo ""
        echo "请手动在工作目录根（与 operators/ 平级）执行："
        echo "  git clone https://gitcode.com/cann/catlass.git"
        echo ""
        echo "⚠️  禁止：将 catlass 克隆到 operators/{operator_name}/ 内（C2 检视项）"
        exit 1
    fi
    echo ""
fi

ERRORS=0
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; ERRORS=$((ERRORS + 1)); }

if [ -d "$CATLASS_DIR/include" ]; then
    ok "catlass/include 存在"
else
    err "catlass/include 不存在"
fi

if [ -d "$CATLASS_DIR/examples" ]; then
    EX_COUNT=$(find "$CATLASS_DIR/examples" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | wc -l)
    ok "catlass/examples 存在（${EX_COUNT} 个子目录）"
else
    err "catlass/examples 不存在"
fi

if [ -d "$CATLASS_DIR/docs" ]; then
    ok "catlass/docs 存在"
else
    warn "catlass/docs 不存在（不影响开发，但调优时建议参考）"
fi

# 防呆：检查是否有人误把 catlass 克隆到了 operators/{op}/ 内
if find operators -maxdepth 3 -type d -name catlass 2>/dev/null | grep -q .; then
    err "在 operators/ 子目录内发现 catlass 目录（违反 C2 检视项）"
    echo "    问题路径："
    find operators -maxdepth 3 -type d -name catlass 2>/dev/null | sed 's/^/      /'
    echo "    请删除并改为在工作区根 ./catlass/ 克隆"
fi

echo ""
echo "================================================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ catlass 源码就绪${NC}"
else
    echo -e "${RED}✗ catlass 源码未就绪（错误 $ERRORS 项）${NC}"
    exit 1
fi

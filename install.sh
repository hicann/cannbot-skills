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

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "   ____    _    _   _ _   _ ____        _   "
echo "  / ___|  / \\  | \\ | | \\ | | __ )  ___ | |_ "
echo " | |     / _ \\ |  \\| |  \\| |  _ \\ / _ \\| __|"
echo " | |___ / ___ \\| |\\  | |\\  | |_) | (_) | |_ "
echo "  \\____/_/   \\_\\_| \\_|_| \\_|____/ \\___/ \\__|"
echo -e "${NC}"
echo -e "${CYAN}🚀 CANNBot Install Helper 安装脚本${NC}"
echo ""

# 1. 检查 Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 Node.js${NC}"
    echo ""
    echo "请先安装 Node.js 18+:"
    echo "  官网下载: https://nodejs.org"
    echo "  或使用 nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "              nvm install 20"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ 错误: Node.js 版本过低 (当前: $(node --version))${NC}"
    echo ""
    echo "需要 Node.js 18+，请升级:"
    echo "  官网下载: https://nodejs.org"
    echo "  或使用 nvm: nvm install 20"
    exit 1
fi

echo -e "${GREEN}✓${NC} Node.js $(node --version)"

# 2. 检查 npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 npm${NC}"
    echo ""
    echo "npm 应该随 Node.js 一起安装，请检查安装"
    exit 1
fi

echo -e "${GREEN}✓${NC} npm $(npm --version)"
echo ""

# 3. 检查是否已安装
if command -v install-helper &> /dev/null; then
    CURRENT_VERSION=$(install-helper --version 2>/dev/null || echo "unknown")
    echo -e "${YELLOW}ℹ${NC}  检测到已安装 install-helper (版本: $CURRENT_VERSION)"
    echo ""
    read -p "是否更新到最新版本？[Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "${GREEN}✓${NC} 保留当前版本"
        echo ""
        echo "使用方法:"
        echo "  install-helper          # 启动交互式安装向导"
        echo "  install-helper list     # 查看可用插件"
        echo "  install-helper doctor   # 健康检查"
        echo ""
        exit 0
    fi
    echo ""
fi

# 4. 安装 install-helper
echo -e "${CYAN}📦 正在安装 @cannbot-ai/install-helper...${NC}"
echo ""

if npm install -g @cannbot-ai/install-helper 2>&1 | grep -E "(added|changed|up to date)"; then
    echo ""
    echo -e "${GREEN}✅ 安装成功！${NC}"
    echo ""
    echo "使用方法:"
    echo "  install-helper          # 启动交互式安装向导"
    echo "  install-helper list     # 查看可用插件"
    echo "  install-helper doctor   # 健康检查"
    echo ""
    echo "文档: https://gitcode.com/cann/cannbot-skills"
    echo ""
    
    # 询问是否立即运行
    read -p "是否立即运行安装向导？[Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo ""
        install-helper
    fi
else
    echo ""
    echo -e "${RED}❌ 安装失败${NC}"
    echo ""
    echo "可能的原因:"
    echo "  1. 网络连接问题"
    echo "  2. 权限不足"
    echo ""
    echo "解决方案:"
    echo "  尝试使用 sudo:"
    echo "    sudo npm install -g @cannbot-ai/install-helper"
    echo ""
    echo "  或使用 npx 免安装方式:"
    echo "    npx @cannbot-ai/install-helper"
    exit 1
fi

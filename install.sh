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

# Channel selection: latest (default), beta, or explicit version
CHANNEL="${1:-latest}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "   ____    _    _   _ _   _ ____        _   "
echo "  / ___|  / \  | \ | | \ | | __ )  ___ | |_ "
echo " | |     / _ \ |  \| |  \| |  _ \ / _ \| __|"
echo " | |___ / ___ \| |\  | |\  | |_) | (_) | |_ "
echo "  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|"
echo -e "${NC}"
echo -e "${CYAN}🚀 CANNBot Install Helper 安装脚本${NC}"
echo ""

# ========================================
# 路径 A: 有 Node.js 18+ → npm install
# ========================================
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        echo -e "${GREEN}✓${NC} Node.js $(node --version)"

        # 查询 npm registry 指定通道版本
        NPM_MIRRORS_A=(
            "https://registry.npmjs.org"
            "https://registry.npmmirror.com"
        )
        REMOTE_VERSION=""
        if [ "$CHANNEL" = "latest" ] || [ "$CHANNEL" = "beta" ]; then
            for REGISTRY in "${NPM_MIRRORS_A[@]}"; do
                TAG_VERSION=$(curl -fsSL "${REGISTRY}/@cannbot-ai/install-helper/${CHANNEL}" 2>/dev/null | grep -o '"version":"[^"]*"' | sed 's/.*"version":"//;s/"//' | head -1)
                if [ -n "$TAG_VERSION" ]; then
                    REMOTE_VERSION="$TAG_VERSION"
                    break
                fi
            done
        else
            REMOTE_VERSION="$CHANNEL"
        fi

        # 检查本地已安装版本
        LOCAL_VERSION=""
        if command -v install-helper >/dev/null 2>&1; then
            LOCAL_VERSION=$(install-helper --version 2>/dev/null | head -1 | tr -d '[:space:]')
        fi

        if [ -n "$LOCAL_VERSION" ] && [ "$LOCAL_VERSION" = "$REMOTE_VERSION" ]; then
            echo -e "${GREEN}✓${NC} 已安装最新版本 (v${LOCAL_VERSION})"
            echo ""
            echo -e "${CYAN}请运行: install-helper${NC}"
            exit 0
        fi

        if [ -n "$LOCAL_VERSION" ] && [ -n "$REMOTE_VERSION" ]; then
            echo -e "${YELLOW}ℹ${NC}  正在更新 v${LOCAL_VERSION} → v${REMOTE_VERSION}"
        elif [ -n "$LOCAL_VERSION" ]; then
            echo -e "${YELLOW}ℹ${NC}  检测到已安装 install-helper (版本: $LOCAL_VERSION)"
        fi
        echo ""

        echo -e "${CYAN}📦 正在安装 @cannbot-ai/install-helper@${CHANNEL}...${NC}"
        echo ""

        if npm install -g "@cannbot-ai/install-helper@${CHANNEL}" 2>&1; then
            echo ""
            echo -e "${GREEN}✅ 安装成功！${NC}"
            echo ""
            echo -e "${CYAN}请运行: install-helper${NC}"
            exit 0
        else
            echo ""
            echo -e "${RED}❌ npm 安装失败${NC}"
            echo ""
            echo "尝试使用 sudo:"
            echo "  sudo npm install -g @cannbot-ai/install-helper"
            exit 1
        fi
    fi
fi

# ========================================
# 路径 B: 无 Node.js → 下载预编译二进制
# ========================================

# 检测平台
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# macOS 无 Node.js 时直接提示安装 Node.js（二进制会被 Gatekeeper 拦截）
if [ "$OS" = "darwin" ]; then
    echo -e "${YELLOW}ℹ${NC}  macOS 未检测到 Node.js 18+"
    echo ""
    echo "请参照官网教程安装 Node.js 18+:"
    echo "  https://nodejs.org/zh-cn/download"
    echo ""
    echo "安装完成后执行:"
    echo "  npm install -g @cannbot-ai/install-helper"
    exit 1
fi

echo -e "${YELLOW}ℹ${NC}  未检测到 Node.js 18+，使用预编译二进制安装"
echo ""

ARCH=$(uname -m)

PLATFORM="linux"

case "$ARCH" in
    x86_64|amd64)  ARCHITECTURE="x64" ;;
    arm64|aarch64) ARCHITECTURE="arm64" ;;
    *)             ARCHITECTURE="x64" ;;
esac

SUBPKG_NAME="@cannbot-ai/install-helper-${PLATFORM}-${ARCHITECTURE}"
BINARY_NAME="install-helper"

echo -e "${GREEN}✓${NC} 平台: ${PLATFORM}-${ARCHITECTURE}"
echo ""

# 2. 从 npm registry 查指定通道版本
echo -e "${CYAN}📦 正在查询版本 (${CHANNEL})...${NC}"

NPM_MIRRORS=(
    "https://registry.npmjs.org"
    "https://registry.npmmirror.com"
)

VERSION=""
BEST_REGISTRY=""
if [ "$CHANNEL" = "latest" ] || [ "$CHANNEL" = "beta" ]; then
    for REGISTRY in "${NPM_MIRRORS[@]}"; do
        TAG_VERSION=$(curl -fsSL "${REGISTRY}/${SUBPKG_NAME}/${CHANNEL}" 2>/dev/null | grep -o '"version":"[^"]*"' | sed 's/.*"version":"//;s/"//' | head -1)
        if [ -n "$TAG_VERSION" ]; then
            VERSION="$TAG_VERSION"
            BEST_REGISTRY="$REGISTRY"
            break
        fi
    done
else
    VERSION="$CHANNEL"
    BEST_REGISTRY="${NPM_MIRRORS[0]}"
fi

if [ -n "$VERSION" ] && [ "$CHANNEL" != "latest" ]; then
    echo -e "${GREEN}✓${NC} 版本: ${VERSION} (${CHANNEL})"
elif [ -n "$VERSION" ]; then
    echo -e "${GREEN}✓${NC} 版本: ${VERSION} (源: ${BEST_REGISTRY})"
fi

if [ -z "$VERSION" ]; then
    echo -e "${RED}❌ 无法查询版本信息${NC}"
    echo ""
    echo "请安装 Node.js 18+ 后使用:"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "  nvm install 20"
    echo "  npm install -g @cannbot-ai/install-helper"
    exit 1
fi

# 3. 检查本地是否已安装最新版本
INSTALL_DIR="$HOME/.cannbot/bin"
if [ -x "${INSTALL_DIR}/install-helper" ]; then
    LOCAL_VERSION=$("${INSTALL_DIR}/install-helper" --version 2>/dev/null | head -1 | tr -d '[:space:]')
    if [ "$LOCAL_VERSION" = "$VERSION" ]; then
        echo -e "${GREEN}✓${NC} 已安装最新版本 (v${VERSION})"
        echo ""
        # 确保 PATH 已设置
        SHELL_RC="$HOME/.bashrc"
        if [ -n "$ZSH_VERSION" ] || [ "${SHELL##*/}" = "zsh" ]; then
            SHELL_RC="$HOME/.zshrc"
        fi
        if [ -f "$SHELL_RC" ]; then
            if ! grep -q "$INSTALL_DIR" "$SHELL_RC" 2>/dev/null; then
                echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
            fi
        fi
        # 启动向导
        if "${INSTALL_DIR}/install-helper" --version >/dev/null 2>&1; then
            if { true </dev/tty; } 2>/dev/null; then
                exec "${INSTALL_DIR}/install-helper" </dev/tty
            else
                echo "请运行以下命令启动安装向导:"
                echo -e "  ${CYAN}${INSTALL_DIR}/install-helper${NC}"
            fi
        fi
        exit 0
    fi
    if [ -n "$LOCAL_VERSION" ]; then
        echo -e "${YELLOW}ℹ${NC}  更新 v${LOCAL_VERSION} → v${VERSION}"
    fi
fi

# 4. 下载二进制
mkdir -p "$INSTALL_DIR"

TARBALL_FILENAME="${SUBPKG_NAME#*/}-${VERSION}.tgz"

# 优先使用查询阶段确定的最优镜像
rm -f /tmp/ih-tarball.tgz
TARBALL_URL=""
DOWNLOAD_MIRRORS=("$BEST_REGISTRY")
for REGISTRY in "${NPM_MIRRORS[@]}"; do
    if [ "$REGISTRY" != "$BEST_REGISTRY" ]; then
        DOWNLOAD_MIRRORS+=("$REGISTRY")
    fi
done
for REGISTRY in "${DOWNLOAD_MIRRORS[@]}"; do
    TARBALL_URL="${REGISTRY}/${SUBPKG_NAME}/-/${TARBALL_FILENAME}"
    echo ""
    echo -e "${CYAN}📦 正在下载...${NC}"
    if curl -fL "$TARBALL_URL" -o /tmp/ih-tarball.tgz --progress-bar; then
        echo -e "${GREEN}✓${NC} 下载完成"
        break
    fi
    TARBALL_URL=""
done

if [ -z "$TARBALL_URL" ] || [ ! -f /tmp/ih-tarball.tgz ]; then
    echo -e "${RED}❌ 下载失败${NC}"
    echo ""
    echo "请安装 Node.js 18+ 后使用:"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "  nvm install 20"
    echo "  npm install -g @cannbot-ai/install-helper"
    exit 1
fi

# 4. 解压
echo -e "${CYAN}📦 正在安装...${NC}"
TEMP_EXTRACT=$(mktemp -d)
tar xzf /tmp/ih-tarball.tgz -C "$TEMP_EXTRACT" || { rm -rf "$TEMP_EXTRACT" /tmp/ih-tarball.tgz; exit 1; }
cp -f "${TEMP_EXTRACT}/package/${BINARY_NAME}" "${INSTALL_DIR}/install-helper"
chmod +x "${INSTALL_DIR}/install-helper"
rm -rf "$TEMP_EXTRACT" /tmp/ih-tarball.tgz

# 5. 自动添加 PATH
SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ] || [ "${SHELL##*/}" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
fi
if [ -f "$SHELL_RC" ]; then
    if ! grep -q "$INSTALL_DIR" "$SHELL_RC" 2>/dev/null; then
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
    fi
else
    echo -e "${YELLOW}⚠${NC} $SHELL_RC 不存在，请手动添加:"
    echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi

# 6. 安装完成 + 自动启动向导
echo -e "${GREEN}✅ 安装完成！${NC}"
echo ""

# 检测二进制是否可运行
if "${INSTALL_DIR}/install-helper" --version >/dev/null 2>&1; then
    if { true </dev/tty; } 2>/dev/null; then
        exec "${INSTALL_DIR}/install-helper" </dev/tty
    else
        echo "请运行以下命令启动安装向导:"
        echo -e "  ${CYAN}${INSTALL_DIR}/install-helper${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} 原生二进制无法运行"
    echo ""
    echo "请安装 Node.js 18+ 后使用:"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "  nvm install 20"
    echo "  npm install -g @cannbot-ai/install-helper"
fi

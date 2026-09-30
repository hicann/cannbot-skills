#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of the
# CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------

set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

# --- Pre-flight: Check prerequisites (all modes) ---
check_prerequisites() {
    local missing=()
    for tool in curl wget python3 gcc g++ cmake; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    # pip3: check pip3 command first, then python3 -m pip (get-pip.py --user may not be in PATH)
    if ! command -v pip3 >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
        missing+=("pip3")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        err "Missing prerequisites: ${missing[*]}"
        info "Install them first, then re-run:"
        if command -v apt-get >/dev/null 2>&1; then
            echo "  sudo apt-get update && sudo apt-get install -y curl wget python3 python3-pip build-essential cmake"
        elif command -v dnf >/dev/null 2>&1; then
            echo "  sudo dnf install -y curl wget python3 python3-pip gcc gcc-c++ make cmake"
        fi
        exit 1
    fi
    ok "All prerequisites present"
}
check_prerequisites

# --- 0. Parse arguments ---
INSTALL_MODE=false
WORKSPACE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            INSTALL_MODE=true
            shift
            ;;
        --workspace)
            WORKSPACE="${2:-}"
            if [ -z "$WORKSPACE" ]; then
                err "--workspace requires a value"
                exit 1
            fi
            shift 2
            ;;
        *)
            err "Unknown argument: $1"
            echo "Usage: setup-env.sh [--install] --workspace <path>" >&2
            exit 1
            ;;
    esac
done

# Fall back to WORKSPACE environment variable if --workspace not provided
if [ -z "$WORKSPACE" ]; then
    WORKSPACE="${WORKSPACE:-}"
fi

# --- 1. Validate WORKSPACE ---
if [ -z "$WORKSPACE" ]; then
    err "WORKSPACE is required. Provide it via --workspace <path> or WORKSPACE environment variable."
    echo "Usage: setup-env.sh [--install] --workspace <path>" >&2
    exit 1
fi

WORKSPACE="$(realpath "$WORKSPACE")"
if [ ! -d "$WORKSPACE" ]; then
    err "WORKSPACE directory does not exist: $WORKSPACE"
    exit 1
fi
info "WORKSPACE: $WORKSPACE"

# --- 2. Clone hccl repo ---
HCCL_REPO_URL="https://gitcode.com/cann/hccl.git"
HCCL_DIR="$WORKSPACE/hccl"

if [ -d "$HCCL_DIR/src/ops" ]; then
    echo "EXISTS:hccl:$HCCL_DIR"
else
    info "Cloning HCCL repo..."
    git clone "$HCCL_REPO_URL" "$HCCL_DIR"
    ok "HCCL cloned to $HCCL_DIR"
fi

# --- 3. Clone hcomm repo ---
HCOMM_REPO_URL="https://gitcode.com/cann/hcomm.git"
HCOMM_DIR="$WORKSPACE/hcomm"

if [ -d "$HCOMM_DIR/docs" ]; then
    echo "EXISTS:hcomm:$HCOMM_DIR"
else
    info "Cloning HCOMM repo..."
    git clone "$HCOMM_REPO_URL" "$HCOMM_DIR"
    ok "HCOMM cloned to $HCOMM_DIR"
fi

# --- 4. CANN_HOME: install or detect (only $WORKSPACE/Ascend) ---
if $INSTALL_MODE; then
    # --- 4a. Install mode: download + install CANN Toolkit + 950 ops ---
    DOWNLOAD_BASE="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    ARCH=$(uname -m)

    info "Fetching latest CANN build index..."
    LATEST_DIR=$(curl -s "$DOWNLOAD_BASE" | grep -oE '2[0-9]{16}/' | sort -r | head -1 | tr -d '/')
    if [ -z "$LATEST_DIR" ]; then
        err "Failed to fetch latest build directory from $DOWNLOAD_BASE"
        exit 1
    fi
    ok "Latest build: $LATEST_DIR"

    DIR_URL="${DOWNLOAD_BASE}${LATEST_DIR}/"

    TOOLKIT_PKG=$(curl -s "$DIR_URL" | grep -oE "Ascend-cann-toolkit_[^\"']*linux-${ARCH}\.run" | head -1)
    OPS_PKG=$(curl -s "$DIR_URL" | grep -oE "Ascend-cann-950-ops_[^\"']*linux-${ARCH}\.run" | head -1)
    if [ -z "$TOOLKIT_PKG" ]; then
        err "Toolkit package not found for ${ARCH} in $LATEST_DIR"
        exit 1
    fi
    if [ -z "$OPS_PKG" ]; then
        err "950-ops package not found for ${ARCH} in $LATEST_DIR"
        exit 1
    fi
    ok "Packages: $TOOLKIT_PKG / $OPS_PKG"

    TMP_DIR="$WORKSPACE/Ascend/.download"
    mkdir -p "$TMP_DIR"

    info "Downloading toolkit ($TOOLKIT_PKG, ~1.2GB)..."
    wget -c -P "$TMP_DIR" "${DIR_URL}${TOOLKIT_PKG}"
    ok "Toolkit downloaded"
    info "Downloading 950-ops ($OPS_PKG, ~2.7GB)..."
    wget -c -P "$TMP_DIR" "${DIR_URL}${OPS_PKG}"
    ok "950-ops downloaded"

    INSTALL_PATH="$WORKSPACE/Ascend"
    chmod +x "$TMP_DIR/$TOOLKIT_PKG" "$TMP_DIR/$OPS_PKG"

    info "Installing CANN Toolkit to $INSTALL_PATH (quiet mode)..."
    "$TMP_DIR/$TOOLKIT_PKG" --full --quiet --install-path="$INSTALL_PATH"
    ok "Toolkit installed"

    info "Installing 950-ops to $INSTALL_PATH (quiet mode)..."
    "$TMP_DIR/$OPS_PKG" --install --quiet --install-path="$INSTALL_PATH"
    ok "950-ops installed"

    rm -rf "$TMP_DIR"

    CANN_HOME=$(ls -d "$INSTALL_PATH"/cann-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$CANN_HOME" ]; then
        err "CANN installation completed but cann-* directory not found in $INSTALL_PATH"
        exit 1
    fi
    CANN_STATUS="CANN_INSTALLED"
    ok "CANN_HOME: $CANN_HOME"
else
    # --- 4b. Detect mode: scan only $WORKSPACE/Ascend ---
    CANN_HOME=$(ls -d "$WORKSPACE/Ascend"/cann-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$CANN_HOME" ] || [ ! -d "$CANN_HOME" ]; then
        warn "CANN_HOME not found under $WORKSPACE/Ascend"
        echo ""
        echo "RESULT_START"
        echo "STATUS:CANN_NOT_FOUND"
        echo "WORKSPACE:$WORKSPACE"
        echo "HCCL_ROOT:$HCCL_DIR"
        echo "HCOMM_DIR:$HCOMM_DIR"
        echo "MESSAGE:CANN Toolkit not detected under $WORKSPACE/Ascend. Run 'bash setup-env.sh --install --workspace <path>' to download and install."
        echo "RESULT_END"
        exit 0
    fi
    CANN_STATUS="CANN_FOUND"
    ok "CANN_HOME: $CANN_HOME"
fi

# --- 5. Determine derived paths ---
CANN_ARCH="${CANN_ARCH:-x86_64-linux}"
ASCEND_HOME_PATH="$CANN_HOME/$CANN_ARCH"
HCOMM_DOCS="$HCOMM_DIR/docs/zh"

if [ ! -d "$HCOMM_DOCS" ]; then
    HCOMM_DOCS="$HCOMM_DIR/docs"
fi

# --- 6. Write hccl-env.sh ---
ENV_FILE="$WORKSPACE/hccl-env.sh"
cat > "$ENV_FILE" << ENVEOF
# HCCLBot environment variables - generated by setup-env.sh
# Clear system CANN variables to prevent build.sh fallback to /usr/local/Ascend
unset ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH
export HCCL_ROOT="$HCCL_DIR"
export HCOMM_CODE_HOME="$HCOMM_DIR"
export HCOMM_DOCS="$HCOMM_DOCS"
export CANN_HOME="$CANN_HOME"
export ASCEND_HOME_PATH="$ASCEND_HOME_PATH"
# Source CANN environment (set_env.sh re-sets ASCEND_OPP_PATH etc. to local CANN paths)
source ${CANN_HOME}/set_env.sh 2>/dev/null || true
ENVEOF
ok "Environment file: $ENV_FILE"

# --- 7. Output structured result for HCCLBot ---
echo ""
echo "RESULT_START"
echo "STATUS:$CANN_STATUS"
echo "WORKSPACE:$WORKSPACE"
echo "HCCL_ROOT:$HCCL_DIR"
echo "HCOMM_CODE_HOME:$HCOMM_DIR"
echo "HCOMM_DOCS:$HCOMM_DOCS"
echo "CANN_HOME:$CANN_HOME"
echo "ASCEND_HOME_PATH:$ASCEND_HOME_PATH"
echo "ENV_FILE:$ENV_FILE"
echo "RESULT_END"

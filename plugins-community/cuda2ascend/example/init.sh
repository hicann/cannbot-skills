#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# CANNBot Agent Workspace Initialization Script
#
# CANNBot:   https://gitcode.com/cann/cannbot-skills
# This file: https://gitcode.com/cann/cannbot-skills/blob/main/plugins-community/cuda2ascend/example/init.sh

set -e

# ============================================================
# Quick Start content (shown after a successful install).
# Customize this block for your repo. $CLI_NAME is resolved at
# runtime from the target tool (opencode/claude/codex/dsh).
# ============================================================
show_quick_start() {
    echo ""
    echo -e "  ${BOLD}Quick Start:${NC}"
    echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}${CLI_NAME}${NC}"
    echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16，shape 主要是 [1,128]、[4,2048]${NC}"
}

# ============================================================
# Configuration
# ============================================================
CANNBOT_URL="https://gitcode.com/cann/cannbot-skills.git"
PLUGIN_NAME="cuda2ascend"

# ============================================================
# Terminal output helpers
# ============================================================
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
    CYAN='\033[0;36m';  BOLD='\033[1m';     DIM='\033[2m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

# ============================================================
# Base plugin init.sh path resolution & tools passthrough
# ============================================================
# Resolve base plugin init.sh path from SKILLS_REPO (no clone here).
# Echoes the path if found, empty otherwise.
resolve_base_init() {
    if [ -n "${SKILLS_REPO:-}" ] && [ -f "${SKILLS_REPO}/plugins-community/${PLUGIN_NAME}/init.sh" ]; then
        echo "${SKILLS_REPO}/plugins-community/${PLUGIN_NAME}/init.sh"
    fi
}

# Query supported tools from the base plugin init (透传基类 --list-tools 输出).
# Falls back to a hint when the base is not yet available (e.g. first run,
# no cache, offline). The fallback intentionally does NOT hardcode tool names.
query_supported_tools() {
    local base_init tools
    base_init="$(resolve_base_init)"
    if [ -n "${base_init}" ]; then
        tools="$(bash "${base_init}" --list-tools 2>/dev/null)" || tools=""
        if [ -n "${tools}" ]; then
            echo "${tools}"
            return
        fi
    fi
    echo "(运行一次 init 或通过 --repo cannbot-skills:<path> 指定后，此列表由基类 init 动态提供)"
}

# ============================================================
# Usage
# ============================================================
show_help() {
    local tools
    tools="$(query_supported_tools)"
    cat << EOF
CANNBot Agent Workspace Initialization Script

Usage:
  ./init.sh <target> [options]

Arguments:
  target                    Target environment. Supported (from base plugin): ${tools}

Options:
  -h, --help                Show this help message
  --repo <name>:<path>      Use a local repository instead of cloning.
                            Supported names:
                              cannbot-skills  - skills platform (consumed here)
                              asc-devkit / cann-samples / ops-tensor
                                             - passed through to the base plugin

Examples:
  ./init.sh opencode --repo cannbot-skills:~/cannbot-skills
  ./init.sh claude --repo cannbot-skills:~/cannbot-skills
  ./init.sh opencode --repo asc-devkit:~/asc-devkit --repo cann-samples:~/cann-samples
EOF
}

# ============================================================
# Parse arguments (collect -h without exiting; defer help display
# until SKILLS_REPO is resolved so the tools list can be queried)
# ============================================================
HELP_REQUESTED=false
TARGET_ENV=""
CANNBOT_LOCAL_PATH=""
REPO_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            HELP_REQUESTED=true; shift ;;
        --repo)
            if [[ -z "${2:-}" ]] || [[ "$2" != *:* ]]; then
                err "--repo requires name:/path argument, got: '${2:-}'"
                exit 1
            fi
            repo_name="${2%%:*}"
            repo_path="${2#*:}"
            case "$repo_name" in
                cannbot-skills|asc-devkit|cann-samples|ops-tensor) ;;
                *)
                    err "Unknown repo name '$repo_name' " \
                        "(valid: cannbot-skills / asc-devkit / cann-samples / ops-tensor)"
                    exit 1
                    ;;
            esac
            if [ ! -d "$repo_path" ]; then
                err "Repo path not found: $repo_path (--repo $repo_name)"
                exit 1
            fi
            if [[ "$repo_name" == "cannbot-skills" ]]; then
                CANNBOT_LOCAL_PATH="$repo_path"
            else
                REPO_ARGS+=("--repo" "$2")
            fi
            shift 2
            ;;
        -*)
            err "Unknown option: '$1'"
            echo ""
            show_help
            exit 1
            ;;
        *)
            if [ -z "$TARGET_ENV" ]; then
                TARGET_ENV="$1"; shift
            else
                err "Unexpected argument: '$1'"
                exit 1
            fi
            ;;
    esac
done

if [ "$HELP_REQUESTED" = false ] && [ -z "$TARGET_ENV" ]; then
    err "Missing required argument: target (e.g. opencode)"
    echo ""
    show_help
    exit 1
fi

# ============================================================
# Resolve paths
# ============================================================
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
AGENT_DIR="${SCRIPT_DIR}"
REPO_DIR="$(realpath "${SCRIPT_DIR}/..")"
WORKSPACE_NAME="$(basename "$REPO_DIR")"

# Determine cannbot-skills repo location (path only; clone happens later)
if [ -n "$CANNBOT_LOCAL_PATH" ]; then
    CANNBOT_LOCAL_PATH="$(realpath "$CANNBOT_LOCAL_PATH" 2>/dev/null || echo "$CANNBOT_LOCAL_PATH")"
    if [ ! -d "$CANNBOT_LOCAL_PATH" ]; then
        err "cannbot-skills directory not found: $CANNBOT_LOCAL_PATH"
        exit 1
    fi
    SKILLS_REPO="$CANNBOT_LOCAL_PATH"
else
    SKILLS_REPO="$REPO_DIR/.cannbot/cannbot-skills"
fi

# ============================================================
# Help mode: show help immediately, no side effects.
# Tools list is queried only from an already-available SKILLS_REPO;
# if the base is not present (first run / offline), show the fallback
# hint. Help must not create directories or trigger git clone.
# ============================================================
if [ "$HELP_REQUESTED" = true ]; then
    show_help
    exit 0
fi

# ============================================================
# Display configuration
# ============================================================
echo ""
echo -e "  ${BOLD}${WORKSPACE_NAME} agent${NC}"
echo -e "  ${BOLD}Configuration:${NC}"
echo -e "  target env:   ${CYAN}${TARGET_ENV}${NC}"
echo -e "  workspace:    ${CYAN}${WORKSPACE_NAME}${NC}"
echo -e "  repo root:    ${CYAN}${REPO_DIR}${NC}"
if [ -n "$CANNBOT_LOCAL_PATH" ]; then
    echo -e "  cannbot:      ${CYAN}${CANNBOT_LOCAL_PATH}${NC} (local)"
else
    echo -e "  cannbot:      ${CYAN}clone from ${CANNBOT_URL}${NC}"
fi
echo ""

# ============================================================
# Step 1: Setup cannbot-skills
# ============================================================
step "Setting up cannbot-skills..."
if [ -n "$CANNBOT_LOCAL_PATH" ]; then
    ok "Using local cannbot-skills at $SKILLS_REPO"
else
    if [ -d "$SKILLS_REPO/.git" ]; then
        git -C "$SKILLS_REPO" pull --quiet 2>/dev/null || true
        ok "cannbot-skills updated (local cache)"
    else
        mkdir -p "$(dirname "$SKILLS_REPO")"
        if git clone --quiet "$CANNBOT_URL" "$SKILLS_REPO" 2>/dev/null; then
            ok "cannbot-skills cloned from $CANNBOT_URL"
        else
            err "Failed to clone cannbot-skills from $CANNBOT_URL"
            exit 1
        fi
    fi
fi
echo ""

# ============================================================
# Step 2: Invoke base plugin init.sh
# ============================================================
PLUGIN_INIT="$SKILLS_REPO/plugins-community/${PLUGIN_NAME}/init.sh"

if [ ! -f "$PLUGIN_INIT" ]; then
    err "Plugin init.sh not found at: $PLUGIN_INIT"
    exit 1
fi

PLUGIN_ARGS=("${TARGET_ENV}" "$REPO_DIR" "--override" "$AGENT_DIR")
if [[ ${#REPO_ARGS[@]} -gt 0 ]]; then
    PLUGIN_ARGS+=("${REPO_ARGS[@]}")
fi

bash "$PLUGIN_INIT" "${PLUGIN_ARGS[@]}" || {
    err "Plugin '${PLUGIN_NAME}' init failed"
    exit 1
}

# ============================================================
# Step 3: Quick Start (子仓定制；基类在 --override 时不输出)
# ============================================================
CLI_NAME="opencode"
[ "$TARGET_ENV" = "claude" ] && CLI_NAME="claude"
[ "$TARGET_ENV" = "codex" ] && CLI_NAME="codex"
[ "$TARGET_ENV" = "dsh" ] && CLI_NAME="dsh"
show_quick_start
echo ""

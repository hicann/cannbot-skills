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

# Detect TRAE variant by scanning global config directories.
# Sets global: TRAE_VARIANT=(ide|plugin|cli|unknown)
detect_trae_variant() {
    if [ -d "$HOME/.trae-cn" ]; then
        TRAE_VARIANT="ide"
    elif [ -d "$HOME/.marscode" ]; then
        TRAE_VARIANT="plugin"
    elif [ -d "$HOME/.traecli" ]; then
        TRAE_VARIANT="cli"
    else
        TRAE_VARIANT="unknown"
    fi
}

BRAND="cannbot"
VERSION="1.0.0"

INCLUDED_SKILLS="model-infer-migrator model-infer-parallel-analysis model-infer-parallel-impl model-infer-kvcache model-infer-fusion model-infer-quantization model-infer-graph-mode model-infer-precision-debug model-infer-runtime-debug model-infer-multi-stream model-infer-prefetch model-infer-superkernel"
INCLUDED_AGENT_PATTERN="model-infer-*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
MODEL_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../model" && pwd)"

show_help() {
    cat << EOF
CANNBot - NPU Model Inference Optimization Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh project opencode
  init.sh project claude
  init.sh project trae
  init.sh project cursor
  init.sh project copilot
  init.sh global copilot
  init.sh global claude
  init.sh global cursor
  init.sh project opencode /path/to/proj
EOF
}

LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""

for arg in "$@"; do
    case "$arg" in
        --help) show_help; exit 0 ;;
        global|project) LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot) TOOL="$arg" ;;
    esac
done

# If last argument is not a known keyword, treat it as install_path
if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|copilot) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$HOME/.marscode" ;;
            cli)    CONFIG_ROOT="$HOME/.traecli" ;;
            *)      CONFIG_ROOT="$HOME/.trae-cn" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ -n "$INSTALL_PATH" ]; then
        CONFIG_ROOT_BASE="$(cd "$INSTALL_PATH" && pwd)"
    else
        CONFIG_ROOT_BASE="$PWD"
    fi

    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$CONFIG_ROOT_BASE/.marscode" ;;
            cli)    CONFIG_ROOT="$CONFIG_ROOT_BASE/.traecli" ;;
            *)      CONFIG_ROOT="$CONFIG_ROOT_BASE/.trae" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
    else
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi
if [ "$TOOL" = "opencode" ] && [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}NPU Model Inference Optimization${NC}"
  echo ""
}

resolve_skill_src() {
    local skill="$1"
    if [ -d "$MODEL_SKILL_ROOT/$skill" ]; then
        echo "$MODEL_SKILL_ROOT/$skill"
    else
        echo ""
    fi
}

install_skill_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    for skill in $INCLUDED_SKILLS; do
        local src
        src=$(resolve_skill_src "$skill")
        if [ -n "$src" ]; then
            rm -rf "$target_root/$skill"
            ln -sfn "$(realpath "$src")" "$target_root/$skill"
            count=$((count + 1))
        else
            warn "Skill not found: $skill"
        fi
    done
    ok "Skills: $count linked"
}

install_agent_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        local name base
        name=$(basename "$agent_entry")
        base="${name%.md}"
        [[ "$base" != $INCLUDED_AGENT_PATTERN ]] && continue
        rm -f "$target_root/$name"
        ln -sfn "$(realpath "$agent_entry")" "$target_root/$name"
        count=$((count + 1))
    done
    ok "Agents: $count linked"
}

install_config() {
    mkdir -p "$CONFIG_ROOT"
    local config_src="$PLUGIN_ROOT/AGENTS.md"
    local config_target
    if [ "$LEVEL" = "project" ]; then
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
            config_target="$PWD/AGENTS.md"
        else
            config_target="$PWD/CLAUDE.md"
        fi
    else
        if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; then
            config_target="$CONFIG_ROOT/AGENTS.md"
        else
            config_target="$CONFIG_ROOT/CLAUDE.md"
        fi
    fi

    if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "trae" ] || [ "$TOOL" = "cursor" ] || [ "$TOOL" = "copilot" ]; } && [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$PWD" ]; then
        ok "$(basename "$config_target") already in current directory"
    else
        ln -sf "$config_src" "$config_target"
        ok "$(basename "$config_target")"
    fi

    if [ "$LEVEL" = "project" ] && [ "$config_target" != "$CONFIG_ROOT/$(basename "$config_target")" ]; then
        ln -sf "$config_src" "$CONFIG_ROOT/$(basename "$config_target")"
    fi
}

install_workflows() {
    if [ -d "$PLUGIN_ROOT/workflows" ]; then
        mkdir -p "$CONFIG_ROOT"
        ln -sfn "$(realpath "$PLUGIN_ROOT/workflows")" "$CONFIG_ROOT/workflows"
        ok "workflows"
    fi
}

install_hooks() {
    if { [ "$TOOL" = "claude" ] || [ "$TOOL" = "cursor" ]; } && [ -d "$PLUGIN_ROOT/hooks" ]; then
        local hook_target_root="$CONFIG_ROOT/hooks"
        mkdir -p "$hook_target_root"
        for hook_file in "$PLUGIN_ROOT/hooks"/*.py; do
            [ -f "$hook_file" ] || continue
            ln -sfn "$(realpath "$hook_file")" "$hook_target_root/$(basename "$hook_file")"
        done

        local hooks_abs_dir
        hooks_abs_dir="$(cd "$PLUGIN_ROOT/hooks" && pwd)"
        if [ ! -f "$CONFIG_ROOT/settings.json" ]; then
            cat > "$CONFIG_ROOT/settings.json" << EOF
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{"type": "command", "command": "python3 $hooks_abs_dir/pre_tool_use.py"}]
      },
      {
        "matcher": "Edit|Write|Bash",
        "hooks": [{"type": "command", "command": "python3 $hooks_abs_dir/time_reminder.py"}]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Read",
        "hooks": [{"type": "command", "command": "python3 $hooks_abs_dir/post_tool_use.py"}]
      }
    ],
    "SubagentStop": [
      {
        "matcher": "model-infer-implementer|model-infer-reviewer",
        "hooks": [{"type": "command", "command": "python3 $hooks_abs_dir/subagent_stop.py"}]
      }
    ]
  }
}
EOF
            ok "settings.json"
        else
            warn "settings.json already exists, hooks not overwritten"
        fi
    fi
}

write_manifest() {
    local manifest="$CONFIG_ROOT/cannbot-manifest.json"
    local skills_json agents_json
    skills_json=$(printf '%s\n' $INCLUDED_SKILLS | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    agents_json=$(find "$LOCAL_AGENT_ROOT" -maxdepth 1 -name 'model-infer-*.md' -exec basename {} \; | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    cat > "$manifest" << EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "model-infer-optimize",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $skills_json,
  "installed_agents": $agents_json,
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    ok "Manifest: $manifest"
}

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo ""

if [ "$TOOL" = "trae" ] && [ "$LEVEL" = "project" ]; then
    case "$TRAE_VARIANT" in
        ide)
            info "Detected: TRAE IDE (.trae-cn / .trae)"
            ;;
        plugin)
            info "Detected: TRAE Plugin (.marscode)"
            ;;
        cli)
            info "Detected: TRAE CLI (.traecli)"
            ;;
        unknown)
            warn "TRAE variant not detected; defaulting to IDE path"
            warn "If you use TRAE Plugin, ensure ~/.marscode exists before re-running"
            warn "If you use TRAE CLI, ensure ~/.traecli exists before re-running"
            ;;
    esac
    echo ""
fi

step "[1/6] Installing skills and agents..."
mkdir -p "$CANNBOT_DIR"
install_skill_links "$CANNBOT_DIR/skills"
install_agent_links "$CANNBOT_DIR/agents"
echo ""

step "[2/6] Installing configuration..."
install_config
install_workflows
echo ""

step "[3/6] Installing hooks..."
install_hooks
echo ""

step "[4/6] Writing manifest..."
write_manifest
echo ""

step "[5/6] Setting up cann-recipes-infer reference repo..."
REFERENCE_DIR="$PLUGIN_ROOT/cann-recipes-infer"

if [ -d "$REFERENCE_DIR/.git" ]; then
    cd "$REFERENCE_DIR"
    git pull --quiet 2>/dev/null && ok "cann-recipes-infer updated" || warn "git pull failed, using existing version"
    cd "$SCRIPT_DIR"
elif command -v git >/dev/null 2>&1; then
    git clone --quiet --depth 1 https://gitcode.com/cann/cann-recipes-infer.git "$REFERENCE_DIR" 2>/dev/null \
        && ok "cann-recipes-infer cloned" \
        || warn "git clone failed, clone manually: git clone --depth 1 https://gitcode.com/cann/cann-recipes-infer.git $REFERENCE_DIR"
else
    warn "git not found, clone manually: git clone --depth 1 https://gitcode.com/cann/cann-recipes-infer.git $REFERENCE_DIR"
fi

# For global mode: symlink reference repo into CONFIG_ROOT so skill paths
# like `cann-recipes-infer/...` resolve from CONFIG_ROOT cwd
if [ "$LEVEL" = "global" ] && [ -d "$REFERENCE_DIR" ]; then
    ln -sfn "$(realpath "$REFERENCE_DIR")" "$CONFIG_ROOT/cann-recipes-infer"
    ok "cann-recipes-infer → $CONFIG_ROOT/"
fi

# For project mode with custom target: symlink into CONFIG_ROOT_BASE (== install target)
# so relative references from agents/workflows work correctly
if [ "$LEVEL" = "project" ] && [ -d "$REFERENCE_DIR" ] && [ "$CONFIG_ROOT_BASE" != "$PLUGIN_ROOT" ]; then
    ln -sfn "$(realpath "$REFERENCE_DIR")" "$CONFIG_ROOT_BASE/cann-recipes-infer"
    ok "cann-recipes-infer → $CONFIG_ROOT_BASE/"
fi
echo ""

step "[6/6] Health check..."
health_ok=true
[ -d "$CANNBOT_DIR/skills" ] || { err "skills/ missing"; health_ok=false; }
[ -d "$CANNBOT_DIR/agents" ] || { err "agents/ missing"; health_ok=false; }
[ -f "$CONFIG_ROOT/cannbot-manifest.json" ] || { err "manifest missing"; health_ok=false; }

if [ "$health_ok" = true ]; then
    ok "All checks passed"
else
    exit 1
fi

echo ""
echo -e "  ${GREEN}${BOLD}✓ model-infer-optimize installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动"
  echo -e "  ${CYAN}2.${NC} 输入：${GREEN}${BOLD}帮我优化 <model_name> 模型的 NPU 推理性能${NC}"
else
  echo -e "  ${CYAN}1.${NC} 在目标 cann-recipes-infer 或模型仓中启动工具"
  echo -e "  ${CYAN}2.${NC} 输入：${GREEN}${BOLD}帮我优化 <model_name> 模型的 NPU 推理性能${NC}"
fi
echo ""

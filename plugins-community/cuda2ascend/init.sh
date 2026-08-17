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
# Base plugin init script (constructor)
#
# Responsibility:
#   1. Create .cannbot intermediate directory
#   2. Link AGENTS.md config to the workspace (claude 额外链接 CLAUDE.md)
#   3. Link sub-agents (architect / developer / qa, flattened to one level)
#   4. Link skill directories referenced by agent frontmatter
#      (plugin skills/ first, then ops/, then infra/)
#      and link the permission hook (opencode plugin / claude settings.json)
#   5. Clone third-party repos into .cannbot/ (or use local paths via --repo)
#   6. Generate manifest and run health check

set -e

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

# Validate that a path exists (directory), exit with unified error if not.
# $1 = path to check  $2 = parameter name for error message
validate_path() {
    local path="$1"
    local param_name="$2"
    if [ ! -d "${path}" ]; then
        err "${param_name} path not found: ${path}"
        exit 1
    fi
}

# Safe install config file with backup and conflict handling.
# $1 = generated temp file path  $2 = target  $3 = display name  $4 = level
safe_install_file() {
    local tmpfile="$1"
    local target="$2"
    local name="$3"
    local level="$4"

    if [ -e "${target}" ] && diff -q "${tmpfile}" "${target}" > /dev/null 2>&1; then
        info "${name} already up to date"
        rm -f "${tmpfile}"
        return 0
    fi

    if [ -e "${target}" ] || [ -L "${target}" ]; then
        local backup
        backup="${target}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "${target}" "${backup}"
        warn "${name} already exists, backed up to $(basename "${backup}")"

        if [ "${level}" = "global" ] && [ -t 0 ] && [ -t 1 ]; then
            echo ""
            echo -e "  ${BOLD}${YELLOW}⚠  $name 存在自定义内容，请选择操作：${NC}"
            echo -e "    ${BOLD}[O]${NC} 覆盖      - 用插件内容替换（原内容已备份）"
            echo -e "    ${BOLD}[M]${NC} 合并      - 插件内容置顶，保留原自定义内容"
            echo -e "    ${BOLD}[S]${NC} 跳过      - 保持现有文件不变"
            printf "  ${BOLD}${CYAN}→${NC} ${BOLD}请输入选择 [O/M/S]:${NC} "
            read -r choice < /dev/tty
            case "${choice}" in
                [Mm]*)
                    cat "${tmpfile}" > "${target}.new"
                    echo "" >> "${target}.new"
                    echo "<!-- === User custom content below === -->" >> "${target}.new"
                    echo "" >> "${target}.new"
                    cat "${target}" >> "${target}.new"
                    mv "${target}.new" "${target}"
                    ok "${name} (merged with backup)"
                    rm -f "${tmpfile}"
                    return 0
                    ;;
                [Ss]*)
                    info "${name} skipped (backup preserved)"
                    rm -f "${tmpfile}"
                    return 0
                    ;;
                *) ;;
            esac
        fi
    fi

    mv "${tmpfile}" "${target}"
    if [ "${level}" = "global" ]; then
        ok "${name} (absolute paths for global mode)"
    else
        ok "${name} (absolute paths for project mode)"
    fi
}

BRAND="cannbot"
VERSION="1.0.0"
# Supported target tools — single source of truth (modify-init.md 红线 1).
# Subclass init.sh queries this via `--list-tools` instead of hardcoding.
SUPPORTED_TOOLS=("opencode" "claude" "codex" "dsh")

# ============================================================
# Third-party repository registry
# ============================================================
# Format: "name|url|ref|update_strategy|post_hook"
#   name:             repo name (used in --{name} option and DIR variable)
#   url:              git clone URL
#   ref:              branch/tag/commit to checkout (empty = no checkout)
#   update_strategy:  "fetch_merge" | "pull" | "pull_rebase"
#   post_hook:        function to call after setup (empty = none)
REPO_REGISTRY=(
    "asc-devkit|https://gitcode.com/cann/asc-devkit.git|31f3ab38|fetch_merge|post_setup_asc_devkit"
    "cann-samples|https://gitcode.com/cann/cann-samples.git||pull|"
    "ops-tensor|https://gitcode.com/cann/ops-tensor.git||pull|"
)

# Post-setup hook for asc-devkit: clean markdown files
post_setup_asc_devkit() {
    local repo_dir="$1"
    CLEAN_SCRIPT="${SHARED_SKILL_ROOT}/ascendc-docs-search/scripts/clean_markdown.py"
    if [ -f "${CLEAN_SCRIPT}" ]; then
        python3 "${CLEAN_SCRIPT}" --dir "${repo_dir}" --no-backup --quiet > /dev/null 2>&1 \
            || warn "markdown cleanup failed"
    else
        warn "clean_markdown.py not found, skipping"
    fi
}

# Setup a third-party repository
# $1 = repo name  $2 = url  $3 = ref  $4 = update_strategy  $5 = post_hook  $6 = local_path (optional)
setup_repo() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local strategy="$4"
    local post_hook="$5"
    local local_path="$6"

    local repo_dir="${CANNBOT_MID_DIR}/${name}"

    # Use local path if provided
    if [ -n "${local_path}" ]; then
        validate_path "${local_path}" "--${name}"
        rm -rf "${repo_dir}"
        ln -sfn "$(realpath "${local_path}")" "${repo_dir}"
        ok "${name} → $(realpath "${local_path}") (local, linked)"

        if [ -n "${post_hook}" ] && type "${post_hook}" >/dev/null 2>&1; then
            "${post_hook}" "${repo_dir}"
        fi
        return 0
    fi

    # Update existing repo
    if [ -d "${repo_dir}/.git" ]; then
        cd "${repo_dir}"
        git checkout . 2>/dev/null || true

        case "${strategy}" in
            fetch_merge)
                git fetch --quiet 2>/dev/null || true
                if [ -n "${ref}" ]; then
                    git checkout --quiet "${ref}" 2>/dev/null || warn "checkout ${ref} failed, using existing"
                fi
                ;;
            pull)
                git pull --quiet 2>/dev/null || warn "git pull failed, using existing version"
                ;;
            pull_rebase)
                git pull --quiet --rebase 2>/dev/null || warn "git pull --rebase failed, using existing version"
                ;;
        esac

        cd "${SCRIPT_DIR}"
        if [ -n "${ref}" ]; then
            ok "${name} updated (@${ref})"
        else
            ok "${name} updated"
        fi
    else
        # Clone new repo
        local clone_opts=""
        [ "${strategy}" = "pull" ] && clone_opts="--depth 1"

        if git clone --quiet ${clone_opts} "${url}" "${repo_dir}" 2>/dev/null; then
            if [ -n "${ref}" ]; then
                cd "${repo_dir}" && git checkout --quiet "${ref}" 2>/dev/null || warn "checkout ${ref} failed"
                cd "${SCRIPT_DIR}"
                ok "${name} cloned (@${ref})"
            else
                ok "${name} cloned"
            fi
        else
            warn "git clone failed, skipping ${name}"
            return 0
        fi
    fi

    # Run post-hook if specified
    if [ -n "${post_hook}" ] && type "${post_hook}" >/dev/null 2>&1; then
        "${post_hook}" "${repo_dir}"
    fi
}

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_NAME="$(basename "${SCRIPT_DIR}")"
PLUGIN_ROOT="${SCRIPT_DIR}"
LOCAL_AGENT_ROOT="${PLUGIN_ROOT}/agents"
LOCAL_SKILL_ROOT="${PLUGIN_ROOT}/skills"
SHARED_SKILL_ROOT="$(cd "${PLUGIN_ROOT}/../../ops" 2>/dev/null && pwd || true)"
INFRA_SKILL_ROOT="$(cd "${PLUGIN_ROOT}/../../infra" 2>/dev/null && pwd || true)"

# ============================================================
# Usage
# ============================================================
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
    echo -e "  ${BOLD}cuda2ascend Installer${NC}"
    echo ""
}

show_help() {
    cat << EOF
cuda2ascend - base workflow installer

Usage: init.sh [level] [tool] [install_path] [options]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: one of ${SUPPORTED_TOOLS[*]} (default: ${SUPPORTED_TOOLS[0]})
  install_path - Project-level install directory (default: current working directory)

Options:
  --override <dir>         - Override root directory: use skills under <dir>/skills/ to
                                replace linked skills of the same name (adds any not
                                already present in the base; errors if there is no
                                skills/ subdirectory). If <dir>/AGENTS.md exists, it is
                                linked as the PM entry instead of the base AGENTS.md
  --plugin-enable <name> <on|off>
                             - Enable or disable a pluggable workflow plugin (plugin-* skill)
                                in .cannbot/settings.json (plugins.<name>.enabled)
  --mode <interactive|silent>
                           - Workflow mode written to .cannbot/settings.json:
                                interactive (default) = questionnaire prompts & progress
                                reports stay as-is; silent = unattended run (no prompts,
                                no progress output, only permission warning, final summary
                                and blocked reports). Existing mode is kept when omitted.
  --list-tools             - Print supported target tools (used by subclass init scripts)
  --repo <name>:<path>     - Use a locally cloned third-party repository (linked
                                instead of cloned). Can be specified multiple times,
                                e.g. --repo asc-devkit:... --repo cann-samples:...
                                Supported repo names: asc-devkit / cann-samples / ops-tensor
  --help                   - Show this help message

Examples:
  init.sh                                  # project-level opencode
  init.sh claude                           # project-level claude
  init.sh global                           # global-level
  init.sh project /path/to/proj            # custom install directory
  init.sh project claude /path/to/proj     # custom install directory + claude
  init.sh --repo asc-devkit:~/repos/asc-devkit --repo cann-samples:~/repos/cann-samples
  init.sh project /path/to/repo --override /path/to/repo/agent
  init.sh --plugin-enable plugin-pr-submit off

Installation paths:
  opencode: .opencode/{skills,agents,plugin}/   + AGENTS.md in project root
  claude:   .claude/{skills,agents,hooks}/      + settings.json + AGENTS.md/CLAUDE.md in project root
  codex:    .agents/skills/ + .codex/agents/     + AGENTS.md in project root (no permission hook)
  dsh:      .dsh/skills/ + .dsh/agents/          + AGENTS.md in project root
                skills discovered natively at <projectRoot>/.dsh/skills (DSH project-dsh root);
                agents are role reference only (DSH subagents are prompt-driven);
                no project-level permission hook — optional deployment-level guard:
                hooks/dsh/install.sh (Cordis plugin via $DSH_HOME/cordis.patch.yml)

Intermediate directory:
  .cannbot/              workflow intermediate files & state
  .cannbot/asc-devkit    asc-devkit repo
  .cannbot/cann-samples  cann-samples repo
  .cannbot/ops-tensor    ops-tensor repo
  .cannbot/settings.json      workflow config & plugin registration (mode, surveyed, plugins)
EOF
}

# ============================================================
# Parse arguments
# ============================================================
LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""
OVERRIDE_SKILL_DIR=""
OVERRIDE_DIR=""
MODE=""

# Local path overrides for registered repos, keyed by repo name.
declare -A REPO_LOCAL

# Check whether a name is a registered repo (returns 0 if yes)
is_registered_repo() {
    local target="$1"
    local entry
    for entry in "${REPO_REGISTRY[@]}"; do
        [ "${entry%%|*}" = "${target}" ] && return 0
    done
    return 1
}

# Check whether a name is a supported target tool (returns 0 if yes)
is_supported_tool() {
    local target="$1"
    local t
    for t in "${SUPPORTED_TOOLS[@]}"; do
        [ "${t}" = "${target}" ] && return 0
    done
    return 1
}

POSITIONAL=()
while [ $# -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --list-tools)
            echo "${SUPPORTED_TOOLS[*]}"; exit 0 ;;
        --help)
            show_help; exit 0 ;;
        --override)
            OVERRIDE_DIR="$2"; shift 2; continue ;;
        --override=*)
            OVERRIDE_DIR="${1#*=}"; shift; continue ;;
        --mode)
            MODE="$2"; shift 2; continue ;;
        --mode=*)
            MODE="${1#*=}"; shift; continue ;;
        --repo)
            local_arg="$2"; shift 2
            if [[ "${local_arg}" != *:* ]]; then
                err "--repo format invalid: must be name:/path, got: ${local_arg}"
                exit 1
            fi
            local_name="${local_arg%%:*}"
            local_path="${local_arg#*:}"
            if ! is_registered_repo "${local_name}"; then
                err "--repo unknown name: ${local_name} (valid: ${REPO_REGISTRY[*]%%|*})"
                exit 1
            fi
            REPO_LOCAL["${local_name}"]="${local_path}"
            continue ;;
        --plugin-enable)
            PLUGIN_ENABLE_NAME="$2"
            PLUGIN_ENABLE_STATE="$3"
            if [ -z "${PLUGIN_ENABLE_NAME}" ] || [ -z "${PLUGIN_ENABLE_STATE}" ]; then
                err "--plugin-enable format invalid: must be <name> <on|off>"
                exit 1
            fi
            if [ "${PLUGIN_ENABLE_STATE}" != "on" ] && [ "${PLUGIN_ENABLE_STATE}" != "off" ]; then
                err "--plugin-enable state must be on or off, got: ${PLUGIN_ENABLE_STATE}"
                exit 1
            fi
            shift 3; continue ;;
        --repo=*)
            local_arg="${1#*=}"; shift
            if [[ "${local_arg}" != *:* ]]; then
                err "--repo format invalid: must be name:/path, got: ${local_arg}"
                exit 1
            fi
            local_name="${local_arg%%:*}"
            local_path="${local_arg#*:}"
            if ! is_registered_repo "${local_name}"; then
                err "--repo unknown name: ${local_name} (valid: ${REPO_REGISTRY[*]%%|*})"
                exit 1
            fi
            REPO_LOCAL["${local_name}"]="${local_path}"
            continue ;;
        --*)
            err "unknown option: $1"
            exit 1 ;;
        global|project)
            LEVEL="$1"; shift; continue ;;
        opencode)      TOOL="$arg"; shift; continue ;;
        claude)        TOOL="$arg"; shift; continue ;;
        codex)         TOOL="$arg"; shift; continue ;;
        dsh)           TOOL="$arg"; shift; continue ;;
        *)
            # 兜底：动态校验 SUPPORTED_TOOLS，支持未来新增工具
            if is_supported_tool "$1"; then
                TOOL="$1"; shift; continue
            fi
            POSITIONAL+=("$1"); shift; continue ;;
    esac
done

# 位置参数只允许剩一个（install_path）。多余位置参数通常是未知工具名
# （如未来新增工具而基类未升级），报错而非静默误当安装路径。
if [ ${#POSITIONAL[@]} -gt 1 ]; then
    err "无法识别的参数: ${POSITIONAL[*]:0:$((${#POSITIONAL[@]}-1))}（当前支持的工具: ${SUPPORTED_TOOLS[*]}）"
    exit 1
fi
if [ ${#POSITIONAL[@]} -eq 1 ]; then
    INSTALL_PATH="${POSITIONAL[0]}"
fi

# Normalize & validate override dir (absolute path for stable symlinks)
if [ -n "${OVERRIDE_DIR}" ]; then
    validate_path "${OVERRIDE_DIR}" "--override"
    if [ ! -d "${OVERRIDE_DIR}/skills" ]; then
        err "--override path has no skills/ subdirectory: ${OVERRIDE_DIR}"
        exit 1
    fi
    OVERRIDE_SKILL_DIR="$(cd "${OVERRIDE_DIR}/skills" && pwd)"
fi

# Validate workflow mode (interactive | silent)
if [ -n "${MODE}" ] && [ "${MODE}" != "interactive" ] && [ "${MODE}" != "silent" ]; then
    err "--mode invalid: ${MODE} (valid: interactive / silent)"
    exit 1
fi

# ============================================================
# Determine config root
# ============================================================
if [ "${LEVEL}" = "global" ]; then
    if [ "${TOOL}" = "opencode" ]; then
        CONFIG_ROOT="${HOME}/.config/opencode"
    elif [ "${TOOL}" = "claude" ]; then
        CONFIG_ROOT="${HOME}/.claude"
    elif [ "${TOOL}" = "codex" ]; then
        CONFIG_ROOT="${HOME}/.codex"
    elif [ "${TOOL}" = "dsh" ]; then
        # DSH（DeepSeek Harness）用户数据根：$DSH_HOME，默认 ~/.dsh
        # （对齐 dsh-home-paths 的 resolveDshHome 优先级：显式配置 > $DSH_HOME > ~/.dsh）
        CONFIG_ROOT="${DSH_HOME:-${HOME}/.dsh}"
    fi
    INSTALL_BASE="${HOME}"
else
    if [ -n "${INSTALL_PATH}" ]; then
        validate_path "${INSTALL_PATH}" "install_path"
        INSTALL_BASE="$(cd "${INSTALL_PATH}" && pwd)"
    else
        INSTALL_BASE="${PWD}"
    fi
    if [ "${TOOL}" = "opencode" ]; then
        CONFIG_ROOT="${INSTALL_BASE}/.opencode"
    elif [ "${TOOL}" = "claude" ]; then
        CONFIG_ROOT="${INSTALL_BASE}/.claude"
    elif [ "${TOOL}" = "codex" ]; then
        CONFIG_ROOT="${INSTALL_BASE}/.codex"
    elif [ "${TOOL}" = "dsh" ]; then
        # DSH 项目级配置根：<install>/.dsh。
        # 其下的 skills/ 正是 DSH 的 project-dsh 发现根（<projectRoot>/.dsh/skills，
        # 最近 .git 祖先判定，rank 100），无需额外配置即可被扫描。
        CONFIG_ROOT="${INSTALL_BASE}/.dsh"
    fi
fi

CANNBOT_DIR="${CONFIG_ROOT}"
CANNBOT_MID_DIR="${INSTALL_BASE}/.cannbot"

# Codex discovers skills from .agents/skills/ (not .codex/skills/).
# Skills and agents are linked to separate discovery roots.
#
# dsh（DeepSeek Harness）走默认分支即正确：
#   - skills 链接到 <install>/.dsh/skills —— 即 DSH 的 project-dsh 发现根
#     （<projectRoot>/.dsh/skills，rank 100；global 为 $DSH_HOME/skills，rank 400）
#   - agents 链接到 <install>/.dsh/agents —— DSH 无原生 agent 注册（子 Agent 纯
#     prompt 驱动），此处仅为角色定义参考文件，供 PM 读取后组装子 Agent prompt
SKILL_DISCOVERY_ROOT="${CANNBOT_DIR}/skills"
AGENT_DISCOVERY_ROOT="${CANNBOT_DIR}/agents"
if [ "${TOOL}" = "codex" ]; then
    if [ "${LEVEL}" = "global" ]; then
        SKILL_DISCOVERY_ROOT="${HOME}/.agents/skills"
    else
        SKILL_DISCOVERY_ROOT="${INSTALL_BASE}/.agents/skills"
    fi
    AGENT_DISCOVERY_ROOT="${CANNBOT_DIR}/agents"
fi

# ============================================================
# Clean up legacy dirs
# ============================================================
if [ -e "${CONFIG_ROOT}/${BRAND}" ] || [ -L "${CONFIG_ROOT}/${BRAND}" ]; then
    rm -rf "${CONFIG_ROOT}/${BRAND}"
fi
if [ -L "${CONFIG_ROOT}/teams" ]; then
    rm -f "${CONFIG_ROOT}/teams"
fi

# ============================================================
# Dependency resolution
# ============================================================

# Parse YAML list from a config file (handles `skills:` / `agents:` sections)
parse_yaml_list() {
    local file="$1"
    local key="$2"
    awk -v key="${key}" '
        $0 ~ "^"key":" { flag=1; next }
        /^[^ ]/ { flag=0 }
        flag && /^ +- / { sub(/^ +- /, ""); print }
    ' "${file}" 2>/dev/null | grep -v '^$' || true
}

# Collect all agent files under agents/ (recurse into subdirs), flattened.
collect_agents() {
    AGENT_FILES=()
    while IFS= read -r f; do
        AGENT_FILES+=("${f}")
    done < <(find "${LOCAL_AGENT_ROOT}" -type f -name '*.md' 2>/dev/null | sort)
}

# Resolve skill source path: local skills/ first (plugin-* nests sub-skills one level
# down, e.g. plugin-<name>/<sub-skill>), then shared ops/, then infra/.
resolve_skill_src() {
    local skill="$1"
    if [ -d "${LOCAL_SKILL_ROOT}/${skill}" ]; then
        echo "${LOCAL_SKILL_ROOT}/${skill}"; return
    fi
    local plugin_dir
    for plugin_dir in "${LOCAL_SKILL_ROOT}"/plugin-*/; do
        [ -d "${plugin_dir}" ] || continue
        if [ -d "${plugin_dir}${skill}" ] && [ -f "${plugin_dir}${skill}/SKILL.md" ]; then
            echo "${plugin_dir}${skill}"; return
        fi
    done
    # Skills from the override dir (linked in the override pass) — resolve them
    # here too so frontmatter-registered override skills are not flagged missing.
    if [ -n "${OVERRIDE_SKILL_DIR}" ]; then
        if [ -d "${OVERRIDE_SKILL_DIR}/${skill}" ]; then
            echo "${OVERRIDE_SKILL_DIR}/${skill}"; return
        fi
        for plugin_dir in "${OVERRIDE_SKILL_DIR}"/plugin-*/; do
            [ -d "${plugin_dir}" ] || continue
            if [ -d "${plugin_dir}${skill}" ] && [ -f "${plugin_dir}${skill}/SKILL.md" ]; then
                echo "${plugin_dir}${skill}"; return
            fi
        done
    fi
    if [ -n "${SHARED_SKILL_ROOT}" ] && [ -d "${SHARED_SKILL_ROOT}/${skill}" ]; then
        echo "${SHARED_SKILL_ROOT}/${skill}"; return
    fi
    if [ -n "${INFRA_SKILL_ROOT}" ] && [ -d "${INFRA_SKILL_ROOT}/${skill}" ]; then
        echo "${INFRA_SKILL_ROOT}/${skill}"; return
    fi
    echo ""
}

# Collect skills to link: union of
#   - all skills under the plugin's skills/ (linked by default), and
#   - skills registered by AGENTS.md and every agent's frontmatter `skills:`.
# Deduplicated.
collect_skills() {
    local raw=""
    # 1) All local skills under skills/ (default-linked); plugin-* may nest
    #    sub-skills one level down (e.g. plugin-<name>/<sub-skill>) — those
    #    are linked at top level as well.
    if [ -d "${LOCAL_SKILL_ROOT}" ]; then
        for skill_dir in "${LOCAL_SKILL_ROOT}"/*/; do
            [ -d "${skill_dir}" ] || continue
            case "$(basename "${skill_dir}")" in
                plugin-*)
                    # plugin-* 须根有 SKILL.md 才算插件，其嵌套子 skill 顶层链接
                    [ -f "${skill_dir}/SKILL.md" ] || continue
                    raw="${raw}
$(basename "${skill_dir}")"
                    for sub_dir in "${skill_dir}"*/; do
                        [ -d "${sub_dir}" ] || continue
                        [ -f "${sub_dir}/SKILL.md" ] || continue
                        raw="${raw}
$(basename "${sub_dir}")"
                    done
                    ;;
                *)
                    raw="${raw}
$(basename "${skill_dir}")"
                    ;;
            esac
        done
    fi
    # 2) Skills registered by AGENTS.md + each agent frontmatter.
    #    With --override, an AGENTS.md in the override dir wins as the PM entry.
    local agents_file="${PLUGIN_ROOT}/AGENTS.md"
    if [ -n "${OVERRIDE_DIR}" ] && [ -f "${OVERRIDE_DIR}/AGENTS.md" ]; then
        agents_file="${OVERRIDE_DIR}/AGENTS.md"
    fi
    [ -f "${agents_file}" ] && raw="${raw}
$(parse_yaml_list "${agents_file}" "skills")"
    for f in "${AGENT_FILES[@]}"; do
        local s
        s="$(parse_yaml_list "${f}" "skills")"
        [ -n "${s}" ] && raw="${raw}
${s}"
    done
    ALL_SKILLS=$(echo "${raw}" | sort -u | grep -v '^$' || true)
    if [ -z "${ALL_SKILLS}" ]; then
        ALL_SKILL_COUNT=0
    else
        ALL_SKILL_COUNT=$(echo "${ALL_SKILLS}" | grep -c '.')
    fi
    return 0
}

# ============================================================
# Main
# ============================================================

show_banner
echo "  Team:      ${TEAM_NAME}"
echo "  Tool:      ${TOOL}"
echo "  Level:     ${LEVEL}"
echo "  Config:    ${CONFIG_ROOT}"
echo "  .cannbot:    ${CANNBOT_MID_DIR}"
echo ""

collect_agents
collect_skills

# ============================================================
# Step 1: Create .cannbot directory
# ============================================================
step "[1/6] Creating .cannbot directory..."
mkdir -p "${CANNBOT_MID_DIR}"
ok ".cannbot → ${CANNBOT_MID_DIR}"
echo ""

# ============================================================
# Step 2: Install config file (AGENTS.md)
# ============================================================
step "[2/6] Installing configuration..."
mkdir -p "${CONFIG_ROOT}"

config_src="${PLUGIN_ROOT}/AGENTS.md"
# 子仓可用 --override 目录下的 AGENTS.md 覆写基类 PM 入口
if [ -n "${OVERRIDE_DIR}" ] && [ -f "${OVERRIDE_DIR}/AGENTS.md" ]; then
    config_src="$(cd "${OVERRIDE_DIR}" && pwd)/AGENTS.md"
fi
config_name="AGENTS.md"
if [ "${LEVEL}" = "project" ]; then
    config_target="${INSTALL_BASE}/${config_name}"
else
    config_target="${CONFIG_ROOT}/${config_name}"
fi

if [ ! -f "${config_src}" ]; then
    warn "AGENTS.md not found in plugin root, skipping config"
elif [ "${config_src}" = "${config_target}" ]; then
    info "${config_name} already at target location"
else
    ln -sf "${config_src}" "${config_target}"
    ok "${config_name} → ${config_target}"
fi

# Claude Code 主记忆文件为 CLAUDE.md，链接到同一 PM 配置源
if [ "${TOOL}" = "claude" ] && [ -f "${config_src}" ]; then
    claude_target="$(dirname "${config_target}")/CLAUDE.md"
    if [ "${config_src}" = "${claude_target}" ]; then
        info "CLAUDE.md already at target location"
    else
        ln -sf "${config_src}" "${claude_target}"
        ok "CLAUDE.md → ${claude_target}"
    fi
fi
echo ""

# ============================================================
# Step 3: Link agents (flattened)
# ============================================================
step "[3/6] Linking agents..."
AGENTS_LINK_DIR="${AGENT_DISCOVERY_ROOT}"
rm -rf "${AGENTS_LINK_DIR}"
mkdir -p "${AGENTS_LINK_DIR}"

agent_count=0
if [ "${TOOL}" = "codex" ]; then
    # Codex uses standalone TOML agent definitions (agents/codex/*.toml).
    # Each TOML has a __CANNBOT_AGENT_SOURCE__ placeholder resolved to the
    # canonical .md agent file, so codex reads the same role instructions.
    CODEX_AGENT_ROOT="${PLUGIN_ROOT}/hooks/codex"
    if [ -d "${CODEX_AGENT_ROOT}" ]; then
        for toml in "${CODEX_AGENT_ROOT}"/*.toml; do
            [ -f "${toml}" ] || continue
            name="$(basename "${toml}")"
            base="${name%.toml}"
            canonical="${LOCAL_AGENT_ROOT}/${base}.md"
            tmpfile=$(mktemp)
            sed "s|__CANNBOT_AGENT_SOURCE__|${canonical}|g" "${toml}" > "${tmpfile}"
            mv "${tmpfile}" "${AGENTS_LINK_DIR}/${name}"
            agent_count=$((agent_count + 1))
        done
        ok "Agents: ${agent_count} codex TOML files"
    else
        warn "No codex agent TOMLs found at ${CODEX_AGENT_ROOT}, skipping agents"
    fi
else
    for f in "${AGENT_FILES[@]}"; do
        link_name="$(basename "${f}")"
        if [ -e "${AGENTS_LINK_DIR}/${link_name}" ] || [ -L "${AGENTS_LINK_DIR}/${link_name}" ]; then
            warn "agent name conflict, overwriting: ${link_name}"
        fi
        ln -sfn "$(realpath "${f}")" "${AGENTS_LINK_DIR}/${link_name}"
        agent_count=$((agent_count + 1))
    done
    ok "Agents: ${agent_count} linked (flattened)"
fi
if [ "${TOOL}" = "dsh" ]; then
    info "dsh: agents linked to .dsh/agents/ as role reference (DSH subagents are prompt-driven, no native agent registration)"
fi
echo ""

# ============================================================
# Step 4: Link skills registered by agents
# ============================================================
step "[4/6] Linking skills..."
SKILLS_LINK_DIR="${SKILL_DISCOVERY_ROOT}"
if [ "${TOOL}" = "dsh" ] && [ "${LEVEL}" = "global" ]; then
    warn "global 安装将重建 ${SKILLS_LINK_DIR}（DSH 的 user-dsh 技能根，rank 400）——"
    warn "该目录下既有的非 cannbot 全局技能会被移除（重跑 init 前请确认无自定义技能）"
fi
rm -rf "${SKILLS_LINK_DIR}"
mkdir -p "${SKILLS_LINK_DIR}"

skill_count=0
missing_skills=""
for skill in ${ALL_SKILLS}; do
    src="$(resolve_skill_src "${skill}")"
    if [ -n "${src}" ]; then
        ln -sfn "$(realpath "${src}")" "${SKILLS_LINK_DIR}/${skill}"
        skill_count=$((skill_count + 1))
    else
        missing_skills="${missing_skills} ${skill}"
    fi
done
ok "Skills: ${skill_count} linked"
[ -n "${missing_skills}" ] && warn "Missing skills:${missing_skills}"

# Override pass: replace/add domain skills from --override/<dir>/skills (subclass repos).
# Runs right after the base skills are linked, so overriding skills win by same name.
if [ -n "${OVERRIDE_SKILL_DIR}" ]; then
    override_replaced=0
    override_added=0
    for skill_dir in "${OVERRIDE_SKILL_DIR}"/*/; do
        [ -d "${skill_dir}" ] || continue
        [ -f "${skill_dir}/SKILL.md" ] || continue
        name="$(basename "${skill_dir}")"
        target="${SKILLS_LINK_DIR}/${name}"
        if [ -e "${target}" ] || [ -L "${target}" ]; then
            rm -rf "${target}"
            ln -sfn "$(realpath "${skill_dir}")" "${target}"
            override_replaced=$((override_replaced + 1))
        else
            ln -sfn "$(realpath "${skill_dir}")" "${target}"
            skill_count=$((skill_count + 1))
            override_added=$((override_added + 1))
        fi
        # plugin-* from override may nest sub-skills; link them at top level too
        case "${name}" in
            plugin-*)
                for sub_dir in "${skill_dir}"*/; do
                    [ -d "${sub_dir}" ] || continue
                    [ -f "${sub_dir}/SKILL.md" ] || continue
                    sub_name="$(basename "${sub_dir}")"
                    sub_target="${SKILLS_LINK_DIR}/${sub_name}"
                    if [ -e "${sub_target}" ] || [ -L "${sub_target}" ]; then
                        rm -rf "${sub_target}"
                    fi
                    ln -sfn "$(realpath "${sub_dir}")" "${sub_target}"
                done
                ;;
        esac
    done
    if [ "${override_replaced}" -eq 0 ] && [ "${override_added}" -eq 0 ]; then
        warn "Override dir has no skills (expected subdirs with SKILL.md): ${OVERRIDE_SKILL_DIR}"
    else
        info "Overridden from ${OVERRIDE_SKILL_DIR}: ${override_replaced} replaced, ${override_added} added"
    fi
fi
echo ""

# ============================================================
# Step 4.5: Generate permission config from workflow-agent-permissions skill
# ============================================================
step "[4.5] Generating permission config..."
PERM_TEMPLATE="${SKILLS_LINK_DIR}/workflow-agent-permissions/hooks"
PERM_TARGET="${CANNBOT_MID_DIR}/permissions"
if [ ! -d "${PERM_TEMPLATE}" ]; then
    warn "workflow-agent-permissions template not found, hook will use built-in defaults"
elif [ -d "${PERM_TARGET}" ]; then
    perm_count=$(ls -1A "${PERM_TARGET}" 2>/dev/null | wc -l)
    info "permissions/ already exists (${perm_count} files), keeping workspace config"
else
    mkdir -p "${CANNBOT_MID_DIR}"
    cp -r "${PERM_TEMPLATE}" "${PERM_TARGET}"
    perm_count=$(ls -1A "${PERM_TARGET}" | wc -l)
    ok "permissions/ generated from workflow-agent-permissions skill (${perm_count} files)"
fi
echo ""

# Step 4+: Link permission-guard hook (tool-specific)
# ============================================================
if [ "${TOOL}" = "claude" ]; then
    # Claude Code：PreToolUse hook，脚本链接到 .claude/hooks/，
    # 并在 settings.json 幂等注册 hook 组
    CLAUDE_HOOK_SRC="${PLUGIN_ROOT}/hooks/claude/permission-guard.js"
    if [ -f "${CLAUDE_HOOK_SRC}" ]; then
        HOOK_LINK_DIR="${CONFIG_ROOT}/hooks"
        mkdir -p "${HOOK_LINK_DIR}"
        ln -sfn "$(realpath "${CLAUDE_HOOK_SRC}")" "${HOOK_LINK_DIR}/permission-guard.js"
        ok "claude hook: permission-guard.js linked"

        SETTINGS_FILE="${CONFIG_ROOT}/settings.json"
        # project 级用 ${CLAUDE_PROJECT_DIR} 占位符（随项目迁移不失效）；
        # global 级项目目录不固定，用绝对路径
        HOOK_REF='${CLAUDE_PROJECT_DIR}/.claude/hooks/permission-guard.js'
        [ "${LEVEL}" = "global" ] && HOOK_REF="${HOOK_LINK_DIR}/permission-guard.js"
        if command -v python3 > /dev/null 2>&1; then
            python3 - "${SETTINGS_FILE}" "${HOOK_REF}" << 'SETTINGS_PY'
import json, os, sys

path, hook_ref = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"  [warn] settings.json 内容不是 JSON 对象，保留原文件未注册 hook: {path}", file=sys.stderr)
            sys.exit(0)
    except Exception:
        print(f"  [warn] settings.json 解析失败，保留原文件未注册 hook: {path}", file=sys.stderr)
        sys.exit(0)

pre = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
for group in pre:
    matcher = group.get("matcher", "")
    for h in group.get("hooks", []):
        blob = h.get("command", "") + " " + " ".join(h.get("args", []))
        if "permission-guard.js" in blob:
            if "Question" in matcher:
                print("  [info] settings.json 已注册 permission-guard hook，跳过")
            else:
                group["matcher"] = matcher + "|Question"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                print("  [ok] settings.json 已注册 permission-guard hook（matcher 补充 Question，静默问卷拦截生效）")
            sys.exit(0)

pre.append({
    "matcher": "Write|Edit|MultiEdit|NotebookEdit|Question",
    "hooks": [{
        "type": "command",
        "command": "node",
        "args": [hook_ref],
    }],
})
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("  [ok] settings.json 注册 permission-guard PreToolUse hook")
SETTINGS_PY
        else
            warn "python3 not found, 请手动在 ${SETTINGS_FILE} 注册 PreToolUse hook（见 hooks/claude/permission-guard.js 注释）"
        fi
    else
        warn "claude permission-guard.js not found, skipping"
    fi
    echo ""
elif [ "${TOOL}" = "codex" ]; then
    # Codex does not support PreToolUse / tool.execute.before hooks.
    # Permission enforcement is downgraded to prompt-only constraints.
    echo -e "  ${YELLOW}${BOLD}⚠ WARNING: Codex does not support permission-guard hooks.${NC}"
    echo -e "  ${DIM}Role-based write restrictions (PM only-schedule, developer-code code-only, etc.)${NC}"
    echo -e "  ${DIM}are enforced by AGENTS.md prompt, NOT by a hard hook. Subagent directory${NC}"
    echo -e "  ${DIM}isolation is best-effort, not guaranteed.${NC}"
    echo ""
elif [ "${TOOL}" = "dsh" ]; then
    # DSH（DeepSeek Harness）无项目级 PreToolUse / tool.execute.before 拦截点，
    # 不部署 permission-guard hook。角色写权限隔离默认由 AGENTS.md prompt 约束
    # （非机制保证）；DSH 自身的沙箱模式与审批策略（sandbox / approval）在会话/
    # 部署层配置，不属于本项目文件机制。.cannbot/permissions/ 仍生成，供 PM 启动
    # 闸口与 workflow-agent-permissions skill 使用。
    # 可选机制升级：hooks/dsh/install.sh 安装部署级 Cordis 守卫插件（挂载到
    # $DSH_HOME/cordis.patch.yml，对所有 profile 生效，仅作用于 cuda2ascend
    # 初始化的工作区），恢复按角色的写权限隔离与静默问卷拦截。
    echo -e "  ${YELLOW}${BOLD}⚠ WARNING: DSH does not support project-level permission-guard hooks.${NC}"
    echo -e "  ${DIM}Role-based write restrictions (PM only-schedule, developer-code code-only, etc.)${NC}"
    echo -e "  ${DIM}are enforced by AGENTS.md prompt, NOT by a hard hook. DSH sandbox / approval${NC}"
    echo -e "  ${DIM}policies are configured at session/deployment level, not by project files.${NC}"
    echo -e "  ${DIM}可选机制升级: ${GREEN}hooks/dsh/install.sh${NC}${DIM} 安装部署级权限守卫${NC}"
    echo -e "  ${DIM}(Cordis 插件 → ${DIM}\$DSH_HOME/cordis.patch.yml，恢复角色写权限隔离与静默问卷拦截)${NC}"
    echo ""
else
    OC_PLUGIN_SRC="${PLUGIN_ROOT}/hooks/opencode/permission-guard.js"
    if [ -f "${OC_PLUGIN_SRC}" ]; then
        PLUGIN_LINK_DIR="${CONFIG_ROOT}/plugin"
        mkdir -p "${PLUGIN_LINK_DIR}"
        ln -sfn "$(realpath "${OC_PLUGIN_SRC}")" "${PLUGIN_LINK_DIR}/permission-guard.js"
        ok "opencode plugin: permission-guard.js linked"
    else
        warn "opencode permission-guard.js not found, skipping"
    fi
    echo ""
fi

# ============================================================

# ============================================================
# Step 5.5: Generate workflow config (.cannbot/settings.json)
#   settings.json 是工作流运行时配置的唯一文件：mode（工作流模式）、surveyed（插件启用
#   是否已询问）、plugins（插件注册信息：hook/stages/standalone/enabled，由 plugin-* 的
#   frontmatter 扫描生成，重扫重写保留 enabled/surveyed、并入新增、剔除失效）、元数据。
#   旧版 .cannbot/plugin-registry.json 存在时一次性迁移并入（迁移后不再生成该文件）。
# ============================================================
step "[5.5] Generating workflow config..."
wf_rc=""
if command -v python3 > /dev/null 2>&1; then
    # --- 扫描插件 frontmatter（基类 + override，override 同名优先）---
    parse_yaml_value() {
        local file="$1"
        local key="$2"
        awk -v key="${key}" '
            $0 ~ "^"key":" { sub("^"key":[[:space:]]*", ""); gsub(/^"|"$/, ""); print; exit }
        ' "${file}" 2>/dev/null || true
    }
    PLUGIN_RECORDS="$(mktemp)"
    : > "${PLUGIN_RECORDS}"
    declare -A PLUGIN_SRC=()
    if [ -d "${LOCAL_SKILL_ROOT}" ]; then
        for pdir in "${LOCAL_SKILL_ROOT}"/plugin-*/; do
            [ -d "${pdir}" ] || continue
            if [ ! -f "${pdir}/SKILL.md" ]; then
                warn "plugin-* dir without SKILL.md skipped: $(basename "${pdir}")"
                continue
            fi
            PLUGIN_SRC["$(basename "${pdir}")"]="$(cd "${pdir}" && pwd)"
        done
    fi
    if [ -n "${OVERRIDE_SKILL_DIR}" ] && [ -d "${OVERRIDE_SKILL_DIR}" ]; then
        for pdir in "${OVERRIDE_SKILL_DIR}"/plugin-*/; do
            [ -d "${pdir}" ] || continue
            if [ ! -f "${pdir}/SKILL.md" ]; then
                warn "plugin-* dir without SKILL.md skipped: $(basename "${pdir}")"
                continue
            fi
            PLUGIN_SRC["$(basename "${pdir}")"]="$(cd "${pdir}" && pwd)"
        done
    fi
    # 校验：workflow-hook 格式与挂载点存在性（对照基类流程表）、workflow-stages 必填
    FLOW_SKILL="${LOCAL_SKILL_ROOT}/ops-direct-invoke-workflow/SKILL.md"
    flow_stages=""
    if [ -f "${FLOW_SKILL}" ]; then
        flow_stages=$(awk -F'|' '/^\| *[⛔]? *[0-9CP]/ { gsub(/[⛔ ]/,"",$2); print $2 }' "${FLOW_SKILL}" | sort -u)
    fi
    for pname in "${!PLUGIN_SRC[@]}"; do
        pdir="${PLUGIN_SRC[${pname}]}"
        hook="$(parse_yaml_value "${pdir}/SKILL.md" "workflow-hook")"
        standalone="$(parse_yaml_value "${pdir}/SKILL.md" "standalone")"
        stages="$(parse_yaml_list "${pdir}/SKILL.md" "workflow-stages" | paste -sd, -)"
        if [ -z "${hook}" ]; then
            warn "plugin ${pname}: frontmatter missing workflow-hook, not registered"
            continue
        fi
        if [[ ! "${hook}" =~ ^(after|before):[0-9A-Za-z]+(\.[0-9A-Za-z]+)*$ ]]; then
            warn "plugin ${pname}: invalid workflow-hook '${hook}' (expect after|before:<stage>), not registered"
            continue
        fi
        hook_target="${hook#*:}"
        if [ -n "${flow_stages}" ] && ! echo "${flow_stages}" | grep -qx "${hook_target}"; then
            warn "plugin ${pname}: workflow-hook target '${hook_target}' not found in base flow table, not registered"
            continue
        fi
        if [ -z "${stages}" ]; then
            warn "plugin ${pname}: frontmatter missing workflow-stages, not registered"
            continue
        fi
        [ "${standalone}" = "true" ] || standalone="false"
        echo "${pname}|${hook}|${stages}|${standalone}" >> "${PLUGIN_RECORDS}"
    done

    MODE_ARG="${MODE}" CONFIG_TARGET="${CANNBOT_MID_DIR}/settings.json" \
    PLUGIN_RECORDS_FILE="${PLUGIN_RECORDS}" \
    PLUGIN_ENABLE_NAME_ARG="${PLUGIN_ENABLE_NAME:-}" PLUGIN_ENABLE_STATE_ARG="${PLUGIN_ENABLE_STATE:-}" \
    python3 - << 'CONFIG_PY' || wf_rc=$?
import json, os, sys

path = os.environ["CONFIG_TARGET"]
mode_arg = os.environ["MODE_ARG"]
records_path = os.environ["PLUGIN_RECORDS_FILE"]
enable_name = os.environ.get("PLUGIN_ENABLE_NAME_ARG", "")
enable_state = os.environ.get("PLUGIN_ENABLE_STATE_ARG", "")

records = {}
with open(records_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, hook, stages_csv, standalone = line.split("|")
        records[name] = {
            "hook": hook,
            "stages": [s for s in stages_csv.split(",") if s],
            "standalone": standalone == "true",
        }

config = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            print(f"  [warn] settings.json 内容不是 JSON 对象，将重建: {path}", file=sys.stderr)
            config = {}
    except Exception as e:
        print(f"  [warn] settings.json 解析失败，将重建: {e}", file=sys.stderr)
        config = {}

# 既有状态：settings.json 的 plugins（保留 enabled）；旧 plugin-registry.json 一次性迁移
existing = {}
surveyed = bool(config.get("surveyed", False))
old_plugins = config.get("plugins")
if not isinstance(old_plugins, dict):
    old_plugins = {}
for name, entry in old_plugins.items():
    if isinstance(entry, dict):
        existing[name] = entry
if not existing:
    registry = os.path.join(os.path.dirname(path), "plugin-registry.json")
    if os.path.exists(registry):
        try:
            with open(registry, encoding="utf-8") as f:
                reg = json.load(f)
            if isinstance(reg, dict):
                surveyed = bool(reg.get("surveyed", surveyed))
                for item in reg.get("plugins", []):
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        existing[item["name"]] = {
                            "hook": item.get("hook", ""),
                            "stages": item.get("stages", []),
                            "standalone": bool(item.get("standalone")),
                            "enabled": bool(item.get("enabled", True)),
                        }
            os.remove(registry)
            print(f"  [ok] 旧 plugin-registry.json 已迁移并入 settings.json 并删除", file=sys.stderr)
        except Exception as e:
            print(f"  [warn] 旧 plugin-registry.json 迁移失败: {e}", file=sys.stderr)

# 重扫重写：保留 enabled，并入新增（新增复位 surveyed），剔除失效插件
plugins = {}
has_new = False
for name in sorted(records):
    rec = records[name]
    old = existing.get(name)
    rec["enabled"] = bool(old.get("enabled", True)) if old else True
    if old is None:
        has_new = True
    plugins[name] = rec
if has_new:
    surveyed = False

if enable_name:
    if enable_name in plugins:
        plugins[enable_name]["enabled"] = enable_state == "on"
    else:
        sys.exit(2)

config = {
    "version": 2,
    "mode": mode_arg or config.get("mode") or "interactive",
    "surveyed": surveyed,
    "plugins": plugins,
    "updated_at": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
}

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"  [ok] settings.json → {path} (mode={config['mode']}, {len(plugins)} plugins)")
CONFIG_PY
    if [ -n "${wf_rc:-}" ] && [ "${wf_rc}" -eq 2 ]; then
        warn "--plugin-enable target not registered: ${PLUGIN_ENABLE_NAME}"
    elif [ -n "${wf_rc:-}" ] && [ "${wf_rc}" -ne 0 ]; then
        warn "workflow config generation failed (rc=${wf_rc})"
    fi
    unset wf_rc
    rm -f "${PLUGIN_RECORDS}"
else
    warn "python3 not found, skipping workflow config generation"
    if [ -n "${PLUGIN_ENABLE_NAME:-}" ]; then
        warn "--plugin-enable ignored: python3 not found"
    fi
fi
echo ""

# ============================================================
# Step 5: Setup third-party repositories
# ============================================================
step "[5/6] Setting up third-party repositories..."
repo_total=${#REPO_REGISTRY[@]}
repo_idx=1
for entry in "${REPO_REGISTRY[@]}"; do
    IFS='|' read -r name url ref strategy post_hook <<< "${entry}"

    info "(${repo_idx}/${repo_total}) ${name}"

    # Get local path override if provided
    local_path="${REPO_LOCAL[${name}]:-}"

    # Setup the repo
    setup_repo "${name}" "${url}" "${ref}" "${strategy}" "${post_hook}" "${local_path}"

    # Export {NAME}_DIR variable for downstream use
    repo_dir="${CANNBOT_MID_DIR}/${name}"
    name_upper=$(echo "${name}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    eval "${name_upper}_DIR=\"${repo_dir}\""

    repo_idx=$((repo_idx + 1))
done
echo ""

# ============================================================
# Step 6: Generate manifest + health check
# ============================================================
step "[6/6] Generating manifest and running health check..."
MANIFEST="${CONFIG_ROOT}/cannbot-manifest.json"
SKILLS_JSON=$(echo "${ALL_SKILLS}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
AGENTS_JSON=$(printf '%s\n' "${AGENT_FILES[@]}" | while read -r f; do [ -n "${f}" ] && { f="${f##*/}"; echo "${f%.md}"; }; done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")

# Build repo paths JSON from registry (key = {name}_dir with dashes to underscores)
REPO_DIRS_JSON=""
for entry in "${REPO_REGISTRY[@]}"; do
    name="${entry%%|*}"
    key="$(echo "${name}" | tr '-' '_')_dir"
    REPO_DIRS_JSON="${REPO_DIRS_JSON}  \"${key}\": \"${CANNBOT_MID_DIR}/${name}\",
"
done

cat > "${MANIFEST}" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "${VERSION}",
  "team": "${TEAM_NAME}",
  "level": "${LEVEL}",
  "tool": "${TOOL}",
  "installed_skills": ${SKILLS_JSON},
  "installed_agents": ${AGENTS_JSON},
  "brand_dir": "${CONFIG_ROOT}",
  "skill_discovery_root": "${SKILL_DISCOVERY_ROOT}",
  "agent_discovery_root": "${AGENT_DISCOVERY_ROOT}",
  "mid_dir": "${CANNBOT_MID_DIR}",
${REPO_DIRS_JSON}  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

health_ok=true
health_errors=""

for sub in skills agents; do
    if [ "${sub}" = "skills" ]; then
        target="${SKILL_DISCOVERY_ROOT}"
    else
        target="${AGENT_DISCOVERY_ROOT}"
    fi
    if [ -d "${target}" ]; then
        count=$(ls -1A "${target}" 2>/dev/null | wc -l)
        [ "${count}" -eq 0 ] && health_errors="${health_errors}\n  ${YELLOW}⚠${NC} ${sub}/ is empty"
    else
        health_errors="${health_errors}\n  ${RED}✗${NC} ${sub}/ missing"
        health_ok=false
    fi
done

[ -d "${CANNBOT_MID_DIR}" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} .cannbot/ missing"; health_ok=false; }

PERM_TARGET="${CANNBOT_MID_DIR}/permissions"
if [ -d "${PERM_TARGET}" ]; then
    perm_count=$(ls -1A "${PERM_TARGET}" 2>/dev/null | grep -c '\.js$' || true)
    [ "${perm_count}" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} .cannbot/permissions/ is empty"; }
else
    health_errors="${health_errors}\n  ${YELLOW}⚠${NC} .cannbot/permissions/ missing (PM startup check will block execution)"
fi


# Check all registered repos
for entry in "${REPO_REGISTRY[@]}"; do
    name="${entry%%|*}"
    repo_dir="${CANNBOT_MID_DIR}/${name}"
    [ -d "${repo_dir}" ] || health_errors="${health_errors}\n  ${YELLOW}⚠${NC} ${name} not available"
done

[ -f "${config_target}" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} ${config_name} missing"; health_ok=false; }
[ -f "${MANIFEST}" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }
[ -f "${CANNBOT_MID_DIR}/settings.json" ] || health_errors="${health_errors}\n  ${YELLOW}⚠${NC} .cannbot/settings.json missing (workflow runs in default interactive mode)"

# Registered plugins must all be linked as installed skills
SETTINGS_FILE="${CANNBOT_MID_DIR}/settings.json"
if [ -f "${SETTINGS_FILE}" ] && command -v python3 > /dev/null 2>&1; then
    while IFS= read -r pname; do
        [ -n "${pname}" ] || continue
        [ -e "${SKILLS_LINK_DIR}/${pname}" ] || health_errors="${health_errors}\n  ${YELLOW}⚠${NC} settings plugin not linked: ${pname}"
    done < <(python3 -c "import json; print('\n'.join(json.load(open('${SETTINGS_FILE}')).get('plugins',{}).keys()))" 2>/dev/null || true)
fi

if [ "${TOOL}" = "claude" ]; then
    [ -e "${CONFIG_ROOT}/hooks/permission-guard.js" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} .claude/hooks/permission-guard.js missing"; health_ok=false; }
    [ -e "$(dirname "${config_target}")/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
    if [ -f "${CONFIG_ROOT}/settings.json" ]; then
        grep -q "permission-guard.js" "${CONFIG_ROOT}/settings.json" \
            || health_errors="${health_errors}\n  ${YELLOW}⚠${NC} settings.json 未注册 permission-guard hook"
    else
        health_errors="${health_errors}\n  ${YELLOW}⚠${NC} settings.json missing (hook 未注册)"
    fi
fi

if [ "${health_ok}" = true ] && [ -z "${health_errors}" ]; then
    ok "All checks passed"
else
    echo -e "${health_errors}"
    [ "${health_ok}" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
fi

# ============================================================
# Summary
# ============================================================
CLI_NAME="opencode"
[ "${TOOL}" = "claude" ] && CLI_NAME="claude"
[ "${TOOL}" = "codex" ] && CLI_NAME="codex"
[ "${TOOL}" = "dsh" ] && CLI_NAME="dsh"
echo ""
echo -e "  ${GREEN}${BOLD}✓ ${TEAM_NAME} installed!${NC}"
echo -e "  ${DIM}Skills: ${skill_count} | Agents: ${agent_count}${NC}"

# Quick Start 仅在基类直接调用（无 --override）时输出。
# 子类 init.sh 通过 --override 调用基类时，由子类自行输出本段，
# 以便各子仓定制 Quick Start 内容（见 example/init.sh 的 QUICK_START 函数）。
if [ -z "${OVERRIDE_DIR}" ]; then
    echo ""
    echo -e "  ${BOLD}Quick Start:${NC}"
    echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}${CLI_NAME}${NC}"
    echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}帮我开发一个 abs 算子，支持 float16，shape 主要是 [1,128]、[4,2048]${NC}"
fi
echo ""

#!/usr/bin/env bash
# common.sh — shared helpers for Z-Search loop hooks.
#
# Sourced by:
#   hooks/loop-stop.sh
#   hooks/loop-bash-safety.sh
#
# Provides:
#   - HOOK_DISABLED — 1 if LINGXI_LOOP_HOOK_DISABLE=1 (downgrades blocks to warn)
#   - hook_log "msg"            — stderr log with prefix
#   - hook_block "msg" [code]   — emit block reason, exit 2 (or 0 if HOOK_DISABLED=1)
#   - hook_allow                — exit 0 silently
#   - find_evo_dir              — echo absolute path containing state.json or empty
#   - read_state_field FIELD    — echo state.json field via python json.load
#   - read_stdin_json           — slurp stdin JSON into $STDIN_JSON
#   - json_get FIELD            — extract field from $STDIN_JSON (uses python)

if [[ -n "${LOOP_HOOK_COMMON_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_COMMON_SOURCED=1

HOOK_NAME="${HOOK_NAME:-loop-hook}"
HOOK_DISABLED=0
if [[ "${LINGXI_LOOP_HOOK_DISABLE:-}" == "1" ]]; then
    HOOK_DISABLED=1
fi

# Locate the plugin root. This script lives at <plugin_root>/hooks/lib/common.sh
# — walk up 2 levels. Works both for the plugin layout (${CLAUDE_PLUGIN_ROOT})
# and for init.sh-installed layouts (the hook command points into the plugin).
_resolve_plugin_root() {
    local self
    self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local root
    root="$(cd "${self}/../.." && pwd)"
    # Normalize MSYS/Git-Bash paths (e.g. /c/...) so Windows-hosted python
    # can consume them inside embedded string interpolation.
    if command -v cygpath >/dev/null 2>&1; then
        root="$(cygpath -m "$root")"
    fi
    echo "$root"
}
PLUGIN_ROOT="$(_resolve_plugin_root)"

# Locate state_ops.py relative to the plugin root (skills are plugin-bundled).
STATE_OPS_PY="${PLUGIN_ROOT}/skills/evolution-world-model/scripts/state_ops.py"

# Locate the user project root. CLAUDE_PROJECT_DIR is set by Claude Code;
# fall back to the hook process cwd when invoked outside Claude Code.
# Normalized like PLUGIN_ROOT so downstream globs and embedded python
# string interpolation stay safe on MSYS/Git Bash + Windows.
_resolve_project_root() {
    local root
    if [[ -n "${CLAUDE_PROJECT_DIR:-}" && -d "${CLAUDE_PROJECT_DIR}" ]]; then
        root="${CLAUDE_PROJECT_DIR}"
    else
        root="$PWD"
    fi
    if command -v cygpath >/dev/null 2>&1; then
        root="$(cygpath -m "$root")"
    fi
    echo "$root"
}

PROJECT_ROOT="$(_resolve_project_root)"

hook_log() {
    echo "[${HOOK_NAME}] $*" >&2
}

# Emit a block message. Always exits the process.
#   exit 2 in strict mode (default)
#   exit 0 (warn-only) when HOOK_DISABLED=1
hook_block() {
    local reason="$1"
    local code="${2:-2}"
    if [[ "${HOOK_DISABLED}" == "1" ]]; then
        hook_log "WARN (hook disabled, would have blocked): ${reason}"
        exit 0
    fi
    hook_log "BLOCK: ${reason}"
    exit "${code}"
}

hook_allow() {
    exit 0
}

# Walk upward from $1 (or cwd) looking for state.json.
# Echoes the directory containing it, or empty string if not found.
find_evo_dir() {
    local start="${1:-$PWD}"
    # Normalize MSYS/Git-Bash paths (e.g. /c/...) for Windows-hosted python.
    if command -v cygpath >/dev/null 2>&1; then
        start="$(cygpath -m "$start")"
    fi
    if [[ ! -f "${STATE_OPS_PY}" ]]; then
        # state_ops.py not present (e.g. fresh clone before installation) → noop
        echo ""
        return
    fi
    python3 -c "
import sys
sys.path.insert(0, '${PLUGIN_ROOT}/skills/evolution-world-model/scripts')
from state_ops import find_evo_dir
d = find_evo_dir('${start}')
# Normalize backslashes so the result is safe to re-embed into quoted
# python one-liners by callers (a literal '\\U' would break parsing).
print((d or '').replace(chr(92), '/'))
" 2>/dev/null || echo ""
}

# Read a single field from state.json in $1 (evo_dir).
# Echoes the field value, or empty string if missing.
# Field path uses dot notation, e.g. "stage" or "partial_status.0".
read_state_field() {
    local evo_dir="$1"
    local field="$2"
    if [[ ! -f "${evo_dir}/state.json" ]]; then
        echo ""
        return
    fi
    python3 -c "
import json, sys
try:
    with open('${evo_dir}/state.json') as f:
        s = json.load(f)
    parts = '${field}'.split('.')
    v = s
    for p in parts:
        if isinstance(v, dict):
            v = v.get(p, '')
        else:
            v = ''
            break
    if isinstance(v, (dict, list)):
        print(json.dumps(v, ensure_ascii=False))
    else:
        print(v if v is not None else '')
except Exception as e:
    print('', end='')
" 2>/dev/null
}

# Slurp stdin into $STDIN_JSON.
# Hooks receive JSON on stdin per Claude Code hook protocol.
read_stdin_json() {
    STDIN_JSON="$(cat)"
}

# Extract a dotted field from $STDIN_JSON.
json_get() {
    local field="$1"
    python3 -c "
import json, sys
try:
    s = json.loads(sys.stdin.read())
    parts = '${field}'.split('.')
    v = s
    for p in parts:
        if isinstance(v, dict):
            v = v.get(p, '')
        else:
            v = ''
            break
    if isinstance(v, (dict, list)):
        print(json.dumps(v, ensure_ascii=False))
    else:
        print(v if v is not None else '')
except Exception:
    print('', end='')
" <<< "${STDIN_JSON}" 2>/dev/null
}

export -f hook_log hook_block hook_allow find_evo_dir read_state_field
export PLUGIN_ROOT PROJECT_ROOT STATE_OPS_PY HOOK_DISABLED HOOK_NAME

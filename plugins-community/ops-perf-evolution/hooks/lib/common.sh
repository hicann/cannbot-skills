#!/usr/bin/env bash
# common.sh — shared helpers for Z-Search loop hooks.
#
# Sourced by:
#   .claude/hooks/loop-stop.sh
#   .claude/hooks/loop-bash-safety.sh
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

# Locate state_ops.py relative to repo root.
# CLAUDE_PROJECT_DIR is set by Claude Code; fallback to script-relative resolution.
_resolve_project_root() {
    if [[ -n "${CLAUDE_PROJECT_DIR:-}" && -d "${CLAUDE_PROJECT_DIR}" ]]; then
        echo "${CLAUDE_PROJECT_DIR}"
        return
    fi
    # script lives at <root>/.claude/hooks/lib/common.sh — walk up 3 levels
    local self
    self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$(cd "${self}/../../.." && pwd)"
}

PROJECT_ROOT="$(_resolve_project_root)"
STATE_OPS_PY="${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/state_ops.py"

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
    if [[ ! -f "${STATE_OPS_PY}" ]]; then
        # state_ops.py not present (e.g. fresh clone before installation) → noop
        echo ""
        return
    fi
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts')
from state_ops import find_evo_dir
d = find_evo_dir('${start}')
print(d or '')
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
export PROJECT_ROOT STATE_OPS_PY HOOK_DISABLED HOOK_NAME

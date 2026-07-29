#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# guard-baseline-paths.sh
#
# Claude Code PreToolUse hook that blocks Edit/Write tool calls targeting protected
# baseline paths (user-provided benchmark sources, hook self-files).
#
# Input: stdin JSON from Claude Code, shape:
#   {"tool_name":"Edit","tool_input":{"file_path":"/abs/path"}, ...}
#
# Output: on deny, stdout JSON:
#   {"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny",
#                          "permissionDecisionReason":"..."}}
#   plus exit 0. On allow: no output, exit 0.
#
# Fail-open policy: if jq is missing, stdin is malformed, or config is unreadable,
# the hook allows the call (prints nothing, exits 0) rather than block all edits.
# Protected paths still get denied using hardcoded fallback patterns when the
# config file is missing but jq is available.
# ----------------------------------------------------------------------------------------------------------

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/guard-config.json"

# ---- helpers ----

emit_deny() {
    # $1 = reason
    if command -v jq >/dev/null 2>&1; then
        jq -nc --arg reason "$1" '{
            hookSpecificOutput: {
                hookEventName: "PreToolUse",
                permissionDecision: "deny",
                permissionDecisionReason: $reason
            }
        }'
    else
        # jq missing: fail-open by printing nothing. We lose deny capability,
        # but only when jq is absent from the environment entirely.
        :
    fi
}

# ---- fail fast if jq missing: fail-open ----

if ! command -v jq >/dev/null 2>&1; then
    # No way to safely parse stdin or emit JSON; allow the call.
    exit 0
fi

# ---- read stdin ----

INPUT="$(cat)"

TOOL_NAME="$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')"
FILE_PATH="$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')"

# Only Edit/Write carry a file_path we care about. matcher is set in settings.json
# to "Edit|Write", but double-check here for safety.
case "$TOOL_NAME" in
    Edit|Write) ;;
    *) exit 0 ;;
esac

# No file_path → nothing to check, allow.
[ -z "$FILE_PATH" ] && exit 0

# ---- load config (or fallback) ----

if [ -f "$CONFIG_PATH" ]; then
    PROTECTED_PATTERNS="$(jq -r '.protected[]?' "$CONFIG_PATH" 2>/dev/null)"
    ALLOWLIST_PATTERNS="$(jq -r '.allowlist[]?' "$CONFIG_PATH" 2>/dev/null)"
    SELF_GUARD_PATTERNS="$(jq -r '.self_guard[]?' "$CONFIG_PATH" 2>/dev/null)"
else
    PROTECTED_PATTERNS="$(jq -r '._fallback_when_config_missing.protected[]?' <<<'{"_fallback_when_config_missing":{"protected":["/home/.*/npu_benchmark/.*\\.py$","/home/.*/ascendc-kernelgen-data.*/.*\\.py$"]}}' 2>/dev/null)"
    ALLOWLIST_PATTERNS="$(jq -r '._fallback_when_config_missing.allowlist[]?' <<<'{"_fallback_when_config_missing":{"allowlist":[".*/triton_ascend_output/.*"]}}' 2>/dev/null)"
    SELF_GUARD_PATTERNS="$(jq -r '._fallback_when_config_missing.self_guard[]?' <<<'{"_fallback_when_config_missing":{"self_guard":[".*/\\.claude/settings\\.json$",".*/\\.claude/hooks/guard-baseline-paths\\.sh$"]}}' 2>/dev/null)"
fi

# ---- match function: any pattern in $1 matches $2 ----

match_any() {
    # $1 = newline-separated patterns, $2 = path
    local patterns="$1" path="$2" pat
    while IFS= read -r pat; do
        [ -z "$pat" ] && continue
        # bash regex; if pattern itself is invalid, skip it (fail-open for that line)
        if [[ $path =~ $pat ]]; then
            return 0
        fi
    done <<<"$patterns"
    return 1
}

# ---- decision order: self_guard → allowlist → protected → allow ----

# 1. self_guard first: never let the agent disable the guard
if match_any "$SELF_GUARD_PATTERNS" "$FILE_PATH"; then
    emit_deny "Blocked by baseline guard: this file is part of the guard/hook infrastructure and cannot be modified ($FILE_PATH). Modifying it would disable tamper protection."
    exit 0
fi

# 2. allowlist: explicit overrides (e.g. working dir, skill sources)
if match_any "$ALLOWLIST_PATTERNS" "$FILE_PATH"; then
    exit 0
fi

# 3. protected: benchmark source paths
if match_any "$PROTECTED_PATTERNS" "$FILE_PATH"; then
    emit_deny "Blocked by baseline guard: $FILE_PATH is a user-provided benchmark source path (read-only). Modifying the baseline would invalidate all downstream verify/benchmark results. If the source has a bug, report it and fail the task rather than patching the baseline."
    exit 0
fi

# 4. default allow
exit 0

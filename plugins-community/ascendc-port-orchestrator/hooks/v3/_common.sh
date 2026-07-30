#!/bin/bash
# Common helpers for V3 hook scripts.
# Source from other hook scripts: `. "$(dirname "$0")/_common.sh"`

# Find the workspace most recently modified (within last 4h).
# Echoes the workspace path (e.g., workspace/adain_backward), or empty if none.
#
# Resolution order:
#   1. If CLAUDE_ACTIVE_WORKSPACE env var points to a workspace with PROGRESS.md, use it.
#   2. Otherwise, scan $WORKSPACE_ROOT for PROGRESS.md within last 4h, pick most recent
#      by mtime (NOT find's default scan order — explicitly sort numeric desc).
find_active_workspace() {
    local root="${WORKSPACE_ROOT:-workspace}"
    [ -d "$root" ] || { echo ""; return 0; }

    # Strict priority: explicit WORKSPACE env takes precedence (orchestrator can set)
    if [ -n "$CLAUDE_ACTIVE_WORKSPACE" ] && [ -f "$CLAUDE_ACTIVE_WORKSPACE/PROGRESS.md" ]; then
        dirname "$CLAUDE_ACTIVE_WORKSPACE/PROGRESS.md"
        return 0
    fi

    # Fallback: most-recent by mtime (sort numeric desc)
    local latest
    latest=$(find "$root" -name "PROGRESS.md" -mmin -240 -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | awk '{print $2}')
    [ -z "$latest" ] && { echo ""; return 0; }
    dirname "$latest"
}

# Emit failure message to stderr and exit blocking (2).
fail_block() {
    local agent="$1"; shift
    echo "❌ $agent: $*" >&2
    exit 2
}

# Emit warning (non-blocking) to stderr.
warn() {
    echo "⚠️ $*" >&2
}

# Log a logging-compliance gap to BOTH stderr (current session) AND PROGRESS.md
# (durable record for orchestrator/post-hoc analysis). Never blocks (exit 0 still expected).
#
# Args:
#   $1: agent name (e.g. "aog-kernel-worker", "aog-precision-probe")
#   $2: gap description (one line)
#   $3: PROGRESS.md path
#
# Writes one line to PROGRESS.md under a "## Logging gaps (auto-captured)" section
# (creates section if absent). Idempotent within a session — same gap emitted twice
# in same minute writes only once.
log_gap_to_progress() {
    local agent="$1"
    local gap="$2"
    local prog="$3"

    # stderr warning (always)
    echo "⚠️ ${agent} logging gap: ${gap}" >&2
    echo "    (recorded to ${prog} for external monitoring; not blocking)" >&2

    [ -f "$prog" ] || return 0

    local ts
    ts=$(date +%H:%M)
    local line="- [${ts}] hook-system: ${agent} did not log: ${gap}"

    # Skip exact duplicate within this minute
    if grep -qF "$line" "$prog" 2>/dev/null; then
        return 0
    fi

    # Append section header if missing
    if ! grep -qF "## Logging gaps (auto-captured)" "$prog" 2>/dev/null; then
        printf '\n## Logging gaps (auto-captured)\n\n' >> "$prog"
    fi

    printf '%s\n' "$line" >> "$prog"
}

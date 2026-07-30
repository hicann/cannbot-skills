#!/bin/bash
# Shared hook: every agent must sign PROGRESS.md before exit.
# The chat-room pattern: each Timeline entry starts with `### [timestamp] {agent-name}`.
#
# 2026-04-16: verified with official docs that $CLAUDE_AGENT_NAME does NOT exist
# (see SKILLS_DESIGN V3.1 appendix). Correct source is the stdin JSON's `agent_type` field.
# If no JSON on stdin (e.g., manual invocation), fall back to "any signed entry" check.
set -e
. "$(dirname "$0")/_common.sh"

WS="$(find_active_workspace)"
[ -z "$WS" ] && exit 0

PROGRESS="$WS/PROGRESS.md"
[ -f "$PROGRESS" ] || fail_block "orchestrator" "PROGRESS.md missing at $PROGRESS"

# Try to read agent name from stdin JSON (Claude Code hook input format).
# If no stdin or non-JSON, AGENT stays empty and we degrade gracefully.
AGENT=""
if [ -p /dev/stdin ] || [ ! -t 0 ]; then
    STDIN_JSON=$(timeout 2 cat 2>/dev/null || true)
    if [ -n "$STDIN_JSON" ]; then
        AGENT=$(printf '%s' "$STDIN_JSON" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get('agent_type', ''))
except Exception:
    print('')
" 2>/dev/null || true)
    fi
fi

# All logging-compliance failures here are WARN-only (per 2026-04-17 user
# feedback). They record gaps to PROGRESS.md so external monitors / orchestrator
# can see + fix them, but never block agent exit.
#
# The contract is "everyone writes" — and when someone doesn't, the gap is
# visible in the same shared log they failed to write to.

if [ -n "$AGENT" ]; then
    # Strict mode: my own signed entry should exist
    if ! grep -qE "^### \[.*\] *${AGENT}\b" "$PROGRESS"; then
        log_gap_to_progress "$AGENT" "no signed PROGRESS entry (expected '### [timestamp] $AGENT (...)')" "$PROGRESS"
    fi
else
    # Degraded mode: any signed entry counts
    LAST_SIGS=$(grep -E "^### \[.*\] *[a-z-]+" "$PROGRESS" 2>/dev/null | wc -l)
    if [ "$LAST_SIGS" -lt 1 ]; then
        log_gap_to_progress "unknown-agent" "PROGRESS.md has no signed entries from any agent" "$PROGRESS"
    fi
fi

# DIAG enforcement: warn if flag set but no DIAG sections
DIAG_FLAG="$WS/.diag_enabled"
if [ -f "$DIAG_FLAG" ]; then
    DIAG_COUNT=$(grep -E "^### DIAG:" "$PROGRESS" 2>/dev/null | wc -l)
    if [ "$DIAG_COUNT" -lt 1 ]; then
        log_gap_to_progress "${AGENT:-agent}" "DIAGNOSTIC mode on but no '### DIAG:' sections written" "$PROGRESS"
    fi
    LOG_COUNT=$(grep -E "^- *\[[0-9]{2}:[0-9]{2}\]" "$PROGRESS" 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -lt 2 ]; then
        log_gap_to_progress "${AGENT:-agent}" "DIAG mode requires Log entries [HH:MM] ACTION/RESULT — found $LOG_COUNT (need >=2)" "$PROGRESS"
    fi
fi

exit 0

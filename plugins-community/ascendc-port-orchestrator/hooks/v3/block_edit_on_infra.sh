#!/bin/bash
# PreToolUse hook: block Edit/Write on kernel/* files when the last build
# classified as infra failure. Forces agent to resolve infra state (retry
# deploy, or BLOCKED-exit) before touching kernel code.
#
# Input: JSON on stdin with tool name + parameters
# Output: exit 2 + stderr to block the tool call. exit 0 allows.
#
# Markers (written by deploy_to_a5.sh, per-workspace primary + /tmp fallback):
#   $WORKSPACE/.last_build.class   — "infra" | "compile" | empty
#   $WORKSPACE/.last_build.stderr  — full stderr of last attempt
#   /tmp/ascendc_last_build.{class,stderr} — legacy fallback
#
# Only blocks when class == "infra" AND tool is Edit/Write AND path contains
# /kernel/ and ends in .h/.cpp/.cc. Other edits (PROGRESS.md, analysis.md,
# verification.json) proceed normally — agent NEEDS to write BLOCKED entry.

set -e
. "$(dirname "$0")/_common.sh"

# Prefer per-workspace markers, fallback to /tmp
WS="$(find_active_workspace)"
CLASS_MARKER=""
STDERR_MARKER=""
if [ -n "$WS" ] && [ -f "$WS/.last_build.class" ]; then
    CLASS_MARKER="$WS/.last_build.class"
    STDERR_MARKER="$WS/.last_build.stderr"
elif [ -f /tmp/ascendc_last_build.class ]; then
    CLASS_MARKER=/tmp/ascendc_last_build.class
    STDERR_MARKER=/tmp/ascendc_last_build.stderr
fi

# No marker or not infra → allow
[ -z "$CLASS_MARKER" ] && exit 0
CLASS=$(cat "$CLASS_MARKER" 2>/dev/null || echo "")
[ "$CLASS" != "infra" ] && exit 0

# Read tool input from stdin
INPUT=$(cat)

# Parse with python (jq not guaranteed available). Pass hook JSON on stdin so
# payload contents can never become part of the Python program.
PARSED=$(printf '%s' "$INPUT" | python3 -c '
import json
import sys

try:
    d = json.load(sys.stdin)
except Exception:
    print("UNKNOWN|")
    sys.exit(0)
tool = d.get("tool_name") or d.get("tool") or ""
# Parameters live under 'tool_input' in standard PreToolUse payload
params = d.get("tool_input") or d.get("parameters") or {}
path = params.get("file_path") or params.get("path") or ""
print(f"{tool}|{path}")
' 2>/dev/null)
TOOL="${PARSED%%|*}"
PATH_ARG="${PARSED#*|}"

# Only gate Edit/Write/MultiEdit
case "$TOOL" in
    Edit|Write|MultiEdit) ;;
    *) exit 0 ;;
esac

# Only gate kernel source files
if ! echo "$PATH_ARG" | grep -qE "/kernel/.*\.(h|cpp|cc|hpp)$"; then
    exit 0
fi

# Block with explicit remediation guidance
cat >&2 <<EOF
❌ aog-kernel-worker: infra block — last deploy/build was classified as INFRA failure.
Last stderr (tail):
$(tail -5 "$STDERR_MARKER" 2>/dev/null || echo "(empty)")

Rule (from aog-kernel-worker.md Fault Tolerance): do NOT Edit kernel/* until infra is resolved.
Options:
  1. Retry: bash src/scripts/deploy_to_a5.sh --build (it retries 3× with backoff automatically).
     Success clears the marker and unblocks Edits.
  2. If infra is persistently down: write BLOCKED entry to PROGRESS.md and exit agent
     (this hook only blocks kernel edits, not PROGRESS edits).

Do not Edit $PATH_ARG on an infra-blocked iteration — this is a V3.1 Bug-1 reinforcement.
EOF
exit 2

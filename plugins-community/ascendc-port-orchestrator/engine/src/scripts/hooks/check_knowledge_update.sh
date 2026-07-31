#!/bin/bash
# Knowledge update gate — blocks agent Stop/commit if knowledge_update.md is missing.
#
# After each completed op, the agent MUST produce knowledge_update.md documenting
# what was learned and what knowledge base files were updated (or explicitly marked
# "no change").
#
# Exit codes:
#   0 = pass (allow)
#   1 = error (logged only)
#   2 = BLOCKING (agent must fix before proceeding)

set -e

WORKSPACE_ROOT="${WORKSPACE_ROOT:-workspace}"
if [ ! -d "$WORKSPACE_ROOT" ]; then
    exit 0  # No workspace — not in a generation workflow
fi

# Find knowledge_update.md modified within last 4 hours
KU_FILE=$(find "$WORKSPACE_ROOT" -name "knowledge_update.md" -mmin -240 2>/dev/null | head -1)

# Only enforce if there's an active workflow (gate_decision.md exists)
GATE_FILE=$(find "$WORKSPACE_ROOT" -name "gate_decision.md" -mmin -240 2>/dev/null | head -1)
if [ -z "$GATE_FILE" ]; then
    exit 0  # No active workflow
fi

if [ -z "$KU_FILE" ]; then
    echo "⚠️ KNOWLEDGE UPDATE MISSING: No knowledge_update.md found in workspace." >&2
    echo "  After completing an op, write knowledge_update.md with:" >&2
    echo "    - ERROR_CORRECTIONS: new entries or 'no change'" >&2
    echo "    - PATTERN_INDEX: validated/new patterns or 'no change'" >&2
    echo "    - OPERATIONAL_KNOWLEDGE: lessons learned or 'no change'" >&2
    echo "    - PLATFORM_BUGS: new bugs found or 'no change'" >&2
    exit 2  # BLOCKING
fi

# Check required sections exist
ERRORS=0
MESSAGES=""

for section in "ERROR_CORRECTIONS" "PATTERN_INDEX" "OPERATIONAL_KNOWLEDGE" "PLATFORM_BUGS"; do
    if ! grep -qi "$section" "$KU_FILE" 2>/dev/null; then
        MESSAGES="$MESSAGES\n  - Missing section: $section"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo -e "⚠️ KNOWLEDGE UPDATE INCOMPLETE ($ERRORS missing sections):$MESSAGES" >&2
    echo "  Each section must have a concrete update or explicit 'no change'." >&2
    exit 2  # BLOCKING
fi

exit 0

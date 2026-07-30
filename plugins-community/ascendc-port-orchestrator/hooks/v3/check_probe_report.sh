#!/bin/bash
# aog-precision-probe SubagentStop hook: probe_report.md must exist with required sections.
set -e
. "$(dirname "$0")/_common.sh"

WS="$(find_active_workspace)"
[ -z "$WS" ] && exit 0

REPORT="$WS/probe_report.md"
if [ ! -f "$REPORT" ]; then
    fail_block "aog-precision-probe" "probe_report.md not found at $REPORT"
fi

REQUIRED=(
    "## Hypothesis"
    "## Minimal repros run"
    "## Classification"
    "## Recommendation"
)
MISSING=()
for s in "${REQUIRED[@]}"; do
    if ! grep -qF "$s" "$REPORT"; then
        MISSING+=("$s")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    fail_block "aog-precision-probe" "probe_report.md missing sections: ${MISSING[*]}"
fi

# Must record at least one actual bit-diff measurement
if ! grep -qE "max_diff[[:space:]]*[:=]" "$REPORT"; then
    fail_block "aog-precision-probe" "probe_report.md has no 'max_diff' measurement — probes not executed?"
fi
if ! grep -qE "bit_identical" "$REPORT"; then
    fail_block "aog-precision-probe" "probe_report.md has no 'bit_identical' field — classify each repro"
fi

# Also verify probes directory has scripts (audit trail)
PROBES_DIR="$WS/probes"
if [ ! -d "$PROBES_DIR" ] || [ -z "$(ls -A "$PROBES_DIR" 2>/dev/null)" ]; then
    fail_block "aog-precision-probe" "probes/ dir empty — must save probe scripts to workspace/{op}/probes/"
fi

exit 0

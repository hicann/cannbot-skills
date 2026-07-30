#!/bin/bash
# aog-fused-optimizer SubagentStop hook: fused_analysis.md must exist with KB Manifest + at least one Iter entry;
# last precision line must be PASS; verification.json (if updated) must show precision.status=PASS.
set -e
. "$(dirname "$0")/_common.sh"

WS="$(find_active_workspace)"
[ -z "$WS" ] && exit 0

FANAL="$WS/fused_analysis.md"
if [ ! -f "$FANAL" ]; then
    fail_block "aog-fused-optimizer" "fused_analysis.md not found at $FANAL"
fi

# Must contain KB Manifest section (parity with worker/optimizer)
if ! grep -qE "^## KB Manifest" "$FANAL"; then
    fail_block "aog-fused-optimizer" "fused_analysis.md missing '## KB Manifest' section — required for audit trail"
fi
if ! grep -qE "^### LOADED" "$FANAL"; then
    fail_block "aog-fused-optimizer" "fused_analysis.md missing '### LOADED' subsection under KB Manifest"
fi

# Must contain at least one Iter entry OR explicit early-exit verdict (e.g. pilot cannot instrument)
if ! grep -qE "^### (Iter[0-9]+|Iter 0|Iter0|Early exit|EARLY EXIT)" "$FANAL"; then
    fail_block "aog-fused-optimizer" "fused_analysis.md has no '### Iter{N}' entry and no explicit early-exit verdict"
fi

# Last "Precision:" line must be PASS — pilot/early-exit with NO edits made is also OK
# (absence of Precision line means no edits were made, which is a legitimate outcome)
LAST_PREC=$(grep -E "^[[:space:]]*Precision:" "$FANAL" | tail -1 || true)
if [ -n "$LAST_PREC" ]; then
    if ! echo "$LAST_PREC" | grep -qE "PASS"; then
        fail_block "aog-fused-optimizer" "last precision line indicates regression: $LAST_PREC — must revert before Stop"
    fi
fi

# If verification.json was updated (mtime after this workspace's analysis.md), it must show PASS
# V3.8.x extension (2026-05-14, fo-2 incident, parity with ko-2 11:18Z patch to check_optimizer_artifacts.sh):
#   PARTIAL is accepted on V3.8.x PARTIAL_PERSIST passthrough flow when fused_analysis.md carries
#   an explicit Early-exit verdict AND verification.json mtime predates fo's own activity. The
#   "fo introduced no edits" claim is content-level checkable from workspace kernel md5 audit +
#   fused_analysis.md early-exit sentinel.
VJSON="$WS/verification.json"
if [ -f "$VJSON" ] && [ "$VJSON" -nt "$WS/analysis.md" ]; then
    STATUS=$(python3 - "$VJSON" 2>/dev/null <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    print(json.load(f)["precision"]["status"])
PY
    ) || STATUS="UNKNOWN"
    case "$STATUS" in
        PASS|PASS_WITHIN_TOLERANCE)
            : ;;  # original happy path preserved
        PARTIAL|FAIL)
            # V3.8.x PARTIAL_PERSIST passthrough: accept when fused_analysis.md has an Early-exit
            # verdict AND verification.json is older than fused_analysis.md (i.e. fo did NOT
            # rewrite verification.json this spawn — the PARTIAL/FAIL is inherited from the worker).
            # FAIL added 2026-06-21 (flash_attention_score_causal_grad fo-2 incident): a directive-
            # emit early-exit can legitimately inherit a worker precision.status=FAIL that is ABOVE
            # the gate floor (e.g. 51/54 vs floor 50/54, 3 dtype-floor-ratio degeneracies). The
            # guard below (early-exit verdict + verification.json mtime < fused_analysis.md mtime,
            # i.e. fo introduced no edits) validates the no-regression claim identically for FAIL as
            # for PARTIAL. fo is FORBIDDEN from flipping FAIL→PASS to dodge this (masking, P5) — the
            # honest inherited-FAIL passthrough is the correct path, perf handled via the directive.
            if grep -qE "^### (Early exit|EARLY EXIT)" "$FANAL" && [ "$FANAL" -nt "$VJSON" ]; then
                echo "⚠️  aog-fused-optimizer: $STATUS precision passthrough accepted (V3.8.x) — fused_analysis.md early-exit verdict present, verification.json predates fused_analysis.md (fo introduced no edits this spawn)" >&2
            else
                fail_block "aog-fused-optimizer" "verification.json updated but precision.status=$STATUS — not PASS (V3.8.x passthrough requires Early-exit verdict in fused_analysis.md AND verification.json mtime < fused_analysis.md mtime)"
            fi
            ;;
        *)
            fail_block "aog-fused-optimizer" "verification.json updated but precision.status=$STATUS — not PASS"
            ;;
    esac
fi

# Must have a Handoff line (handoff contract parity with aog-kernel-optimizer)
if ! grep -qE "^##[[:space:]]+Handoff|Handoff:" "$FANAL"; then
    fail_block "aog-fused-optimizer" "fused_analysis.md missing Handoff section/line — required to tell orchestrator the next step"
fi

exit 0

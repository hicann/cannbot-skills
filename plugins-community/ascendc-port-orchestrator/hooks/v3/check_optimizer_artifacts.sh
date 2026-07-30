#!/bin/bash
# aog-kernel-optimizer SubagentStop hook: optimization_log.md must exist and last entry must show precision PASS.
set -e
. "$(dirname "$0")/_common.sh"

WS="$(find_active_workspace)"
[ -z "$WS" ] && exit 0

OPTLOG="$WS/optimization_log.md"
if [ ! -f "$OPTLOG" ]; then
    fail_block "aog-kernel-optimizer" "optimization_log.md not found at $OPTLOG"
fi

# Must contain at least one "## Opt" entry
if ! grep -qE "^## Opt" "$OPTLOG"; then
    fail_block "aog-kernel-optimizer" "optimization_log.md has no '## Opt{N}' entry"
fi

# Last "Precision:" line in the file must be PASS (i.e., current state is not broken)
LAST_PREC=$(grep -E "^[[:space:]]*Precision:" "$OPTLOG" | tail -1 || true)
if [ -z "$LAST_PREC" ]; then
    fail_block "aog-kernel-optimizer" "no Precision line in optimization_log.md"
fi
if ! echo "$LAST_PREC" | grep -qE "PASS"; then
    fail_block "aog-kernel-optimizer" "last optimization iteration broke precision: $LAST_PREC — must revert before Stop"
fi

# Final verification.json must reflect 50/50 or all-PASS state.
#
# V3.8.x extension (2026-05-14, ko apply_adam_w_quant-ko-2 incident):
#   The V3.8.x state machine allows ko to be spawned on PARTIAL_PERSIST handoffs
#   (e.g. perf-gate-misread, defensive pre-finalize routing, or fo→ko escalation
#   reverse path). In that case ko correctly performs a baseline-only audit,
#   makes no edits, and emits PARTIAL_PERSIST passthrough. The original check
#   below (STATUS != "PASS") would erroneously block this valid flow.
#
#   Resolution: accept PARTIAL ONLY when optimization_log.md carries an explicit
#   `Precision: ... NO_REGRESSION ...` sentinel — this proves ko did NOT introduce
#   the PARTIAL state (it was inherited from the worker's prior baseline). The
#   hook's actual guarantee ("ko's edits did not break precision") is preserved
#   under this case because the sentinel asserts ko made no precision-impacting
#   change. PASS / PASS_WITHIN_TOLERANCE remain unconditionally accepted. FAIL
#   and UNKNOWN remain blocked. PARTIAL without sentinel remains blocked.
VJSON="$WS/verification.json"
if [ -f "$VJSON" ]; then
    STATUS=$(python3 - "$VJSON" 2>/dev/null <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    print(json.load(f)["precision"]["status"])
PY
    ) || STATUS="UNKNOWN"
    case "$STATUS" in
        PASS|PASS_WITHIN_TOLERANCE)
            : # OK — original happy path
            ;;
        PARTIAL)
            if grep -qE "^[[:space:]]*Precision:.*NO_REGRESSION" "$OPTLOG"; then
                warn "ko spawn passed through PARTIAL precision state (NO_REGRESSION sentinel in optimization_log.md) — accepted per V3.8.x extension"
            else
                fail_block "aog-kernel-optimizer" "verification.json precision.status=PARTIAL after optimization, and optimization_log.md lacks 'Precision: ... NO_REGRESSION ...' sentinel — either restore precision to PASS, or document the no-edit passthrough with the NO_REGRESSION sentinel"
            fi
            ;;
        *)
            fail_block "aog-kernel-optimizer" "verification.json precision.status=$STATUS after optimization — not PASS"
            ;;
    esac
fi

exit 0

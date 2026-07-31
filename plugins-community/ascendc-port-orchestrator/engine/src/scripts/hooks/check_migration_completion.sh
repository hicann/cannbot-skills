#!/bin/bash
# Migration completion gate — blocks agent Stop/commit if work is incomplete.
#
# Exit codes:
#   0 = pass (allow)
#   1 = error (logged only)
#   2 = BLOCKING (agent must fix before proceeding)
#
# Checks:
#   1. Static checker passed (10/10)
#   2. Precision gate decision exists and is PASS
#   3. Performance benchmark data exists
#   4. msprof profiling data exists (if optimization claimed)

set -e

# Locate scripts directory (sibling to hooks/ in deployed skill)
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$HOOK_DIR")"
SCRIPTS_DIR="$SKILL_DIR/scripts"

# Find the most recent workspace (modified within last 4 hours)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-workspace}"
if [ ! -d "$WORKSPACE_ROOT" ]; then
    exit 0  # No workspace directory at all — not in a migration workflow
fi

LATEST=$(find "$WORKSPACE_ROOT" -name "gate_decision.md" -mmin -240 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    # No recent gate_decision.md — check if there are any iter_* dirs (active work)
    ACTIVE_ITERS=$(find "$WORKSPACE_ROOT" -maxdepth 3 -name "iter_*" -type d -mmin -240 2>/dev/null | head -1)
    if [ -z "$ACTIVE_ITERS" ]; then
        exit 0  # No active migration workflow
    fi
    # Active work exists but no gate_decision yet — that's OK for Stage 1 in progress
    exit 0
fi

LATEST_DIR=$(dirname "$LATEST")

ERRORS=0
MESSAGES=""

# --- Check 1: Static checker ---
# Find generated kernel files
GEN_DIR=$(find "$WORKSPACE_ROOT" -name "generated" -type d -mmin -240 2>/dev/null | head -1)
if [ -n "$GEN_DIR" ]; then
    STATIC_RESULT=$(python3 $SCRIPTS_DIR/ascendc_static_check.py "$GEN_DIR" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('PASS' if d.get('passed', False) else 'FAIL')
except:
    print('ERROR')
" 2>/dev/null)
    if [ "$STATIC_RESULT" = "FAIL" ]; then
        MESSAGES="$MESSAGES\n⚠️ Static checker FAIL — run: python3 $SCRIPTS_DIR/ascendc_static_check.py $GEN_DIR"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Check 2: Precision gate decision ---
if [ -f "$LATEST" ]; then
    GATE=$(head -1 "$LATEST" | tr -d '[:space:]')
    if [ "$GATE" != "PASS" ] && [ "$GATE" != "WAIVER" ]; then
        MESSAGES="$MESSAGES\n⚠️ Precision gate is '$GATE' — must be PASS or WAIVER before commit"
        ERRORS=$((ERRORS + 1))
    fi
else
    MESSAGES="$MESSAGES\n⚠️ No gate_decision.md found in active workspace"
    ERRORS=$((ERRORS + 1))
fi

# --- Check 3: Performance benchmark ---
PERF_FILE=$(find "$LATEST_DIR" -name "performance_report.md" -o -name "qa_report.md" 2>/dev/null | head -1)
if [ -z "$PERF_FILE" ]; then
    MESSAGES="$MESSAGES\n⚠️ No performance_report.md or qa_report.md — benchmark not run"
    ERRORS=$((ERRORS + 1))
fi

# --- Check 4: msprof data (only if optimization iteration > 0) ---
# Count iter_* dirs in the parent of LATEST_DIR (workspace/{kernel}/)
KERNEL_DIR=$(dirname "$LATEST_DIR")
ITER_DIRS=$(find "$KERNEL_DIR" -maxdepth 1 -name "iter_*" -type d 2>/dev/null | wc -l)
if [ "$ITER_DIRS" -gt 1 ]; then
    # Stage 2+ (optimization) — msprof should exist
    MSPROF_FILE=$(find "$LATEST_DIR" -name "bottleneck_diagnosis.md" 2>/dev/null | head -1)
    if [ -z "$MSPROF_FILE" ]; then
        MESSAGES="$MESSAGES\n⚠️ Optimization iterations exist but no msprof bottleneck_diagnosis.md"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Result ---
if [ "$ERRORS" -gt 0 ]; then
    echo -e "MIGRATION WORKFLOW INCOMPLETE ($ERRORS issues):$MESSAGES" >&2
    exit 2  # BLOCKING
fi

exit 0

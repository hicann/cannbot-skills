#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# aog-kernel-worker SubagentStop hook — merged replacement for check_analysis + check_static
# + check_pybind_purity + check_build_status + check_verification.
#
# Runs whenever a worker attempts to stop. Checks all contracts at once; exits 2 with stderr
# if any missing so worker sees feedback and retries in-agent.
#
# Behavior depends on what phase the worker has produced:
#   - analysis.md present → check its schema
#   - kernel/ present → check pybind purity + static check + kernel.h has AscendC APIs
#   - verification.json present → check schema + status coherent
#
# This lets the worker exit at handoff points (stuck for probe/optimizer) without
# being forced to complete all phases — the hook only rejects when produced artifacts
# are malformed, not when phases are incomplete.
set -e
. "$(dirname "$0")/_common.sh"

WS="$(find_active_workspace)"
[ -z "$WS" ] && exit 0

ERRORS=0
MESSAGES=""

# 2026-08-26 (flash_attention_score_oc, npubench route): several legacy
# CC-era checks below encode the old worker-owned verification contract.
# For NPUKernelBench workspaces the O5 runner owns precision/performance
# verification.json fields and the worker's turn legitimately revisits
# analysis.md AFTER kernel edits (phase A/B interleaved), so the mtime
# ordering and the precision.status schema checks are wrong for them.
# Detect the provider by the frozen bundle marker; these checks keep their
# full weight for every other workspace.
NPUBENCH_WS=""
if [ -d "$WS/reference_inputs/npubench" ]; then
    NPUBENCH_WS=1
fi

# --- Check 1: analysis.md (if present) ---
ANALYSIS="$WS/analysis.md"
if [ -f "$ANALYSIS" ]; then
    REQUIRED=("## Source & references" "## Dtypes & shapes" "## Precision-critical ops" \
              "## Architecture decision" "## KB Manifest" "## Precision traps" "## UB budget estimate")
    for s in "${REQUIRED[@]}"; do
        if ! grep -qF "$s" "$ANALYSIS" 2>/dev/null; then
            MESSAGES="$MESSAGES\n  - analysis.md missing section: $s"
            ERRORS=$((ERRORS + 1))
        fi
    done
    for field in "algorithm_family:" "choice:" "dtypes:"; do
        if ! grep -qE "^- *$field" "$ANALYSIS" 2>/dev/null; then
            MESSAGES="$MESSAGES\n  - analysis.md missing field: $field  (expected bullet line form: \`- $field <value>\`; a \`## $field\` heading does NOT count)"
            ERRORS=$((ERRORS + 1))
        fi
    done
    if grep -qE "(TBD|PLACEHOLDER|\{op\}|\{path\})" "$ANALYSIS"; then
        MESSAGES="$MESSAGES\n  - analysis.md has unfilled placeholders"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Check 1b (DEBT-046, 2026-04-23): kernel/*.{h,cpp} exists but analysis.md absent ---
# Phase A completion gate: worker must NOT write any kernel/ source before analysis.md exists.
KERNEL_DIR="$WS/kernel"
if [ -d "$KERNEL_DIR" ] && [ ! -f "$ANALYSIS" ]; then
    KERNEL_SRC_COUNT=$(find "$KERNEL_DIR" -maxdepth 2 \( -name "*.h" -o -name "*.cpp" -o -name "*.cc" \) 2>/dev/null | wc -l)
    if [ "$KERNEL_SRC_COUNT" -gt 0 ]; then
        MESSAGES="$MESSAGES\n  - Phase-A ordering violation (DEBT-046): kernel/ has $KERNEL_SRC_COUNT source file(s) but analysis.md missing.\n    Rule: analysis.md MUST be written and contain the mandatory sections BEFORE any kernel/*.h|.cpp write.\n    Fix: write workspace/{op}/analysis.md with the 7 required sections, then re-run Phase B."
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Check 1c (DEBT-046, 2026-04-23): kernel/* mtime ordering vs analysis.md ---
# Phase A must complete before Phase B. The ordering violation is:
# kernel source mtime >10s OLDER than analysis.md (worker wrote kernel before analysis).
# Kernel mtime NEWER than analysis is the expected/correct ordering — never flag it.
# Caught 2026-04-24 (moefinalize-kw-1 regression): earlier version concatenated a
# `-newer "$ANALYSIS" -prune` find that emitted the (correct) newer-files list as
# STALE, forcing workers to equalize all mtimes via explicit touch. Bug removed.
# npubench workspace: skipped (worker turn revisits analysis.md legitimately;
# kernel-exists-without-analysis is still enforced by Check 1b).
if [ -z "$NPUBENCH_WS" ] && [ -f "$ANALYSIS" ] && [ -d "$KERNEL_DIR" ]; then
    ANALYSIS_MTIME=$(stat -c%Y "$ANALYSIS" 2>/dev/null || echo 0)
    STALE_KERNEL=$(find "$KERNEL_DIR" -maxdepth 2 \( -name "*.h" -o -name "*.cpp" -o -name "*.cc" \) 2>/dev/null \
                   | while read f; do
                        FM=$(stat -c%Y "$f" 2>/dev/null || echo 0)
                        if [ "$FM" -gt 0 ] && [ "$((ANALYSIS_MTIME - FM))" -gt 10 ]; then echo "$f"; fi
                     done | head -3)
    if [ -n "$STALE_KERNEL" ]; then
        MESSAGES="$MESSAGES\n  - Phase-A ordering violation (DEBT-046, mtime check): kernel source(s) written >10s BEFORE analysis.md.\n    Stale file(s): $(echo $STALE_KERNEL | tr '\n' ' ')\n    Rule: Phase A (analysis.md) must complete before Phase B (kernel/ writes)."
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Check 2: kernel/ files (pybind purity + static check) ---
if [ -d "$KERNEL_DIR" ]; then
    PYBIND=$(find "$KERNEL_DIR" -maxdepth 2 -name pybind11.cpp 2>/dev/null | head -1)
    if [ -n "$PYBIND" ] && [ -f "$PYBIND" ]; then
        # Pybind purity grep (comments stripped to avoid false hits on docs)
        PYBIND_CODE="$(sed 's://.*$::' "$PYBIND")"
        BLACKLIST=('\.sum\(' '\.mean\(' '\.max\(' '\.min\(' '\.var\(' '\.std\(' \
                   '\.norm\(' '\.argmax\(' '\.argmin\(' '\.topk\(' '\.prod\(' '\.cumsum\(' \
                   '\.exp\(' '\.log\(' '\.sqrt\(' '\.rsqrt\(' '\.relu\(' '\.sigmoid\(' '\.tanh\(' \
                   '\.cpu[[:space:]]*\(' '\.to[[:space:]]*\([[:space:]]*at::kCPU' \
                   'torch::pow' 'torch::exp' 'torch::log' 'torch::sqrt' 'torch::matmul' \
                   'torch::sum' 'torch::mean' 'torch::var' 'torch::norm' 'torch::cat' 'torch::stack' \
                   'torch::bmm' 'torch::mm' 'torch::ones_like' 'at::ones_like' \
                   'at::matmul' 'at::pow' 'at::exp' 'at::log' \
                   'at::sum' 'at::mean' 'F::')
        HITS=""
        for pat in "${BLACKLIST[@]}"; do
            matches=$(printf '%s\n' "$PYBIND_CODE" | grep -nE "$pat" || true)
            [ -n "$matches" ] && HITS+="  - $pat in pybind11.cpp:\n    $matches\n"
        done
        if [ -n "$HITS" ]; then
            MESSAGES="$MESSAGES\n  Pybind compute blacklist violations (DEBT-002):\n$HITS  Rule: pybind may only do tensor metadata/alloc/contiguous. All compute in kernel."
            ERRORS=$((ERRORS + 1))
        fi
        if ! printf '%s\n' "$PYBIND_CODE" | grep -qE '#include[[:space:]]*[<"]torch_npu/csrc/core/npu/NPUStream\.h[>"]'; then
            MESSAGES="$MESSAGES\n  - pybind11.cpp is missing torch_npu NPUStream.h; use c10_npu::getCurrentNPUStream() for AscendC launch."
            ERRORS=$((ERRORS + 1))
        fi
        if ! printf '%s\n' "$PYBIND_CODE" | grep -qE 'c10_npu::getCurrentNPUStream[[:space:]]*\('; then
            MESSAGES="$MESSAGES\n  - pybind11.cpp is missing c10_npu::getCurrentNPUStream() launch handoff."
            ERRORS=$((ERRORS + 1))
        fi
    fi

    # Static check. Resolve the harness repo root at runtime (env override →
    # git toplevel → hook-relative ../../..) instead of a hardcoded home path.
    REPO_ROOT="${LOCAL_PROJECT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
    [ -z "$REPO_ROOT" ] && REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
    STATIC_SCRIPT="$REPO_ROOT/src/scripts/ascendc_static_check.py"
    if [ -f "$STATIC_SCRIPT" ]; then
        if ! python3 "$STATIC_SCRIPT" "$KERNEL_DIR" >/dev/null 2>&1; then
            OUT=$(python3 "$STATIC_SCRIPT" "$KERNEL_DIR" 2>&1 | tail -15)
            MESSAGES="$MESSAGES\n  Static check FAIL on $KERNEL_DIR:\n$OUT"
            ERRORS=$((ERRORS + 1))
        fi
    fi

    # kernel.h must have genuine AscendC primitives (prove it's not CANN wrapper)
    KERNEL_H=$(find "$KERNEL_DIR" -name "*_kernel.h" | head -1)
    if [ -n "$KERNEL_H" ]; then
        if ! grep -qE "DataCopy|TQue|TBuf" "$KERNEL_H" 2>/dev/null; then
            MESSAGES="$MESSAGES\n  - $KERNEL_H has no DataCopy/TQue/TBuf — not a real AscendC kernel"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

# --- Check 3: verification.json (if present) ---
VJSON="$WS/verification.json"
if [ -f "$VJSON" ] && [ -z "$NPUBENCH_WS" ]; then
    if ! VERR=$(python3 - "$VJSON" 2>&1 <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p) as f: d = json.load(f)
except Exception as e:
    print(f"verification.json invalid JSON: {e}", file=sys.stderr)
    sys.exit(1)
errors = []
prec = d.get("precision")
if not isinstance(prec, dict):
    errors.append("missing 'precision' dict")
else:
    for k in ("pass", "total", "status"):
        if k not in prec:
            errors.append(f"precision missing '{k}'")
    if prec.get("status") not in ("PASS", "FAIL"):
        errors.append(f"precision.status must be PASS|FAIL, got {prec.get('status')}")
perf = d.get("performance")
if not isinstance(perf, dict):
    errors.append("missing 'performance' dict")
else:
    for k in ("reference_median_us", "ascendc_median_us", "ratio", "status"):
        if k not in perf:
            errors.append(f"performance missing '{k}'")
# Optional: determinism dimension (V3.2+). Validate shape if present; absence is
# not an error yet (Group 3 will make it mandatory when worker spec is updated).
det = d.get("determinism")
if det is not None:
    if not isinstance(det, dict):
        errors.append("'determinism' must be dict if present")
    else:
        policy = det.get("policy")
        if policy not in ("required", "best_effort", "n/a"):
            errors.append(f"determinism.policy must be required|best_effort|n/a, got {policy!r}")
        if policy != "n/a":
            if "observed_deterministic" not in det:
                errors.append("determinism missing 'observed_deterministic' (required when policy != n/a)")
            if "policy_satisfied" not in det:
                errors.append("determinism missing 'policy_satisfied'")
if errors:
    for e in errors: print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
    ); then
        MESSAGES="$MESSAGES\n  verification.json schema errors:\n$VERR"
        ERRORS=$((ERRORS + 1))
    fi

    # Coherence: if worker claims done, precision must be PASS
    if grep -qE "→ orchestrator: done" "$WS/PROGRESS.md" 2>/dev/null; then
        STATUS=$(python3 - "$VJSON" 2>/dev/null <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    print(json.load(f)["precision"]["status"])
PY
        ) || STATUS="UNKNOWN"
        if [ "$STATUS" != "PASS" ]; then
            MESSAGES="$MESSAGES\n  worker claims done but verification.json precision.status=$STATUS"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

# --- Check 4: DIAG iter-level enforcement (only when .diag_enabled + verification.json) ---
# DIAG iter-level checks: WARN-only (per 2026-04-17 user feedback — blocking 3h
# of work to enforce logging is bad ROI). Gaps recorded to PROGRESS.md so
# orchestrator + external monitoring can see compliance issues without losing work.
if [ -f "$WS/.diag_enabled" ] && [ -f "$WS/verification.json" ]; then
    PROG="$WS/PROGRESS.md"

    # Rule 1: at least 1 Phase D iter entry (verification.json exists = Phase D ran)
    # grep -c emits "0" on no-match AND exits 1 → use `|| true` to keep set -e happy
    D_ITERS=$(grep -cE "^### \[[0-9:]+\] aog-kernel-worker \(Phase D iter [0-9]+\)" "$PROG" 2>/dev/null || true)
    D_ITERS=${D_ITERS:-0}
    if [ "$D_ITERS" -lt 1 ]; then
        log_gap_to_progress "aog-kernel-worker" "verification.json present but no 'Phase D iter N' entries (DIAG mode)" "$PROG"
    fi

    # Rule 2: each Phase [CD] iter entry must be followed by a matching DIAG section
    MISSING=$(python3 - "$PROG" 2>/dev/null <<'PY'
import re, sys
text = open(sys.argv[1]).read()
iters = re.findall(r'^### \[[0-9:]+\] aog-kernel-worker \(Phase ([CD]) iter (\d+)\)', text, re.M)
diags = set(re.findall(r'^### DIAG: Phase ([CD]) iter (\d+)', text, re.M))
missing = [f"{p}{n}" for p,n in iters if (p,n) not in diags]
print(','.join(missing) if missing else '')
PY
)
    if [ -n "$MISSING" ]; then
        log_gap_to_progress "aog-kernel-worker" "Phase iters without matching DIAG section: $MISSING" "$PROG"
    fi

    # Rule 3: iter numbers monotonic from 1 (no gaps, no repeats per phase)
    for P in C D; do
        NUMS=$(grep -oE "Phase $P iter [0-9]+" "$PROG" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | uniq)
        [ -z "$NUMS" ] && continue
        EXPECTED=1
        for n in $NUMS; do
            if [ "$n" != "$EXPECTED" ]; then
                log_gap_to_progress "aog-kernel-worker" "Phase $P iter numbers non-monotonic: $(echo $NUMS | tr '\n' ' ')" "$PROG"
                break
            fi
            EXPECTED=$((EXPECTED + 1))
        done
    done

    # Rule 4 (lazy-load contract): if Phase D ran ≥ 2 iter (meaning at least 1 fail
    # happened), worker must have loaded precision.md per agent def Phase D step a0.
    # Soft warn only — aog-kernel-worker may have grokked precision from Phase A KB.
    D_MAX=$(grep -oE "Phase D iter [0-9]+" "$PROG" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1)
    D_MAX=${D_MAX:-0}
    if [ "$D_MAX" -ge 2 ] && ! grep -qF "precision.md loaded" "$PROG" 2>/dev/null; then
        log_gap_to_progress "aog-kernel-worker" "Phase D iter ≥2 but no 'precision.md loaded' marker (lazy-load step a0 skipped — see worker.md §Phase D step 4a0)" "$PROG"
    fi
fi

# --- Emit result ---
if [ "$ERRORS" -gt 0 ]; then
    echo -e "❌ aog-kernel-worker: $ERRORS contract violation(s):$MESSAGES" >&2
    echo "Fix these before exiting. You're still in-agent; Edit and retry are available." >&2
    exit 2
fi

exit 0

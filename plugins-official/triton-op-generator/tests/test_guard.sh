#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# test_guard.sh — UT for guard-baseline-paths.sh
#
# Runs without external test framework. Each case feeds a synthetic PreToolUse
# JSON payload into the hook script and checks exit code + stdout.
#
# Usage:  bash test_guard.sh
# Exit:   0 if all pass, 1 if any fail
# ----------------------------------------------------------------------------------------------------------

set -u

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
HOOK_DIR="$PLUGIN_ROOT/.claude/hooks"
HOOK="$HOOK_DIR/guard-baseline-paths.sh"
CONFIG="$HOOK_DIR/guard-config.json"

PASS=0
FAIL=0
FAILED_CASES=()

# helpers --------------------------------------------------------------------

# Run hook with given stdin JSON; capture stdout and exit code.
run_hook() {
    local stdin_json="$1"
    HOOK_STDOUT="$(printf '%s' "$stdin_json" | bash "$HOOK" 2>/dev/null)"
    HOOK_EXIT=$?
}

# Assert last run produced a deny decision.
expect_deny() {
    if [ "$HOOK_EXIT" -ne 0 ]; then
        return 1
    fi
    printf '%s' "$HOOK_STDOUT" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null 2>&1
}

# Assert last run allowed (no deny in output, exit 0).
expect_allow() {
    if [ "$HOOK_EXIT" -ne 0 ]; then
        return 1
    fi
    # Either empty output, or output that does NOT contain deny
    if [ -z "$HOOK_STDOUT" ]; then
        return 0
    fi
    ! printf '%s' "$HOOK_STDOUT" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null 2>&1
}

# Record result
record() {
    # $1 = case name, $2 = 0 (pass) or 1 (fail)
    local name="$1" ok="$2"
    if [ "$ok" -eq 0 ]; then
        PASS=$((PASS + 1))
        echo "  PASS  $name"
    else
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("$name")
        echo "  FAIL  $name"
        echo "        exit=$HOOK_EXIT stdout=${HOOK_STDOUT:-<empty>}"
    fi
}

# Test case runner: $1=name, $2=stdin json, $3=expected (deny|allow)
case_() {
    local name="$1" json="$2" expected="$3"
    run_hook "$json"
    if [ "$expected" = "deny" ]; then
        expect_deny && record "$name" 0 || record "$name" 1
    else
        expect_allow && record "$name" 0 || record "$name" 1
    fi
}

# PreToolUse input builder
mk_input() {
    # $1 = tool_name, $2 = file_path
    jq -nc --arg tool "$1" --arg path "$2" \
        '{tool_name:$tool, tool_input:{file_path:$path}}'
}

echo "=== test_guard.sh ==="
echo "  hook:  $HOOK"
echo "  config: $CONFIG"
echo ""

# ---------- protected paths (must live under ascendc-kernelgen-data*/) ----------

case_ "protected: kernelgen-data/npu_benchmark/level4 .py" \
    "$(mk_input Edit /home/w00934874/agent/code/run_batch/ascendc-kernelgen-data/npu_benchmark/level4/7_SparseFlashAttention.py)" \
    deny

case_ "protected: kernelgen-data_old/npu_benchmark/level3 .py" \
    "$(mk_input Edit /home/foo/ascendc-kernelgen-data_old/npu_benchmark/level3/10_ConvTranspose2d.py)" \
    deny

case_ "protected: Write tool also blocked" \
    "$(mk_input Write /home/u/ascendc-kernelgen-data/npu_benchmark/x.py)" \
    deny

case_ "protected: nested level path still blocked" \
    "$(mk_input Edit /home/u/ascendc-kernelgen-data/npu_benchmark/level1/level2/op.py)" \
    deny

case_ "protected: .json config also blocked" \
    "$(mk_input Write /home/u/ascendc-kernelgen-data/npu_benchmark/level1/1_GELU.json)" \
    deny

# ---------- regression: 'benchmark' in unrelated dirs must NOT be blocked ----------

case_ "regression: my_benchmark/ (not kernelgen-data) → allow" \
    "$(mk_input Edit /home/user/projects/my_benchmark/foo.py)" \
    allow

case_ "regression: benchmark_utils/ (not kernelgen-data) → allow" \
    "$(mk_input Edit /home/user/code/benchmark_utils/op.py)" \
    allow

case_ "regression: npu_benchmark NOT under kernelgen-data → allow" \
    "$(mk_input Edit /home/user/some_other_repo/npu_benchmark/op.py)" \
    allow

# ---------- allowlist paths ----------

case_ "allowlist: triton_ascend_output" \
    "$(mk_input Edit /tmp/triton_ascend_output/op_x/output/iter_0/generated_code.py)" \
    allow

case_ "allowlist: .claude/skills/" \
    "$(mk_input Edit /home/u/.claude/skills/triton-op-coding/SKILL.md)" \
    allow

case_ "allowlist: .claude/template/" \
    "$(mk_input Edit /home/u/.claude/template/softmax.md)" \
    allow

case_ "allowlist: arbitrary non-protected path" \
    "$(mk_input Edit /tmp/random/op.py)" \
    allow

# ---------- self_guard ----------

case_ "self_guard: settings.json" \
    "$(mk_input Edit /home/u/.claude/settings.json)" \
    deny

case_ "self_guard: settings.local.json" \
    "$(mk_input Edit /home/u/.claude/settings.local.json)" \
    deny

case_ "self_guard: hook script itself" \
    "$(mk_input Edit /home/u/.claude/hooks/guard-baseline-paths.sh)" \
    deny

case_ "self_guard: guard-config.json" \
    "$(mk_input Write /home/u/.claude/hooks/guard-config.json)" \
    deny

# ---------- non-Edit/Write tools ----------

case_ "non-Edit tool (Read) on protected path → allow (matcher handles it)" \
    "$(mk_input Read /home/x/npu_benchmark/op.py)" \
    allow

# ---------- malformed input (fail-open) ----------

case_ "empty file_path → allow" \
    '{"tool_name":"Edit","tool_input":{}}' \
    allow

case_ "missing tool_input → allow" \
    '{"tool_name":"Edit"}' \
    allow

# ---------- config fallback ----------

# Temporarily move config away; protected path should still be denied via fallback.
echo ""
echo "  -- with config moved aside (fallback path) --"
mv "$CONFIG" "$CONFIG.bak"
case_ "fallback: protected npu_benchmark still denied without config" \
    "$(mk_input Edit /home/x/ascendc-kernelgen-data/npu_benchmark/op.py)" \
    deny
case_ "fallback: arbitrary path allowed without config" \
    "$(mk_input Edit /tmp/random.py)" \
    allow
mv "$CONFIG.bak" "$CONFIG"

# ---------- summary ----------

echo ""
echo "=== summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "  Failed cases:"
    for c in "${FAILED_CASES[@]}"; do
        echo "    - $c"
    done
    exit 1
fi

echo "  All tests passed."
exit 0

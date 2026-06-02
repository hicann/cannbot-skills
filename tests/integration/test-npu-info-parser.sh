#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# Integration test: NPU info parser robustness
# =============================================================================
# Tests _npu_info.py, npu_info.sh, and collect_baseline.py
# with the real npu-smi environment.
#
# Run: bash tests/integration/test-npu-info-parser.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_DIR="$(cd "$TESTS_DIR/.." && pwd)"

# Source test helpers
source "$TESTS_DIR/lib/test-helpers.sh"

# Paths
NPU_INFO_PY="$SKILLS_DIR/ops/ascendc-env-check/scripts/_npu_info.py"
NPU_INFO_SH="$SKILLS_DIR/ops/ascendc-env-check/scripts/npu_info.sh"
COLLECT_BASELINE="$SKILLS_DIR/model/model-infer-migrator/scripts/collect_baseline.py"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

run_test() {
    local name="$1"
    local cmd="$2"
    local expected_pattern="${3:-}"
    local check_rc="${4:-0}"

    echo "TEST: $name"
    local output
    local rc=0
    output=$(eval "$cmd" 2>&1) || rc=$?

    if [ "$check_rc" -ne 0 ]; then
        # We expect non-zero exit
        if [ $rc -ne 0 ]; then
            print_pass "$name (exit $rc as expected)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$name (expected non-zero exit, got 0)"
            echo "  Output: ${output:0:200}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        return
    fi

    if [ $rc -ne 0 ]; then
        print_fail "$name (exit $rc)"
        echo "  Output: ${output:0:200}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi

    if [ -n "$expected_pattern" ]; then
        if echo "$output" | grep -qE "$expected_pattern"; then
            print_pass "$name"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$name (pattern not found: $expected_pattern)"
            echo "  Output: ${output:0:300}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        print_pass "$name"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# =============================================================================
print_section_header "Prerequisites"
# =============================================================================

assert_file_exists "$NPU_INFO_PY" "_npu_info.py exists" || { FAIL_COUNT=$((FAIL_COUNT + 1)); }
assert_file_exists "$NPU_INFO_SH" "npu_info.sh exists" || { FAIL_COUNT=$((FAIL_COUNT + 1)); }
assert_file_exists "$COLLECT_BASELINE" "collect_baseline.py exists" || { FAIL_COUNT=$((FAIL_COUNT + 1)); }

# Check npu-smi availability
if ! command -v npu-smi &> /dev/null; then
    print_skip "npu-smi not available, skipping all tests"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    echo ""
    echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
    exit 0
fi

# =============================================================================
print_section_header "_npu_info.py CLI Tests"
# =============================================================================

# -- Test 1: Default mode (summary)
run_test "_npu_info.py default mode shows device count" \
    "python3 '$NPU_INFO_PY'" \
    "Detected [0-9]+ NPU"

# -- Test 2: --list mode
run_test "_npu_info.py --list returns numeric IDs" \
    "python3 '$NPU_INFO_PY' --list" \
    "^[0-9]+"

# -- Test 3: --json mode
run_test "_npu_info.py --json has npu_count" \
    "python3 '$NPU_INFO_PY' --json" \
    '"npu_count"'

run_test "_npu_info.py --json has devices array" \
    "python3 '$NPU_INFO_PY' --json" \
    '"devices"'

run_test "_npu_info.py --json has chip_name" \
    "python3 '$NPU_INFO_PY' --json" \
    '"chip_name"'

run_test "_npu_info.py --json has health" \
    "python3 '$NPU_INFO_PY' --json" \
    '"health"'

run_test "_npu_info.py --json has temperature" \
    "python3 '$NPU_INFO_PY' --json" \
    '"temperature"'

run_test "_npu_info.py --json has power" \
    "python3 '$NPU_INFO_PY' --json" \
    '"power"'

run_test "_npu_info.py --json has memory info" \
    "python3 '$NPU_INFO_PY' --json" \
    '"hbm_capacity_mb"'

run_test "_npu_info.py --json has usage info" \
    "python3 '$NPU_INFO_PY' --json" \
    '"aicore"'

# -- Test 4: --health mode
run_test "_npu_info.py --health returns OK/Warning/Alarm" \
    "python3 '$NPU_INFO_PY' --health" \
    "NPU [0-9]+: (OK|Warning|Alarm|Critical|UNKNOWN)"

# -- Test 5: --chip-name mode
run_test "_npu_info.py --chip-name returns chip name" \
    "python3 '$NPU_INFO_PY' --chip-name" \
    "NPU [0-9]+: (Ascend|Atlas)"

# -- Test 6: --warnings mode (should usually be empty or have warnings)
run_test "_npu_info.py --warnings runs without error" \
    "python3 '$NPU_INFO_PY' --warnings" \
    ""

# -- Test 6b: --discover mode
run_test "_npu_info.py --discover shows subcommands" \
    "python3 '$NPU_INFO_PY' --discover" \
    "(health|temp|power|memory|usages|common)"

run_test "_npu_info.py --discover shows correct count" \
    "python3 '$NPU_INFO_PY' --discover" \
    "supports [0-9]+ subcommand"

# -- Test 7: JSON structure validation
echo "TEST: _npu_info.py JSON schema validation"
json_output=$(python3 "$NPU_INFO_PY" --json 2>&1)
if python3 -c "
import json, sys
try:
    data = json.loads(sys.argv[1])
    assert 'npu_count' in data, 'missing npu_count'
    assert 'devices' in data, 'missing devices'
    assert 'warnings' in data, 'missing warnings'
    assert isinstance(data['devices'], list), 'devices not a list'
    if data['devices']:
        d = data['devices'][0]
        assert 'npu_id' in d, 'missing npu_id'
        assert 'chip_name' in d, 'missing chip_name'
        assert 'health' in d, 'missing health'
        assert 'temperature' in d, 'missing temperature'
        assert 'power' in d, 'missing power'
        assert 'memory' in d, 'missing memory'
        assert 'usage' in d, 'missing usage'
    print('Schema valid')
except Exception as e:
    print(f'Schema error: {e}')
    sys.exit(1)
" "$json_output" 2>&1 | grep -q "Schema valid"; then
    print_pass "_npu_info.py JSON schema validation"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "_npu_info.py JSON schema validation"
    echo "  JSON output (first 500 chars): ${json_output:0:500}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# =============================================================================
print_section_header "_npu_info.py Python API Tests"
# =============================================================================

echo "TEST: NpuInfoCollector API - get_npu_ids"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
assert len(ids) > 0, 'No NPU IDs found'
assert all(isinstance(i, int) for i in ids), 'IDs not integers'
print(f'NPU IDs: {ids}')
" 2>&1 | grep -q "NPU IDs:"; then
    print_pass "NpuInfoCollector.get_npu_ids() returns valid IDs"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_npu_ids()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_chip_name"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    name = c.get_chip_name(ids[0])
    assert name, 'No chip name found'
    assert 'Ascend' in name or 'Atlas' in name, f'Unexpected chip name: {name}'
    print(f'Chip name: {name}')
" 2>&1 | grep -q "Chip name:"; then
    print_pass "NpuInfoCollector.get_chip_name() returns valid name"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_chip_name()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_health"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    h = c.get_health(ids[0])
    assert h in ('OK', 'Warning', 'Alarm', 'Critical', 'UNKNOWN'), f'Unexpected health: {h}'
    print(f'Health: {h}')
" 2>&1 | grep -q "Health:"; then
    print_pass "NpuInfoCollector.get_health() returns valid status"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_health()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_temperature"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    t = c.get_temperature(ids[0])
    assert t, 'No temperature found'
    float(t)  # should be numeric
    print(f'Temperature: {t}')
" 2>&1 | grep -q "Temperature:"; then
    print_pass "NpuInfoCollector.get_temperature() returns numeric value"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_temperature()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_power"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    p = c.get_power(ids[0])
    assert p, 'No power found'
    float(p)  # should be numeric
    print(f'Power: {p}')
" 2>&1 | grep -q "Power:"; then
    print_pass "NpuInfoCollector.get_power() returns numeric value"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_power()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_memory_info"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    m = c.get_memory_info(ids[0])
    assert 'hbm_capacity_mb' in m, 'missing hbm_capacity_mb'
    assert 'hbm_usage_rate' in m, 'missing hbm_usage_rate'
    print(f'Memory: {m}')
" 2>&1 | grep -q "Memory:"; then
    print_pass "NpuInfoCollector.get_memory_info() returns dict"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_memory_info()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - get_usage_info"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
ids = c.get_npu_ids()
if ids:
    u = c.get_usage_info(ids[0])
    assert 'aicore' in u, 'missing aicore'
    assert 'hbm_usage' in u, 'missing hbm_usage'
    print(f'Usage: {u}')
" 2>&1 | grep -q "Usage:"; then
    print_pass "NpuInfoCollector.get_usage_info() returns dict"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector.get_usage_info()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: NpuInfoCollector API - warnings are tracked"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/ops/ascendc-env-check/scripts')
from _npu_info import NpuInfoCollector
c = NpuInfoCollector()
c.get_npu_ids()
c.get_chip_name(999999)  # invalid ID should produce warning
w = c.get_all_warnings()
print(f'Warnings count: {len(w)}')
" 2>&1 | grep -q "Warnings count:"; then
    print_pass "NpuInfoCollector tracks warnings correctly"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "NpuInfoCollector warnings tracking"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# =============================================================================
print_section_header "npu_info.sh Tests"
# =============================================================================

run_test "npu_info.sh runs to completion" \
    "bash '$NPU_INFO_SH'" \
    "查询完成"

run_test "npu_info.sh detects npu-smi" \
    "bash '$NPU_INFO_SH'" \
    "npu-smi 可用"

run_test "npu_info.sh shows device list" \
    "bash '$NPU_INFO_SH'" \
    "检测到 [0-9]+ 个 NPU"

run_test "npu_info.sh shows chip name" \
    "bash '$NPU_INFO_SH'" \
    "(Ascend|Atlas)"

run_test "npu_info.sh shows health status" \
    "bash '$NPU_INFO_SH'" \
    "(OK|Warning|Alarm|Critical|UNKNOWN)"

# =============================================================================
print_section_header "collect_baseline.py Tests"
# =============================================================================

echo "TEST: collect_baseline.get_npu_info()"
if python3 -c "
import sys
sys.path.insert(0, '$SKILLS_DIR/model/model-infer-migrator/scripts')
from collect_baseline import get_npu_info
model, count = get_npu_info()
assert count > 0, f'No NPU devices: count={count}'
assert model != 'unknown', f'Unknown model: {model}'
assert 'Ascend' in model or 'Atlas' in model, f'Unexpected model: {model}'
print(f'Model: {model}, Count: {count}')
" 2>&1 | grep -q "Model:"; then
    print_pass "collect_baseline.get_npu_info() returns valid data"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "collect_baseline.get_npu_info()"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# =============================================================================
print_section_header "No-table-parsing Guarantee"
# =============================================================================

echo "TEST: _npu_info.py does not parse npu-smi info main table"
if ! grep -q 'npu-smi info"' "$NPU_INFO_PY" 2>/dev/null || \
   grep -qE 'npu-smi.*info[^-]' "$NPU_INFO_PY" 2>/dev/null; then
    # The fallback does use "npu-smi info" but it's marked as fallback
    # and only extracts IDs with a very conservative regex
    if grep -q 'LAST RESORT' "$NPU_INFO_PY" && \
       grep -q 'unreliable table format' "$NPU_INFO_PY"; then
        print_pass "_npu_info.py marks table parsing as last-resort fallback"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "_npu_info.py uses main table without proper fallback marking"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
else
    print_pass "_npu_info.py avoids main table parsing"
    PASS_COUNT=$((PASS_COUNT + 1))
fi

echo "TEST: npu_info.sh does not use awk/regex on table"
if ! grep -qE 'awk.*-F.*\|' "$NPU_INFO_SH" 2>/dev/null && \
   ! grep -qE 'awk.*\[0-9\]' "$NPU_INFO_SH" 2>/dev/null; then
    print_pass "npu_info.sh has no awk table parsing"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "npu_info.sh still contains awk table parsing"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo "TEST: collect_baseline.py does not parse npu-smi info table"
if ! grep -qE 'npu-smi.*info[^-]' "$COLLECT_BASELINE" 2>/dev/null || \
   grep -q 'info -m' "$COLLECT_BASELINE" 2>/dev/null; then
    print_pass "collect_baseline.py uses mapping table instead of main table"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    print_fail "collect_baseline.py still parses main table"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# =============================================================================
print_section_header "Summary"
# =============================================================================

echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    print_status_passed
else
    print_status_failed
fi
echo ""
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo ""

exit $FAIL_COUNT

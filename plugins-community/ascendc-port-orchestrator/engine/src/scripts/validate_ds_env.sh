#!/usr/bin/env bash
# validate_ds_env.sh — preflight check for DeepSeek-env A3 deploy readiness.
# Runs once per session before any orchestrator spawn.
# Saves 30+ min per op of "discover-at-infra-step-14" pattern.
#
# Usage: bash src/scripts/validate_ds_env.sh [--json]
#   --json  emit JSON verdict line for machine consumption
#
# Checks:
#   (i)   SSH to A3_HOST reachable
#   (ii)  $CANN_PATH/lib64 populated + libascendcl.so loadable
#   (iii) torch_npu importable on A3 container
#   (iv)  aclrtLaunchKernel symbol resolvable via nm
#   (v)   NPU device visible and idle
#
# Exit code: 0 = all green, 1 = WARN, 2 = BLOCK

set -euo pipefail

JSON_OUT=false
if [[ "${1:-}" == "--json" ]]; then
    JSON_OUT=true
fi

# Load target env (TARGET, A3_HOST, A3_CONTAINER, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../../workspace/.ascendc_env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

A3_HOST="${A3_HOST:-198.51.100.92}"
A3_CONTAINER="${A3_CONTAINER:-npu-a3}"
CANN_PATH="${A3_CANN_PATH:-/usr/local/Ascend/cann}"
NPU_DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-2}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

WARNINGS=0
FAILURES=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARNINGS=$((WARNINGS + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILURES=$((FAILURES + 1)); }

echo "=== DS Env Preflight ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "  TARGET: ${TARGET:-unset}"
echo "  A3_HOST: $A3_HOST"
echo "  Container: $A3_CONTAINER"
echo "  CANN: $CANN_PATH"
echo "  NPU device: $NPU_DEVICE"
echo ""

# ---- (i) SSH reachable ----
echo "--- (i) SSH to A3_HOST ---"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 root@${A3_HOST}"
if $SSH_CMD "echo OK" >/dev/null 2>&1; then
    log_pass "SSH to $A3_HOST reachable"
else
    log_fail "SSH to $A3_HOST failed — check connectivity / key / perms"
fi

# ---- (ii) CANN libs ----
echo "--- (ii) CANN lib64 ---"
CANN_LIB64="${CANN_PATH}/lib64"
if $SSH_CMD "test -d $CANN_LIB64 && ls $CANN_LIB64/libascendcl.so* 2>/dev/null" >/dev/null 2>&1; then
    log_pass "$CANN_LIB64 populated, libascendcl.so found"
else
    log_fail "$CANN_LIB64 missing or no libascendcl.so"
fi

# ---- (iii) torch_npu importable ----
echo "--- (iii) torch_npu import ---"
if $SSH_CMD "docker exec $A3_CONTAINER bash -c 'source ${CANN_PATH}/set_env.sh 2>/dev/null; python3 -c \"import torch; import torch_npu; print(torch_npu.__version__)\" 2>&1'" 2>&1 | grep -qE '[0-9]+\.[0-9]+'; then
    log_pass "torch_npu importable on $A3_CONTAINER"
else
    log_fail "torch_npu import failed on $A3_CONTAINER"
fi

# ---- (iv) aclrtLaunchKernel symbol resolution ----
echo "--- (iv) aclrtLaunchKernel symbol ---"
if $SSH_CMD "docker exec $A3_CONTAINER bash -c 'nm -D ${CANN_PATH}/lib64/libascendcl.so 2>/dev/null | grep -q aclrtLaunchKernel'" 2>&1; then
    log_pass "aclrtLaunchKernel symbol found in libascendcl.so"
else
    log_warn "aclrtLaunchKernel symbol NOT found — check LD_LIBRARY_PATH"
fi

# ---- (v) NPU device visible + idle ----
echo "--- (v) NPU device $NPU_DEVICE ---"
NPU_INFO=$($SSH_CMD "docker exec $A3_CONTAINER npu-smi info -t 0 -i $NPU_DEVICE 2>/dev/null" 2>/dev/null || echo "")
if echo "$NPU_INFO" | grep -q "OK"; then
    log_pass "NPU $NPU_DEVICE visible and healthy"
else
    log_warn "NPU $NPU_DEVICE status unclear — check npu-smi"
fi

echo ""
echo "--- Summary ---"
echo "  Passes:  $((5 - WARNINGS - FAILURES))/5"
echo "  Warnings: $WARNINGS"
echo "  Failures: $FAILURES"

if $JSON_OUT; then
    if [[ $FAILURES -gt 0 ]]; then
        VERDICT="BLOCK"
    elif [[ $WARNINGS -gt 0 ]]; then
        VERDICT="WARN"
    else
        VERDICT="PASS"
    fi
    echo "{\"verdict\":\"$VERDICT\",\"passes\":$((5-WARNINGS-FAILURES)),\"warnings\":$WARNINGS,\"failures\":$FAILURES,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
fi

if [[ $FAILURES -gt 0 ]]; then
    exit 2
elif [[ $WARNINGS -gt 0 ]]; then
    exit 1
fi
exit 0

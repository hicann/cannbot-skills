#!/bin/bash
# resolve_target.sh — Multi-target resolver for AscendC harness (V3.4)
#
# Reads workspace/.ascendc_env, looks at $TARGET (default: a5), and exports
# the unprefixed `HOST`, `USER`, `PASSWORD`, `CONTAINER`, `CANN_PATH`,
# `SOC_VERSION`, plus a derived `PLATFORM_SIMT` capability flag.
#
# Usage (source-style; preferred):
#   source src/scripts/resolve_target.sh
#   echo "Target=$TARGET host=$HOST soc=$SOC_VERSION simt=$PLATFORM_SIMT"
#
# Usage (override target via env, no edit needed):
#   TARGET=a3 source src/scripts/resolve_target.sh
#
# Capability rules (derived, not user-editable):
#   PLATFORM_SIMT=true    only when TARGET=a5
#   PLATFORM_SIMT=false   for a2/a3 (V220 arch — SIMD-only)
#   ARCH_CODE=arch35      for a5
#   ARCH_CODE=arch22      for a2/a3
#   UB_PER_AIV_KB=256     for a5
#   UB_PER_AIV_KB=192     for a2/a3
set -e

# Find .ascendc_env relative to caller's PROJECT root (LOCAL_PROJECT) or cwd
ENV_FILE="${ASCENDC_ENV_FILE:-}"
if [ -z "$ENV_FILE" ]; then
    for candidate in \
        "$PWD/workspace/.ascendc_env" \
        "$(dirname "${BASH_SOURCE[0]}")/../../workspace/.ascendc_env"; do
        if [ -f "$candidate" ]; then
            ENV_FILE="$candidate"
            break
        fi
    done
fi

if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
    echo "resolve_target.sh: cannot locate workspace/.ascendc_env (set ASCENDC_ENV_FILE to override)" >&2
    return 1 2>/dev/null || exit 1
fi

# Preserve a TARGET passed in via environment (e.g. `TARGET=a3 source resolve_target.sh`).
# The .ascendc_env file also sets TARGET, but env override wins so /ascendc-op-gen --target=
# can switch chips without rewriting the config file.
_TARGET_ENV_OVERRIDE="${TARGET:-}"

# shellcheck disable=SC1090
set -a
. "$ENV_FILE"
set +a

# Restore env override if one was set
if [ -n "$_TARGET_ENV_OVERRIDE" ]; then
    TARGET="$_TARGET_ENV_OVERRIDE"
fi
unset _TARGET_ENV_OVERRIDE

# Default target
TARGET="${TARGET:-a5}"
TARGET="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"

# Normalize DS-backend isolation targets: a3-ds → a3, a2-ds → a2
# These are backend-isolation targets (DeepSeek V4 CC backend — separate instance of
# the harness sharing the same A3/A2 hardware). Strip the -ds suffix so HOST/CONTAINER
# resolution uses the base target's env vars (A3_HOST, A3_CONTAINER, etc.).
case "$TARGET" in
    *-ds) TARGET="${TARGET%-ds}" ;;
esac

case "$TARGET" in
    a5|a3|a2) ;;
    *)
        echo "resolve_target.sh: invalid TARGET='$TARGET' (must be a5|a3|a2)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

UPPER="$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')"

# Indirect expansion via eval (works under bash and POSIX-ish sh)
eval "HOST=\${${UPPER}_HOST:-}"
eval "USER=\${${UPPER}_USER:-root}"
eval "PASSWORD=\${${UPPER}_PASSWORD:-}"
eval "CONTAINER=\${${UPPER}_CONTAINER:-}"
eval "CANN_PATH=\${${UPPER}_CANN_PATH:-}"
eval "SOC_VERSION=\${${UPPER}_SOC_VERSION:-}"
# NPU-python bin dir + extra run-env LD prefix (target-specific, fall back to generic).
# EXTRA_LD_LIBRARY_PATH: prepended to the run env's LD_LIBRARY_PATH; needed on multi-CANN boxes
# where the runtime acl/hccl libs live in a different CANN than the compile toolkit (see
# ascendc_env.template + phase_o5_runner._resolve_extra_ld). Both default empty.
eval "NPU_PYTHON_BIN=\${${UPPER}_NPU_PYTHON_BIN:-\${NPU_PYTHON_BIN:-}}"
eval "EXTRA_LD_LIBRARY_PATH=\${${UPPER}_EXTRA_LD_LIBRARY_PATH:-\${EXTRA_LD_LIBRARY_PATH:-}}"
# SSH key (2026-06-20, FA-grad .141 V351 lane): an OPTIONAL explicit ssh identity for
# key-auth lanes (the .141/.171 V351 hosts use id_ca_team, not a password). Target-
# specific `{TARGET}_SSH_KEY` falls back to a generic `SSH_KEY`; empty (default) →
# unchanged legacy auth (password or default key) in deploy_to_npu.sh.
eval "SSH_KEY=\${${UPPER}_SSH_KEY:-\${SSH_KEY:-}}"

# Capability derivation — single source of truth for chip-arch facts.
# Update this block (and references/hardware/INDEX.md) if a new SOC family is added.
case "$TARGET" in
    a5)
        PLATFORM_SIMT=true
        ARCH_CODE=arch35
        NPU_ARCH=3510
        UB_PER_AIV_KB=256
        L0C_KB=256
        ;;
    a3|a2)
        PLATFORM_SIMT=false
        ARCH_CODE=arch22
        NPU_ARCH=2201
        UB_PER_AIV_KB=192
        L0C_KB=128
        ;;
esac

export TARGET HOST USER PASSWORD CONTAINER CANN_PATH SOC_VERSION
export NPU_PYTHON_BIN EXTRA_LD_LIBRARY_PATH SSH_KEY
export PLATFORM_SIMT ARCH_CODE NPU_ARCH UB_PER_AIV_KB L0C_KB

# Validate required fields for active target
if [ -z "$HOST" ] || [ -z "$CONTAINER" ] || [ -z "$SOC_VERSION" ]; then
    echo "resolve_target.sh: TARGET=$TARGET selected but ${UPPER}_HOST/${UPPER}_CONTAINER/${UPPER}_SOC_VERSION not configured. Update workspace/.ascendc_env from its template (created by this plugin's init.sh)." >&2
    return 1 2>/dev/null || exit 1
fi

# When invoked directly (not sourced), print the resolved set
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    cat <<EOF
TARGET=$TARGET
HOST=$HOST
USER=$USER
CONTAINER=$CONTAINER
CANN_PATH=$CANN_PATH
SOC_VERSION=$SOC_VERSION
NPU_PYTHON_BIN=$NPU_PYTHON_BIN
EXTRA_LD_LIBRARY_PATH=$EXTRA_LD_LIBRARY_PATH
PLATFORM_SIMT=$PLATFORM_SIMT
ARCH_CODE=$ARCH_CODE
NPU_ARCH=$NPU_ARCH
UB_PER_AIV_KB=$UB_PER_AIV_KB
L0C_KB=$L0C_KB
EOF
fi

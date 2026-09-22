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
#   PLATFORM_SIMT=false   for a2/a3 (V220 arch — SIMD-only) and 310p (dav-m200 — no SIMT)
#   ARCH_CODE=arch35      for a5
#   ARCH_CODE=arch22      for a2/a3
#   ARCH_CODE=arch20     for 310p
#   UB_PER_AIV_KB=256     for a5
#   UB_PER_AIV_KB=192     for a2/a3
#   UB_PER_AIV_KB=256     for 310p
#   L0C_KB=256            for 310p
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
    a5|a3|a2|310p) ;;
    *)
        echo "resolve_target.sh: invalid TARGET='$TARGET' (must be a5|a3|a2|310p)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# 310p env keys use the ASCEND310P_ prefix ("310P_*" starts with a digit —
# not a legal shell identifier, so ${310P_HOST}-style lookups can never
# resolve). Fail loud if the legacy spelling is present so a silent A5_*
# fallback can never name the wrong chip.
case "$(grep -cE '^310P_[A-Za-z0-9_]+=' "$ENV_FILE" 2>/dev/null)" in
    0) ;;
    *) echo "resolve_target.sh: LEGACY_310P_ENV_KEYS in $ENV_FILE — '310P_*' keys are not legal; use ASCEND310P_*" >&2
       return 1 2>/dev/null || exit 1
       ;;
esac

# Env-key prefix per target. Explicit case — never derived by tr 'a-z' 'A-Z'
# (310p → "310P" would produce the illegal ${310P_*} expansions above).
case "$TARGET" in
    310p) UPPER="ASCEND310P" ;;
    *) UPPER="$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')" ;;
esac

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
# Update this block and the external hardware knowledge when a new SOC family is added.
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
    310p)
        # Ascend310P3 (Atlas 300I Duo) — 200x / DAV_2002 / dav-m200,
        # __CCE_AICORE__==200. Fuse chip: cube+vec on one core, no SIMT, no
        # bf16, no fp64. UB/L0C verified on-target: both 256 KB, from the CANN
        # compiler codegen config platform_config/Ascend310P3.ini (INI-derived
        # single source; see cannbot-knowledge target_ascend310p.md).
        PLATFORM_SIMT=false
        ARCH_CODE=arch20
        NPU_ARCH=2002
        COMPILER_ARCH=dav-m200
        UB_PER_AIV_KB=256
        L0C_KB=256
        ;;
esac

export TARGET HOST USER PASSWORD CONTAINER CANN_PATH SOC_VERSION
export NPU_PYTHON_BIN EXTRA_LD_LIBRARY_PATH SSH_KEY
export PLATFORM_SIMT ARCH_CODE NPU_ARCH UB_PER_AIV_KB L0C_KB
[ -n "${COMPILER_ARCH:-}" ] && export COMPILER_ARCH

# Validate required fields for active target
if [ -z "$HOST" ] || [ -z "$CONTAINER" ] || [ -z "$SOC_VERSION" ]; then
    echo "resolve_target.sh: TARGET=$TARGET selected but ${UPPER}_HOST/${UPPER}_CONTAINER/${UPPER}_SOC_VERSION not configured. Update workspace/.ascendc_env from its template (created by this plugin's init.sh)." >&2
    return 1 2>/dev/null || exit 1
fi

# 310p: SoC must name the 310P family (prefix check, not a closed SKU list —
# the family has multiple SKUs on the market). Wrong-family SoCs mean the
# env inherited another target's block; fail closed with the exact key named.
if [ "$TARGET" = "310p" ]; then
    case "$SOC_VERSION" in
        Ascend310P*) ;;
        *)
            echo "resolve_target.sh: TARGET_SOC_MISMATCH: TARGET=310p requires ASCEND310P_SOC_VERSION=Ascend310P*; got '$SOC_VERSION'" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
fi
# 310p requires an explicit CANN path (no generic CANN_PATH fallback —
# a multi-CANN host could otherwise resolve 9.1.0-vs-8.x by accident).
if [ "$TARGET" = "310p" ] && [ -z "$CANN_PATH" ]; then
    echo "resolve_target.sh: TARGET=310p requires ASCEND310P_CANN_PATH (no generic fallback)" >&2
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

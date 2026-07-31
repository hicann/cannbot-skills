#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Overlay the DEBT-20 build_ascendc.py patch onto an EXISTING A3/A5 container
# without re-provisioning the whole AscendOpGenAgent tree.
#
# Use when: setup_a3_isolated_container.sh has already been run for this host,
# but the container was provisioned before DEBT-20 landed. Re-running the full
# setup script is overkill; this just overlays the one patched file.
#
# Usage:
#   bash src/scripts/install_build_ascendc_patch.sh <host> <slice_home>
#
# Example:
#   bash src/scripts/install_build_ascendc_patch.sh 198.51.100.92 /home/npu_user_arch35
#
# After install, the next `python3 /home/npu_user/AscendOpGenAgent/utils/build_ascendc.py <op>`
# inside the container (mapped from <slice_home>/AscendOpGenAgent/) honors
# kernel/build_overrides.json per-source COMPILE_DEFINITIONS.

set -eo pipefail

HOST="${1:-}"
SLICE_HOME="${2:-}"

if [[ -z "$HOST" || -z "$SLICE_HOME" ]]; then
    echo "usage: $0 <host> <slice_home>" >&2
    echo "example: $0 198.51.100.92 /home/npu_user_arch35" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="${SCRIPT_DIR}/patches/build_ascendc.py"

if [[ ! -f "$PATCH" ]]; then
    echo "FATAL: patch not found at $PATCH" >&2
    exit 3
fi

TARGET="${SLICE_HOME}/AscendOpGenAgent/utils/build_ascendc.py"

echo "[debt20-patch] overlaying $PATCH → root@${HOST}:${TARGET}" >&2

scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$PATCH" "root@${HOST}:${TARGET}" \
    || { echo "FATAL: scp failed" >&2; exit 4; }

# Sanity check: verify the patched file has the marker
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${HOST}" \
    "grep -q 'DEBT-20' '${TARGET}'" \
    || { echo "FATAL: patched file present but missing DEBT-20 marker" >&2; exit 5; }

echo "[debt20-patch] OK — patch installed. Add kernel/build_overrides.json to enable per-source COMPILE_DEFINITIONS." >&2
echo "[debt20-patch] schema:" >&2
echo '  {' >&2
echo '    "per_source_defines": {' >&2
echo '      "<filename.cpp>": ["MACRO1=value", "MACRO2"]' >&2
echo '    }' >&2
echo '  }' >&2

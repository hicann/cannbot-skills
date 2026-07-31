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
# DEBT-111: Provision missing AscendOpGenAgent/utils/ to A3 containers.
#
# All configured arch22 source-NPU containers
# were deployed without /root/AscendOpGenAgent/utils/, causing deploy_to_npu.sh:333
# to fail. This script provisions the utilities from vendor.
#
# Usage:
#   bash src/scripts/setup_a3_utils.sh <host> <container>
#   bash src/scripts/setup_a3_utils.sh 198.51.100.92 npu-a3
set -euo pipefail

HOST="${1:-}"
CONTAINER="${2:-}"
if [ -z "$HOST" ] || [ -z "$CONTAINER" ]; then
    echo "Usage: $0 <host> <container>"
    echo "  $0 198.51.100.92 npu-a3"
    exit 1
fi

# Credential from env — NEVER hardcoded (security; owner: creds live only in
# workspace/.ascendc_env). Export A5_PASSWORD or `source workspace/.ascendc_env`.
: "${A5_PASSWORD:?A5_PASSWORD not set — export it or 'source workspace/.ascendc_env' (credential is no longer hardcoded)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_UTILS="$(cd "$SCRIPT_DIR/../../vendor/AscendOpGenAgent/utils" && pwd)"

echo "=== DEBT-111: Provisioning AscendOpGenAgent/utils to $HOST:$CONTAINER ==="

TMP_TAR="/tmp/a3_utils_provision_$(date +%s).tar.gz"
cd "$VENDOR_UTILS" && tar czf "$TMP_TAR" *.py *.sh

echo "[1/3] Copying utils to $HOST..."
sshpass -p "$A5_PASSWORD" scp -o StrictHostKeyChecking=no "$TMP_TAR" "root@${HOST}:/tmp/utils_provision.tar.gz"

echo "[2/3] Installing on $CONTAINER..."
sshpass -p "$A5_PASSWORD" ssh -o StrictHostKeyChecking=no "root@${HOST}" \
    "docker cp /tmp/utils_provision.tar.gz ${CONTAINER}:/tmp/ &&
     docker exec ${CONTAINER} bash -c '
        mkdir -p /root/AscendOpGenAgent/utils
        cd /root/AscendOpGenAgent/utils
        tar xzf /tmp/utils_provision.tar.gz
        echo \"Provisioned: \$(ls *.py *.sh | wc -l) files\"
     '"

echo "[3/3] Verifying..."
sshpass -p "$A5_PASSWORD" ssh -o StrictHostKeyChecking=no "root@${HOST}" \
    "docker exec ${CONTAINER} ls /root/AscendOpGenAgent/utils/build_ascendc.py 2>/dev/null && echo 'VERIFIED' || echo 'FAILED'"

rm -f "$TMP_TAR"
echo "=== DEBT-111 fix complete for $HOST:$CONTAINER ==="

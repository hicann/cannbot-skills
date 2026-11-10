#!/usr/bin/env bash
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

# =============================================================================
# Test: install.sh npm 失败回退看护（!1213 × !1654 合并链路）
# =============================================================================
# !1213：npm 全局安装失败（如全局目录不可写）时 Linux 自动回退用户目录
# 预编译二进制安装（全程无需 sudo）；!1654：install_binary 内含 registry
# dist.integrity (sha512) 载荷校验。本看护固化两套特性的衔接链路：
# npm EACCES 失败 → 回退文案 → install_binary（含完整性校验）→ 载荷落地执行。
#
# Static rules:
#   FB-01: install_binary 存在且被路径 A 失败分支调用
#   FB-02: EACCES / permission denied 特征识别存在
#   FB-03: 全文无 sudo 引导（仅"无需 sudo"说明文案）
#
# Behavioral rules（mock E2E，仅 Linux）:
#   FB-04: 假 node + npm EACCES 失败 + mock registry →
#          exit 0 + 回退文案 + EACCES 原因识别 + 二进制落地 + 载荷执行
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INSTALL_SH="$SKILLS_DIR/install.sh"

PASS_COUNT=0
FAIL_COUNT=0

run_check() {
    local name="$1"
    shift
    if "$@"; then
        print_pass "$name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "$name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# =============================================================================
# Static rules
# =============================================================================
print_section "Static: npm 失败回退结构"

fb01_fallback_wired() {
    grep -q '^install_binary()' "$INSTALL_SH" && grep -qF 'install_binary || exit 1' "$INSTALL_SH"
}
run_check "FB-01: install_binary 存在且被回退分支调用" fb01_fallback_wired

fb02_eacces_detect() {
    grep -q 'EACCES' "$INSTALL_SH" && grep -q 'permission denied' "$INSTALL_SH"
}
run_check "FB-02: EACCES 特征识别存在" fb02_eacces_detect

fb03_no_sudo() {
    ! grep -q 'sudo npm install' "$INSTALL_SH"
}
run_check "FB-03: 全文无 sudo 引导" fb03_no_sudo

# =============================================================================
# Behavioral rule：npm EACCES → 回退 → install_binary mock E2E
# =============================================================================
print_section "Behavior: npm EACCES → 回退链路"

BEHAVIOR_OK=true
if [ "$(uname -s)" != "Linux" ]; then
    print_skip "行为用例仅支持 Linux（当前: $(uname -s)）"
    BEHAVIOR_OK=false
elif ! command -v python3 >/dev/null 2>&1; then
    print_skip "行为用例需要 python3（mock server + 哈希计算）"
    BEHAVIOR_OK=false
elif ! command -v curl >/dev/null 2>&1; then
    print_skip "行为用例需要 curl"
    BEHAVIOR_OK=false
fi

if $BEHAVIOR_OK; then
    MOCK_ROOT="$(mktemp -d)"
    SANDBOX_BIN="$(mktemp -d)"
    SERVER_PID=""
    cleanup_mock() {
        if [ -n "$SERVER_PID" ]; then
            kill "$SERVER_PID" 2>/dev/null || true
        fi
        rm -rf "$MOCK_ROOT" "$SANDBOX_BIN"
    }
    trap cleanup_mock EXIT

    # --- 平台推导（与 install.sh 一致） ---
    case "$(uname -m)" in
        x86_64|amd64)  B_ARCH="x64" ;;
        arm64|aarch64) B_ARCH="arm64" ;;
        *)             B_ARCH="x64" ;;
    esac
    B_VERSION="9.9.9"
    MAIN_PKG_DIR="${MOCK_ROOT}/@cannbot-ai/install-helper"
    SUBPKG_DIR="${MOCK_ROOT}/@cannbot-ai/install-helper-linux-${B_ARCH}"
    mkdir -p "${MAIN_PKG_DIR}" "${SUBPKG_DIR}/-" "${MOCK_ROOT}/build/package"

    # --- 假载荷：执行时落 marker（shell 内建，无外部命令依赖） ---
    cat > "${MOCK_ROOT}/build/package/install-helper" <<PAYLOAD_EOF
#!/bin/sh
: > "${MOCK_ROOT}/payload-ran.marker"
exit 0
PAYLOAD_EOF
    chmod +x "${MOCK_ROOT}/build/package/install-helper"
    tar -czf "${SUBPKG_DIR}/-/install-helper-linux-${B_ARCH}-${B_VERSION}.tgz" -C "${MOCK_ROOT}/build" package

    B_INTEGRITY="$(python3 - "${SUBPKG_DIR}/-/install-helper-linux-${B_ARCH}-${B_VERSION}.tgz" <<'PYEOF'
import base64, hashlib, sys
with open(sys.argv[1], "rb") as f:
    digest = hashlib.sha512(f.read()).digest()
print("sha512-" + base64.b64encode(digest).decode())
PYEOF
)"

    # --- packuments：主包（路径 A 查询）+ 平台子包（install_binary 查询） ---
    printf '{"version":"%s"}' "$B_VERSION" > "${MAIN_PKG_DIR}/latest"
    printf '{"version":"%s","dist":{"integrity":"%s"}}' "$B_VERSION" "$B_INTEGRITY" > "${SUBPKG_DIR}/latest"

    # --- mock registry ---
    MOCK_PORT="$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")"
    (cd "$MOCK_ROOT" && exec python3 -m http.server "$MOCK_PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
    SERVER_PID=$!

    server_ready=false
    for _ in $(seq 1 40); do
        if curl --noproxy '*' -fsS "http://127.0.0.1:${MOCK_PORT}/@cannbot-ai/install-helper/latest" >/dev/null 2>&1; then
            server_ready=true
            break
        fi
        sleep 0.25
    done

    if ! $server_ready; then
        print_fail "mock registry 未就绪（FB-04 无法执行）"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        # --- 沙箱 PATH：真实工具 + 假 node（v20）+ 假 npm（EACCES 失败） ---
        for tool in bash sh curl tar gzip uname mktemp cp chmod rm mkdir grep sed head cut tr cat openssl python3; do
            tool_path="$(command -v "$tool" 2>/dev/null || true)"
            if [ -n "$tool_path" ]; then
                ln -s "$tool_path" "${SANDBOX_BIN}/${tool}"
            fi
        done
        printf '#!/bin/sh\necho "v20.11.0"\n' > "${SANDBOX_BIN}/node"
        printf '#!/bin/sh\necho "npm ERR! Error: EACCES: permission denied, access /usr/local/lib/node_modules"\nexit 1\n' > "${SANDBOX_BIN}/npm"
        chmod +x "${SANDBOX_BIN}/node" "${SANDBOX_BIN}/npm"

        DEAD_PROXY="http://127.0.0.1:9"
        FB_HOME="$(mktemp -d)"

        FB_RC=0
        FB_OUT="$(env -i \
            HOME="$FB_HOME" \
            PATH="$SANDBOX_BIN" \
            CANNBOT_REGISTRY="http://127.0.0.1:${MOCK_PORT}" \
            http_proxy="$DEAD_PROXY" https_proxy="$DEAD_PROXY" \
            no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
            bash "$INSTALL_SH" </dev/null 2>&1)" || FB_RC=$?

        run_check "FB-04a: npm EACCES 失败 → 回退后成功退出（exit 0）" \
            test "$FB_RC" -eq 0
        fb_fallback_msg() { printf '%s' "$FB_OUT" | grep -q '回退到用户目录预编译二进制安装'; }
        run_check "FB-04b: 输出回退文案" fb_fallback_msg
        fb_reason_msg() { printf '%s' "$FB_OUT" | grep -q '全局目录不可写'; }
        run_check "FB-04c: EACCES 原因被识别" fb_reason_msg
        fb_integrity() { printf '%s' "$FB_OUT" | grep -q '完整性校验通过'; }
        run_check "FB-04d: 回退载荷经 sha512 完整性校验" fb_integrity
        run_check "FB-04e: 二进制落地 ~/.cannbot/bin" \
            test -f "${FB_HOME}/.cannbot/bin/install-helper"
        run_check "FB-04f: 载荷实际执行（marker 证据）" \
            test -f "${MOCK_ROOT}/payload-ran.marker"

        rm -rf "$FB_HOME"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Install Fallback Test Summary"
echo "========================================"
echo "  Passed:  $PASS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi

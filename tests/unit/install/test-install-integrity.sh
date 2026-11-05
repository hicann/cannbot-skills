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
# Test: install.sh / install.ps1 Payload Integrity (#832)
# =============================================================================
# L4 security issue #832: curl | bash 信任模型加固的看护测试。
#
# 入口一键命令保持业界标准形态（rustup / opencode 同款）不回退（SEC-05），
# 完整性保障落在脚本内置的载荷校验：安装脚本对下载的二进制 tarball 按
# npm registry 元数据 dist.integrity (sha512) 校验，失败即中止（fail-closed）。
#
# Static rules:
#   SEC-01: install.sh 解析 packument 的 dist.integrity 并校验下载的 tarball
#   SEC-02: install.sh tarball 落地用 mktemp，不用固定 /tmp 路径
#   SEC-03: install.sh 支持 CANNBOT_REGISTRY 环境变量覆盖 registry（企业镜像/测试）
#   SEC-04: install.ps1 对下载 tarball 做同等 SHA512 完整性校验
#   SEC-05: 四份安装文档保留一键命令（回归看护：防过度修复移除标准入口）
#
#   注：文档保持业界标准裸命令形态（opencode / Claude Code / Trae 均无
#   安全提示块，已调研核实），完整性保障全部内置于脚本——合法载荷静默
#   通过，用户无需手动执行任何校验步骤，故不设文档提示类断言。
#
# Behavioral rules（mock registry E2E，fail-closed 实证，仅 Linux）:
#   SEC-07: 篡改的 tarball → 非零退出 + 完整性校验失败输出 + 载荷未落地未执行
#   SEC-08: 匹配的 tarball → exit 0 + 二进制落地且执行（marker 证据）
#   SEC-09: 显式版本通道（install.sh <version>）mock E2E——版本化 packument
#           → integrity → 下载校验 → 落地执行全链路
#
# Static rules（评审加固，PR !1654 机器人评审意见落地）:
#   SEC-10: install.sh 显式版本分支遍历全部镜像（P2-1：单点查询在 npmjs
#           不可达而 npmmirror 可达的受限网络下会误伤安装，需与 latest
#           分支及 install.ps1 的镜像回退契约一致）
#   SEC-11: CANNBOT_REGISTRY 为 http:// 时两脚本给出明文链路告警（P3-4）
#   SEC-12: mock 就绪探测隔离代理环境（P3-5：企业代理下不误判为用例失败）
#
# 行为用例通过死代理（http_proxy 指向不可达端口）隔离公网 registry，
# 断言 CANNBOT_REGISTRY 指定的 mock registry 生效：
#   - 若脚本未实现 CANNBOT_REGISTRY（RED），公网不可达 → 版本查询失败退出，
#     SEC-07b/SEC-08 与预期不符而失败；
#   - 若已实现（GREEN），载荷全部来自 mock，可精确控制篡改/合法两种形态。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INSTALL_SH="$SKILLS_DIR/install.sh"
INSTALL_PS1="$SKILLS_DIR/install.ps1"

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

print_section "Static: install.sh payload integrity"

# SEC-01: packument integrity 解析 + 下载后校验 + fail-closed
run_check "SEC-01a: install.sh 解析 dist.integrity (sha512)" \
    grep -q '"integrity":"sha512-' "$INSTALL_SH"
run_check "SEC-01b: install.sh 校验失败即中止（完整性校验失败消息）" \
    grep -q '完整性校验失败' "$INSTALL_SH"

# SEC-02: mktemp 代替固定 /tmp 路径（消除共享主机 symlink 面与陈旧文件面）
sec02_no_fixed_tmp() {
    ! grep -q '/tmp/ih-tarball' "$INSTALL_SH"
}
run_check "SEC-02a: install.sh 不再使用固定 /tmp/ih-tarball 路径" sec02_no_fixed_tmp
sec02_tarball_mktemp() {
    grep -qF 'TARBALL_FILE="$(mktemp' "$INSTALL_SH"
}
run_check "SEC-02b: install.sh tarball 下载目标本身来自 mktemp" sec02_tarball_mktemp

# SEC-03: CANNBOT_REGISTRY 覆盖
run_check "SEC-03: install.sh 支持 CANNBOT_REGISTRY 环境变量" \
    grep -q 'CANNBOT_REGISTRY' "$INSTALL_SH"

# SEC-04: install.ps1 同等校验
sec04_ps1_verify() {
    grep -q 'Get-FileHash' "$INSTALL_PS1" && grep -q 'integrity' "$INSTALL_PS1"
}
run_check "SEC-04: install.ps1 对 tarball 做 SHA512 完整性校验" sec04_ps1_verify

# SEC-10: 显式版本分支必须遍历镜像并回传命中镜像（P2-1）
# 锚点：BEST_REGISTRY="$REGISTRY" —— latest 分支已有 1 处；显式分支修复后
# 命中 integrity 时同步 BEST_REGISTRY（镜像回退下载的必要条件），计数应为 2
sec10_explicit_mirror_loop() {
    [ "$(grep -cF 'BEST_REGISTRY="$REGISTRY"' "$INSTALL_SH")" -ge 2 ]
}
run_check "SEC-10: install.sh 显式版本分支遍历镜像并回传命中镜像（非单点）" sec10_explicit_mirror_loop

# SEC-11: http:// CANNBOT_REGISTRY 明文链路告警（P3-4）
sec11_http_warn() {
    grep -q 'CANNBOT_REGISTRY.*http://' "$INSTALL_SH" && grep -q 'CANNBOT_REGISTRY.*http://' "$INSTALL_PS1"
}
run_check "SEC-11: http:// registry 覆盖时两脚本给出明文链路告警" sec11_http_warn

# SEC-12: 就绪探测须隔离代理（P3-5）
run_check "SEC-12: mock 就绪探测使用 --noproxy（企业代理环境不误判）" \
    grep -q -- '--noproxy' "${BASH_SOURCE[0]}"

# SEC-05 / SEC-06: 四份文档
DOCS=(
    "README.md"
    "docs/installation-guide.md"
    "plugins-community/install-helper/README.md"
    "plugins-community/triton-optimizer/quickstart.md"
)

print_section "Static: docs keep one-liner + security note"

for doc in "${DOCS[@]}"; do
    doc_path="$SKILLS_DIR/$doc"
    run_check "SEC-05: $doc 保留 curl | bash 一键命令" \
        grep -q 'install.sh | bash' "$doc_path"
done

# =============================================================================
# Behavioral rules（mock registry E2E）
# =============================================================================

print_section "Behavior: mock registry fail-closed verification"

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

    # --- 平台与包名（与 install.sh 的推导一致） ---
    case "$(uname -m)" in
        x86_64|amd64)  B_ARCH="x64" ;;
        arm64|aarch64) B_ARCH="arm64" ;;
        *)             B_ARCH="x64" ;;
    esac
    SUBPKG="@cannbot-ai/install-helper-linux-${B_ARCH}"
    SUBDIR_NAME="install-helper-linux-${B_ARCH}"
    B_VERSION="9.9.9"
    TARBALL_NAME="${SUBDIR_NAME}-${B_VERSION}.tgz"

    MOCK_NPM="${MOCK_ROOT}/${SUBPKG}"
    mkdir -p "${MOCK_NPM}/-"

    # --- 构造假载荷：可执行脚本，被实际执行时落 marker（仅用 shell 内建，无外部命令依赖） ---
    BUILD_DIR="${MOCK_ROOT}/build"
    mkdir -p "${BUILD_DIR}/package"
    cat > "${BUILD_DIR}/package/install-helper" <<PAYLOAD_EOF
#!/bin/sh
: > "${MOCK_ROOT}/payload-ran.marker"
exit 0
PAYLOAD_EOF
    chmod +x "${BUILD_DIR}/package/install-helper"
    tar -czf "${MOCK_NPM}/-/${TARBALL_NAME}" -C "${BUILD_DIR}" package

    # --- 计算 registry dist.integrity（sha512-base64） ---
    B_INTEGRITY="$(python3 - "${MOCK_NPM}/-/${TARBALL_NAME}" <<'PYEOF'
import base64, hashlib, sys
with open(sys.argv[1], "rb") as f:
    digest = hashlib.sha512(f.read()).digest()
print("sha512-" + base64.b64encode(digest).decode())
PYEOF
)"

    # --- mock packument（minified，与 registry 响应同构）：latest + 显式版本两个路径 ---
    printf '{"version":"%s","dist":{"integrity":"%s"}}' \
        "$B_VERSION" "$B_INTEGRITY" > "${MOCK_NPM}/latest"
    printf '{"version":"%s","dist":{"integrity":"%s"}}' \
        "$B_VERSION" "$B_INTEGRITY" > "${MOCK_NPM}/${B_VERSION}"

    # --- 起本地 mock registry ---
    MOCK_PORT="$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")"
    (cd "$MOCK_ROOT" && exec python3 -m http.server "$MOCK_PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
    SERVER_PID=$!

    server_ready=false
    for _ in $(seq 1 40); do
        if curl --noproxy '*' -fsS "http://127.0.0.1:${MOCK_PORT}/${SUBPKG}/latest" >/dev/null 2>&1; then
            server_ready=true
            break
        fi
        sleep 0.25
    done

    if ! $server_ready; then
        print_fail "mock registry 未就绪（SEC-07/SEC-08 无法执行）"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        # --- 沙箱 PATH：只含 install.sh Path B 所需工具，排除 node ---
        # 注意：tar 解 .tgz 会 spawn gzip 子进程，必须一并放入沙箱
        for tool in bash sh curl tar gzip uname mktemp cp chmod rm mkdir grep sed head cut tr cat openssl python3; do
            tool_path="$(command -v "$tool" 2>/dev/null || true)"
            if [ -n "$tool_path" ]; then
                ln -s "$tool_path" "${SANDBOX_BIN}/${tool}"
            fi
        done

        # 死代理隔离公网 registry；no_proxy 放行本地 mock
        DEAD_PROXY="http://127.0.0.1:9"

        run_install() {
            local home_dir="$1"
            local channel="${2:-latest}"
            env -i \
                HOME="$home_dir" \
                PATH="$SANDBOX_BIN" \
                CANNBOT_REGISTRY="http://127.0.0.1:${MOCK_PORT}" \
                http_proxy="$DEAD_PROXY" https_proxy="$DEAD_PROXY" \
                no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
                bash "$INSTALL_SH" "$channel" </dev/null 2>&1
        }

        # ---- Case 1: 篡改 tarball（integrity 与内容不匹配） ----
        # 先保存原始载荷（Case 2 直接恢复原文件，避免重建引入 gzip 时间戳差异）
        cp "${MOCK_NPM}/-/${TARBALL_NAME}" "${MOCK_ROOT}/tarball-valid.tgz"

        TAMPER_HOME="$(mktemp -d)"
        printf 'TAMPERED-PAYLOAD' >> "${MOCK_NPM}/-/${TARBALL_NAME}"

        TAMPER_RC=0
        TAMPER_OUT="$(run_install "$TAMPER_HOME")" || TAMPER_RC=$?

        run_check "SEC-07a: 篡改载荷 → 安装非零退出（fail-closed）" \
            test "$TAMPER_RC" -ne 0
        tampered_msg_check() { printf '%s' "$TAMPER_OUT" | grep -q '完整性校验失败'; }
        run_check "SEC-07b: 篡改载荷 → 输出完整性校验失败（失败原因可辨识）" \
            tampered_msg_check
        run_check "SEC-07c: 篡改载荷 → 二进制未落地用户目录" \
            test ! -e "${TAMPER_HOME}/.cannbot/bin/install-helper"
        run_check "SEC-07d: 篡改载荷 → 载荷从未执行" \
            test ! -e "${MOCK_ROOT}/payload-ran.marker"

        rm -rf "$TAMPER_HOME"

        # ---- Case 2: 合法 tarball（恢复原始文件，保证与 integrity 匹配） ----
        cp "${MOCK_ROOT}/tarball-valid.tgz" "${MOCK_NPM}/-/${TARBALL_NAME}"

        VALID_HOME="$(mktemp -d)"
        VALID_RC=0
        VALID_OUT="$(run_install "$VALID_HOME")" || VALID_RC=$?

        run_check "SEC-08a: 合法载荷 → 安装成功退出（exit 0）" \
            test "$VALID_RC" -eq 0
        run_check "SEC-08b: 合法载荷 → 二进制落地 ~/.cannbot/bin" \
            test -f "${VALID_HOME}/.cannbot/bin/install-helper"
        run_check "SEC-08c: 合法载荷 → 载荷实际执行过（marker 证据）" \
            test -f "${MOCK_ROOT}/payload-ran.marker"

        rm -rf "$VALID_HOME"

        # ---- Case 3: 显式版本通道（bash install.sh 9.9.9）mock E2E ----
        rm -f "${MOCK_ROOT}/payload-ran.marker"
        EXPL_HOME="$(mktemp -d)"
        EXPL_RC=0
        EXPL_OUT="$(run_install "$EXPL_HOME" "$B_VERSION")" || EXPL_RC=$?

        run_check "SEC-09a: 显式版本通道 → 安装成功退出（exit 0）" \
            test "$EXPL_RC" -eq 0
        run_check "SEC-09b: 显式版本通道 → 二进制落地 ~/.cannbot/bin" \
            test -f "${EXPL_HOME}/.cannbot/bin/install-helper"
        run_check "SEC-09c: 显式版本通道 → 载荷实际执行（marker 证据）" \
            test -f "${MOCK_ROOT}/payload-ran.marker"

        rm -rf "$EXPL_HOME"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Install Integrity Test Summary"
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

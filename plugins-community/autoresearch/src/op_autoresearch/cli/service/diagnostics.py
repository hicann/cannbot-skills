# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map remote probe facts to findings and render them."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Iterable, Optional

from op_autoresearch.utils.console import emit


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    result: str
    suggest: str


@dataclass(frozen=True)
class ClassificationPolicy:
    port: int
    ascend_backend: bool
    needs_triton: bool
    for_start: bool


def _ssh_suggest(error: str) -> str:
    lower = error.lower()
    if "could not resolve hostname" in lower or "name or service not known" in lower:
        return "检查 ~/.ssh/config 中的 Host 别名和主机名"
    network_errors = ("timed out", "no route to host", "network is unreachable")
    if any(text in lower for text in network_errors):
        return "检查 VPN、路由和远端主机在线状态"
    if "connection refused" in lower:
        return "检查远端 sshd 和 22 端口防火墙"
    if "connection closed" in lower or "connection reset" in lower:
        return "查看远端认证日志及 MaxSessions/防火墙限制"
    if "permission denied" in lower or "publickey" in lower:
        return "检查 IdentityFile 与 authorized_keys 免密配置"
    if "host key verification failed" in lower:
        return "确认主机身份后更新 known_hosts 中的旧 host key"
    return "手动运行 ssh <alias>，检查别名、网络与免密配置"


def _append_env_finding(findings: list[Finding], facts: dict) -> None:
    env_path = facts.get("ENV_PATH") or ""
    if not env_path:
        findings.append(Finding(
            "info", "env_script", "未配置",
            "若默认 shell 未加载 CANN/torch_npu，请在 config.yaml 配置 env_script",
        ))
    elif facts.get("ENV_OK") == "yes":
        findings.append(Finding("ok", "env_script", env_path, ""))
    else:
        findings.append(Finding(
            "fatal", "env_script", f"配置为 {env_path}，但文件不存在",
            "修正 remote_worker.hosts.<alias>.env_script",
        ))


def _append_runtime_findings(findings: list[Finding], facts: dict,
                             policy: ClassificationPolicy) -> None:
    torch_result = facts.get("TORCH_NPU") or ""
    if torch_result == "ok":
        findings.append(Finding("ok", "torch_npu", "importable", ""))
    elif policy.ascend_backend:
        findings.append(Finding(
            "fatal", "torch_npu", torch_result[:120] or "import 失败",
            "检查 env_script 是否加载 CANN 环境，以及 torch_npu 是否安装",
        ))
    triton_result = facts.get("TRITON") or ""
    if triton_result == "ok":
        findings.append(Finding("ok", "triton", "importable", ""))
        return
    severity = "fatal" if policy.needs_triton else "warn"
    suggestion = ("triton_* DSL 必须安装 Triton"
                  if policy.needs_triton else "仅 triton_* DSL 需要 Triton")
    findings.append(Finding(
        severity, "triton", triton_result[:80] or "import 失败", suggestion,
    ))


def _append_device_tool_finding(findings: list[Finding], facts: dict,
                                policy: ClassificationPolicy) -> None:
    if facts.get("NPU_SMI") == "ok":
        findings.append(Finding("ok", "npu-smi", "in PATH", ""))
    elif policy.ascend_backend:
        findings.append(Finding(
            "fatal", "npu-smi", "not in PATH",
            "检查 env_script 是否加载 CANN set_env.sh",
        ))


def _append_arch_finding(findings: list[Finding], facts: dict,
                         policy: ClassificationPolicy) -> None:
    arch = (facts.get("ARCH") or "").strip()
    if arch:
        findings.append(Finding("ok", "npu arch", arch.lower(), ""))
    elif policy.ascend_backend:
        findings.append(Finding(
            "warn", "npu arch", "无法从 npu-smi 推断",
            "通过 --arch 显式指定架构，例如 ascend910b3",
        ))


def _integer_fact(facts: dict, key: str) -> int:
    try:
        return int(facts.get(key) or "0")
    except (TypeError, ValueError):
        return 0


def _append_capacity_findings(findings: list[Finding], facts: dict,
                              policy: ClassificationPolicy) -> None:
    device_count = _integer_fact(facts, "DEVICES")
    if device_count > 0:
        findings.append(Finding(
            "ok", "npu devices", f"{device_count} visible", ""))
    elif policy.ascend_backend:
        findings.append(Finding(
            "fatal", "npu devices", "0 visible",
            "检查驱动状态并在 SSH 会话中运行 npu-smi info",
        ))
    free_mb = _integer_fact(facts, "DISK_FREE_MB")
    if free_mb >= 500:
        findings.append(Finding("ok", "disk free", f"{free_mb} MB", ""))
    elif free_mb > 0:
        findings.append(Finding(
            "fatal", "disk free", f"only {free_mb} MB",
            "清理 /tmp 和旧日志后重试，避免 daemon 写日志时 ENOSPC",
        ))


def _append_port_finding(findings: list[Finding], facts: dict,
                         policy: ClassificationPolicy) -> None:
    process_id = (facts.get("PORT_PID") or "").strip()
    if not process_id:
        findings.append(Finding(
            "ok", f"remote :{policy.port}", "free", ""))
        return
    severity = "fatal" if policy.for_start else "warn"
    if policy.for_start:
        suggestion = (f"端口会导致 bind 冲突；确认后终止 PID {process_id} "
                      "或更换端口")
    else:
        suggestion = f"确认 PID {process_id} 是否为残留 daemon，或更换端口"
    findings.append(Finding(
        severity,
        f"remote :{policy.port}",
        f"held by PID {process_id}",
        suggestion,
    ))


def classify(facts: dict, port: int, *,
             backend: Optional[str] = None,
             dsl: Optional[str] = None,
             for_start: bool = False) -> list[Finding]:
    """Classify raw probe facts under the requested backend/DSL policy."""
    ssh_error = facts.get("_SSH_ERROR")
    if ssh_error:
        return [Finding(
            "fatal", "ssh", ssh_error[:160], _ssh_suggest(ssh_error))]
    policy = ClassificationPolicy(
        port=port,
        ascend_backend=backend in (None, "", "ascend"),
        needs_triton=(dsl or "").startswith("triton"),
        for_start=for_start,
    )
    findings: list[Finding] = []
    _append_env_finding(findings, facts)
    _append_runtime_findings(findings, facts, policy)
    _append_device_tool_finding(findings, facts, policy)
    _append_arch_finding(findings, facts, policy)
    _append_capacity_findings(findings, facts, policy)
    _append_port_finding(findings, facts, policy)
    return findings


def has_fatal(findings: Iterable[Finding]) -> bool:
    return any(finding.severity == "fatal" for finding in findings)


def render_findings(findings: Iterable[Finding], log_tail: str = "") -> None:
    """Print findings and an optional daemon log tail to stderr."""
    emit("Remote diagnostics:", file=sys.stderr)
    for finding in findings:
        line = (
            f"[{finding.severity.upper()}] "
            f"{finding.check}: {finding.result}"
        )
        if finding.suggest:
            line += f" | {finding.suggest}"
        emit(line, file=sys.stderr)
    stripped_tail = log_tail.strip()
    if stripped_tail and not stripped_tail.startswith("(no log"):
        emit(f"daemon log tail:\n{log_tail}", file=sys.stderr)

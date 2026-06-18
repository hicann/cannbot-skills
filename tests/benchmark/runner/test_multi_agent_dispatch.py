#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""多 Agent 协同 dispatch 验证测试。

验证 opencode serve 模式下 CANNBot 多 Agent 工作流的基础设施是否正确：
  1. serve 启动后 AGENTS.md 中的 primary agent 是否被注册
  2. .opencode/agents/ 下的 subagent 是否被注册
  3. 发送 prompt 后 primary agent 是否 dispatch subagent（通过 SSE 监控）

用法:
  # 快速验证（仅检查注册，不触发 LLM）
  pytest runner/test_multi_agent_dispatch.py -v -k "not dispatch"

  # 完整验证（含 LLM dispatch 测试，耗时较长）
  pytest runner/test_multi_agent_dispatch.py -v

  # 自定义 serve 端口
  OPENCODE_SERVE_PORT=4097 pytest runner/test_multi_agent_dispatch.py -v

依赖:
  - opencode CLI 已安装
  - requests, pytest, pyyaml
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import requests
import yaml

# ── 路径常量 ──────────────────────────────────────────────────────────
RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(RUNNER_DIR)
SKILLS_FORK = os.path.dirname(os.path.dirname(
    os.path.dirname(RUNNER_DIR)))

WORKFLOW_ROOT = os.path.join(SKILLS_FORK, "plugins-official", "ops-direct-invoke")
WORKFLOW_INIT = os.path.join(WORKFLOW_ROOT, "init.sh")

SERVE_PORT = int(os.environ.get("OPENCODE_SERVE_PORT", "4096"))
SERVE_URL = f"http://127.0.0.1:{SERVE_PORT}"

# ── 期望注册的 agent ──────────────────────────────────────────────────
# 文件名（带 .md，用于检查 .opencode/agents/ 目录）
EXPECTED_SUBAGENT_FILES = {
    "ascendc-kernel-architect.md",
    "ascendc-kernel-developer.md",
    "ascendc-kernel-reviewer.md",
}
# API 名称（不带 .md，用于检查 /agent 端点）
EXPECTED_SUBAGENT_NAMES = {f.replace(".md", "") for f in EXPECTED_SUBAGENT_FILES}

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_workdir():
    """创建隔离测试工作目录，运行 init.sh 安装 CANNBot 工作流。

    同时复制 operators/exp/ 算子定义，对齐 run_eval.py 的工作目录结构。
    """
    workdir = tempfile.mkdtemp(prefix="test-multi-agent-")
    try:
        result = subprocess.run(
            ["bash", WORKFLOW_INIT, "project", "opencode", workdir],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"init.sh 失败 (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:1000]}\nstderr: {result.stderr[:1000]}"
        )

        # 验证关键文件已安装
        agents_md = os.path.join(workdir, "AGENTS.md")
        agents_dir = os.path.join(workdir, ".opencode", "agents")
        assert os.path.isfile(agents_md), f"AGENTS.md 未安装: {agents_md}"
        assert os.path.isdir(agents_dir), f".opencode/agents/ 未创建: {agents_dir}"

        installed_agents = set(os.listdir(agents_dir))
        missing = EXPECTED_SUBAGENT_FILES - installed_agents
        assert not missing, f"subagent 定义缺失: {missing}"

        cannbot_link = os.path.join(agents_dir, "cannbot.md")
        if os.path.lexists(cannbot_link):
            os.remove(cannbot_link)
        os.symlink(os.path.realpath(agents_md), cannbot_link)

        # 复制 operators/exp/ 算子定义（对齐 run_eval.py）
        op_src = os.path.join(
            BENCHMARK_DIR, "cann-bench", "tasks", "level1", "exp")
        op_dst = os.path.join(workdir, "operators", "exp")
        if os.path.isdir(op_src):
            os.makedirs(op_dst, exist_ok=True)
            for fname in os.listdir(op_src):
                src = os.path.join(op_src, fname)
                dst = os.path.join(op_dst, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="module")
def serve(test_workdir):
    """启动 opencode serve，返回 (url, workdir, process)。"""
    log_path = os.path.join(test_workdir, "serve.log")
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(SERVE_PORT),
         "--hostname", "127.0.0.1"],
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=os.path.realpath(test_workdir),
    )

    # 等待 serve 就绪
    ready = False
    for _ in range(30):
        time.sleep(2)
        try:
            resp = requests.get(f"{SERVE_URL}/global/health", timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            continue

    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        log_fh.close()
        with open(log_path) as f:
            log_content = f.read()
        pytest.fail(f"serve 启动失败，日志:\n{log_content[:2000]}")

    yield {
        "url": SERVE_URL,
        "workdir": test_workdir,
        "proc": proc,
        "log_path": log_path,
    }

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_fh.close()


def _get_agents(serve_url: str) -> list[dict]:
    """从 serve API 获取 agent 列表。"""
    resp = requests.get(f"{serve_url}/agent", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _find_agent(agents: list[dict], name: str) -> dict | None:
    """在 agent 列表中查找指定名称的 agent。"""
    for a in agents:
        if a.get("name") == name:
            return a
    return None


# ── 测试用例：Agent 注册验证 ──────────────────────────────────────────

class TestAgentRegistration:
    """验证 serve 启动后 agent 注册是否正确。"""

    def test_serve_health(self, serve):
        """serve 健康检查端点返回 200。"""
        resp = requests.get(f"{serve['url']}/global/health", timeout=5)
        assert resp.status_code == 200

    def test_agents_api_returns_list(self, serve):
        """/agent API 返回非空 agent 列表。"""
        agents = _get_agents(serve["url"])
        assert isinstance(agents, list)
        assert len(agents) > 0

    def test_builtin_agents_registered(self, serve):
        """内置 agent (build, explore, general) 已注册。"""
        agents = _get_agents(serve["url"])
        names = {a["name"] for a in agents}
        for builtin in ("build", "explore", "general"):
            assert builtin in names, f"内置 agent '{builtin}' 未注册。已注册: {names}"

    def test_subagents_registered(self, serve):
        """CANNBot subagent (architect/developer/reviewer) 已注册。"""
        agents = _get_agents(serve["url"])
        names = {a["name"] for a in agents}
        for subagent in EXPECTED_SUBAGENT_NAMES:
            assert subagent in names, (
                f"subagent '{subagent}' 未注册。\n"
                f"已注册: {sorted(names)}\n"
                f"缺失: {EXPECTED_SUBAGENT_NAMES - names}"
            )

    def test_subagents_mode_is_subagent(self, serve):
        """subagent 的 mode 字段为 'subagent'。"""
        agents = _get_agents(serve["url"])
        for subagent in EXPECTED_SUBAGENT_NAMES:
            agent = _find_agent(agents, subagent)
            assert agent is not None, f"subagent '{subagent}' 未找到"
            assert agent.get("mode") == "subagent", (
                f"subagent '{subagent}' mode={agent.get('mode')}，期望 'subagent'"
            )

    def test_subagents_not_native(self, serve):
        """subagent 标记为非 native（来自 .opencode/agents/）。"""
        agents = _get_agents(serve["url"])
        for subagent in EXPECTED_SUBAGENT_NAMES:
            agent = _find_agent(agents, subagent)
            assert agent is not None
            assert agent.get("native") is False, (
                f"subagent '{subagent}' 不应是 native agent"
            )

    def test_cannbot_agent_registered(self, serve):
        """CANNBot primary agent 已注册为命名 agent。

        这是多 Agent 协同的关键：CANNBot 必须作为命名 agent 注册，
        serve 才能将其选为 primary agent 来 dispatch subagent。
        """
        agents = _get_agents(serve["url"])
        names = {a["name"] for a in agents}

        # 检查 cannbot 是否注册
        cannbot = _find_agent(agents, "cannbot")
        if cannbot is None:
            pytest.fail(
                f"CANNBot 未注册为命名 agent。\n"
                f"已注册: {sorted(names)}\n"
                f"需要: AGENTS.md 中声明 name: cannbot，"
                f"且 benchmark runner 将其链接到 .opencode/agents/cannbot.md"
            )

        assert cannbot.get("mode") in ("primary", "all"), (
            f"CANNBot mode={cannbot.get('mode')}，期望 'primary' 或 'all'"
        )

    def test_agents_md_exists_in_workdir(self, serve):
        """隔离工作目录中存在 AGENTS.md。"""
        agents_md = os.path.join(serve["workdir"], "AGENTS.md")
        assert os.path.isfile(agents_md)

    def test_agents_md_has_cannbot_content(self, serve):
        """AGENTS.md 包含 CANNBot 工作流定义。"""
        agents_md = os.path.join(serve["workdir"], "AGENTS.md")
        with open(agents_md) as f:
            content = f.read()
        assert "CANNBot" in content
        assert "ascendc-kernel-architect" in content
        assert "ascendc-kernel-developer" in content
        assert "ascendc-kernel-reviewer" in content


# ── 测试用例：Session 创建 ────────────────────────────────────────────

class TestSessionCreation:
    """验证 serve 模式下 session 创建是否正常。"""

    def test_create_session(self, serve):
        """POST /session 创建 session 成功。"""
        resp = requests.post(
            f"{serve['url']}/session",
            json={"title": "test-multi-agent"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        assert "id" in data
        assert data["id"].startswith("ses_")

    def test_list_sessions(self, serve):
        """GET /session 返回 session 列表。"""
        resp = requests.get(f"{serve['url']}/session", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert isinstance(data, list)


# ── 测试用例：Subagent Dispatch 验证 ──────────────────────────────────

@pytest.mark.slow
@pytest.mark.dispatch
class TestSubagentDispatch:
    """验证 primary agent 是否 dispatch subagent。

    这些测试会实际调用 LLM，耗时较长。
    使用 -k "dispatch" 或 --runslow 运行。
    """

    @pytest.fixture
    def session(self, serve):
        """创建测试 session。"""
        resp = requests.post(
            f"{serve['url']}/session",
            json={"title": "test-dispatch"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _monitor_sse(self, serve_url: str, session_id: str,
                     stop_event: threading.Event, events_log: list):
        """后台线程：监控 SSE 事件，记录 subagent dispatch。"""
        try:
            resp = requests.get(
                f"{serve_url}/session/{session_id}/event",
                stream=True, timeout=300,
            )
            for line in resp.iter_lines(decode_unicode=True):
                if stop_event.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                try:
                    evt = json.loads(line[5:].strip())
                    events_log.append(evt)

                    etype = evt.get("type", "")
                    tool_name = evt.get("toolName") or evt.get("tool_name", "")
                    if etype == "tool-invocation" and tool_name == "task":
                        inp = evt.get("input", {})
                        agent = inp.get("subagent_type") or inp.get("subagentType", "?")
                        events_log.append({"_dispatch": True, "agent": agent})
                except (json.JSONDecodeError, AttributeError):
                    pass
        except Exception as e:
            events_log.append({"_error": str(e)})

    def _send_and_poll_subagent(
        self, serve_url: str, session_id: str, payload: dict,
        dispatch_timeout: int,
    ) -> list[dict]:
        """异步发送 message，轮询 /session API 检测 subagent 创建。

        将 /message 请求放在后台线程，主线程轮询 /session API 查找
        parentID 为 session_id 的子 session。检测到或超时后 abort 并返回。
        """
        def _send_message():
            try:
                requests.post(
                    f"{serve_url}/session/{session_id}/message",
                    json=payload,
                    timeout=dispatch_timeout + 60,
                )
            except Exception:
                pass

        msg_thread = threading.Thread(target=_send_message, daemon=True)
        msg_thread.start()

        subagent_sessions = []
        deadline = time.time() + dispatch_timeout
        while time.time() < deadline:
            time.sleep(5)
            try:
                resp = requests.get(f"{serve_url}/session", timeout=10)
                resp.raise_for_status()
                all_sessions = resp.json()
                for s in all_sessions:
                    if s.get("parentID") == session_id:
                        subagent_sessions.append(s)
                        return subagent_sessions
            except Exception:
                pass

        try:
            requests.post(
                f"{serve_url}/session/{session_id}/abort", timeout=5)
        except Exception:
            pass

        return subagent_sessions

    @pytest.mark.xfail(
        reason="LLM dispatch 行为不稳定，可能因模型版本或 prompt 变化导致 flaky",
        strict=False,
    )
    def test_dispatch_with_cannbot_agent(self, serve, session):
        """使用 run_eval.py 相同的 API 调用方式，验证 subagent dispatch。

        对齐 run_eval.py 的调用方式：
          - message payload 包含 "agent": "cannbot"
          - prompt 为 eval 风格（任务描述型，不显式引导工作流）
        验证 CANNBot 收到 eval prompt 后是否按 AGENTS.md 调度 subagent。
        """
        prompt = (
            "请完成 exp 算子的开发，不需要询问我，"
            "自动执行下一步直到完成所有开发任务并提供最终交付件。\n\n"
            "## 算子信息\n\n"
            "算子定义位于 operators/exp/ 目录，包含：\n"
            "- desc.md：算子数学定义、输入输出、精度标准\n"
            "- cases.yaml：测试用例\n"
            "- golden.py：对标竞品实现\n\n"
            "## 参考工程\n\n"
            "example/ 目录是 cann-bench 参考工程。\n\n"
            "## 交付件\n\n"
            "最终交付件为 cann_bench-xxx.whl。"
        )

        dispatch_timeout = int(os.environ.get("DISPATCH_TIMEOUT", "300"))

        subagent_sessions = self._send_and_poll_subagent(
            serve["url"], session,
            payload={
                "agent": "cannbot",
                "parts": [{"type": "text", "text": prompt}],
            },
            dispatch_timeout=dispatch_timeout,
        )

        subagent_names = {s.get("agent") for s in subagent_sessions}

        assert len(subagent_sessions) > 0, (
            f"未检测到任何 subagent session（等待 {dispatch_timeout}s）。\n"
            f"可能原因:\n"
            f"  1. cannbot agent 未正确加载 AGENTS.md 工作流定义\n"
            f"  2. LLM 未遵循 AGENTS.md 工作流，直接自行编写代码\n"
            f"  3. 模型能力不足，无法按复杂工作流调度"
        )

        assert "ascendc-kernel-architect" in subagent_names, (
            f"architect subagent 未被 dispatch。\n"
            f"已 dispatch: {subagent_names}\n"
            f"期望包含: ascendc-kernel-architect"
        )

    def test_cannbot_identity_via_agent_field(self, serve):
        """指定 agent=cannbot 发送简单 prompt，验证 CANNBot 身份响应。

        对齐 run_eval.py 使用 "agent": "cannbot" 的调用方式，
        验证 cannbot agent 确实被选中并正确响应其身份。
        """
        resp = requests.post(
            f"{serve['url']}/session",
            json={"title": "test-cannbot-identity"},
            timeout=10,
        )
        resp.raise_for_status()
        session_id = resp.json()["id"]

        prompt = "列出当前目录的文件结构，然后告诉我你的身份和工作流程。"

        try:
            resp = requests.post(
                f"{serve['url']}/session/{session_id}/message",
                json={
                    "agent": "cannbot",
                    "parts": [{"type": "text", "text": prompt}],
                },
                timeout=120,
            )
            resp.raise_for_status()
            response_data = resp.json()
        except requests.Timeout:
            pytest.skip("message 超时，跳过身份验证测试")
            return

        text_parts = [p.get("text", "") for p in response_data.get("parts", [])
                      if p.get("type") == "text"]
        response_text = "\n".join(text_parts)

        assert "CANNBot" in response_text or "cannbot" in response_text.lower(), (
            f"响应中未提及 CANNBot，agent=cannbot 可能未生效。\n"
            f"响应前 500 字符: {response_text[:500]}"
        )

# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""三个 I/O 管道函数的 mock 测试（守护重构：行为必须保持不变）。

mock 掉真实 LLM/claude CLI/subprocess，用合成轨迹 + 预设工具调用序列驱动
run_agent_pipeline / run_claude_code_pipeline / run_v4_pipeline 的主流程。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import trajectory_analyzer as ta
from trajectory_analyzer import AIProviderConfig, AnalysisError, RunContext

TRAJ_MD = """# Session

## §1 Assistant
*Skill: my-skill (invoke) ✅
<skill_content name="my-skill">
SKILL BODY
</skill_content>
**Tool: skill**
input
**Output:**
result

## §2 User
*Skill: sub (dispatch) ✅
**Tool: task**
do the subtask
<task_result>
RESULT
</task_result>

## Stats
| Turns | 2 | x | 2 |
| Subagents | 1 | x | 1 |
| Cost | 0.1 | x | 0.1 |
"""


def _valid_analysis() -> dict:
    return {
        "sessionSummary": "ok",
        "workflowMeta": {"skillName": "my-skill"},
        "sessionMeta": {"cpsExecuted": [], "cpsMissing": [], "phasesNotReached": []},
        "flow": [],
        "skillQuality": [],
        "workflowLevelIssues": [],
        "optimizationPriorities": [],
    }


def _provider() -> AIProviderConfig:
    return AIProviderConfig(base_url="http://127.0.0.1:1", api_key="k", model="m")


# ---------------- run_agent_pipeline ----------------

def test_run_agent_pipeline_success(tmp_path):
    analysis_path = tmp_path / "sess-analysis.json"
    calls = [
        ta.LlmResponse(content=json.dumps({
            "tool": "write_file",
            "args": {"path": str(analysis_path),
                     "content": json.dumps(_valid_analysis(), ensure_ascii=False)},
            "thought": "write skeleton",
        })),
        ta.LlmResponse(content=json.dumps({"tool": "finish", "args": {}, "thought": "done"})),
    ]
    events: list = []
    with patch.object(ta, "call_llm", side_effect=calls):
        result = ta.run_agent_pipeline(
            TRAJ_MD, "P", _provider(),
            RunContext(str(tmp_path), "sess", events.append, None),
        )

    assert result.steps == 2
    assert result.output_path == str(analysis_path)
    assert result.analysis["sessionSummary"] == "ok"
    assert analysis_path.exists()
    stages = [e["stage"] for e in events]
    assert "skeleton" in stages
    assert "key-sections" in stages
    assert "agent" in stages
    assert "done" in stages


def test_run_agent_pipeline_finish_without_file_raises(tmp_path):
    calls = [ta.LlmResponse(content=json.dumps({"tool": "finish", "args": {}}))]
    with patch.object(ta, "call_llm", side_effect=calls):
        with pytest.raises(AnalysisError):
            ta.run_agent_pipeline(
                TRAJ_MD, "P", _provider(),
                RunContext(str(tmp_path), "sess", logger=None),
            )


# ---------------- run_claude_code_pipeline ----------------

class _FakeProc:
    def __init__(self, stdout_lines):
        self._lines = list(stdout_lines)
        self.stdout = self._iter()
        self.stderr = MagicMock()
        self.stderr.read = MagicMock(return_value="")
        self.returncode = 0

    @staticmethod
    def wait(timeout=None):
        return 0

    def kill(self):
        pass

    def _iter(self):
        for line in self._lines:
            yield line


def test_run_claude_code_pipeline_success(tmp_path):
    analysis_path = tmp_path / "sess-analysis.json"
    valid = _valid_analysis()
    lines = [
        json.dumps({"type": "system", "model": "m"}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": str(analysis_path)}}]}}),
        json.dumps({"type": "result", "subtype": "success", "num_turns": 1,
                    "total_cost_usd": 0.01}),
    ]

    def fake_popen(*a, **k):
        # 模拟 claude 在执行中写出 analysis.json（函数此前已 unlink 旧文件）
        analysis_path.write_text(json.dumps(valid, ensure_ascii=False), encoding="utf-8")
        return _FakeProc(lines)

    events: list = []
    with patch.object(ta.subprocess, "Popen", side_effect=fake_popen):
        result = ta.run_claude_code_pipeline(
            TRAJ_MD, "P", None,
            RunContext(str(tmp_path), "sess", events.append, None),
        )

    assert result.analysis["sessionSummary"] == "ok"
    assert result.output_path == str(analysis_path)
    assert result.steps >= 1
    stages = [e["stage"] for e in events]
    assert "claude-start" in stages
    assert "claude-done" in stages


def test_run_claude_code_pipeline_no_json_raises(tmp_path):
    # claude 未产出 analysis.json → AnalysisError
    lines = [json.dumps({"type": "result", "subtype": "success", "num_turns": 0, "total_cost_usd": 0})]

    def fake_popen(*a, **k):
        return _FakeProc(lines)

    with patch.object(ta.subprocess, "Popen", side_effect=fake_popen):
        with pytest.raises(AnalysisError):
            ta.run_claude_code_pipeline(
                TRAJ_MD, "P", None,
                RunContext(str(tmp_path), "sess", logger=None),
            )


# ---------------- run_v4_pipeline ----------------

def _agent_io() -> dict:
    env = {"latencySec": 1, "tokensKt": 1, "turnCount": 1, "toolCallCount": 1,
           "errorCount": 0, "retryCount": 0, "reasoningTokensKt": 0}
    return {
        "taskQuery": "do X",
        "agents": [
            {"id": "main", "role": "main", "name": "main", "parentId": None,
             "inputSummary": "", "outputSummary": "", "artifacts": [], "envelope": env},
            {"id": "sub", "role": "subagent", "name": "g", "parentId": "main",
             "inputSummary": "", "outputSummary": "", "artifacts": [], "envelope": env},
        ],
    }


def test_run_v4_pipeline_success(tmp_path):
    def fake_audit(provider, agent, subagent_dir=None, tool_use_id=None):
        return ({"completion": {"rating": "pass", "note": "ok"},
                "quality": {"rating": "pass", "note": "ok"},
                "efficiency": {"note": "lean"}}, {})

    def fake_agg(provider, task_query, summary_list, logger=None):
        return ({"sessionSummary": "all ok", "crossIssues": [], "optimizationPriorities": []}, {})

    events: list = []
    with patch.object(ta, "_audit_one_agent", side_effect=fake_audit), \
         patch.object(ta, "_aggregate_session", side_effect=fake_agg):
        result = ta.run_v4_pipeline(
            _agent_io(), "P",
            RunContext(str(tmp_path), "sess", events.append, None),
            provider=_provider(),
        )

    analysis_path = tmp_path / "sess-analysis.json"
    agentio_path = tmp_path / "sess-agentio.json"
    assert analysis_path.exists()
    assert agentio_path.exists()
    assert result.steps == 2
    assert result.analysis["sessionSummary"] == "all ok"
    agents = result.analysis["agents"]
    assert len(agents) == 2
    assert agents[0]["dimensions"]["efficiency"]["rating"] == "pass"
    assert agents[1]["dimensions"]["completion"]["rating"] == "pass"
    stages = [e["stage"] for e in events]
    assert "llm-start" in stages
    assert "llm-done" in stages
    assert "done" in stages


def test_run_v4_pipeline_empty_agents_raises(tmp_path):
    with patch.object(ta, "_audit_one_agent"), patch.object(ta, "_aggregate_session"):
        with pytest.raises(AnalysisError):
            ta.run_v4_pipeline(
                {"taskQuery": "x", "agents": []}, "P",
                RunContext(str(tmp_path), "sess", logger=None),
                provider=_provider(),
            )

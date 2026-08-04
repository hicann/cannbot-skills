# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Audit v4 纯逻辑测试：schema 校验、agent_io 合并、efficiency 确定性评级。

不启动 claude CLI（那在端到端手测覆盖）。
"""
from __future__ import annotations

import json

import pytest

from trajectory_analyzer import (
    validate_v4_schema,
    inject_v4_merge,
    _efficiency_rating,
    SchemaError,
)


def test_efficiency_rating_thresholds():
    assert _efficiency_rating({"errorCount": 0, "retryCount": 0, "turnCount": 5}) == "pass"
    assert _efficiency_rating({"errorCount": 2, "retryCount": 0, "turnCount": 5}) == "weak"
    assert _efficiency_rating({"errorCount": 0, "retryCount": 2, "turnCount": 5}) == "weak"
    assert _efficiency_rating({"errorCount": 0, "retryCount": 0, "turnCount": 30}) == "weak"
    assert _efficiency_rating({"errorCount": 5, "retryCount": 0, "turnCount": 5}) == "fail"
    assert _efficiency_rating({"errorCount": 0, "retryCount": 5, "turnCount": 5}) == "fail"
    assert _efficiency_rating({"errorCount": 0, "retryCount": 0, "turnCount": 80}) == "fail"


def _dim(rating, note=""):
    return {"rating": rating, "note": note}


def test_validate_v4_schema_accepts_minimal():
    obj = {
        "sessionSummary": "ok",
        "agents": [{"id": "main", "dimensions": {
            "completion": _dim("pass"), "quality": _dim("pass")}}],
    }
    validate_v4_schema(obj)  # 不抛


def test_validate_v4_schema_rejects_bad():
    with pytest.raises(SchemaError):
        validate_v4_schema({"agents": []})  # 缺 sessionSummary
    with pytest.raises(SchemaError):
        validate_v4_schema({"sessionSummary": "x", "agents": []})  # 空 agents
    with pytest.raises(SchemaError):
        validate_v4_schema({"sessionSummary": "x", "agents": [{"id": "main"}]})  # 缺 dimensions
    with pytest.raises(SchemaError):
        validate_v4_schema({"sessionSummary": "x", "agents": [
            {"id": "main", "dimensions": {"completion": _dim("bogus"), "quality": _dim("pass")}}]})  # 非法 rating


def test_inject_v4_merge_overwrites_envelope_and_fills_efficiency():
    agent_io = {
        "sessionId": "s1", "taskQuery": "do X", "agents": [
            {"id": "main", "parentId": None, "role": "main", "name": "main",
             "inputSummary": "task", "outputSummary": "done", "artifacts": ["Write: a.py"],
             "envelope": {"latencySec": 12.3, "tokensKt": 50, "turnCount": 4,
                          "toolCallCount": 9, "errorCount": 0, "retryCount": 0,
                          "reasoningTokensKt": 2}},
            {"id": "sub-a", "parentId": "main", "role": "subagent", "name": "general-purpose",
             "inputSummary": "dispatch", "outputSummary": "resp", "artifacts": ["Edit: b.md"],
             "envelope": {"latencySec": 45, "tokensKt": 150, "turnCount": 12,
                          "toolCallCount": 20, "errorCount": 3, "retryCount": 2,
                          "reasoningTokensKt": 5}},
        ]
    }
    llm = {"sessionSummary": "ok", "agents": [
        {"id": "main", "dimensions": {
            "completion": _dim("pass", "done"), "quality": _dim("pass", "good"),
            "efficiency": {"note": "lean"}}},
        {"id": "sub-a", "dimensions": {
            "completion": _dim("weak", "partial"), "quality": _dim("fail", "bad")}},
    ]}

    merged = inject_v4_merge(llm, agent_io)
    main = next(a for a in merged["agents"] if a["id"] == "main")
    sub = next(a for a in merged["agents"] if a["id"] == "sub-a")

    # envelope / artifacts / name 来自 agent_io
    assert main["envelope"]["turnCount"] == 4
    assert main["artifacts"] == ["Write: a.py"]
    assert sub["envelope"]["turnCount"] == 12
    assert sub["artifacts"] == ["Edit: b.md"]
    assert sub["name"] == "general-purpose"

    # efficiency.rating 服务端确定性填，LLM note 保留
    assert main["dimensions"]["efficiency"]["rating"] == "pass"
    assert main["dimensions"]["efficiency"]["note"] == "lean"
    assert sub["dimensions"]["efficiency"]["rating"] == "weak"

    # LLM 的 completion/quality 保留
    assert sub["dimensions"]["completion"]["rating"] == "weak"
    assert sub["dimensions"]["quality"]["rating"] == "fail"


def test_inject_v4_merge_stubs_missing_llm_agents():
    agent_io = {"agents": [
        {"id": "main", "parentId": None, "role": "main", "name": "main",
         "inputSummary": "", "outputSummary": "", "artifacts": [],
         "envelope": {"latencySec": 0, "tokensKt": 0, "turnCount": 1,
                      "toolCallCount": 0, "errorCount": 0, "retryCount": 0, "reasoningTokensKt": 0}},
        {"id": "sub-a", "parentId": "main", "role": "subagent", "name": "x",
         "inputSummary": "", "outputSummary": "", "artifacts": [],
         "envelope": {"latencySec": 0, "tokensKt": 0, "turnCount": 1,
                      "toolCallCount": 0, "errorCount": 0, "retryCount": 0, "reasoningTokensKt": 0}},
    ]}
    llm = {"sessionSummary": "ok", "agents": [
        {"id": "main", "dimensions": {"completion": _dim("pass"), "quality": _dim("pass")}}]}

    merged = inject_v4_merge(llm, agent_io)
    sub = next(a for a in merged["agents"] if a["id"] == "sub-a")
    # LLM 漏了 sub-a → 补 n-a stub
    assert sub["dimensions"]["completion"]["rating"] == "n-a"
    assert sub["dimensions"]["quality"]["rating"] == "n-a"
    assert sub["dimensions"]["efficiency"]["rating"] == "pass"  # envelope 干净
    assert json.dumps(merged, ensure_ascii=False)  # 结果可序列化

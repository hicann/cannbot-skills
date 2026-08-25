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

"""Token 采集单元测试 (mock, 不调 LLM):

回归背景: 反馈"任务完成后没有真正采集到 token 消耗"。
实测 opencode serve 1.18.10:
  - POST /session/{id}/message 响应的 info.tokens 只覆盖最后一条 assistant 消息,
    不含 task 工具分发的 subagent 消息 → 单看该字段会严重低估消耗;
  - GET /session/{id} 的顶层 tokens/cost 为会话级累计口径
    (所有 assistant 消息求和, 含 subagent), 才是任务完成后的真实消耗;
  - 会话级 tokens 无 "total" 键 (input/output/reasoning + 嵌套 cache)。

本测试模拟上述两种口径, 校验 run_via_serve 采集到的是会话级累计值。

用法:
  pytest runner/test_token_collection.py -v
"""

import os
import sys

import pytest
import requests as real_requests

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNER_DIR)

import report  # noqa: E402
import run_eval  # noqa: E402


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data

    def iter_lines(self, decode_unicode=True):
        return iter([])


# 实测抓取的两种口径 (含 subagent 任务的真实比例):
# 消息级 = 最后一条 assistant 消息, 会话级 = 全会话累计
MESSAGE_LEVEL_TOKENS = {
    "total": 7954,
    "input": 144,
    "output": 2,
    "reasoning": 0,
    "cache": {"write": 0, "read": 7808},
}
MESSAGE_LEVEL_COST = 4.25824e-05
SESSION_LEVEL_TOKENS = {
    "input": 278,
    "output": 88,
    "reasoning": 0,
    "cache": {"read": 15488, "write": 0},
}
SESSION_LEVEL_COST = 0.0001069264
SESSION_TOTAL = (278 + 88 + 0 + 15488 + 0)  # 会话级无 total, 需含 cache 求和


class _FakeServe:
    """模拟 opencode serve: 会话/消息/SSE 事件。

    消息响应 info.tokens 为消息级口径 (subagent 消耗不在其中),
    会话查询 (GET /session/{id}) 返回会话级累计口径。
    """
    # 异常类型需与真实 requests 一致 (except 子句按模块属性解析)
    Timeout = real_requests.Timeout
    HTTPError = real_requests.HTTPError
    ConnectionError = real_requests.ConnectionError

    def __init__(self, session_tokens=SESSION_LEVEL_TOKENS,
                 session_cost=SESSION_LEVEL_COST,
                 fail_session_get=False):
        self.session_id = "ses_token_test"
        self.session_tokens = session_tokens
        self.session_cost = session_cost
        self.fail_session_get = fail_session_get
        self.session_gets = 0

    def post(self, url, json=None, timeout=None, **kw):
        if url.endswith("/session"):
            return _FakeResp({
                "id": self.session_id,
                "tokens": {"input": 0, "output": 0, "reasoning": 0,
                           "cache": {"read": 0, "write": 0}},
            })
        if url.endswith("/message"):
            return _FakeResp({
                "parts": [{"type": "text", "text": "ok"}],
                "info": {
                    "role": "assistant",
                    "providerID": "deepseek",
                    "modelID": "deepseek-v4-flash",
                    "tokens": MESSAGE_LEVEL_TOKENS,
                    "cost": MESSAGE_LEVEL_COST,
                },
            })
        return _FakeResp({})  # abort / delete 等

    def get(self, url, **kw):
        if "/event" in url:
            return _FakeResp({})  # SSE 流 (空)
        self.session_gets += 1
        if self.fail_session_get:
            raise real_requests.ConnectionError("session get down")
        return _FakeResp({
            "id": self.session_id,
            "tokens": self.session_tokens,
            "cost": self.session_cost,
        })


def run_via_serve(monkeypatch, fake):
    monkeypatch.setattr(run_eval, "requests", fake)
    monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)
    return run_eval.run_via_serve("tasks/level2/x", "p", "/tmp/opencode",
                                   timeout=60)


# ══════════════════════════════════════════════════════════════════════
#  run_via_serve: 采集会话级累计 token (含 subagent)
# ══════════════════════════════════════════════════════════════════════

class TestServeTokenCollection:
    def test_collects_session_level_tokens(self, monkeypatch):
        """任务完成后采集会话级累计 tokens/cost (含 subagent 消耗)。

        消息响应 info.tokens 只有最后一条 assistant 消息的 2 个输出 token,
        会话级累计输出 88; 仅采消息级会把多 agent 工作流的消耗漏掉。
        """
        fake = _FakeServe()
        result = run_via_serve(monkeypatch, fake)
        assert result["status"] == "success"
        assert result["tokens"] == SESSION_LEVEL_TOKENS
        assert result["cost"] == SESSION_LEVEL_COST
        # 防呆: 会话级口径确实含 subagent 消耗 (否则断言无意义)
        assert SESSION_LEVEL_TOKENS["output"] > MESSAGE_LEVEL_TOKENS["output"]
        assert SESSION_LEVEL_TOKENS["input"] > MESSAGE_LEVEL_TOKENS["input"]
        assert SESSION_LEVEL_COST > MESSAGE_LEVEL_COST

    def test_message_level_would_undercount_by_an_order(self):
        """数据关系基线: 消息级 output=2 与会话级 output=88 相差一个数量级,
        证明仅采消息响应 info.tokens 必然漏掉 subagent 消耗。
        """
        assert MESSAGE_LEVEL_TOKENS["output"] * 10 < SESSION_LEVEL_TOKENS["output"]
        assert MESSAGE_LEVEL_TOKENS["input"] < SESSION_LEVEL_TOKENS["input"]
        assert MESSAGE_LEVEL_COST < SESSION_LEVEL_COST

    def test_session_get_failure_falls_back_to_message_info(self, monkeypatch):
        """GET /session/{id} 失败时回退消息级 info.tokens, 不中断任务。"""
        fake = _FakeServe(fail_session_get=True)
        result = run_via_serve(monkeypatch, fake)
        assert result["status"] == "success"
        assert result["tokens"] == MESSAGE_LEVEL_TOKENS
        assert result["cost"] == MESSAGE_LEVEL_COST


# ══════════════════════════════════════════════════════════════════════
#  report._token_total: 会话级 tokens 结构 (无 total 键, 嵌套 cache)
# ══════════════════════════════════════════════════════════════════════

class TestTokenTotalSessionShape:
    def test_session_shape_includes_cache(self):
        """会话级 tokens 无 total 键: input+output+reasoning+cache 求和。"""
        assert report._token_total(SESSION_LEVEL_TOKENS) == SESSION_TOTAL

    def test_message_shape_total_key_still_wins(self):
        """消息级 tokens 有 total 键时仍直接取 total (向后兼容)。"""
        assert report._token_total({"total": 100, "input": 90, "output": 10,
                                    "cache": {"read": 999}}) == 100

    def test_collect_stats_session_shape(self):
        """collect_stats 用会话级 tokens 汇总时计入 cache。"""
        stats = report.collect_stats([
            {"op_name": "a", "status": "success", "duration_s": 1,
             "tokens": SESSION_LEVEL_TOKENS, "cost": SESSION_LEVEL_COST},
        ])
        assert stats["tokens"] == SESSION_TOTAL

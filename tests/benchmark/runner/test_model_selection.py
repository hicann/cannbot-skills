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

"""--model 模型指定功能测试。

分两层:
  1. 单元测试 (mock, 不调 LLM):
     - parse_model 格式解析
     - prompt_for_model 交互问询 (非 TTY 报错 / TTY 输入)
     - ServeManager 启动命令携带 -m
     - run_via_serve POST body 携带 model 字段
     - run_via_pipe 命令携带 -m
  2. 在线验证 (slow, 真实调 LLM, 默认 MiniMax-M3):
     - POST body 指定 model → 响应 info 中确为该模型
     - 工作目录 .opencode/opencode.json 钉住模型 → 不带 model 的请求默认用该模型

  注: opencode serve 不支持 -m 参数, serve 模式模型通过 opencode.json 下发;
      opencode run (pipe 回退模式) 支持 -m。

用法:
  # 单元测试 (默认, 不触发 LLM)
  pytest runner/test_model_selection.py -v

  # 在线验证 (真实调 LLM)
  pytest runner/test_model_selection.py -v --runslow -k live

  # 换模型/端口
  TEST_MODEL=zhipuai-coding-plan/glm-5.2 TEST_MODEL_SERVE_PORT=4098 \
      pytest runner/test_model_selection.py -v --runslow -k live

依赖:
  - 在线验证需 opencode CLI 已安装且对应 provider 已认证
"""

import os
import shutil
import sys
import tempfile

import pytest
import requests

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNER_DIR)

import run_eval  # noqa: E402
from run_eval import ServeManager, parse_model, prompt_for_model  # noqa: E402

# ── 在线验证参数 ──────────────────────────────────────────────────────
LIVE_MODEL = os.environ.get("TEST_MODEL", "minimax-cn-coding-plan/MiniMax-M3")
LIVE_PORT = int(os.environ.get("TEST_MODEL_SERVE_PORT", "4099"))
LIVE_URL = f"http://127.0.0.1:{LIVE_PORT}"
MSG_TIMEOUT = int(os.environ.get("TEST_MODEL_MSG_TIMEOUT", "180"))


def _extract_model(msg: dict) -> tuple[str, str]:
    """从 serve 消息响应中提取实际使用的 (providerID, modelID)。"""
    info = msg.get("info", {})
    nested = info.get("model", {}) if isinstance(info.get("model"), dict) else {}
    return (info.get("providerID") or nested.get("providerID", ""),
            info.get("modelID") or nested.get("modelID", ""))


def _read_serve_log(mgr) -> str:
    """读取 ServeManager 日志 (启动失败诊断); 无日志返回空串。"""
    if mgr.log_path and os.path.isfile(mgr.log_path):
        with open(mgr.log_path) as f:
            return f.read()
    return ""


# ══════════════════════════════════════════════════════════════════════
#  单元测试: parse_model / prompt_for_model
# ══════════════════════════════════════════════════════════════════════

class TestParseModel:
    @staticmethod
    def test_valid():
        assert parse_model("minimax-cn-coding-plan/MiniMax-M3") == \
            ("minimax-cn-coding-plan", "MiniMax-M3")

    @staticmethod
    def test_valid_multi_segment():
        # modelID 允许再含 / (如 alibaba-cn/MiniMax/MiniMax-M2.7)
        assert parse_model("alibaba-cn/MiniMax/MiniMax-M2.7") == \
            ("alibaba-cn", "MiniMax/MiniMax-M2.7")

    @pytest.mark.parametrize("bad", ["", "nomodel", "/onlymodel", "onlyprovider/", "/"])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_model(bad)


class TestPromptForModel:
    @staticmethod
    def test_non_tty_exits(monkeypatch):
        """非交互环境 (CI) 直接报错, 指引用 --model。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(ValueError):
            prompt_for_model()

    @staticmethod
    def test_interactive_input(monkeypatch):
        """TTY 环境下列出可用模型并接受用户输入。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        class FakeCompleted:
            stdout = "minimax-cn-coding-plan/MiniMax-M3\nzhipuai-coding-plan/glm-5.2\n"
        monkeypatch.setattr(run_eval.subprocess, "run",
                            lambda *a, **k: FakeCompleted())
        monkeypatch.setattr("builtins.input",
                            lambda prompt="": " minimax-cn-coding-plan/MiniMax-M3 ")
        assert prompt_for_model() == "minimax-cn-coding-plan/MiniMax-M3"

    @staticmethod
    def test_interactive_empty_retries(monkeypatch):
        """空输入会被拒绝并重新问询。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        class FakeCompleted:
            stdout = ""
        monkeypatch.setattr(run_eval.subprocess, "run",
                            lambda *a, **k: FakeCompleted())
        answers = iter(["", "  ", "zhipuai-coding-plan/glm-5.2"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        assert prompt_for_model() == "zhipuai-coding-plan/glm-5.2"


# ══════════════════════════════════════════════════════════════════════
#  单元测试: serve / pipe / POST 链路携带模型
# ══════════════════════════════════════════════════════════════════════

class TestServeManagerModelFlag:
    """验证 ServeManager 的模型下发方式。

    opencode serve 不支持 -m, 模型通过工作目录 .opencode/opencode.json 钉住。
    """

    def test_model_written_to_opencode_json(self, monkeypatch, tmp_path):
        cmd, workdir = self._launch(
            monkeypatch, tmp_path, "minimax-cn-coding-plan/MiniMax-M3")
        assert "-m" not in cmd  # serve 不支持 -m
        import json
        cfg_path = os.path.join(workdir, ".opencode", "opencode.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        assert cfg["model"] == "minimax-cn-coding-plan/MiniMax-M3"

    def test_model_config_merges_existing(self, monkeypatch, tmp_path):
        """已有 opencode.json 的其他配置项必须保留。"""
        import json
        workdir = tmp_path / "workdir"
        cfg_dir = workdir / ".opencode"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "opencode.json").write_text(json.dumps(
            {"permission": {}, "compaction": {"auto": True}}))

        self._launch(monkeypatch, tmp_path, "minimax-cn-coding-plan/MiniMax-M3")
        with open(cfg_dir / "opencode.json") as f:
            cfg = json.load(f)
        assert cfg["model"] == "minimax-cn-coding-plan/MiniMax-M3"
        assert cfg["compaction"] == {"auto": True}

    def test_no_model_no_config(self, monkeypatch, tmp_path):
        cmd, workdir = self._launch(monkeypatch, tmp_path, None)
        assert "-m" not in cmd
        assert not os.path.exists(os.path.join(workdir, ".opencode", "opencode.json"))

    @staticmethod
    def _launch(monkeypatch, tmp_path, model):
        monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)

        captured = {}

        class FakeProc:
            @staticmethod
            def poll():
                return None

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return FakeProc()
        monkeypatch.setattr(run_eval.subprocess, "Popen", fake_popen)

        # 启动前健康检查 False (走到启动分支), 启动后 True (就绪退出)
        checks = iter([False, True])
        monkeypatch.setattr(ServeManager, "_health_check",
                            lambda self: next(checks))

        workdir = str(tmp_path / "workdir")
        os.makedirs(workdir, exist_ok=True)
        mgr = ServeManager(port=59999, url="http://127.0.0.1:59999", model=model)
        assert mgr.ensure_running(cwd=workdir) is True
        return captured["cmd"], workdir


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data

    @staticmethod
    def iter_lines(decode_unicode=True):
        return iter([])


class TestRunViaServePayload:
    """验证 run_via_serve POST body 的 model 字段与实际模型记录。"""

    def test_payload_includes_model(self, monkeypatch):
        result, posted = self._run(monkeypatch, "minimax-cn-coding-plan/MiniMax-M3")
        assert posted["message"]["model"] == {
            "providerID": "minimax-cn-coding-plan", "modelID": "MiniMax-M3"}
        assert result["status"] == "success"
        assert result["model"] == "minimax-cn-coding-plan/MiniMax-M3"
        assert result["model_actual"] == "minimax-cn-coding-plan/MiniMax-M3"

    def test_payload_without_model(self, monkeypatch):
        result, posted = self._run(monkeypatch, None)
        assert "model" not in posted["message"]
        assert result["model"] == "default"

    @staticmethod
    def _run(monkeypatch, model):
        posted = {}

        class FakeRequests:
            @staticmethod
            def post(url, timeout=None, **kw):
                body = kw.get("json")
                if url.endswith("/session"):
                    return _FakeResp({"id": "ses_test_model"})
                if url.endswith("/message"):
                    posted["message"] = body
                    return _FakeResp({
                        "parts": [{"type": "text", "text": "ok"}],
                        "info": {
                            "tokens": {}, "cost": 0,
                            "providerID": "minimax-cn-coding-plan",
                            "modelID": "MiniMax-M3",
                        },
                    })
                return _FakeResp({})

            @staticmethod
            def get(url, **kw):
                return _FakeResp({})

        monkeypatch.setattr(run_eval, "requests", FakeRequests)
        return run_eval.run_via_serve(
            "tasks/level3/add_rms_norm_dynamic_quant", "prompt", "/tmp",
            model=model), posted


class TestRunViaPipeModelFlag:
    """验证 pipe 回退模式的 opencode run 命令。"""

    def test_cmd_includes_model(self, monkeypatch, tmp_path):
        cmd = self._run(monkeypatch, tmp_path, "minimax-cn-coding-plan/MiniMax-M3")
        i = cmd.index("-m")
        assert cmd[i + 1] == "minimax-cn-coding-plan/MiniMax-M3"

    def test_cmd_without_model(self, monkeypatch, tmp_path):
        cmd = self._run(monkeypatch, tmp_path, None)
        assert "-m" not in cmd

    @staticmethod
    def _run(monkeypatch, tmp_path, model):
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeCompleted()
        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("prompt")
        run_eval.run_via_pipe("op", "prompt", out_dir, model=model)
        return captured["cmd"]


# ══════════════════════════════════════════════════════════════════════
#  在线验证: 真实 opencode serve + MiniMax-M3
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestLiveModelSelection:
    """通过 ServeManager 真实启动 serve, 验证模型指定生效 (默认 minimax-cn-coding-plan/MiniMax-M3)。

    走与 run_eval.py 生产环境完全相同的路径:
      ServeManager(model=...) → 写 .opencode/opencode.json → opencode serve → POST /message
    """

    @pytest.fixture(scope="class")
    def live_serve(self):
        provider_id, model_id = parse_model(LIVE_MODEL)
        workdir = tempfile.mkdtemp(prefix="test-model-selection-")
        try:
            mgr = ServeManager(port=LIVE_PORT, url=LIVE_URL, model=LIVE_MODEL)
            if not mgr.ensure_running(cwd=workdir):
                log = _read_serve_log(mgr)
                pytest.fail(f"serve 启动失败 (model={LIVE_MODEL}):\n{log[:2000]}")

            yield {"provider_id": provider_id, "model_id": model_id}
        finally:
            mgr.shutdown()
            shutil.rmtree(workdir, ignore_errors=True)

    def test_payload_model_applies(self, live_serve):
        """POST body 指定 model → 响应 info 中确为该模型。"""
        session_id = self._new_session("test-payload-model")
        data = self._ask(session_id, {
            "model": {"providerID": live_serve["provider_id"],
                      "modelID": live_serve["model_id"]},
        })
        provider_id, model_id = _extract_model(data)
        assert (provider_id, model_id) == \
            (live_serve["provider_id"], live_serve["model_id"]), (
            f"POST 指定 {LIVE_MODEL} 未生效, 实际: {provider_id}/{model_id}\n"
            f"info keys: {list(data.get('info', {}).keys())}"
        )

    def test_workdir_config_model_applies(self, live_serve):
        """opencode.json 钉住的模型 → 不带 model 的请求默认用该模型。"""
        session_id = self._new_session("test-workdir-config-model")
        data = self._ask(session_id, {})
        provider_id, model_id = _extract_model(data)
        assert (provider_id, model_id) == \
            (live_serve["provider_id"], live_serve["model_id"]), (
            f"opencode.json model={LIVE_MODEL} 未生效, 实际: {provider_id}/{model_id}\n"
            f"info keys: {list(data.get('info', {}).keys())}"
        )

    @staticmethod
    def _new_session(title: str) -> str:
        resp = requests.post(f"{LIVE_URL}/session", json={"title": title},
                             timeout=10)
        resp.raise_for_status()
        return resp.json()["id"]

    @staticmethod
    def _ask(session_id: str, payload_extra: dict) -> dict:
        payload = {"parts": [{"type": "text",
                              "text": "只回复 OK 两个字母, 不要说任何其他内容。"}]}
        payload.update(payload_extra)
        resp = requests.post(f"{LIVE_URL}/session/{session_id}/message",
                             json=payload, timeout=MSG_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

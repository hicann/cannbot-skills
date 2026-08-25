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

"""kimi harness 单元测试与连通性验证。

分两层:
  1. 单元测试 (mock, 不调 LLM):
     - --harness 参数解析 (choices/default)
     - parse_model harness 感知 (kimi 不要求 provider/model 格式)
     - prompt_for_model 非 opencode harness 交互问询
     - run_via_pipe kimi CLI 命令构建
     - run_single_op harness 路由 (kimi 不走 serve)
  2. 连通性验证 (slow, 真实调 kimi CLI):
     - kimi -p 基础连通 (默认 k3 模型)

用法:
  # 单元测试 (默认, 不触发 LLM)
  pytest runner/test_kimi_harness.py -v

  # 连通性验证 (真实调 kimi CLI)
  pytest runner/test_kimi_harness.py -v --runslow -k connectivity

  # 换模型/超时
  KIMI_MODEL=k2 KIMI_TIMEOUT=120 \
      pytest runner/test_kimi_harness.py -v --runslow -k connectivity

依赖:
  - 连通性验证需 kimi CLI 已安装且在 PATH 中
"""

import os
import sys

import pytest

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNER_DIR)

import run_eval  # noqa: E402
from run_eval import (HARNESS_CHOICES, HARNESS_DEFAULT,  # noqa: E402
                       parse_model, prompt_for_model, run_via_pipe,
                       run_single_op)

# ── 连通性验证参数 ──────────────────────────────────────────────────────
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-code/k3")
KIMI_TIMEOUT = int(os.environ.get("KIMI_TIMEOUT", "60"))


# ══════════════════════════════════════════════════════════════════════
#  单元测试: --harness 参数
# ══════════════════════════════════════════════════════════════════════

class TestHarnessCli:
    """验证 harness 参数定义。"""

    def test_choices(self):
        assert "opencode" in HARNESS_CHOICES
        assert "claude" in HARNESS_CHOICES
        assert "kimi" in HARNESS_CHOICES
        assert len(HARNESS_CHOICES) == 3

    def test_default(self):
        assert HARNESS_DEFAULT == "opencode"

    def test_argparse_accepts_harness(self):
        """验证 --harness 参数被 argparse 接受。"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--harness", default=HARNESS_DEFAULT,
                            choices=HARNESS_CHOICES)
        # 默认值
        args = parser.parse_args([])
        assert args.harness == "opencode"
        # 显式指定
        args = parser.parse_args(["--harness", "kimi"])
        assert args.harness == "kimi"
        args = parser.parse_args(["--harness", "claude"])
        assert args.harness == "claude"

    def test_argparse_rejects_invalid(self):
        """无效 harness 被 argparse 拒绝。"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--harness", default=HARNESS_DEFAULT,
                            choices=HARNESS_CHOICES)
        with pytest.raises(SystemExit):
            parser.parse_args(["--harness", "invalid"])


# ══════════════════════════════════════════════════════════════════════
#  单元测试: parse_model 对 kimi harness 的适配
# ══════════════════════════════════════════════════════════════════════

class TestParseModelHarness:
    """parse_model harness 感知: 非 opencode 不解析 provider/model。"""

    def test_opencode_requires_provider_format(self):
        provider, model_id = parse_model("p/m", harness="opencode")
        assert provider == "p"
        assert model_id == "m"

    def test_opencode_rejects_plain_model(self):
        with pytest.raises(ValueError):
            parse_model("k3", harness="opencode")

    def test_kimi_accepts_plain_model(self):
        provider, model_id = parse_model("k3", harness="kimi")
        assert provider == ""
        assert model_id == "k3"

    def test_claude_accepts_plain_model(self):
        provider, model_id = parse_model("sonnet", harness="claude")
        assert provider == ""
        assert model_id == "sonnet"

    def test_kimi_accepts_slash_in_name(self):
        """带 / 的模型名在 kimi harness 下不拆分为 provider/model。"""
        provider, model_id = parse_model("kimi-for-coding/k3", harness="kimi")
        assert provider == ""
        assert model_id == "kimi-for-coding/k3"


# ══════════════════════════════════════════════════════════════════════
#  单元测试: prompt_for_model 非 opencode 交互
# ══════════════════════════════════════════════════════════════════════

class TestPromptForModelHarness:
    def test_kimi_prompts_for_model_name(self, monkeypatch):
        """kimi harness 不调 opencode models, 直接提示输入模型名。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "k3")
        result = prompt_for_model(harness="kimi")
        assert result == "k3"

    def test_kimi_respects_allowed(self, monkeypatch):
        """白名单: kimi harness 下拒绝非白名单模型。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        answers = iter(["evil-model", "k3"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        result = prompt_for_model(harness="kimi", allowed=["k3", "k2"])
        assert result == "k3"

    def test_claude_prompts_for_model_name(self, monkeypatch):
        """claude harness 不调 opencode models, 直接提示输入模型名。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "sonnet")
        result = prompt_for_model(harness="claude")
        assert result == "sonnet"


# ══════════════════════════════════════════════════════════════════════
#  单元测试: run_via_pipe kimi CLI 命令构建
# ══════════════════════════════════════════════════════════════════════

class TestKimiCommand:
    """验证 kimi harness 的 CLI 命令构建。"""

    def test_kimi_basic_command(self, monkeypatch, tmp_path):
        """kimi harness: 构建 kimi -p <prompt> 命令。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "kimi response"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kw"] = kw
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("hello world")

        result = run_eval.run_via_pipe(
            "test_op", "hello world", out_dir,
            harness="kimi",
        )
        assert result["status"] == "success"
        assert "kimi" in captured["cmd"]
        assert "-p" in captured["cmd"]
        assert "hello world" in captured["cmd"]
        assert captured["kw"]["cwd"] == out_dir

    def test_kimi_command_with_model(self, monkeypatch, tmp_path):
        """kimi harness: --model k3 追加到命令。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("prompt")

        run_eval.run_via_pipe(
            "test_op", "prompt", out_dir,
            model="k3", harness="kimi",
        )
        i = captured["cmd"].index("--model")
        assert captured["cmd"][i + 1] == "k3"
        # 无 CLAUDE.md 时不追加 --agent-file
        assert "--agent-file" not in captured["cmd"]

    def test_kimi_command_no_model(self, monkeypatch, tmp_path):
        """kimi harness: 未指定 model 时不带 --model。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("prompt")

        run_eval.run_via_pipe(
            "test_op", "prompt", out_dir,
            harness="kimi",
        )
        assert "--model" not in captured["cmd"]
        assert "--agent-file" not in captured["cmd"]

    def test_kimi_agent_file(self, monkeypatch, tmp_path):
        """kimi harness: 当 CLAUDE.md 存在时追加 --agent-file。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kw"] = kw
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "CLAUDE.md").write_text("---\nname: cannbot\n---\n")
        (work_dir / "prompt.txt").write_text("prompt")

        run_eval.run_via_pipe(
            "test_op", "prompt", str(work_dir),
            harness="kimi", work_dir=str(work_dir),
        )
        assert "--agent-file" in captured["cmd"]
        assert "CLAUDE.md" in captured["cmd"][captured["cmd"].index("--agent-file") + 1]
        assert "--agent-file" in captured["cmd"]

    def test_kimi_uses_workdir_as_cwd(self, monkeypatch, tmp_path):
        """kimi harness: cwd 指向 work_dir 而非 output_dir。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["kw"] = kw
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "prompt.txt").write_text("prompt")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        run_eval.run_via_pipe(
            "test_op", "prompt", str(out_dir),
            harness="kimi", work_dir=str(work_dir),
        )
        assert captured["kw"]["cwd"] == str(work_dir)

    def test_claude_basic_command(self, monkeypatch, tmp_path):
        """claude harness: 构建 claude -p <prompt> 命令。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "claude response"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("hello")

        run_eval.run_via_pipe(
            "test_op", "hello", out_dir,
            harness="claude",
        )
        assert "claude" in captured["cmd"]
        assert "-p" in captured["cmd"]

    def test_claude_uses_workdir_as_cwd(self, monkeypatch, tmp_path):
        """claude harness: cwd 指向 work_dir (含 .claude/ + CLAUDE.md) 而非 output_dir。

        回归 PR#755 review: claude 此前用 cwd=output_dir (results/{op}, 无 .claude 配置),
        等于以裸 agent 运行, 完全不加载 CANNBot 多 Agent 工作流。
        """
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["kw"] = kw
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "prompt.txt").write_text("prompt")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        run_eval.run_via_pipe(
            "test_op", "prompt", str(out_dir),
            harness="claude", work_dir=str(work_dir),
        )
        assert captured["kw"]["cwd"] == str(work_dir), \
            "claude harness 必须以 work_dir 为 cwd 才能加载 CANNBot 工作流"

    def test_claude_agent_file(self, monkeypatch, tmp_path):
        """claude harness: 当 work_dir 下存在 CLAUDE.md 时追加 --agent-file。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kw"] = kw
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "CLAUDE.md").write_text("---\nname: cannbot\n---\n")
        (work_dir / "prompt.txt").write_text("prompt")

        run_eval.run_via_pipe(
            "test_op", "prompt", str(work_dir),
            harness="claude", work_dir=str(work_dir),
        )
        assert "--agent-file" in captured["cmd"]
        agent_path = captured["cmd"][captured["cmd"].index("--agent-file") + 1]
        assert agent_path.endswith("CLAUDE.md")
        assert captured["kw"]["cwd"] == str(work_dir)

    def test_claude_no_agent_file_when_missing(self, monkeypatch, tmp_path):
        """claude harness: work_dir 无 CLAUDE.md 时不追加 --agent-file。"""
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeCompleted()

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "prompt.txt").write_text("prompt")

        run_eval.run_via_pipe(
            "test_op", "prompt", str(out_dir),
            harness="claude", work_dir=str(out_dir),
        )
        assert "--agent-file" not in captured["cmd"]

    def test_unknown_harness_errors(self, tmp_path):
        """未知 harness 返回 error 状态。"""
        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("p")

        result = run_eval.run_via_pipe(
            "test_op", "p", out_dir,
            harness="unknown",
        )
        assert result["status"] == "error"
        assert "unknown" in result["stderr"]

    def test_harness_recorded_in_result(self, monkeypatch, tmp_path):
        """result 中记录 harness 字段。"""

        class FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        monkeypatch.setattr(run_eval.subprocess, "run",
                            lambda *a, **k: FakeCompleted())

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("p")

        result = run_eval.run_via_pipe(
            "test_op", "p", out_dir,
            harness="kimi",
        )
        assert result["harness"] == "kimi"

    def test_kimi_cli_not_found(self, monkeypatch, tmp_path):
        """kimi CLI 未安装时返回 error 状态。"""
        def fake_run(*a, **kw):
            raise FileNotFoundError("kimi not found")
        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("p")

        result = run_eval.run_via_pipe(
            "test_op", "p", out_dir,
            harness="kimi",
        )
        assert result["status"] == "error"
        assert "kimi" in result["stderr"]

    def test_kimi_timeout(self, monkeypatch, tmp_path):
        """kimi harness 超时返回 timeout 状态。"""
        import subprocess

        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["kimi"], timeout=1)
        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

        out_dir = str(tmp_path)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write("p")

        result = run_eval.run_via_pipe(
            "test_op", "p", out_dir,
            harness="kimi", timeout=1,
        )
        assert result["status"] == "timeout"


# ══════════════════════════════════════════════════════════════════════
#  单元测试: run_single_op harness 路由
# ══════════════════════════════════════════════════════════════════════

class TestRunSingleOpHarness:
    """run_single_op 根据 harness 选择执行模式。"""

    def test_kimi_skips_serve(self, monkeypatch, tmp_path):
        """kimi harness 不走 serve, 直接走 pipe。"""
        serve_checked = {"called": False}

        def fake_check(self, cwd=None):
            serve_checked["called"] = True
            return True

        monkeypatch.setattr(run_eval.ServeManager, "ensure_running", fake_check)

        pipe_called = {"harness": None}

        def fake_pipe(*a, **kw):
            pipe_called["harness"] = kw.get("harness")
            return {"status": "success", "op_name": "x"}

        monkeypatch.setattr(run_eval, "run_via_pipe", fake_pipe)

        out_dir = str(tmp_path)
        run_single_op("test_op", "prompt", out_dir,
                       work_dir=str(tmp_path),
                       cfg=run_eval.OpRunConfig(use_serve=True, harness="kimi"))
        assert not serve_checked["called"], \
            "kimi harness 不应调用 serve"
        assert pipe_called["harness"] == "kimi"

    def test_opencode_uses_serve(self, monkeypatch, tmp_path):
        """opencode harness 优先用 serve。"""
        serve_checked = {"called": False}

        def fake_check(self, cwd=None):
            serve_checked["called"] = True
            return True

        monkeypatch.setattr(run_eval.ServeManager, "ensure_running", fake_check)

        serve_called = {"called": False}

        def fake_serve(*a, **kw):
            serve_called["called"] = True
            return {"status": "success"}

        monkeypatch.setattr(run_eval, "run_via_serve", fake_serve)

        out_dir = str(tmp_path)
        run_single_op("test_op", "prompt", out_dir,
                       work_dir=str(tmp_path),
                       cfg=run_eval.OpRunConfig(use_serve=True, harness="opencode"))
        assert serve_checked["called"], \
            "opencode harness 应调用 serve check"
        assert serve_called["called"], \
            "opencode harness serve 就绪后应走 run_via_serve"


# ══════════════════════════════════════════════════════════════════════
#  连通性验证: 真实 kimi CLI
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestKimiConnectivity:
    """验证 kimi CLI 基础连通性。

    使用环境变量配置:
      KIMI_MODEL   - 模型名 (默认 k3)
      KIMI_TIMEOUT - 超时秒数 (默认 60)
    """

    def test_kimi_version(self):
        """kimi CLI 可执行且输出版本/帮助信息。"""
        import subprocess
        try:
            # 尝试 --version 或 --help
            for flag in ["--version", "-v", "--help"]:
                try:
                    proc = subprocess.run(
                        ["kimi", flag],
                        capture_output=True, text=True, timeout=10,
                    )
                    if proc.returncode == 0:
                        return  # 成功
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            pytest.fail(
                "kimi CLI 不可用: 请确认 kimi 已安装且在 PATH 中。\n"
                "安装参考: https://kimi.moonshot.cn/")
        except FileNotFoundError:
            pytest.fail(
                "kimi CLI 未找到: 请确认 kimi 已安装且在 PATH 中。\n"
                "安装参考: https://kimi.moonshot.cn/")

    def test_kimi_basic_prompt(self):
        """kimi -p 基础连通: 发送简单 prompt, 验证返回码 0 且响应非空。"""
        import subprocess

        model = KIMI_MODEL
        timeout = KIMI_TIMEOUT
        prompt = "请只回复 OK 两个字母, 不要说任何其他内容。"

        try:
            proc = subprocess.run(
                ["kimi", "-p", prompt, "--model", model],
                capture_output=True, text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            pytest.fail(
                "kimi CLI 未找到: 请确认 kimi 已安装且在 PATH 中。")
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"kimi 响应超时 ({timeout}s): model={model}。\n"
                f"可设置 KIMI_TIMEOUT 环境变量调整超时。")

        assert proc.returncode == 0, (
            f"kimi 返回非零: rc={proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}\n"
            f"stdout: {proc.stdout[:500]}"
        )
        assert proc.stdout.strip(), (
            f"kimi 返回空响应 (model={model})。\n"
            f"stderr: {proc.stderr[:500]}"
        )

    def test_kimi_with_k3_model(self):
        """使用 k3 模型做一次简单问答, 验证模型连通。"""
        import subprocess

        model = KIMI_MODEL
        timeout = KIMI_TIMEOUT
        prompt = (
            "请回答以下问题: 1+1 等于几? "
            "只回复数字, 不要其他内容。"
        )

        try:
            proc = subprocess.run(
                ["kimi", "-p", prompt, "--model", model],
                capture_output=True, text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            pytest.fail("kimi CLI 未找到。")
        except subprocess.TimeoutExpired:
            pytest.fail(f"kimi 响应超时 ({timeout}s)。")

        assert proc.returncode == 0, (
            f"kimi (model={model}) 返回非零: rc={proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )
        assert proc.stdout.strip(), (
            f"kimi (model={model}) 返回空响应。"
        )
        # 验证响应中包含预期内容 (数字 2)
        assert "2" in proc.stdout, (
            f"kimi (model={model}) 回答了 1+1 但未包含预期答案 '2':\n"
            f"stdout: {proc.stdout[:500]}"
        )

    def test_kimi_responds_in_chinese(self):
        """kimi 中文问答连通: 发送中文 prompt 验证中文响应。"""
        import subprocess

        model = KIMI_MODEL
        timeout = KIMI_TIMEOUT
        prompt = "用中文回答: 你好, 请介绍你自己。一句话即可。"

        try:
            proc = subprocess.run(
                ["kimi", "-p", prompt, "--model", model],
                capture_output=True, text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            pytest.fail("kimi CLI 未找到。")
        except subprocess.TimeoutExpired:
            pytest.fail(f"kimi 响应超时 ({timeout}s)。")

        assert proc.returncode == 0, (
            f"kimi 返回非零: rc={proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )
        assert len(proc.stdout.strip()) > 0, (
            "kimi 中文问答返回空响应。"
        )
        # 宽松检查: 响应非空且包含中文或 kimi 关键词
        stdout_lower = proc.stdout.lower()
        has_content = (
            "kimi" in stdout_lower
            or "你好" in proc.stdout
            or "我是" in proc.stdout
            or "助手" in proc.stdout
            or len(proc.stdout.strip()) > 5  # 至少有一句话
        )
        assert has_content, (
            f"kimi 中文问答响应内容异常:\n"
            f"stdout: {proc.stdout[:500]}"
        )

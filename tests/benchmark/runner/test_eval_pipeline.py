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

"""评测管线新功能单元测试 (mock, 不调 LLM):

  - resolve_op_timeout      per-op > category > default > env 超时优先级
  - 模型白名单               resolve/validate/prompt_for_model 问询拦截
  - run_via_serve           超时纳入重试 (Timeout 不再直接 break)
  - delivery_complete       交付件 .whl 完整性检查
  - persist_op_code         dist/*.whl 持久化
  - report                  md/html 汇总报告生成
  - archive_run             交付工程组装 + 格式校验
  - scan_all_ops            --all 扫描与 config example_path 对齐 (模板统一)
  - 钉版本                  PINNED_COMMIT resolve/checkout/verify 显式报警
  - 隔离基线                基线算子按 git ls-files 自动发现 + examples/ 树基线签名

用法:
  pytest runner/test_eval_pipeline.py -v
"""

import os
import subprocess
import sys

import pytest
import requests as real_requests

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNER_DIR)

import archive_run  # noqa: E402
import isolation_check  # noqa: E402
import report  # noqa: E402
import run_eval  # noqa: E402
import setup_cann_bench  # noqa: E402
from progress import EvalProgress  # noqa: E402


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


def _make_timeout_fake_requests(calls, fail_times, get_resp, success_info):
    """构造一个 /message POST 前 fail_times 次抛 Timeout 的 FakeRequests。

    被 TestServeTimeoutRetry / TestServeSseThreadCleanup 共用 (消除重复 FakeRequests 类)。
    """

    class FakeRequests:
        Timeout = real_requests.Timeout
        HTTPError = real_requests.HTTPError
        ConnectionError = real_requests.ConnectionError

        @staticmethod
        def post(url, json=None, timeout=None, **kw):
            if url.endswith("/session"):
                return _FakeResp({"id": f"ses_{calls['message']}"})
            if url.endswith("/message"):
                calls["message"] += 1
                if calls["message"] <= fail_times:
                    raise real_requests.Timeout("boom")
                return _FakeResp(
                    {
                        "parts": [{"type": "text", "text": "ok"}],
                        "info": success_info,
                    }
                )
            return _FakeResp({})  # abort / delete 等

        @staticmethod
        def get(url, **kw):
            return get_resp

    return FakeRequests


# ══════════════════════════════════════════════════════════════════════
#  显式工作流选择与安装失败隔离
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def workflow_sandbox(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    plugin = repo / "plugins-official" / "fixture-workflow"
    plugin.mkdir(parents=True)
    init = plugin / "init.sh"
    init.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$3/.$2/agents"\n'
        'printf "%s" "$2" > "$3/selected-tool.txt"\n'
        'if [ "$2" = claude ]; then config=CLAUDE.md; else config=AGENTS.md; fi\n'
        'printf "# Fixture workflow\\n" > "$3/$config"\n'
    )
    task = tmp_path / "task"
    task.mkdir()
    (task / "desc.md").write_text("fixture operator")
    example = tmp_path / "example"
    example.mkdir()
    (example / "CMakeLists.txt").write_text("fixture example")
    monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(repo))
    monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(run_eval, "OPS_CACHE_DIR", str(tmp_path / "cache"))
    return init, task, example


class TestWorkflowInstallation:
    @staticmethod
    def test_cli_requires_explicit_workflow():
        with pytest.raises(SystemExit) as exc:
            run_eval._build_arg_parser().parse_args(["--all"])
        assert exc.value.code == 2

    @staticmethod
    def test_missing_workflow_stops_before_setup(monkeypatch, workflow_sandbox):
        monkeypatch.setattr(
            sys, "argv", ["run_eval.py", "--all", "--workflow", "missing"]
        )
        monkeypatch.setattr(
            run_eval,
            "prompt_for_model",
            lambda **kw: pytest.fail("工作流缺失时不应选择模型"),
        )
        monkeypatch.setattr(
            run_eval,
            "ensure_cann_bench",
            lambda **kw: pytest.fail("工作流缺失时不应下载评测仓"),
        )
        # 本地无 + 应用仓也拉取失败 → _workflow_init 抛 ValueError → parser.error
        captured = {}

        def fake_fetch(workflow, commit=None):
            captured["commit"] = commit
            raise ValueError("missing")

        monkeypatch.setattr(run_eval, "_fetch_plugin_init", fake_fetch)
        with pytest.raises(SystemExit) as exc:
            run_eval.main()
        assert exc.value.code == 2
        # 未传 --cannbot-app-commit 时, commit 应为 None, 而非被静默忽略
        assert captured["commit"] is None

    @staticmethod
    def test_missing_workflow_raises_when_fetch_fails(monkeypatch, workflow_sandbox):
        """本地无、应用仓也拉不到 → prepare_isolated_work_dir 抛 ValueError。"""
        _, task, example = workflow_sandbox
        captured = {}

        def fake_fetch(workflow, commit=None):
            captured["commit"] = commit
            raise ValueError("missing")

        monkeypatch.setattr(run_eval, "_fetch_plugin_init", fake_fetch)
        with pytest.raises(ValueError, match="missing"):
            run_eval.prepare_isolated_work_dir(
                "fixture", str(task), str(example), "run", workflow="missing"
            )
        assert captured["commit"] is None
        assert not os.path.exists(run_eval.RESULTS_DIR)

    @staticmethod
    def test_app_commit_forwarded_to_workflow_init(monkeypatch, workflow_sandbox):
        """prepare_isolated_work_dir 应把 app_commit 原样转发给 _workflow_init 的 commit。"""
        _, task, example = workflow_sandbox
        captured = {}

        def fake_workflow_init(workflow, commit=None):
            captured["commit"] = commit
            raise ValueError("stop")

        monkeypatch.setattr(run_eval, "_workflow_init", fake_workflow_init)
        with pytest.raises(ValueError, match="stop"):
            run_eval.prepare_isolated_work_dir(
                "fixture",
                str(task),
                str(example),
                "run",
                workflow="fixture-workflow",
                app_commit="deadbeef",
            )
        assert captured["commit"] == "deadbeef"

    @staticmethod
    def test_install_failure_stops_before_copying_operator(workflow_sandbox):
        init, task, example = workflow_sandbox
        init.write_text('#!/bin/sh\necho "fixture installation failed" >&2\nexit 7\n')
        with pytest.raises(RuntimeError, match="exit 7.*fixture installation failed"):
            run_eval.prepare_isolated_work_dir(
                "fixture", str(task), str(example), "run", workflow="fixture-workflow"
            )
        work = os.path.join(run_eval.RESULTS_DIR, ".workdir", "run", "fixture")
        assert not os.path.exists(os.path.join(work, "operators"))
        assert not os.path.exists(os.path.join(work, "example"))

    @staticmethod
    def test_second_operator_install_failure_closes_progress_and_serve(
        monkeypatch, workflow_sandbox
    ):
        from unittest.mock import Mock

        init, task, example = workflow_sandbox
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_eval.py",
                "--all",
                "--workflow",
                "fixture-workflow",
                "--model",
                "fixture/model",
                "--skip-isolation-check",
            ],
        )
        monkeypatch.delenv("OPS_FILTER", raising=False)
        monkeypatch.setattr(
            run_eval, "ensure_cann_bench", lambda **kw: str(task.parent)
        )
        monkeypatch.setattr(
            run_eval,
            "_load_ops",
            lambda *a: [{"op_name": "first"}, {"op_name": "second"}],
        )
        progress, serve, finalize = Mock(), Mock(), Mock()
        monkeypatch.setattr(run_eval, "EvalProgress", lambda total: progress)
        monkeypatch.setattr(run_eval, "_serve_mgr", serve)
        monkeypatch.setattr(run_eval, "_finalize_run", finalize)
        attempts = []

        def attempt(op, ctx):
            attempts.append(op["op_name"])
            if op["op_name"] == "second":
                init.write_text('#!/bin/sh\necho "second install failed" >&2\nexit 7\n')
            run_eval.prepare_isolated_work_dir(
                op["op_name"],
                str(task),
                str(example),
                ctx.run_id,
                workflow=ctx.workflow,
            )
            os.makedirs(
                os.path.join(run_eval.RESULTS_DIR, op["op_name"]), exist_ok=True
            )
            return {
                "op_name": op["op_name"],
                "status": "success",
                "duration_s": 1,
            }, True

        monkeypatch.setattr(run_eval, "_attempt_op", attempt)
        with pytest.raises(RuntimeError, match="second install failed"):
            run_eval.main()
        assert attempts == ["first", "second"]
        progress.finish_op.assert_called_once_with("success", 1)
        progress.close.assert_called_once()
        serve.shutdown.assert_called_once()
        finalize.assert_not_called()

    @staticmethod
    @pytest.mark.parametrize(
        "harness,tool",
        [("opencode", "opencode"), ("claude", "claude"), ("kimi", "claude")],
    )
    def test_selected_workflow_keeps_harness_mapping_and_task_isolation(
        workflow_sandbox, harness, tool
    ):
        _, task, example = workflow_sandbox
        work = run_eval.prepare_isolated_work_dir(
            "fixture",
            str(task),
            str(example),
            "run",
            workflow="fixture-workflow",
            harness=harness,
        )
        assert open(os.path.join(work, "selected-tool.txt")).read() == tool
        assert (
            open(os.path.join(work, "operators", "fixture", "desc.md")).read()
            == "fixture operator"
        )
        assert os.path.isfile(os.path.join(work, "example", "CMakeLists.txt"))
        if harness == "opencode":
            agent = os.path.join(work, ".opencode", "agents", "cannbot.md")
            assert os.path.realpath(agent) == os.path.join(work, "AGENTS.md")


@pytest.mark.parametrize(
    "arguments,op,timeout,status",
    [
        ([], None, None, 2),
        (["fixture-workflow"], "level1/exp", "3600", 0),
        (["fixture-workflow", "level2/softmax", "90"], "level2/softmax", "90", 7),
    ],
)
def test_run_bench_forwards_explicit_workflow(
    monkeypatch, tmp_path, arguments, op, timeout, status
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$BENCH_TEST_ARGS"\n'
        'printf "%s\\n" "$OPS_FILTER" "$OP_TIMEOUT" > "$BENCH_TEST_ENV"\n'
        'exit "$BENCH_TEST_STATUS"\n'
    )
    python.chmod(0o755)
    args_file = tmp_path / "arguments"
    env_file = tmp_path / "environment"
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BENCH_TEST_ARGS", str(args_file))
    monkeypatch.setenv("BENCH_TEST_ENV", str(env_file))
    monkeypatch.setenv("BENCH_TEST_STATUS", str(status))
    result = run_eval.subprocess.run(
        ["bash", os.path.join(RUNNER_DIR, "run_bench.sh"), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == status
    if not arguments:
        assert "工作流目录名" in result.stderr
        assert not args_file.exists()
    else:
        assert args_file.read_text().splitlines() == [
            "-u",
            "runner/run_eval.py",
            "--workflow",
            "fixture-workflow",
            "--all",
            "--skip-isolation-check",
        ]
        assert env_file.read_text().splitlines() == [op, timeout]


# ══════════════════════════════════════════════════════════════════════
#  超时解析: per-op > category > default > env
# ══════════════════════════════════════════════════════════════════════


class TestResolveOpTimeout:
    @staticmethod
    def test_per_op_wins():
        op = {"timeout": 100, "category": "Normalization"}
        cfg = {"category_timeouts": {"Normalization": 200}, "default_timeout": 50}
        assert run_eval.resolve_op_timeout(op, cfg) == 100

    @staticmethod
    def test_category_beats_default():
        op = {"category": "FusedComposite"}
        cfg = {"category_timeouts": {"FusedComposite": 21600}, "default_timeout": 10800}
        assert run_eval.resolve_op_timeout(op, cfg) == 21600

    @staticmethod
    def test_default_when_category_unlisted():
        op = {"category": "Unknown"}
        cfg = {"category_timeouts": {"FusedComposite": 21600}, "default_timeout": 10800}
        assert run_eval.resolve_op_timeout(op, cfg) == 10800

    @staticmethod
    def test_env_fallback_without_config():
        assert run_eval.resolve_op_timeout({}, None) == run_eval.OP_TIMEOUT
        assert run_eval.resolve_op_timeout({"category": "X"}, {}) == run_eval.OP_TIMEOUT

    @staticmethod
    def test_real_configs():
        """随仓配置文件: 每个算子都能解出正超时, mla per-op 覆盖生效。"""
        import yaml

        for name in ("eval_config_mini.yaml",):
            path = os.path.join(RUNNER_DIR, "..", "config", name)
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert run_eval.resolve_allowed_models(cfg, None)
            for op in cfg["ops"]:
                assert run_eval.resolve_op_timeout(op, cfg) > 0
        with open(
            os.path.join(RUNNER_DIR, "..", "config", "eval_config_mini.yaml")
        ) as f:
            mini = yaml.safe_load(f)
        mla = [op for op in mini["ops"] if op["op_name"].endswith("/mla")][0]
        assert run_eval.resolve_op_timeout(mla, mini) == 32400


# ══════════════════════════════════════════════════════════════════════
#  算子短名 (跨 level 防碰撞)
# ══════════════════════════════════════════════════════════════════════


class TestOpShortName:
    @staticmethod
    def test_level_path_gets_prefix():
        assert run_eval.op_short_name("cann-bench/tasks/level2/top_k") == "level2_top_k"
        assert run_eval.op_short_name("cann-bench/tasks/level4/mla") == "level4_mla"

    @staticmethod
    def test_different_levels_do_not_collide():
        a = run_eval.op_short_name("cann-bench/tasks/level2/top_k")
        b = run_eval.op_short_name("cann-bench/tasks/level3/top_k")
        assert a != b
        assert a == "level2_top_k" and b == "level3_top_k"

    @staticmethod
    def test_non_level_path_falls_back_to_basename():
        assert run_eval.op_short_name("foo") == "foo"
        assert run_eval.op_short_name("some/where/bar") == "bar"

    @staticmethod
    def test_case_insensitive_level():
        assert run_eval.op_short_name("x/Level2/foo") == "Level2_foo"


# ══════════════════════════════════════════════════════════════════════
#  build_prompt (不污染 op 字典)
# ══════════════════════════════════════════════════════════════════════


class TestBuildPrompt:
    @staticmethod
    def test_example_path_override(tmp_path):
        """example_path 显式覆盖, 不改写 op["example_path_abs"]。"""
        tpl = tmp_path / "tpl.txt"
        tpl.write_text("name={op_short_name} ex={example_path}")
        op = {
            "op_name": "cann-bench/tasks/level2/foo",
            "op_name_abs": "/abs/foo",
            "example_path_abs": "/original/example",
            "cann_bench_root": "/cb",
        }
        out = run_eval.build_prompt(str(tpl), op, example_path="/iso/example")
        assert "ex=/iso/example" in out
        assert "name=level2_foo" in out
        # op 字典未被污染
        assert op["example_path_abs"] == "/original/example"

    @staticmethod
    def test_example_path_defaults_to_op(tmp_path):
        tpl = tmp_path / "tpl.txt"
        tpl.write_text("ex={example_path}")
        op = {
            "op_name": "foo",
            "op_name_abs": "/abs",
            "example_path_abs": "/original/example",
        }
        out = run_eval.build_prompt(str(tpl), op)
        assert "ex=/original/example" in out


# ══════════════════════════════════════════════════════════════════════
#  resolve_agent (agent 判定优先级)
# ══════════════════════════════════════════════════════════════════════


class TestResolveAgent:
    @staticmethod
    def _make_work_dir(tmp_path, files):
        """构造 work_dir/.opencode/agents 并写入给定文件名列表。"""
        agents_dir = tmp_path / ".opencode" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            (agents_dir / name).write_text("")
        return str(tmp_path)

    @staticmethod
    def test_env_overrides(monkeypatch, tmp_path):
        monkeypatch.setenv("CANNBOT_AGENT", "custom")
        work = TestResolveAgent._make_work_dir(
            tmp_path, ["cannbot.md", "ops-direct-invoke-architect.md"]
        )
        assert run_eval.resolve_agent(work) == "custom"

    @staticmethod
    def test_new_plugin_build(monkeypatch, tmp_path):
        monkeypatch.delenv("CANNBOT_AGENT", raising=False)
        work = TestResolveAgent._make_work_dir(
            tmp_path, ["ops-direct-invoke-architect.md"]
        )
        assert run_eval.resolve_agent(work) == run_eval.NEW_PLUGIN_AGENT

    @staticmethod
    def test_other_workflow_role_agents_build(monkeypatch, tmp_path):
        # 非 ops-direct-invoke 前缀的角色 agent 也应判定为新插件 (build)
        monkeypatch.delenv("CANNBOT_AGENT", raising=False)
        work = TestResolveAgent._make_work_dir(
            tmp_path, ["architect.md", "developer.md"]
        )
        assert run_eval.resolve_agent(work) == run_eval.NEW_PLUGIN_AGENT

    @staticmethod
    def test_old_plugin_cannbot(monkeypatch, tmp_path):
        monkeypatch.delenv("CANNBOT_AGENT", raising=False)
        work = TestResolveAgent._make_work_dir(tmp_path, ["cannbot.md"])
        assert run_eval.resolve_agent(work) == run_eval.DEFAULT_AGENT

    @staticmethod
    def test_old_plugin_with_role_agents_cannbot(monkeypatch, tmp_path):
        # 旧插件: cannbot.md 与角色 agent 同在, 应优先 cannbot
        monkeypatch.delenv("CANNBOT_AGENT", raising=False)
        work = TestResolveAgent._make_work_dir(
            tmp_path, ["cannbot.md", "architect.md", "developer.md"]
        )
        assert run_eval.resolve_agent(work) == run_eval.DEFAULT_AGENT

    @staticmethod
    def test_no_agents_default(monkeypatch, tmp_path):
        monkeypatch.delenv("CANNBOT_AGENT", raising=False)
        work = TestResolveAgent._make_work_dir(tmp_path, [])
        assert run_eval.resolve_agent(work) == run_eval.DEFAULT_AGENT


# ══════════════════════════════════════════════════════════════════════
#  模型白名单
# ══════════════════════════════════════════════════════════════════════


class TestAllowedModels:
    @staticmethod
    def test_cli_overrides_config():
        cfg = {"allowed_models": ["a/1", "b/2"]}
        assert run_eval.resolve_allowed_models(cfg, "c/3, d/4 ,") == ["c/3", "d/4"]

    @staticmethod
    def test_config_list():
        assert run_eval.resolve_allowed_models({"allowed_models": ["a/1"]}, None) == [
            "a/1"
        ]

    @staticmethod
    def test_empty():
        assert run_eval.resolve_allowed_models(None, None) == []
        assert run_eval.resolve_allowed_models({}, "") == []

    @staticmethod
    def test_validate():
        run_eval.validate_model_allowed("a/1", ["a/1"])
        run_eval.validate_model_allowed("x/9", [])  # 空白名单放行
        with pytest.raises(ValueError, match="白名单"):
            run_eval.validate_model_allowed("x/9", ["a/1"])

    @staticmethod
    def test_prompt_rejects_non_whitelisted(monkeypatch):
        """交互问询: 白名单外输入被拒绝并重新问询。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        class FakeCompleted:
            stdout = "a/1\nb/2\n"

        monkeypatch.setattr(run_eval.subprocess, "run", lambda *a, **k: FakeCompleted())
        answers = iter(["evil/model", "a/1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        assert run_eval.prompt_for_model(allowed=["a/1", "b/2"]) == "a/1"

    @staticmethod
    def test_prompt_hides_non_whitelisted_models(monkeypatch, capsys):
        """白名单与 opencode models 无交集时, 不把非白名单模型列为可用。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        class FakeCompleted:
            stdout = "evil/1\nother/2\n"

        monkeypatch.setattr(run_eval.subprocess, "run", lambda *a, **k: FakeCompleted())
        monkeypatch.setattr("builtins.input", lambda prompt="": "a/1")
        assert run_eval.prompt_for_model(allowed=["a/1"]) == "a/1"
        out = capsys.readouterr().out
        assert "evil/1" not in out and "other/2" not in out
        assert "白名单" in out


# ══════════════════════════════════════════════════════════════════════
#  超时纳入重试 (run_via_serve)
# ══════════════════════════════════════════════════════════════════════


class TestServeTimeoutRetry:
    @staticmethod
    def test_timeout_retried_then_success(monkeypatch, tmp_path):
        fake, calls = TestServeTimeoutRetry._fake_requests(fail_times=2)
        monkeypatch.setattr(run_eval, "requests", fake)
        monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)
        result = run_eval.run_via_serve(
            "tasks/level2/x", "p", str(tmp_path), timeout=60
        )
        assert result["status"] == "success"
        assert result["attempts"] == 3
        assert calls["message"] == 3
        assert result["timeout_s"] == 60

    @staticmethod
    def test_all_timeouts_give_timeout_status(monkeypatch, tmp_path):
        fake, calls = TestServeTimeoutRetry._fake_requests(fail_times=99)
        monkeypatch.setattr(run_eval, "requests", fake)
        monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)
        result = run_eval.run_via_serve(
            "tasks/level2/x", "p", str(tmp_path), timeout=60
        )
        assert result["status"] == "timeout"
        assert result["attempts"] == run_eval.SERVE_RETRY
        assert calls["message"] == run_eval.SERVE_RETRY
        assert "超时" in result["stderr"]

    @staticmethod
    def test_timeout_aborts_correct_session(monkeypatch, tmp_path):
        """超时重试: abort 必须命中刚建的真实 session_id (而非 stale/None)。

        回归保护: 重构后 _serve_post_message 在 message POST 前已把 session_id
        写入 result; abort 应取该值, 否则 serve 侧任务不会被取消。
        """
        calls = {"message": 0}
        aborted = []
        info = {"tokens": {"total": 10}, "cost": 0}

        class FakeRequests:
            Timeout = real_requests.Timeout
            HTTPError = real_requests.HTTPError
            ConnectionError = real_requests.ConnectionError

            @staticmethod
            def post(url, timeout=None, **kw):
                if url.endswith("/session"):
                    return _FakeResp({"id": f"ses_{calls['message']}"})
                if url.endswith("/message"):
                    calls["message"] += 1
                    if calls["message"] <= 1:
                        raise real_requests.Timeout("boom")
                    return _FakeResp(
                        {"parts": [{"type": "text", "text": "ok"}], "info": info}
                    )
                if url.endswith("/abort"):
                    aborted.append(url.rsplit("/", 2)[-2])
                return _FakeResp({})

            @staticmethod
            def get(url, **kw):
                return _FakeResp({})

        monkeypatch.setattr(run_eval, "requests", FakeRequests)
        monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)
        result = run_eval.run_via_serve(
            "tasks/level2/x", "p", str(tmp_path), timeout=60
        )
        assert result["status"] == "success"
        # 首次 message 超时 → 对应 ses_0 必须被 abort
        assert "ses_0" in aborted

    @staticmethod
    def _fake_requests(fail_times):
        calls = {"message": 0}
        info = {"tokens": {"total": 10}, "cost": 0}
        return _make_timeout_fake_requests(
            calls, fail_times, _FakeResp({}), info
        ), calls


class TestServeSseThreadCleanup:
    @staticmethod
    def test_no_sse_thread_leak_on_timeout_retry(monkeypatch, tmp_path):
        """超时重试: 旧 session 的 SSE 监控线程必须停止, 不得泄漏。"""
        import itertools
        import threading

        class EndlessResp(_FakeResp):
            @staticmethod
            def iter_lines(decode_unicode=True):
                # 无限事件流: 只有 stop_event 能让监控线程退出
                return itertools.repeat('data: {"type": "ping"}')

        calls = {"message": 0}
        fake = _make_timeout_fake_requests(calls, 1, EndlessResp({}), {})
        monkeypatch.setattr(run_eval, "requests", fake)
        monkeypatch.setattr(run_eval.time, "sleep", lambda *a, **k: None)
        result = run_eval.run_via_serve(
            "tasks/level2/x", "p", str(tmp_path), timeout=60
        )
        assert result["status"] == "success"
        assert result["attempts"] == 2
        alive = [
            t
            for t in threading.enumerate()
            if t.name.startswith("sse-") and t.is_alive()
        ]
        assert alive == []


# ══════════════════════════════════════════════════════════════════════
#  交付完整性 + dist 持久化
# ══════════════════════════════════════════════════════════════════════


class TestDelivery:
    @staticmethod
    def test_delivery_complete_from_workdir(tmp_path):
        example_dist = tmp_path / "example" / "dist"
        example_dist.mkdir(parents=True)
        (example_dist / "cann_bench-1.0.0-x.whl").write_text("bin")
        assert run_eval.delivery_complete(str(tmp_path), {}) is True

    @staticmethod
    def test_delivery_complete_from_task_dir(tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        task = tmp_path / "task"
        (task / "dist").mkdir(parents=True)
        (task / "dist" / "x.whl").write_text("bin")
        assert run_eval.delivery_complete(str(work), {"op_name_abs": str(task)}) is True

    @staticmethod
    def test_delivery_incomplete(tmp_path):
        assert run_eval.delivery_complete(str(tmp_path), {}) is False

    @staticmethod
    def test_delivery_complete_from_persisted(monkeypatch, tmp_path):
        """重跑续作场景 (op_attempt>1): workdir 无 whl 但 operators/ 持久化 dist 有 → 完整。"""
        cache = tmp_path / "operators"
        monkeypatch.setattr(run_eval, "OPS_CACHE_DIR", str(cache))
        dist = cache / "level2_foo" / "dist"
        dist.mkdir(parents=True)
        (dist / "a.whl").write_text("bin")
        work = tmp_path / "work"
        work.mkdir()
        assert (
            run_eval.delivery_complete(
                str(work), {"op_name": "cann-bench/tasks/level2/foo"}, op_attempt=2
            )
            is True
        )

    @staticmethod
    def test_delivery_first_attempt_ignores_persisted(monkeypatch, tmp_path):
        """首次尝试: 即使 operators/ 缓存有 whl, workdir 无本轮产出仍判不完整。

        防止缓存残留 whl 绕过 prompt 硬性门禁。
        """
        cache = tmp_path / "operators"
        monkeypatch.setattr(run_eval, "OPS_CACHE_DIR", str(cache))
        dist = cache / "level2_foo" / "dist"
        dist.mkdir(parents=True)
        (dist / "a.whl").write_text("bin")
        work = tmp_path / "work"
        work.mkdir()
        assert (
            run_eval.delivery_complete(
                str(work), {"op_name": "cann-bench/tasks/level2/foo"}, op_attempt=1
            )
            is False
        )

    @staticmethod
    def test_restore_dist_whl(monkeypatch, tmp_path):
        """restore_op_code 恢复持久化的 dist/*.whl 到 example/dist。"""
        cache = tmp_path / "operators"
        monkeypatch.setattr(run_eval, "OPS_CACHE_DIR", str(cache))
        dist = cache / "level2_foo" / "dist"
        dist.mkdir(parents=True)
        (dist / "a.whl").write_text("bin")
        work = tmp_path / "work"
        (work / "example").mkdir(parents=True)
        run_eval.restore_op_code(str(work), "cann-bench/tasks/level2/foo")
        assert (work / "example" / "dist" / "a.whl").is_file()

    @staticmethod
    def test_persist_dist_whl(monkeypatch, tmp_path):
        cache = tmp_path / "operators"
        monkeypatch.setattr(run_eval, "OPS_CACHE_DIR", str(cache))

        work = tmp_path / "work"
        (work / "example" / "dist").mkdir(parents=True)
        (work / "example" / "dist" / "a.whl").write_text("bin")
        task = tmp_path / "task"
        (task / "dist").mkdir(parents=True)
        (task / "dist" / "b.whl").write_text("bin")

        run_eval.persist_op_code(
            str(work), "cann-bench/tasks/level2/foo", op_name_abs=str(task)
        )
        dist = cache / "level2_foo" / "dist"
        assert (dist / "a.whl").is_file()
        assert (dist / "b.whl").is_file()


# ══════════════════════════════════════════════════════════════════════
#  汇总报告
# ══════════════════════════════════════════════════════════════════════


class TestReport:
    @staticmethod
    def test_collect_stats():
        stats = report.collect_stats(TestReport._results())
        assert stats["total"] == 2
        assert stats["success"] == 1
        assert stats["timeout"] == 1
        assert stats["pass_rate"] == pytest.approx(0.5)
        assert stats["tokens"] == 100
        assert stats["duration_s"] == pytest.approx(10800.0)

    @staticmethod
    def test_generate_md_html(tmp_path):
        paths = report.generate_report(
            TestReport._results(), "test001", "a/1", "ops-direct-invoke", str(tmp_path)
        )
        md = open(paths["md"], encoding="utf-8").read()
        html_doc = open(paths["html"], encoding="utf-8").read()
        assert "通过率" in md and "50.0%" in md
        assert "softmax" in md and "mla" in md
        assert "⏱️" in md  # timeout 图标
        assert "<table>" in html_doc and "st-timeout" in html_doc

    @staticmethod
    def test_delivery_column(tmp_path):
        """交付完整性: 明细列 + 汇总行。"""
        results = [
            {
                "op_name": "x",
                "status": "success",
                "duration_s": 1,
                "tokens": {},
                "cost": 0,
                "model": "a/1",
                "delivery_ok": False,
                "op_attempt": 2,
            }
        ]
        paths = report.generate_report(results, "del001", "a/1", "wf", str(tmp_path))
        md = open(paths["md"], encoding="utf-8").read()
        assert "缺whl" in md
        assert "交付完整 | 0/1" in md
        assert "| x |" in md and "| 2 |" in md  # 尝试列 = op_attempt

    @staticmethod
    def test_dirty_metrics_do_not_break_report(tmp_path):
        """脏指标 (非数值 cost/duration) 不拖垮整份报告。"""
        results = [
            {
                "op_name": "x",
                "status": "success",
                "duration_s": "bad",
                "tokens": None,
                "cost": "not-a-number",
                "model": "a/1",
            }
        ]
        paths = report.generate_report(results, "dirty01", "a/1", "wf", str(tmp_path))
        md = open(paths["md"], encoding="utf-8").read()
        assert "x" in md and "0.0000" in md

    @staticmethod
    def _results():
        return [
            {
                "op_name": "cann-bench/tasks/level2/softmax",
                "status": "success",
                "duration_s": 3600.0,
                "tokens": {"total": 100},
                "cost": 0.5,
                "model": "a/1",
                "model_actual": "a/1",
                "attempts": 1,
            },
            {
                "op_name": "cann-bench/tasks/level4/mla",
                "status": "timeout",
                "duration_s": 7200.0,
                "tokens": {},
                "cost": 0,
                "model": "a/1",
                "attempts": 3,
            },
        ]


# ══════════════════════════════════════════════════════════════════════
#  进度条 (tqdm 未安装时的文本回退)
# ══════════════════════════════════════════════════════════════════════


class TestProgress:
    @staticmethod
    def test_text_fallback(capsys):
        p = EvalProgress(2)
        if p.use_tqdm:
            pytest.skip("环境已安装 tqdm, 文本回退不生效")
        p.start_op(0, "cann-bench/tasks/level2/softmax")
        p.finish_op("success", 60.0)
        p.finish_op("timeout", 120.0)
        p.close()
        out = capsys.readouterr().out
        assert "[1/2]" in out or "1/2" in out
        assert "[####################] 2/2 (100%)" in out
        assert "ETA" in out


# ══════════════════════════════════════════════════════════════════════
#  归档: 交付工程组装 + 格式校验
# ══════════════════════════════════════════════════════════════════════


def _make_template(root):
    """最小 direct_launch_example 模板 (构建文件算子无关)。"""
    for fn in ("CMakeLists.txt", "setup.py", "build.sh"):
        (root / fn).write_text("# template")
    (root / "cann_bench").mkdir()
    (root / "cann_bench" / "__init__.py").write_text("# template init")
    ops = root / "csrc" / "ops"
    (ops / "add").mkdir(parents=True)
    (ops / "CMakeLists.txt").write_text("# auto discovery")
    (ops / "add" / "add_kernel.cpp").write_text("// baseline add")
    (root / "tests" / "add").mkdir(parents=True)
    (root / "tests" / "add" / "test_add.py").write_text("# baseline test")


def _make_persisted_op(ops_root, slug, with_tests=True, with_whl=True, with_init=True):
    op = ops_root / slug
    kernel = op / "csrc" / "ops" / slug / "op_kernel"
    kernel.mkdir(parents=True)
    (kernel / f"{slug}_kernel.cpp").write_text("// kernel\n" * 40)
    if with_init:
        (op / "cann_bench").mkdir(parents=True)
        (op / "cann_bench" / "__init__.py").write_text(f"# init for {slug}")
    if with_tests:
        (op / "tests" / slug).mkdir(parents=True)
        (op / "tests" / slug / f"test_{slug}.py").write_text("# test")
    if with_whl:
        (op / "dist").mkdir(parents=True)
        (op / "dist" / f"cann_bench-1.0.0-{slug}.whl").write_text("bin")
    return op


class TestArchive:
    @staticmethod
    def test_find_persisted_ops(tmp_path):
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo")
        (ops_root / "empty").mkdir(parents=True)  # 无 csrc/ops → 排除
        assert archive_run.find_persisted_ops(str(ops_root)) == ["foo"]
        assert archive_run.find_persisted_ops(str(tmp_path / "nope")) == []

    @staticmethod
    def test_stage_and_validate_ok(tmp_path):
        template = tmp_path / "template"
        template.mkdir()
        _make_template(template)
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo")

        staging = tmp_path / "staging"
        staging.mkdir()
        proj = archive_run.stage_delivery_project(
            "foo", str(ops_root), str(template), str(staging)
        )

        # 模板构建文件保留, 基线算子/测试剔除
        assert os.path.isfile(os.path.join(proj, "CMakeLists.txt"))
        assert not os.path.exists(os.path.join(proj, "csrc", "ops", "add"))
        assert not os.path.exists(os.path.join(proj, "tests", "add"))
        # 算子源码/测试/init/whl 就位
        assert os.path.isfile(
            os.path.join(proj, "csrc", "ops", "foo", "op_kernel", "foo_kernel.cpp")
        )
        assert os.path.isfile(os.path.join(proj, "tests", "foo", "test_foo.py"))
        with open(os.path.join(proj, "cann_bench", "__init__.py")) as f:
            assert f.read() == "# init for foo"
        assert os.path.isfile(os.path.join(proj, "dist", "cann_bench-1.0.0-foo.whl"))

        errors, warnings = archive_run.validate_delivery_project(proj, "foo")
        assert errors == []
        assert warnings == []

    @staticmethod
    def test_validate_missing_optional_warns(tmp_path):
        template = tmp_path / "template"
        template.mkdir()
        _make_template(template)
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo", with_tests=False, with_whl=False)

        staging = tmp_path / "staging"
        staging.mkdir()
        proj = archive_run.stage_delivery_project(
            "foo", str(ops_root), str(template), str(staging)
        )
        errors, warnings = archive_run.validate_delivery_project(proj, "foo")
        assert errors == []
        assert any("tests" in w for w in warnings)
        assert any("whl" in w for w in warnings)

    @staticmethod
    def test_validate_missing_required_errors(tmp_path):
        template = tmp_path / "template"
        template.mkdir()
        _make_template(template)
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo", with_init=False)

        staging = tmp_path / "staging"
        staging.mkdir()
        proj = archive_run.stage_delivery_project(
            "foo", str(ops_root), str(template), str(staging)
        )
        # __init__.py 未持久化 → 保留模板件, 不缺; 删掉模拟缺失
        os.remove(os.path.join(proj, "cann_bench", "__init__.py"))
        errors, _ = archive_run.validate_delivery_project(proj, "foo")
        assert any("__init__.py" in e for e in errors)

    @staticmethod
    def test_stage_template_without_csrc_ops(tmp_path):
        """模板缺 csrc/ops 目录: 组装不抛异常, 算子源码仍就位。"""
        template = tmp_path / "template"
        template.mkdir()
        for fn in ("CMakeLists.txt", "setup.py", "build.sh"):
            (template / fn).write_text("# template")
        (template / "cann_bench").mkdir()
        (template / "cann_bench" / "__init__.py").write_text("# init")
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo")

        staging = tmp_path / "staging"
        staging.mkdir()
        proj = archive_run.stage_delivery_project(
            "foo", str(ops_root), str(template), str(staging)
        )
        assert os.path.isfile(
            os.path.join(proj, "csrc", "ops", "foo", "op_kernel", "foo_kernel.cpp")
        )
        errors, _ = archive_run.validate_delivery_project(proj, "foo")
        assert errors == []

    @staticmethod
    def test_validate_baseline_api_residue_warns(tmp_path):
        """init 未持久化而保留模板注册入口: 残留 add/sqrt API 给 warning。"""
        template = tmp_path / "template"
        template.mkdir()
        _make_template(template)
        (template / "cann_bench" / "__init__.py").write_text(
            "def add(x, y):\n    return torch.ops.cann_bench.add(x, y)\n"
        )
        ops_root = tmp_path / "operators"
        _make_persisted_op(ops_root, "foo", with_init=False)

        staging = tmp_path / "staging"
        staging.mkdir()
        proj = archive_run.stage_delivery_project(
            "foo", str(ops_root), str(template), str(staging)
        )
        errors, warnings = archive_run.validate_delivery_project(proj, "foo")
        assert errors == []
        assert any("基线" in w for w in warnings)


class TestConfirmArchive:
    """_confirm_archive 非交互/dry-run/yes 行为 (回归 PR#755 review ⑤)。"""

    @staticmethod
    def test_dry_run_skips_confirm(monkeypatch):
        """--dry-run 单独即可跳过确认, 不需要 --yes (CI 单纯 dry-run)。"""
        called = {"input": False}
        monkeypatch.setattr(
            "builtins.input", lambda *a, **k: called.__setitem__("input", True)
        )
        assert archive_run._confirm_archive(False, True, False) is True
        assert called["input"] is False

    @staticmethod
    def test_yes_skips_confirm(monkeypatch):
        """--yes 跳过确认 (与 --force 语义分离)。"""
        monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
        assert archive_run._confirm_archive(False, False, True) is True

    @staticmethod
    def test_force_skips_confirm(monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
        assert archive_run._confirm_archive(True, False, False) is True

    @staticmethod
    def test_non_tty_without_flag_refuses(monkeypatch):
        """非交互环境且未显式跳过: 拒绝而非 EOFError 崩溃。"""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        crashed = {"v": False}

        def boom(*a, **k):
            crashed["v"] = True
            raise EOFError

        monkeypatch.setattr("builtins.input", boom)
        assert archive_run._confirm_archive(False, False, False) is False
        assert crashed["v"] is False

    @staticmethod
    def test_tty_decline(monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
        assert archive_run._confirm_archive(False, False, False) is False


class TestArchiveSubmoduleCleanup:
    """archive_submodule finally 清理逻辑 (回归 PR#755 review ④)。

    eval_delivery/ 受 original_branch 跟踪时, 不能 rmtree (会误删受跟踪文件);
    仅 original_branch 未跟踪时才 rmtree。
    """

    @staticmethod
    def _make_staging(tmp_path, slugs=("foo",)):
        staging = tmp_path / "staging"
        delivery = staging / archive_run.ARCHIVE_ROOT
        delivery.mkdir(parents=True)
        for s in slugs:
            (delivery / s).mkdir()
            (delivery / s / "CMakeLists.txt").write_text("#")
        return str(staging)

    @staticmethod
    def test_rmtree_skipped_when_tracked_on_original(monkeypatch, tmp_path):
        """eval_delivery/ 受 original_branch 跟踪: 走 git restore, 不 rmtree。"""
        cann_bench = tmp_path / "cann_bench"
        cann_bench.mkdir()
        rmtree_calls = []
        run_cmds = []

        def fake_run(cmd, cwd=None, check=True):
            run_cmds.append(cmd)

            class FakeProc:
                pass

            # git ls-files 非空 → eval_delivery/ 受跟踪
            r = FakeProc()
            r.stdout = "eval_delivery/foo/CMakeLists.txt"
            r.returncode = 0
            return r

        def fake_rmtree(path, *a, **k):
            rmtree_calls.append(str(path))

        monkeypatch.setattr(archive_run, "_run", fake_run)
        monkeypatch.setattr(archive_run.shutil, "rmtree", fake_rmtree)
        monkeypatch.setattr(archive_run, "get_current_branch", lambda root: "master")

        archive_run.archive_submodule(
            str(cann_bench),
            "eval-run-001",
            "origin",
            TestArchiveSubmoduleCleanup._make_staging(tmp_path),
            ["foo"],
        )
        # finally 走 git restore 还原工作区
        assert any(c[:2] == ["git", "restore"] for c in run_cmds)
        # 不应 rmtree 受跟踪的 archive_abs
        archive_abs = str(cann_bench / archive_run.ARCHIVE_ROOT)
        assert not any(p == archive_abs for p in rmtree_calls), (
            "eval_delivery/ 受跟踪时不应 rmtree (会误删受跟踪文件)"
        )

    @staticmethod
    def test_rmtree_when_untracked(monkeypatch, tmp_path):
        """eval_delivery/ 未被 original_branch 跟踪: 安全 rmtree。"""
        cann_bench = tmp_path / "cann_bench"
        cann_bench.mkdir()
        rmtree_calls = []

        def fake_run(cmd, cwd=None, check=True):
            class FakeProc:
                pass

            # git ls-files 空 → 未跟踪
            r = FakeProc()
            r.stdout = ""
            r.returncode = 0
            return r

        def fake_rmtree(path, *a, **k):
            rmtree_calls.append(str(path))

        monkeypatch.setattr(archive_run, "_run", fake_run)
        monkeypatch.setattr(archive_run.shutil, "rmtree", fake_rmtree)
        monkeypatch.setattr(archive_run, "get_current_branch", lambda root: "master")

        archive_run.archive_submodule(
            str(cann_bench),
            "eval-run-001",
            "origin",
            TestArchiveSubmoduleCleanup._make_staging(tmp_path),
            ["foo"],
        )
        archive_abs = str(cann_bench / archive_run.ARCHIVE_ROOT)
        assert any(p == archive_abs for p in rmtree_calls), (
            "eval_delivery/ 未跟踪时应 rmtree 本次新建目录"
        )


class TestScanAllOps:
    """--all 扫描与 config example_path 对齐 (修复 --all/config 模板分裂)。"""

    def _make_tasks(self, root, *ops):
        for level, name in ops:
            d = root / "tasks" / level / name
            d.mkdir(parents=True)
            (d / "proto.yaml").write_text("operator: {}\n")

    def test_default_example_is_direct(self):
        """DEFAULT_EXAMPLE 与 config (eval_config_mini.yaml) 统一为 direct。"""
        assert run_eval.DEFAULT_EXAMPLE == "cann-bench/examples/direct_launch_example"

    def test_scan_uses_default_example(self, tmp_path):
        self._make_tasks(tmp_path, ("level1", "exp"))
        ops = run_eval.scan_all_ops(str(tmp_path))
        assert ops[0]["example_path"] == run_eval.DEFAULT_EXAMPLE

    def test_scan_merges_config_example(self, tmp_path):
        """config 显式配置过的算子沿用其 example_path, 未配置回退默认。"""
        self._make_tasks(tmp_path, ("level1", "exp"), ("level2", "rms_norm"))
        m = {
            "cann-bench/tasks/level2/rms_norm": "cann-bench/examples/aclnn_launch_example"
        }
        ops = run_eval.scan_all_ops(str(tmp_path), m)
        by_name = {op["op_name"]: op["example_path"] for op in ops}
        assert (
            by_name["cann-bench/tasks/level2/rms_norm"]
            == "cann-bench/examples/aclnn_launch_example"
        )
        assert by_name["cann-bench/tasks/level1/exp"] == run_eval.DEFAULT_EXAMPLE

    def test_scan_skips_non_dir(self, tmp_path):
        self._make_tasks(tmp_path, ("level1", "exp"))
        (tmp_path / "tasks" / "level1" / "not_a_dir").write_text("x")
        ops = run_eval.scan_all_ops(str(tmp_path))
        assert len(ops) == 1


class TestPinnedCommit:
    """钉版本: 上游漂移显式报警 (修复 cann-bench 不钉版本的静默漂移)。"""

    def test_resolve_default(self, monkeypatch):
        monkeypatch.delenv("CANN_BENCH_COMMIT", raising=False)
        assert (
            setup_cann_bench.resolve_pinned_commit() == setup_cann_bench.PINNED_COMMIT
        )

    def test_resolve_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("CANN_BENCH_COMMIT", "env_sha")
        assert setup_cann_bench.resolve_pinned_commit("arg_sha") == "arg_sha"
        assert setup_cann_bench.resolve_pinned_commit() == "env_sha"

    def test_resolve_none_disables(self, monkeypatch):
        monkeypatch.delenv("CANN_BENCH_COMMIT", raising=False)
        assert setup_cann_bench.resolve_pinned_commit("none") == ""
        monkeypatch.setenv("CANN_BENCH_COMMIT", "off")
        assert setup_cann_bench.resolve_pinned_commit() == ""

    def test_verify_passes_on_match(self, monkeypatch):
        monkeypatch.setattr(
            setup_cann_bench, "current_head", lambda: setup_cann_bench.PINNED_COMMIT
        )
        setup_cann_bench.verify_commit(setup_cann_bench.PINNED_COMMIT)

    def test_verify_alarms_on_drift(self, monkeypatch):
        """HEAD != pin → fail-fast 显式报警 (上游漂移场景)。"""
        monkeypatch.setattr(setup_cann_bench, "current_head", lambda: "drifted_sha")
        with pytest.raises(RuntimeError, match="不一致"):
            setup_cann_bench.verify_commit(setup_cann_bench.PINNED_COMMIT)

    def test_verify_skips_when_disabled(self, monkeypatch):
        monkeypatch.setattr(setup_cann_bench, "current_head", lambda: "whatever")
        setup_cann_bench.verify_commit("")

    def test_checkout_commit_fetch_fallback(self, monkeypatch):
        """本地缺对象: 浅 fetch 失败 → 全量 fetch; 都拿不到 → 报警。"""
        calls = []

        class FakeProc:
            def __init__(self, rc=0):
                self.returncode = rc
                self.stdout = ""

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["git", "rev-parse"]:
                return FakeProc(1)  # 本地永远缺对象
            if cmd[:2] == ["git", "fetch"] and "--depth" in cmd:
                return FakeProc(1)  # 浅 fetch 失败
            return FakeProc(0)

        monkeypatch.setattr(setup_cann_bench, "_run", fake_run)
        with pytest.raises(RuntimeError, match="无法获取 pin commit"):
            setup_cann_bench.checkout_commit("deadbeef" * 5)
        assert any(c[:2] == ["git", "fetch"] and "--depth" in c for c in calls)
        assert any(
            c[:3] == ["git", "fetch", "origin"] and "--depth" not in c for c in calls
        )


class TestBaselineDiscovery:
    """基线算子目录按当前模板自动发现 (git ls-files), 上游结构变化无需手工同步。"""

    def test_fallback_matches_current_upstream(self):
        """兜底基线与 pin 版上游模板真实结构一致 (mish 已被上游移除)。"""
        assert isolation_check.BASELINE_OPS_FALLBACK["aclnn_launch_example"] == {
            "add",
            "sqrt",
            "_common",
        }
        assert isolation_check.BASELINE_OPS_FALLBACK["direct_launch_example"] == {
            "add",
            "sqrt",
        }

    def test_discover_from_git(self, monkeypatch):
        class FakeProc:
            returncode = 0
            stdout = (
                "examples/aclnn_launch_example/csrc/ops/add/x.cpp\n"
                "examples/aclnn_launch_example/csrc/ops/sqrt/x.cpp\n"
                "examples/aclnn_launch_example/csrc/ops/_common/h.h\n"
                "examples/aclnn_launch_example/csrc/ops/CMakeLists.txt\n"
            )

        monkeypatch.setattr(
            isolation_check.subprocess, "run", lambda *a, **k: FakeProc()
        )
        assert isolation_check._discover_baseline_ops("/x", "aclnn_launch_example") == {
            "add",
            "sqrt",
            "_common",
        }

    def test_discover_fallback_on_git_failure(self, monkeypatch):
        class FakeProc:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(
            isolation_check.subprocess, "run", lambda *a, **k: FakeProc()
        )
        assert isolation_check._discover_baseline_ops("/x", "aclnn_launch_example") == {
            "add",
            "sqrt",
            "_common",
        }


class TestIsolationCheck:
    """隔离检查: 钉版本校验 + examples/ 树基线签名 (agent 改写参考工程场景)。"""

    def _fake_examples(self, root):
        ops = root / "examples" / "aclnn_launch_example" / "csrc" / "ops"
        for d in ("add", "sqrt", "_common"):
            (ops / d).mkdir(parents=True)
        return root / "examples"

    def _patch_git(self, monkeypatch, *, head="pin_sha", porcelain="", ls_files=""):
        class FakeProc:
            def __init__(self, rc, out):
                self.returncode = rc
                self.stdout = out

        def fake_run(cmd, **kw):
            if cmd[:2] == ["git", "rev-parse"]:
                return FakeProc(0, head + "\n")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return FakeProc(0, porcelain)
            if cmd[:2] == ["git", "ls-files"]:
                return FakeProc(0, ls_files)
            return FakeProc(0, "")

        monkeypatch.setattr(isolation_check.subprocess, "run", fake_run)

    def test_head_drift_fails(self, monkeypatch, tmp_path, capsys):
        """HEAD != pin (上游漂移) → 显式报警失败。"""
        self._fake_examples(tmp_path)
        self._patch_git(monkeypatch, head="drifted")
        monkeypatch.setenv("CANN_BENCH_COMMIT", "pin_sha")
        assert isolation_check.verify_isolation(str(tmp_path)) is False
        assert "不一致" in capsys.readouterr().out

    def test_examples_tree_modified_fails(self, monkeypatch, tmp_path):
        """参考工程被改写 (agent 直接写 examples/) → 基线签名拦截。"""
        self._fake_examples(tmp_path)
        self._patch_git(
            monkeypatch,
            head="pin_sha",
            porcelain=" M examples/aclnn_launch_example/cann_bench/__init__.py\n",
        )
        monkeypatch.setenv("CANN_BENCH_COMMIT", "pin_sha")
        assert isolation_check.verify_isolation(str(tmp_path)) is False

    def test_extra_op_dir_fails(self, monkeypatch, tmp_path):
        """csrc/ops 下多出非基线目录 (agent 把算子写进参考工程) → 拦截。"""
        ex = self._fake_examples(tmp_path)
        (ex / "aclnn_launch_example" / "csrc" / "ops" / "exp").mkdir()
        self._patch_git(
            monkeypatch,
            head="pin_sha",
            ls_files="examples/aclnn_launch_example/csrc/ops/add/x.cpp\n",
        )
        monkeypatch.setenv("CANN_BENCH_COMMIT", "pin_sha")
        assert isolation_check.verify_isolation(str(tmp_path)) is False

    def test_clean_tree_passes(self, monkeypatch, tmp_path):
        self._fake_examples(tmp_path)
        self._patch_git(monkeypatch, head="pin_sha")
        monkeypatch.setenv("CANN_BENCH_COMMIT", "pin_sha")
        assert isolation_check.verify_isolation(str(tmp_path)) is True


class TestPluginFetch:
    """插件按需拉取（本地缺失 → cannbot 应用仓 sparse-checkout）的单元测试。

    git 网络操作经 subprocess.run 打桩，不真连 cannbot 应用仓；缓存目录 monkeypatch
    到 tmp_path，隔离真实 results/.plugins。
    """

    @staticmethod
    def _patch_plugin_dirs(monkeypatch, tmp_path):
        # 大部分用例走"跟随 master"分支, 通过环境变量显式关闭内置钉版, 避免进入
        # 真实 git fetch/checkout 网络路径; 钉版路径由专门用例显式传 commit 覆盖
        monkeypatch.setenv("CANNBOT_APP_COMMIT", "none")
        monkeypatch.setattr(run_eval, "_PLUGIN_CACHE_DIR", str(tmp_path / ".plugins"))

    @staticmethod
    def _fake_git_run(tmp_path, workflow):
        """打桩 subprocess.run：clone 时把插件目录/init.sh 落到 cache_root/repo 下。"""
        real_run = run_eval.subprocess.run

        def fake_run(cmd, *args, **kwargs):
            # clone 命令：把目标插件目录落盘，模拟 sparse-checkout 结果
            if len(cmd) >= 5 and cmd[1:4] == [
                "-C",
                str(tmp_path / ".plugins" / workflow),
                "clone",
            ]:
                repo = tmp_path / ".plugins" / workflow / "repo"
                plugin = repo / "plugins-official" / workflow
                plugin.mkdir(parents=True)
                (plugin / "init.sh").write_text("#!/bin/sh\necho fake init\n")
                from unittest.mock import MagicMock

                return MagicMock(returncode=0, stdout="", stderr="")
            # sparse-checkout set 命令：直接放行
            if any("sparse-checkout" in str(c) for c in cmd):
                from unittest.mock import MagicMock

                return MagicMock(returncode=0, stdout="", stderr="")
            return real_run(cmd, *args, **kwargs)

        return fake_run

    def test_local_plugin_hit_returns_local_path(self, monkeypatch, tmp_path):
        local = tmp_path / "local"
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(local))
        plugin = local / "plugins-official" / "ops-direct-invoke"
        plugin.mkdir(parents=True)
        (plugin / "init.sh").write_text("#!/bin/sh\n")
        assert run_eval._workflow_init("ops-direct-invoke") == str(plugin / "init.sh")

    def test_missing_local_fetches_from_app_repo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            self._fake_git_run(tmp_path, "ops-direct-invoke"),
        )
        init = run_eval._workflow_init("ops-direct-invoke")
        assert init == str(
            tmp_path
            / ".plugins"
            / "ops-direct-invoke"
            / "repo"
            / "plugins-official"
            / "ops-direct-invoke"
            / "init.sh"
        )

    def test_fetch_caches_on_second_call(self, monkeypatch, tmp_path):
        """钉版模式首次拉取后, 第二次命中缓存, 不再 clone（不触发 git）。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            self._fake_git_run(tmp_path, "ops-direct-invoke"),
        )
        pin = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: pin)
        monkeypatch.setattr(
            run_eval, "_checkout_plugin_commit", lambda repo_dir, commit, branch: pin
        )

        first = run_eval._workflow_init("ops-direct-invoke", commit=pin)
        # 第二次：清掉打桩，若还走 clone 会因 real_run 真连网而失败
        called = {"n": 0}

        def counting_run(cmd, *args, **kwargs):
            called["n"] += 1
            raise AssertionError("缓存命中不应再触发 git 拉取")

        monkeypatch.setattr(run_eval.subprocess, "run", counting_run)
        second = run_eval._workflow_init("ops-direct-invoke", commit=pin)
        assert first == second
        assert called["n"] == 0

    def test_follow_master_bypasses_stale_cache(self, monkeypatch, tmp_path):
        """跟随 master (none) 不复用旧缓存, 旧缓存存在也应重新 clone 最新。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)  # setenv none → 跟随 master

        cache_root = tmp_path / ".plugins" / "ops-direct-invoke"
        init = (
            cache_root / "repo" / "plugins-official" / "ops-direct-invoke" / "init.sh"
        )
        init.parent.mkdir(parents=True)
        init.write_text("#!/bin/sh\nstale\n")
        (cache_root / "commit").write_text("stale_sha\n")

        cloned = []
        real_fake = self._fake_git_run(tmp_path, "ops-direct-invoke")

        def counting_run(cmd, *args, **kwargs):
            if len(cmd) >= 5 and cmd[1:4] == ["-C", str(cache_root), "clone"]:
                cloned.append(True)
            return real_fake(cmd, *args, **kwargs)

        monkeypatch.setattr(run_eval.subprocess, "run", counting_run)
        monkeypatch.setattr(
            run_eval,
            "_git_head",
            lambda repo_dir: "feedfeedfeedfeedfeedfeedfeedfeedfeedfeed",
        )

        run_eval._fetch_plugin_init("ops-direct-invoke")
        assert cloned, "跟随 master 模式应绕过旧缓存并重新 clone"

    def test_missing_everywhere_raises(self, monkeypatch, tmp_path):
        """本仓没有 + 应用仓也没有 init.sh → 明确报错。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        from unittest.mock import MagicMock

        def fake_run(cmd, *args, **kwargs):
            if len(cmd) >= 5 and cmd[1:4] == [
                "-C",
                str(tmp_path / ".plugins" / "missing-plugin"),
                "clone",
            ]:
                return MagicMock(returncode=0, stdout="", stderr="")
            if any("sparse-checkout" in str(c) for c in cmd):
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)
        with pytest.raises(ValueError, match="不存在或缺少 init.sh"):
            run_eval._workflow_init("missing-plugin")

    @pytest.mark.parametrize("bad", ["..", ".", "-foo", "a/b", "a\\b", ""])
    def test_rejects_unsafe_workflow_names(self, monkeypatch, tmp_path, bad):
        """目录名校验拒绝 . / .. / 路径分隔符 / '-' 开头 / 空串。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        # 这些非法名在本地也一定找不到，从而走到校验；校验本身应在拉取前拦截
        with pytest.raises(ValueError, match="非法工作流名"):
            run_eval._workflow_init(bad)

    def test_git_timeout_raises_value_error(self, monkeypatch, tmp_path):
        """git clone 超时/缺失等 OSError 应统一转 ValueError（main 与降级分支只接 ValueError）。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)

        def timeout_run(cmd, *args, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 120)

        monkeypatch.setattr(run_eval.subprocess, "run", timeout_run)
        with pytest.raises(ValueError, match="拉取插件失败"):
            run_eval._fetch_plugin_init("ops-direct-invoke")

    def test_oserror_from_cache_prep_raises_value_error(self, monkeypatch, tmp_path):
        """缓存目录准备阶段的 OSError（如 .plugins 被同名文件占用）统一转 ValueError。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        # 用同名文件占用 .plugins，使 os.makedirs 抛 FileExistsError(OSError)
        (tmp_path / ".plugins").write_text("blocked")

        with pytest.raises(ValueError, match="准备插件缓存目录失败"):
            run_eval._fetch_plugin_init("ops-direct-invoke")

    def test_resolve_plugin_commit_precedence(self, monkeypatch):
        """插件钉版解析: 显式参数 > CANNBOT_APP_COMMIT 环境变量 > 内置 PIN。"""
        monkeypatch.delenv("CANNBOT_APP_COMMIT", raising=False)
        assert run_eval._resolve_plugin_commit() == run_eval.CANNBOT_APP_PINNED_COMMIT
        monkeypatch.setenv("CANNBOT_APP_COMMIT", "env_sha")
        assert run_eval._resolve_plugin_commit() == "env_sha"
        assert run_eval._resolve_plugin_commit("arg_sha") == "arg_sha"
        assert run_eval._resolve_plugin_commit("none") == ""
        monkeypatch.setenv("CANNBOT_APP_COMMIT", "off")
        assert run_eval._resolve_plugin_commit() == ""

    def test_pin_commit_checkout_and_record_sha(self, monkeypatch, tmp_path):
        """钉版 commit: 拉取后 checkout 到 pin 并把实际 sha 记录到 commit 文件。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            self._fake_git_run(tmp_path, "ops-direct-invoke"),
        )
        pin = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        checked_out = []
        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: pin)
        monkeypatch.setattr(
            run_eval,
            "_checkout_plugin_commit",
            lambda repo_dir, commit, branch: checked_out.append(commit) or pin,
        )

        init = run_eval._fetch_plugin_init("ops-direct-invoke", commit=pin)
        cache_root = tmp_path / ".plugins" / "ops-direct-invoke"
        assert init == str(
            cache_root / "repo" / "plugins-official" / "ops-direct-invoke" / "init.sh"
        )
        assert checked_out == [pin]
        assert (cache_root / "commit").read_text().strip() == pin

    def test_follow_master_records_sha(self, monkeypatch, tmp_path):
        """显式关闭钉版 (none) 跟随 master: 也记录并落盘本次拉取到的实际 sha。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            self._fake_git_run(tmp_path, "ops-direct-invoke"),
        )
        head = "feedfeedfeedfeedfeedfeedfeedfeedfeedfeed"
        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: head)

        init = run_eval._fetch_plugin_init("ops-direct-invoke")
        cache_root = tmp_path / ".plugins" / "ops-direct-invoke"
        assert init == str(
            cache_root / "repo" / "plugins-official" / "ops-direct-invoke" / "init.sh"
        )
        assert (cache_root / "commit").read_text().strip() == head

    def test_resolve_full_commit_abbreviated(self, monkeypatch):
        """缩写/短 sha 应被解析为完整 sha, 供一致性比较使用。"""
        from unittest.mock import MagicMock

        full = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            lambda *a, **kw: MagicMock(returncode=0, stdout=full + "\n", stderr=""),
        )
        assert run_eval._resolve_full_commit("/repo", "deadbeef") == full

    def test_pin_abbreviated_sha_checkout_not_misjudged(self, monkeypatch):
        """缩写 sha 钉版: checkout 后 HEAD(完整sha) 不应与缩写 pin 误判为不一致。"""
        from unittest.mock import MagicMock

        full = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        monkeypatch.setattr(
            run_eval, "_resolve_full_commit", lambda repo_dir, commit: full
        )
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            lambda *a, **kw: MagicMock(returncode=0, stdout="", stderr=""),
        )
        heads = iter(["cafecafecafecafecafecafecafecafecafe", full])
        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: next(heads))

        assert run_eval._checkout_plugin_commit("/repo", "deadbeef", "master") == full

    def test_checkout_plugin_commit_falls_back_to_unshallow_fetch(self, monkeypatch):
        """浅 clone 缺对象且按 SHA 浅 fetch 被拒时, 兜底 unshallow 全量 fetch 分支。"""
        from unittest.mock import MagicMock

        pin = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        resolves = iter(["", "", pin])
        monkeypatch.setattr(
            run_eval, "_resolve_full_commit", lambda repo_dir, commit: next(resolves)
        )

        commands = []

        def fake_run(cmd, *args, **kwargs):
            commands.append(cmd)
            if "fetch" in cmd and "--depth" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="rejected")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(run_eval.subprocess, "run", fake_run)
        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: pin)

        assert run_eval._checkout_plugin_commit("/repo", pin, "master") == pin
        assert any("--unshallow" in c for c in commands), (
            "应按兜底 unshallow fetch 分支"
        )

    def test_cache_hit_mismatched_pin_refetches(self, monkeypatch, tmp_path):
        """钉版 commit 与缓存 sha 不一致时, 清理过期缓存重新拉取。"""
        monkeypatch.setattr(run_eval, "_SKILLS_FORK", str(tmp_path / "empty-local"))
        self._patch_plugin_dirs(monkeypatch, tmp_path)
        monkeypatch.setattr(
            run_eval.subprocess,
            "run",
            self._fake_git_run(tmp_path, "ops-direct-invoke"),
        )
        pin = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        stale = "cafecafecafecafecafecafecafecafecafe"

        cache_root = tmp_path / ".plugins" / "ops-direct-invoke"
        stale_init = (
            cache_root / "repo" / "plugins-official" / "ops-direct-invoke" / "init.sh"
        )
        stale_init.parent.mkdir(parents=True)
        stale_init.write_text("#!/bin/sh\nstale\n")
        (cache_root / "commit").write_text(stale + "\n")

        monkeypatch.setattr(run_eval, "_git_head", lambda repo_dir: pin)
        monkeypatch.setattr(
            run_eval, "_checkout_plugin_commit", lambda repo_dir, commit, branch: pin
        )

        init = run_eval._fetch_plugin_init("ops-direct-invoke", commit=pin)
        assert init == str(
            cache_root / "repo" / "plugins-official" / "ops-direct-invoke" / "init.sh"
        )
        assert (cache_root / "commit").read_text().strip() == pin

    def test_attempt_op_workflow_failure_degrades_to_error(self, monkeypatch, tmp_path):
        """prepare_isolated_work_dir 抛异常时, _attempt_op 降级为单算子 error, 不中断整批。"""
        monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.cann_bench_root = str(tmp_path)
        ctx.run_id = "run"
        ctx.workflow = "missing"
        ctx.harness = "opencode"
        ctx.template_path = "tpl"
        ctx.use_serve = False
        ctx.model = None
        ctx.keep_isolated = False

        op = {
            "op_name": "cann-bench/tasks/level1/swi_glu",
            "op_name_abs": str(tmp_path / "tasks" / "level1" / "swi_glu"),
            "example_path": "cann-bench/examples/direct_launch_example",
            "example_path_abs": str(tmp_path / "examples"),
            "category": "Elementwise",
        }
        os.makedirs(op["op_name_abs"], exist_ok=True)
        os.makedirs(op["example_path_abs"], exist_ok=True)
        monkeypatch.setattr(
            run_eval,
            "prepare_isolated_work_dir",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("workflow failed")),
        )
        monkeypatch.setattr(run_eval, "clean_task_dist_whl", lambda *a, **kw: 0)
        monkeypatch.setattr(run_eval, "resolve_op_timeout", lambda *a, **kw: 10800)

        result, delivery_ok = run_eval._attempt_op(op, ctx)
        assert result["status"] == "error"
        assert result["op_attempt"] == 1
        assert result["stderr"] == "workflow failed"
        assert delivery_ok is False

    def test_save_result_creates_dir_and_writes_yaml(self, tmp_path):
        """save_result 应能自建 output_dir 并落盘 yaml (覆盖目录不存在场景)。"""
        import yaml

        result = {"op_name": "cann-bench/tasks/level1/swi_glu", "status": "error"}
        output_dir = tmp_path / "results" / "cann-bench_tasks_level1_swi_glu"
        run_eval.save_result(result, str(output_dir))
        saved = output_dir / "cann-bench_tasks_level1_swi_glu.yaml"
        assert os.path.isfile(saved)
        with open(saved) as f:
            loaded = yaml.safe_load(f)
        assert loaded["status"] == "error"

    @pytest.mark.parametrize(
        "exc",
        [
            OSError("bash not found"),
            run_eval.subprocess.TimeoutExpired(cmd=["bash"], timeout=120),
        ],
    )
    def test_attempt_op_workflow_oserror_timeout_degrade(
        self, monkeypatch, tmp_path, exc
    ):
        """工作流准备的 OSError/bash 超时也应降级为单算子失败, 不中断整批。"""
        monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path / "results"))
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.cann_bench_root = str(tmp_path)
        ctx.run_id = "run"
        ctx.workflow = "missing"
        ctx.harness = "opencode"
        ctx.template_path = "tpl"
        ctx.use_serve = False
        ctx.model = None
        ctx.keep_isolated = False

        op = {
            "op_name": "cann-bench/tasks/level1/swi_glu",
            "op_name_abs": str(tmp_path / "tasks" / "level1" / "swi_glu"),
            "example_path": "cann-bench/examples/direct_launch_example",
            "example_path_abs": str(tmp_path / "examples"),
            "category": "Elementwise",
        }
        os.makedirs(op["op_name_abs"], exist_ok=True)
        os.makedirs(op["example_path_abs"], exist_ok=True)
        monkeypatch.setattr(
            run_eval,
            "prepare_isolated_work_dir",
            lambda *a, **kw: (_ for _ in ()).throw(exc),
        )
        monkeypatch.setattr(run_eval, "clean_task_dist_whl", lambda *a, **kw: 0)
        monkeypatch.setattr(run_eval, "resolve_op_timeout", lambda *a, **kw: 10800)

        result, delivery_ok = run_eval._attempt_op(op, ctx)
        assert result["status"] == "error"
        assert result["op_attempt"] == 1
        assert delivery_ok is False

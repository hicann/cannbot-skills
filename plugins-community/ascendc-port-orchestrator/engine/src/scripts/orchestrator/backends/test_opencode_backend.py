#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Focused tests for the opencode harness backend.

These tests intentionally cover the parts that differ from Claude Code:
opencode must receive long prompts on stdin, and its host-hook support is a
plugin artifact rather than a .claude/settings.json file.
"""
from __future__ import annotations

import subprocess
import sys
import json
import os
import stat
import textwrap
import types
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # orchestrator/

from backends.opencode_backend import OpencodeBackend  # noqa: E402


def test_opencode_run_cmd_uses_stdin_and_dir_not_positional_prompt() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(
        session_id="ses_1",
        auto=True,
        cwd="/repo",
        extra_args=["--model", "mini/model"],
    )

    assert cmd[:2] == ["/tmp/opencode", "run"]
    assert "--dir" in cmd
    assert cmd[cmd.index("--dir") + 1] == "/repo"
    assert "--session" in cmd and cmd[cmd.index("--session") + 1] == "ses_1"
    assert "--auto" in cmd
    assert "--model" in cmd and "mini/model" in cmd
    assert "Return marker" not in cmd


def test_opencode_run_cmd_can_pin_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_MODEL", "mini/model")
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(cwd="/repo")

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "mini/model"


def test_opencode_run_cmd_does_not_duplicate_explicit_model(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_MODEL", "mini/default")
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(extra_args=["--model", "mini/explicit"])

    assert cmd.count("--model") == 1
    assert "mini/explicit" in cmd
    assert "mini/default" not in cmd


def test_opencode_run_cmd_can_pin_agent_and_variant_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_AGENT", "a5-ops-kernel-worker")
    monkeypatch.setenv("AOG_OPENCODE_VARIANT", "high")
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(cwd="/repo")

    assert "--agent" in cmd
    assert cmd[cmd.index("--agent") + 1] == "a5-ops-kernel-worker"
    assert "--variant" in cmd
    assert cmd[cmd.index("--variant") + 1] == "high"


def test_opencode_run_cmd_does_not_duplicate_explicit_agent_or_variant(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_AGENT", "default-agent")
    monkeypatch.setenv("AOG_OPENCODE_VARIANT", "high")
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(extra_args=["--agent", "explicit-agent", "--variant", "minimal"])

    assert cmd.count("--agent") == 1
    assert cmd.count("--variant") == 1
    assert "explicit-agent" in cmd
    assert "minimal" in cmd
    assert "default-agent" not in cmd
    assert "high" not in cmd


def test_opencode_model_selection_prefers_target_then_kind_then_global(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_MODEL", "global/model")
    monkeypatch.setenv("AOG_OPENCODE_SKILL_MODEL", "skill/model")
    monkeypatch.setenv("AOG_OPENCODE_MODEL_AOG_KERNEL_WORKER", "worker/model")

    assert getattr(OpencodeBackend, '_select_model')("aog-op-classify", "skill") == "skill/model"
    assert getattr(OpencodeBackend, '_select_model')("aog-kernel-worker", "agent") == "worker/model"
    assert getattr(OpencodeBackend, '_select_model')("aog-precision-probe", "agent") == "global/model"


def test_opencode_agent_and_variant_selection_prefers_target_then_kind_then_global(monkeypatch) -> None:
    monkeypatch.setenv("AOG_OPENCODE_AGENT", "global-agent")
    monkeypatch.setenv("AOG_OPENCODE_SKILL_AGENT", "skill-agent")
    monkeypatch.setenv("AOG_OPENCODE_AGENT_AOG_KERNEL_WORKER", "worker-agent")
    monkeypatch.setenv("AOG_OPENCODE_VARIANT", "global-variant")
    monkeypatch.setenv("AOG_OPENCODE_SKILL_VARIANT", "skill-variant")
    monkeypatch.setenv("AOG_OPENCODE_VARIANT_AOG_KERNEL_WORKER", "worker-variant")

    assert getattr(OpencodeBackend, '_select_agent')("aog-op-classify", "skill") == "skill-agent"
    assert getattr(OpencodeBackend, '_select_agent')("aog-kernel-worker", "agent") == "worker-agent"
    assert getattr(OpencodeBackend, '_select_agent')("aog-precision-probe", "agent") == "global-agent"
    assert getattr(OpencodeBackend, '_select_variant')("aog-op-classify", "skill") == "skill-variant"
    assert getattr(OpencodeBackend, '_select_variant')("aog-kernel-worker", "agent") == "worker-variant"
    assert getattr(OpencodeBackend, '_select_variant')("aog-precision-probe", "agent") == "global-variant"


def test_opencode_streaming_cmd_requests_json_events() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    cmd = getattr(backend, '_build_run_cmd')(cwd="/repo", format_json=True)

    assert "--format" in cmd
    assert cmd[cmd.index("--format") + 1] == "json"


def test_opencode_dispatch_feeds_formatted_prompt_to_stdin(tmp_path) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import sys

            args = sys.argv[1:]
            prompt = sys.stdin.read()
            if args[:1] != ["run"]:
                print(f"bad argv: {args!r}", file=sys.stderr)
                raise SystemExit(2)
            if "Return marker" in args:
                print("prompt leaked into argv", file=sys.stderr)
                raise SystemExit(3)
            if "--format" in args:
                print("unexpected json format", file=sys.stderr)
                raise SystemExit(4)
            if "--dir" not in args:
                print("missing dir", file=sys.stderr)
                raise SystemExit(5)
            if not prompt.startswith("You are running as the a5_ops harness backend"):
                print("missing wrapper", file=sys.stderr)
                raise SystemExit(6)
            if "Return marker" not in prompt or "aog-op-classify/SKILL.md" not in prompt:
                print("missing skill context or user prompt", file=sys.stderr)
                raise SystemExit(7)
            print("OPENCODE_OK")
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch(
        "aog-op-classify",
        "Return marker",
        kind="skill",
        cwd=tmp_path,
        timeout=10,
    )

    assert not env.is_error
    assert env.output_text == "OPENCODE_OK\n"


def test_opencode_background_dispatch_accepts_output_file(tmp_path, monkeypatch) -> None:
    writes: list[str] = []

    class FakeStdin:
        @staticmethod
        def write(value: str) -> None:
            writes.append(value)

        @staticmethod
        def close() -> None:
            return None

    process = types.SimpleNamespace(stdin=FakeStdin())
    calls: list[dict] = []

    def fake_popen(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        kwargs["stdout"].close()
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")
    output_file = tmp_path / "logs" / "opencode.log"

    result = backend.dispatch(
        "aog-kernel-worker",
        "Return marker",
        mode="background",
        output_file=output_file,
        cwd=tmp_path,
    )

    assert result is process
    assert calls[0]["stdout"].name == str(output_file)
    assert output_file.exists()
    assert writes and "Return marker" in writes[0]


def test_opencode_skill_json_format_is_opt_in(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys

            args = sys.argv[1:]
            sys.stdin.read()
            if "--format" not in args or args[args.index("--format") + 1] != "json":
                print("missing json format", file=sys.stderr)
                raise SystemExit(2)
            print(json.dumps({
                "type": "text",
                "sessionID": "ses_skill",
                "part": {"type": "text", "text": "OPENCODE_JSON_OK"}
            }), flush=True)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("AOG_OPENCODE_SKILL_FORMAT", "json")
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch("aog-op-classify", "Return marker", kind="skill", cwd=tmp_path, timeout=10)

    assert not env.is_error
    assert env.output_text == "OPENCODE_JSON_OK"
    assert env.session_id == "ses_skill"


def test_opencode_dispatch_marks_hook_agent_context(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return types.SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    backend.dispatch("aog-kernel-worker", "Return marker", kind="agent", cwd="/repo")

    assert calls
    env = calls[0]["env"]
    assert env["AOG_HOOK_AGENT_ID"] == "opencode:aog-kernel-worker"
    assert env["AOG_HOOK_AGENT_TYPE"] == "aog-kernel-worker"


def test_opencode_kernel_worker_prompt_carries_real_ascendc_guard() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    prompt = getattr(backend, '_format_prompt')("aog-kernel-worker", "Return marker", kind="agent")

    assert "opencode a5_ops runtime smoke guard" in prompt
    assert "/root/miniconda3/envs/py311/bin/python3.11" in prompt
    assert "aclrtLaunchKernelWithHostArgs" in prompt
    assert "libacl_dvpp.so" in prompt
    assert "opencode a5_ops kernel-worker compatibility guard" in prompt
    assert "Do not satisfy precision with CPU/PyTorch/NumPy fallback code" in prompt
    assert "extern \"C\" __global__ __aicore__" in prompt
    assert "OPENVINO_HIDDEN" in prompt
    assert "workspace/<op>/kernel/" in prompt
    assert "aclrtlaunch_<kernel>" in prompt
    assert "ACLRT_LAUNCH_KERNEL(<kernel>)" in prompt
    assert "never include nonexistent `torch/pybind.h`" in prompt
    assert "m.def(\"run_<op>\", &run_<op>)" in prompt
    assert "do not use device-side `__gm__` qualifiers in pybind host code" in prompt
    assert "do not create PR4778" in prompt
    assert "Do not call raw `aclrtLaunchKernel(...)` directly" in prompt
    assert "do not use `py::tensor`" in prompt
    assert "ACLRTLauchKernel.h" in prompt
    assert "`py::object`" in prompt
    assert "not a `uint32_t` launch status" in prompt
    assert "TORCH_CHECK(ret == 0" in prompt
    assert "`_<op>_ext` where `<op>` is the workspace directory basename" in prompt
    assert "pybind11/strict_rcward.h" in prompt
    assert "reinterpret_cast<GM_ADDR>(&tiling)" in prompt
    assert "PYBIND11_MODULE` binding only in `pybind11.cpp" in prompt
    assert "not just declare it" in prompt
    assert "`kernel.h` must not declare or define that extern kernel entry" in prompt
    assert "never `KERNEL_OPERATOR_H`" in prompt
    assert "fixed tile byte-size" in prompt
    assert "never `totalElems * sizeof(float)`" in prompt
    assert "complete DataCopy/Add/DataCopy tile loop over chunks" in prompt
    assert "must be ASCII only" in prompt
    assert "`TPipe` only initializes queue buffers" in prompt
    assert "do not call `InitBuffer(xTbuf_, depth, bytes)`" in prompt
    assert "with each `DeQue` stored in a LocalTensor variable exactly once" in prompt
    assert "do not invent `epilogue_len`" in prompt
    assert "yDequeonge" in prompt
    assert "`EnQue` twice on the same LocalTensor" in prompt
    assert "not as file-scope globals" in prompt
    assert "`pipe.Barrier()`" in prompt
    assert "`blockDim = 8`" in prompt
    assert "8-element aligned" in prompt
    assert "non-overlapping" in prompt
    assert "block0 write 0..7" in prompt
    assert "blockSize = (total + blockNum - 1) / blockNum" in prompt
    assert "`tileLenAligned`" in prompt
    assert "`aclrtlaunch_<kernel>` host stubs should use" in prompt
    assert "`void*` tensor `data_ptr` arguments" in prompt
    assert "never write `kernel_module_t`" in prompt
    assert "reinterpret_cast<__gm__ float*>(offset)" in prompt
    assert "deploy_to_npu_lane.sh --lane <LANE> --build" in prompt
    assert "Do not pipe any `deploy_to_npu*.sh` output" in prompt
    assert "Do not scan project-wide `output/` archives" in prompt
    assert "provenance-tracked prior-art/prestage context" in prompt
    assert "never treat it as truth" in prompt
    assert "Do not read, grep, glob, or Bash-inspect any other `workspace/<op>`" in prompt
    assert "the kernel worker owns code generation" in prompt
    assert "`pass_a_runner.py` and `pass_b_runner.py`" in prompt
    assert "If you create `pass_b_runner.py`" in prompt
    assert "`{'cases': [{'inputs': {...}, 'outputs': {...}}, ...]}`" in prompt
    assert "Do not silently replace that oracle" in prompt
    assert "`verification.json` with `precision`, `determinism`, and `performance`" in prompt
    assert "`performance.independent_re_measure`" in prompt
    assert "check_verification_schema.py" in prompt
    assert "cannot produce honest verification artifacts" in prompt
    assert "do not mkdir, touch, read, or write `op_host/` or `op_kernel/`" in prompt


def test_opencode_precision_probe_prompt_carries_runtime_smoke_guard_only() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    prompt = getattr(backend, '_format_prompt')("aog-precision-probe", "Return marker", kind="agent")

    assert "opencode a5_ops runtime smoke guard" in prompt
    assert "/root/miniconda3/envs/py311/bin/python3.11" in prompt
    assert "aclrtLaunchKernelWithHostArgs" in prompt
    assert "libacl_dvpp.so" in prompt
    assert "opencode a5_ops kernel-worker compatibility guard" not in prompt


def test_opencode_optimizer_prompt_carries_runtime_guard_and_a5_exec_rule() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    prompt = getattr(backend, '_format_prompt')("aog-kernel-optimizer", "Return marker", kind="agent")

    assert "opencode a5_ops runtime smoke guard" in prompt
    assert "/root/miniconda3/envs/py311/bin/python3.11" in prompt
    assert "Do not nest `docker exec` inside an `a5_exec.py` command" in prompt
    assert "missing `libtorch_python.so` means" in prompt
    assert "opencode a5_ops kernel-optimizer compatibility guard" in prompt
    assert "Do not hand-roll deployment with tar/scp/manual copies" in prompt
    assert "deploy_to_npu_lane.sh" in prompt
    assert "opencode a5_ops kernel-worker compatibility guard" not in prompt


def test_opencode_skill_prompt_does_not_carry_kernel_worker_guard() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    prompt = getattr(backend, '_format_prompt')("aog-op-classify", "Return marker", kind="skill")

    assert "opencode a5_ops runtime smoke guard" not in prompt
    assert "opencode a5_ops kernel-worker compatibility guard" not in prompt
    assert "aog-op-classify/SKILL.md" in prompt


def test_opencode_prespawn_critic_timeout_default_is_shorter() -> None:
    env = os.environ.copy()
    env["AOG_HARNESS_BACKEND"] = "opencode"
    env.pop("AOG_PRESPAWN_CRITIC_TIMEOUT_SEC", None)
    env.pop("AOG_OPENCODE_PRESPAWN_CRITIC_TIMEOUT_SEC", None)
    env["PYTHONPATH"] = str(_HERE.parents[1])

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import critic_invoke; print(critic_invoke.PRESPAWN_CRITIC_TIMEOUT_SEC)",
        ],
        cwd=_HERE.parents[4],
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout.strip() == "180"


def test_opencode_dispatch_marks_active_workspace_for_hooks(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return types.SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")
    workspace = "/repo/workspace/opencode_e2e_clean_add_a3"

    backend.dispatch(
        "aog-self-critic",
        f"Evaluate current state in workspace {workspace}.",
        kind="skill",
        cwd="/repo",
    )

    env = calls[0]["env"]
    assert env["ASCENDC_WORKSPACE"] == workspace
    assert env["CLAUDE_ACTIVE_WORKSPACE"] == workspace


def test_opencode_dispatch_keeps_explicit_active_workspace(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return types.SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setenv("ASCENDC_WORKSPACE", "/explicit/workspace")
    monkeypatch.setenv("CLAUDE_ACTIVE_WORKSPACE", "/explicit/workspace")
    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    backend.dispatch(
        "aog-kernel-worker",
        "ASCENDC_WORKSPACE: /repo/workspace/opencode_e2e_clean_add_a3",
        kind="agent",
        cwd="/repo",
    )

    env = calls[0]["env"]
    assert env["ASCENDC_WORKSPACE"] == "/explicit/workspace"
    assert env["CLAUDE_ACTIVE_WORKSPACE"] == "/explicit/workspace"


def test_opencode_streaming_extracts_json_text_events(tmp_path) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys

            args = sys.argv[1:]
            prompt = sys.stdin.read()
            if "--format" not in args or args[args.index("--format") + 1] != "json":
                print("missing json format", file=sys.stderr)
                raise SystemExit(2)
            if "Return marker" not in prompt:
                print("missing prompt", file=sys.stderr)
                raise SystemExit(3)
            print(json.dumps({
                "type": "text",
                "sessionID": "ses_stream",
                "part": {"type": "text", "text": "→ orchestrator: done"}
            }), flush=True)
            print(json.dumps({
                "type": "step_finish",
                "sessionID": "ses_stream",
                "part": {
                    "type": "step-finish",
                    "cost": 0.01,
                    "tokens": {"total": 11}
                }
            }), flush=True)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    tee_path = tmp_path / "events.jsonl"
    progress: list[dict] = []
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch(
        "aog-kernel-worker",
        "Return marker",
        kind="agent",
        mode="streaming",
        cwd=tmp_path,
        timeout=5,
        tee_path=tee_path,
        progress_callback=progress.append,
    )

    assert not env.is_error
    assert env.session_id == "ses_stream"
    assert env.output_text == "→ orchestrator: done"
    first_event = json.loads(tee_path.read_text().splitlines()[0])
    assert first_event["part"]["text"] == "→ orchestrator: done"
    assert progress[0]["message"]["content"][0]["type"] == "text"
    assert progress[0]["message"]["content"][0]["text"] == "→ orchestrator: done"


def test_opencode_streaming_timeout_returns_partial_output(tmp_path) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys
            import time

            sys.stdin.read()
            print(json.dumps({
                "type": "text",
                "sessionID": "ses_timeout",
                "part": {"type": "text", "text": "partial"}
            }), flush=True)
            time.sleep(60)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch(
        "aog-kernel-worker",
        "Return marker",
        kind="agent",
        mode="streaming",
        cwd=tmp_path,
        timeout=1,
    )

    assert env.is_error
    assert env.output_text == "partial"
    assert env.raw_envelope["timed_out"] is True
    assert env.raw_envelope["timeout_sec"] == 1


def test_opencode_streaming_silence_timeout_returns_partial_output(tmp_path) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys
            import time

            sys.stdin.read()
            print(json.dumps({
                "type": "text",
                "sessionID": "ses_silence",
                "part": {"type": "text", "text": "partial"}
            }), flush=True)
            time.sleep(60)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch(
        "aog-precision-probe",
        "Return marker",
        kind="agent",
        mode="streaming",
        cwd=tmp_path,
        timeout=30,
        silence_timeout=1,
    )

    assert env.is_error
    assert env.output_text == "partial"
    assert env.raw_envelope["silence_timed_out"] is True
    assert env.raw_envelope["silence_timeout_sec"] == 1


def test_opencode_streaming_invalid_tool_event_fails_fast(tmp_path) -> None:
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys
            import time

            sys.stdin.read()
            print(json.dumps({
                "type": "text",
                "sessionID": "ses_invalid",
                "part": {"type": "text", "text": "before invalid"}
            }), flush=True)
            print(json.dumps({
                "type": "tool_use",
                "sessionID": "ses_invalid",
                "part": {
                    "type": "tool",
                    "tool": "invalid",
                    "state": {
                        "status": "completed",
                        "input": {"tool": "bash", "error": "JSON parsing failed"}
                    }
                }
            }), flush=True)
            time.sleep(60)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    backend = OpencodeBackend(opencode_bin=str(fake))

    env = backend.dispatch(
        "aog-precision-probe",
        "Return marker",
        kind="agent",
        mode="streaming",
        cwd=tmp_path,
        timeout=30,
    )

    assert env.is_error
    assert env.output_text == "before invalid"
    assert env.raw_envelope["invalid_tool_event"] is True
    assert env.raw_envelope["invalid_tool_limit"] == 1
    assert "JSON parsing failed" in env.raw_envelope["stdout_tail"]


def test_opencode_wire_safety_points_to_host_hook_plugin() -> None:
    wiring = OpencodeBackend(opencode_bin="/tmp/opencode").wire_safety([])

    assert wiring["backend"] == "opencode"
    assert wiring["kind"] == "host-hook-plugin"
    assert "tool.execute.before" in wiring["events"]
    assert "tool.execute.after" in wiring["events"]
    assert Path(wiring["plugin_path"]).is_file()

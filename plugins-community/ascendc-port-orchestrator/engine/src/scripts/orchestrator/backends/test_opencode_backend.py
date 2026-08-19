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
import builtins
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # orchestrator/

from backends.opencode_backend import OpencodeBackend  # noqa: E402
from backends.base import Envelope, STREAM_SILENCE_TIMEOUT_SEC, StreamSilenceTimeout  # noqa: E402

# Private seams under test, bound once (see G.CLS.11 note in test_stop_gate_dispatch.py).
_opencode_config_content = getattr(OpencodeBackend, "_opencode_config_content")
_select_agent = getattr(OpencodeBackend, "_select_agent")
_engine_root = getattr(OpencodeBackend, "_engine_root")
_stream_silence_timeout = getattr(OpencodeBackend, "_stream_silence_timeout")
_parse_frontmatter = getattr(OpencodeBackend, "_parse_frontmatter")
_dispatch_foreground = getattr(OpencodeBackend, "_dispatch_foreground")


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


def test_opencode_stream_silence_uses_shared_default_and_local_override(monkeypatch) -> None:
    monkeypatch.delenv("AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC", raising=False)
    assert _stream_silence_timeout(None) == STREAM_SILENCE_TIMEOUT_SEC

    monkeypatch.setenv("AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC", "15")
    assert _stream_silence_timeout(None) == 15

    monkeypatch.setenv("AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC", "invalid")
    assert _stream_silence_timeout(None) == STREAM_SILENCE_TIMEOUT_SEC


def test_opencode_frontmatter_fallback_works_without_pyyaml(monkeypatch) -> None:
    """A clean OpenCode install must not require a transitive PyYAML package."""
    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("simulated clean environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    frontmatter, body = _parse_frontmatter(
        "---\nname: aog-test\ntools:\n  - Read\n---\nagent body\n")
    assert frontmatter == {"name": "aog-test", "tools": ["Read"]}
    assert body == "agent body\n"


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
    assert calls[0]["start_new_session"] is True
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


def test_backend_prompt_carries_no_semantic_policy() -> None:
    """A backend WIRES the harness; it must not own kernel-authoring rules.

    These rules used to be appended to the prompt here, and only for opencode — so an
    opencode worker and a Claude Code worker were held to different rules for the same job,
    with no way to see the divergence from either side. They now live in canonical KB
    (kb/shared/KERNEL_AUTHORING_GUARDS.md) which both harnesses read, per the boundary
    invariant stated in backends/base.py.
    """
    backend = OpencodeBackend()
    for target in ("aog-kernel-worker", "aog-kernel-optimizer", "aog-precision-probe"):
        prompt = getattr(backend, "_format_prompt")(target, "Return marker", kind="agent")
        for leaked in (
            "compatibility guard",
            "runtime smoke guard",
            "pybind11.cpp",
            "op_host",
            "ACLRT_LAUNCH_KERNEL",
        ):
            assert leaked not in prompt, (
                f"backend re-introduced semantic policy for {target}: {leaked!r} — put it in "
                "kb/shared/KERNEL_AUTHORING_GUARDS.md so both harnesses get the same rules"
            )


def test_kernel_authoring_guards_are_canonical_and_reachable() -> None:
    """The relocated rules must survive, and be reachable by BOTH harnesses.

    Deleting backend-owned policy is only correct if the rules did not vanish with it: they
    are loaded through shared/ALWAYS_LOADED_RULES.md, which every kernel-authoring agent is
    required to read regardless of harness.
    """
    kb = Path(__file__).resolve().parents[5] / "kb" / "shared"
    guards = kb / "KERNEL_AUTHORING_GUARDS.md"
    assert guards.is_file(), "canonical kernel-authoring guards are missing"
    body = guards.read_text()
    for rule in ("pybind", "op_host", "ACLRT_LAUNCH_KERNEL", "aog-kernel-optimizer"):
        assert rule in body, f"guard content lost in the move: {rule!r}"
    always = (kb / "ALWAYS_LOADED_RULES.md").read_text()
    assert "KERNEL_AUTHORING_GUARDS" in always, (
        "guards are not referenced from ALWAYS_LOADED_RULES.md, so no agent is told to read them"
    )


def test_kernel_authoring_guards_carry_no_operator_specific_environment() -> None:
    """Canonical KB is authoring rules, not one operator's machine.

    This file was populated by relocating text out of the backend, and that text had grown an
    environment half: an absolute interpreter path from a private container, a pinned CANN
    root, a hand-built LD_LIBRARY_PATH ordering, and a helper script that lives outside this
    repo. None of it is portable, none of it is knowledge about writing kernels, and once the
    file was wired into ALWAYS_LOADED_RULES.md every Claude Code kernel agent was required to
    read it too. Relocation is the moment such content escapes, so the check lives here.
    """
    kb = Path(__file__).resolve().parents[5] / "kb" / "shared"
    body = (kb / "KERNEL_AUTHORING_GUARDS.md").read_text()
    forbidden = {
        "/root/miniconda3": "absolute interpreter path from one operator's container",
        "cann-9.1.T500": "CANN root pinned to one installed version",
        "/usr/local/Ascend/8.5.0": "second pinned CANN root",
        "a5_exec.py": "helper script that is not part of this plugin",
        "~/.claude/skills": "path into a user's private skill install",
    }
    for needle, why in forbidden.items():
        assert needle not in body, (
            f"canonical KB carries operator-specific environment ({why}): {needle!r}. "
            "Environment belongs in workspace/.ascendc_env and the deploy wrapper."
        )


def test_opencode_skill_prompt_does_not_carry_kernel_worker_guard() -> None:
    backend = OpencodeBackend(opencode_bin="/tmp/opencode")

    prompt = getattr(backend, '_format_prompt')("aog-op-classify", "Return marker", kind="skill")

    assert "kernel-worker authoring guard" not in prompt
    assert "kernel-optimizer authoring guard" not in prompt
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


def test_opencode_dispatch_blocks_when_runtime_check_fails(monkeypatch, tmp_path) -> None:
    """A refused runtime self-check blocks dispatch before it can spawn."""
    monkeypatch.delenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", raising=False)
    monkeypatch.setattr(
        "backends.opencode_runtime.ensure_opencode_runtime",
        lambda bin: types.SimpleNamespace(ok=False, reason="safety-net probe failed", warnings=[]),
    )
    spawned = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: spawned.append(cmd))
    backend = OpencodeBackend(opencode_bin=str(tmp_path / "opencode"))
    env = backend.dispatch("aog-op-classify", "Return marker", kind="skill", mode="foreground")
    assert env.is_error
    assert env.raw_envelope["runtime_check_failed"] is True
    assert "safety-net probe failed" in env.raw_envelope["reason"]
    assert spawned == [], "a refused runtime check must not reach a spawn"


def test_opencode_dispatch_passes_through_ok_runtime_check(monkeypatch, tmp_path) -> None:
    """G5: ok check (with a warning) lets dispatch proceed to the fake opencode spawn."""
    monkeypatch.delenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", raising=False)
    monkeypatch.setattr(
        "backends.opencode_runtime.ensure_opencode_runtime",
        lambda bin: types.SimpleNamespace(ok=True, reason="ok", warnings=["probe SKIP"]),
    )

    class _Popen:
        returncode = 0

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def communicate(_input, timeout=None):
            return "OK\n", ""

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    backend = OpencodeBackend(opencode_bin=str(tmp_path / "opencode"))
    env = backend.dispatch("aog-op-classify", "Return marker", kind="skill", mode="foreground", timeout=10)
    assert not env.is_error


def test_opencode_background_runtime_failure_raises_not_envelope(monkeypatch, tmp_path) -> None:
    """Background callers own a Popen lifecycle, so fail-closed must not return an Envelope."""
    monkeypatch.delenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", raising=False)
    monkeypatch.setattr(
        "backends.opencode_runtime.ensure_opencode_runtime",
        lambda bin: types.SimpleNamespace(ok=False, reason="safety-net probe failed", warnings=[]),
    )
    backend = OpencodeBackend(opencode_bin=str(tmp_path / "opencode"))
    with pytest.raises(RuntimeError, match="runtime self-check failed"):
        backend.dispatch(
            "aog-kernel-worker", "Return marker", kind="agent", mode="background",
            output_file=tmp_path / "opencode.log",
        )


def test_opencode_runtime_skip_requires_exact_one(monkeypatch, tmp_path) -> None:
    """`AOG_OPENCODE_SKIP_RUNTIME_CHECK=0` must not silently disable fail-closed."""
    calls = []
    monkeypatch.setenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", "0")
    monkeypatch.setattr(
        "backends.opencode_runtime.ensure_opencode_runtime",
        lambda bin: calls.append(bin) or types.SimpleNamespace(ok=False, reason="refused", warnings=[]),
    )
    env = OpencodeBackend(opencode_bin=str(tmp_path / "opencode")).dispatch(
        "aog-op-classify", "Return marker", kind="skill")
    assert env.is_error and calls == [str(tmp_path / "opencode")]


def test_o17_uses_opencode_auto_compatible_permissions(monkeypatch, tmp_path) -> None:
    """O1.7 keeps Claude's acceptEdits shape, but OpenCode needs --auto."""
    import phase_o17_classify as o17

    calls = []

    def dispatch(*args, **kwargs):
        calls.append(kwargs)
        return Envelope(is_error=False, output_text="{}", raw_envelope={})

    opencode_backend = types.SimpleNamespace(name="opencode", dispatch=dispatch)
    claude_backend = types.SimpleNamespace(name="claude_code", dispatch=dispatch)
    monkeypatch.setattr(o17, "_backend", opencode_backend)
    invoke_claude_skill = getattr(o17, "_invoke_claude_skill")
    ok, _, _ = invoke_claude_skill(tmp_path)
    assert ok
    assert calls[-1]["permission_mode"] == "bypassPermissions"

    monkeypatch.setattr(o17, "_backend", claude_backend)
    ok, _, _ = invoke_claude_skill(tmp_path)
    assert ok
    assert calls[-1]["permission_mode"] == "acceptEdits"


def test_o17_surfaces_opencode_runtime_refusal_reason(monkeypatch, tmp_path) -> None:
    """The fail-closed runtime reason must survive the O1.7 compatibility wrapper."""
    import phase_o17_classify as o17

    class _Backend:
        name = "opencode"

        @staticmethod
        def dispatch(*args, **kwargs):
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={
                    "runtime_check_failed": True,
                    "reason": "safety-net probe failed: door open",
                },
            )

    monkeypatch.setattr(o17, "_backend", _Backend())
    invoke_claude_skill = getattr(o17, "_invoke_claude_skill")
    ok, _, stderr = invoke_claude_skill(tmp_path)
    assert not ok
    assert "safety-net probe failed" in stderr


def test_opencode_foreground_timeout_terminates_process_group(monkeypatch) -> None:
    """Foreground timeout uses the same G6 process-group cleanup as streaming."""
    calls = []

    class _Popen:

        def __init__(self):
            self.pid = 123
            self.returncode = None
            self.communicate_calls = 0

        def communicate(self, _input=None, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(["opencode", "run"], timeout, output="partial", stderr="late")
            self.returncode = -15
            return "partial", "late"

    proc = _Popen()

    def _fake_popen(*_args, **kwargs):
        calls.append(kwargs)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    terminated = []
    monkeypatch.setattr(
        "backends.opencode_runtime.terminate_process_group", lambda p: terminated.append(p))
    env = _dispatch_foreground(OpencodeBackend(opencode_bin="opencode"),
        ["opencode", "run"], "prompt", {}, 1, None, "agent", "foreground")
    assert env.is_error and env.raw_envelope["timed_out"] is True
    assert env.output_text == "partial"
    assert terminated == [proc]
    assert calls[0]["start_new_session"] is True


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


def _dispatch_env(monkeypatch, inherited: str) -> dict:
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return types.SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setenv("ASCENDC_WORKSPACE", inherited)
    monkeypatch.setenv("CLAUDE_ACTIVE_WORKSPACE", inherited)
    monkeypatch.setattr(subprocess, "run", fake_run)
    OpencodeBackend(opencode_bin="/tmp/opencode").dispatch(
        "aog-kernel-worker",
        "ASCENDC_WORKSPACE: /repo/workspace/opencode_e2e_clean_add_a3",
        kind="agent",
        cwd="/repo",
    )
    return calls[0]["env"]


def test_opencode_dispatch_keeps_an_agreeing_inherited_workspace(monkeypatch) -> None:
    """An inherited value that names the same op is kept (and is a no-op either way)."""
    env = _dispatch_env(monkeypatch, "/repo/workspace/opencode_e2e_clean_add_a3")

    assert env["ASCENDC_WORKSPACE"] == "/repo/workspace/opencode_e2e_clean_add_a3"
    assert env["CLAUDE_ACTIVE_WORKSPACE"] == "/repo/workspace/opencode_e2e_clean_add_a3"


@pytest.mark.parametrize(
    "inherited",
    [
        "/repo/workspace/some_other_op",  # a SIBLING — the worst case, see below
        "/repo/workspace",                # the workspace root
        "/repo",                          # an ancestor of it
        "/explicit/workspace",            # outside the tree entirely
    ],
)
def test_opencode_dispatch_replaces_a_disagreeing_inherited_workspace(monkeypatch, inherited) -> None:
    """Any inherited value naming a different place is replaced by the dispatch's own.

    The door scopes its cross-workspace rules to this variable, so a wrong value does not
    weaken the guard evenly — it relocates it. Measured with the real door while dispatching
    to opA:

      ASCENDC_WORKSPACE=.../opA   read opB -> BLOCK    read own opA -> ALLOW
      ASCENDC_WORKSPACE=.../opB   read opB -> ALLOW    read own opA -> BLOCK
      ASCENDC_WORKSPACE=.../      read opB -> BLOCK (root disarms the matcher; armed by the
                                                     door's own fallback)

    The sibling row is the one that matters: the worker gains another operator's
    `verification.json` — precisely what the anti-cheating layer exists to stop — and loses
    its own files. An earlier version of this check accepted any path under the workspace
    root, so the sibling passed and this test asserted it as intended behaviour.
    """
    env = _dispatch_env(monkeypatch, inherited)

    assert env["ASCENDC_WORKSPACE"] == "/repo/workspace/opencode_e2e_clean_add_a3"
    assert env["CLAUDE_ACTIVE_WORKSPACE"] == "/repo/workspace/opencode_e2e_clean_add_a3"


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

    # G1 harness-decoupling: mid-work silence raises the SHARED StreamSilenceTimeout
    # (backends.base) so the FSM's respawn-budget logic is harness-agnostic — the
    # partial output is attached to the exception, not discarded with the
    # retry signal.
    with pytest.raises(StreamSilenceTimeout) as exc_info:
        backend.dispatch(
            "aog-precision-probe",
            "Return marker",
            kind="agent",
            mode="streaming",
            cwd=tmp_path,
            timeout=30,
            silence_timeout=1,
        )

    assert exc_info.value.agent_type == "aog-precision-probe"
    assert exc_info.value.silent_seconds >= 0.9
    assert exc_info.value.last_event_type is None  # _StreamState has no event type field
    assert exc_info.value.partial_output == "partial"
    assert exc_info.value.raw_envelope["silence_timed_out"] is True


def test_opencode_json_skill_silence_returns_envelope_for_graceful_fallback(tmp_path, monkeypatch) -> None:
    """JSON transport is also used by skills, but only FSM agent work may raise retry signals."""
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            r'''
            import json
            import sys
            import time

            sys.stdin.read()
            print(json.dumps({"type": "text", "sessionID": "ses_skill",
                              "part": {"type": "text", "text": "partial"}}), flush=True)
            time.sleep(60)
            '''
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("AOG_OPENCODE_SKILL_FORMAT", "json")
    env = OpencodeBackend(opencode_bin=str(fake)).dispatch(
        "aog-op-classify", "Return marker", kind="skill", cwd=tmp_path,
        timeout=30, silence_timeout=1,
    )
    assert env.is_error
    assert env.output_text == "partial"
    assert env.raw_envelope["silence_timed_out"] is True


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


def test_generated_config_is_byte_stable_across_processes() -> None:
    """Tool grants must not depend on Python's per-process string hash seed.

    opencode collapses edit/write/patch into ONE write-side group and the LAST key for that
    group wins. Building the record from a set therefore made each agent's write capability
    random per process — measured at 4/10 runs keeping `edit` for aog-kernel-worker. A run
    that lost it did not fail loudly: the model reported "I don't have a write tool" and
    fell back to `bash printf > file`, which no generated-code rule inspects.
    """
    import hashlib
    import subprocess as sp

    src = (
        "import sys; sys.path.insert(0, %r);"
        "from backends.opencode_backend import OpencodeBackend as B;"
        # Inside a string: this runs in a CHILD process, where only `B` exists.
        "print(B._opencode_config_content())"
    ) % str(Path(__file__).resolve().parents[1])
    digests = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = sp.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        digests.add(hashlib.sha256(out.stdout.strip().encode()).hexdigest())
    assert len(digests) == 1, f"config varies with PYTHONHASHSEED: {len(digests)} variants"


def test_write_group_semantics_are_respected() -> None:
    """A kernel author keeps write; an analyzer-only agent does not get it."""
    cfg = json.loads(_opencode_config_content())
    agents = cfg["agent"]

    kw = agents["aog-kernel-worker"]["tools"]
    assert kw.get("edit") is True and kw.get("write") is True, (
        "kernel worker lost its write tools; it will fall back to bash redirection, which "
        "bypasses every generated-code guard"
    )
    # `patch` aliases the same group: emitting it as False switches write off entirely.
    assert "patch" not in kw, "patch must not be emitted — it aliases the edit/write group"

    analyzer = agents["aog-determinism-analyzer"]["tools"]
    assert analyzer.get("edit") is False and analyzer.get("write") is False, (
        "analyzer-only agent was widened to write; door.py's write guards only bind "
        "kernel-author agents, so it could author kernel sources unjudged"
    )

    # Grants must be emitted AFTER denials. opencode resolves the edit/write/patch alias
    # group by LAST key, so reversing this order silently strips write from every agent that
    # should have it — and nothing else in the suite notices.
    keys = list(kw)
    granted = [k for k in keys if kw[k] is True]
    denied_keys = [k for k in keys if kw[k] is False]
    assert denied_keys and granted, "expected both denials and grants for a kernel author"
    assert keys.index(granted[0]) > keys.index(denied_keys[-1]), (
        "grants are not emitted last; under opencode's last-key-wins alias resolution the "
        "write group would resolve to denied"
    )

    # Edit-only is NOT expressible: CC grants aog-kernel-optimizer Edit without Write, but
    # the alias group resolves together, so we grant the group explicitly instead of
    # emitting a restriction the harness discards.
    optimizer = agents["aog-kernel-optimizer"]["tools"]
    assert optimizer.get("edit") is True and optimizer.get("write") is True, (
        "Edit-only agents must be granted the whole write group; emitting write=false here "
        "resolves to edit=false too and leaves the agent unable to work"
    )


def test_skills_are_bindable_agents() -> None:
    """Skill dispatch passes --agent; the name must exist or identity self-contradicts."""
    cfg = json.loads(_opencode_config_content())
    for skill in ("aog-knowledge-maintain", "aog-op-classify", "aog-a3-author"):
        assert skill in cfg["agent"], f"{skill} is dispatched with --agent but not registered"
        assert cfg["agent"][skill]["mode"] == "primary"
    assert _select_agent("aog-knowledge-maintain", "skill") == "aog-knowledge-maintain"


def test_kb_root_is_expanded_for_opencode() -> None:
    """${CLAUDE_PLUGIN_ROOT} is prompt text; opencode does not expand it in tool arguments."""
    cfg = _opencode_config_content()
    assert "${CLAUDE_PLUGIN_ROOT}" not in cfg
    bodies = _engine_root() / "workspace" / ".opencode-agents"
    worker = bodies / "aog-kernel-worker.md"
    if worker.is_file():
        assert "${CLAUDE_PLUGIN_ROOT}" not in worker.read_text(), (
            "agent body still addresses the KB through an unexpanded variable; the worker "
            "would try to open a literal path and give up"
        )


def test_inherited_config_cannot_disarm_the_safety_net(monkeypatch) -> None:
    """An inherited OPENCODE_CONFIG_CONTENT must not suppress our injection.

    It is a user-exportable INPUT variable — the same argument launch_orchestrator.sh makes
    when it refuses to use OPENCODE_CONFIG* as a host fingerprint. Treating it as "already
    configured" disarmed everything: no plugin registration, so the adapter never loaded,
    `--agent` did not resolve, and an answer-bearing output/ read that is BLOCKED in an armed
    run SUCCEEDED with exit 0. One exported variable, whole safety net off, failing OPEN.
    """
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "{}")
    env = OpencodeBackend().build_env("aog-kernel-worker", "brief", kind="agent")
    cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    plugins = cfg.get("plugin") or []
    assert any("a5_ops_hooks" in str(p) for p in plugins), (
        "adapter registration was dropped because the environment already carried a config; "
        "sub-agents would run with no door, no identity guard and no output guard"
    )
    assert "aog-kernel-worker" in cfg.get("agent", {})


def test_unregistered_dispatch_target_is_still_bindable() -> None:
    """`--agent <unknown>` only warns and falls back, so identity would self-contradict.

    kb_auto_promote dispatches Claude Code's built-in "general-purpose" and resume()
    dispatches the pseudo-target "resume"; neither is a plugin agent or skill. Without an
    entry the run binds opencode's default agent while the hook env announces the requested
    name, the identity guard then refuses every tool call, and exit 0 makes the orchestrator
    record the refusal as a successful result.
    """
    for target in ("general-purpose", "resume"):
        cfg = json.loads(_opencode_config_content(extra_agent=target))
        assert target in cfg["agent"], f"{target} is dispatched but not bindable"
        assert cfg["agent"][target]["mode"] == "primary"
        assert _select_agent(target, "agent") == target

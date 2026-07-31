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

"""Unit tests for CCBackend pure methods (normalize / identify_cmd / parse_op_from_cmd) +
Envelope required/optional contract. No claude spawn needed.
run: python3 test_cc_backend.py   (exit 0 = all pass)
"""
import logging
import math
import os
import sys
import types
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # orchestrator/ (parent of backends/), for package import
from backends.base import Backend, Envelope
from backends.cc_backend import CCBackend

FAILS = []
LOGGER = logging.getLogger(__name__)


def check(n, c, d=""):
    LOGGER.info("  [%s] %s%s", "PASS" if c else "FAIL", n,
                f" — {d}" if d and not c else "")
    if not c:
        FAILS.append(n)


def test_envelope_required_contract():
    check(
        "required control-flow fields",
        Envelope.CONTROL_FLOW_FIELDS == ("is_error", "output_text", "api_error_status"),
    )
    e = Envelope(is_error=False, output_text="ok")
    check("optional default None (num_turns)", e.num_turns is None)
    check("optional default None (session_id)", e.session_id is None)


def test_normalize_agentresult():
    ar = types.SimpleNamespace(
        is_error=False, output_text="hi", session_id="ses_1", cost_usd=0.42,
        terminal_reason="stop", duration_ms=1234,
        raw_envelope={"api_error_status": None, "num_turns": 3, "permission_denials": [], "stop_reason": "end_turn"})
    e = CCBackend().normalize(ar)
    check("is_error mapped", e.is_error is False)
    check("output_text mapped", e.output_text == "hi")
    check("total_cost_usd from cost_usd", math.isclose(e.total_cost_usd, 0.42))
    check("num_turns from raw_envelope (optional)", e.num_turns == 3)
    check("stop_reason from raw_envelope", e.stop_reason == "end_turn")
    check("terminal_reason mapped", e.terminal_reason == "stop")


def test_normalize_dict_envelope():
    env = {"is_error": True, "result": "boom", "api_error_status": 429, "num_turns": 1}
    e = CCBackend().normalize(env)
    check("dict is_error", e.is_error is True)
    check("dict output_text from result", e.output_text == "boom")
    check("dict api_error_status (required, 429)", e.api_error_status == 429)


def test_identify_and_parse_cmd():
    b = CCBackend()
    check("identify claude --print", b.identify_cmd("bwrap ... -- claude --print --agent aog-kernel-worker") is True)
    check("identify double-space variant", b.identify_cmd("claude  --print --agent x") is True)
    check("non-claude cmd not identified", b.identify_cmd("python3 orchestrator.py gelu --lane 1") is False)
    check("parse op slug from claude cmd", b.parse_op_from_cmd(
        "claude --print --agent aog-kernel-worker ... gelu-kw-3 ...") == "gelu")
    check("parse op none when no slug", b.parse_op_from_cmd("claude --print --agent x foo bar") is None)


def test_is_a_backend():
    check("CCBackend is a Backend", isinstance(CCBackend(), Backend))
    check("name set", CCBackend().name == "claude_code")


def test_build_skill_cmd():
    b = CCBackend()
    cmd = getattr(b, "_build_skill_cmd")("do the thing")
    check("skill cmd ends with prompt", cmd[-1] == "do the thing")
    check("skill cmd --print + json out", "--print" in cmd and "json" in cmd)
    check("skill cmd bypassPermissions", "bypassPermissions" in cmd)
    check("skill cmd claude bin (default)", cmd[0] == "claude" or cmd[0].endswith("/claude"))
    # sandbox_prefix: OPAQUE verbatim prepend, zero interpretation (main-ratified constraint)
    pfx = ["bwrap", "--ro-bind", "/", "/", "--"]
    cmd2 = getattr(b, "_build_skill_cmd")("p", sandbox_prefix=pfx)
    check("sandbox_prefix verbatim-prepended", cmd2[:5] == pfx)
    check("prefix then base, prompt last", "--print" in cmd2 and cmd2[-1] == "p")


def test_build_skill_cmd_divergent_shape():
    """phase_o17_classify's shape: acceptEdits + plain-text (output_format=None → no --output-format) +
    stdin_prompt (prompt NOT in argv). Must reproduce the original ['claude','--print','--permission-mode',
    'acceptEdits'] byte-for-byte.
    """
    b = CCBackend()
    cmd = getattr(b, "_build_skill_cmd")("/aog-op-classify /ws", permission_mode="acceptEdits",
                             output_format=None, stdin_prompt=True)
    check("divergent cmd byte-identical to phase_o17 original",
          cmd == ["claude", "--print", "--permission-mode", "acceptEdits"] or
          (cmd[1:] == ["--print", "--permission-mode", "acceptEdits"] and cmd[0].endswith("claude")))
    check("stdin_prompt → prompt NOT in argv", "/aog-op-classify /ws" not in cmd)
    check("output_format=None → no --output-format flag", "--output-format" not in cmd)
    # defaults unchanged (backward-compat) — existing skill callers still get json + argv prompt
    d = getattr(b, "_build_skill_cmd")("P")
    check("default still json + bypassPermissions + argv prompt",
          "--output-format" in d and "bypassPermissions" in d and d[-1] == "P")


def test_resume_cmd_and_dispatch():
    """resume: cmd shape (`claude --resume <sid> --print --output-format json <prompt>`) + faithful
    Envelope mapping via a stubbed subprocess.run. No live a5ops caller — interface-completeness only.
    """
    import subprocess as _sp
    b = CCBackend()
    cmd = getattr(b, "_build_resume_cmd")("ses_abc", "keep going")
    check("resume cmd has --resume <sid>", "--resume" in cmd and cmd[cmd.index("--resume") + 1] == "ses_abc")
    check("resume cmd --print + json + prompt-last", "--print" in cmd and "json" in cmd and cmd[-1] == "keep going")

    def _stub_run(*_args, **_kwargs):
        return types.SimpleNamespace(returncode=0, stdout="resumed-out", stderr="")

    _orig_run = _sp.run
    _sp.run = _stub_run
    try:
        env = b.resume("ses_abc", "keep going")
        check("resume → Envelope not-error", isinstance(env, Envelope) and env.is_error is False)
        check("resume output_text mapped", env.output_text == "resumed-out")
    finally:
        _sp.run = _orig_run


def test_dispatch_agent_modes_delegate_faithfully():
    """streaming/background modes delegate to agent_transport faithfully + pass sandbox_prefix VERBATIM +
    thread progress_callback/tee/silence/output_file. No claude spawn (agent_transport stubbed).
    """
    _orig = sys.modules.get("agent_transport")
    at = types.ModuleType("agent_transport")
    calls = {}

    def _mk_result(txt):
        return types.SimpleNamespace(is_error=False, output_text=txt, session_id=None, cost_usd=None,
                                     terminal_reason=None, duration_ms=None, raw_envelope={})

    def _stub_stream(target, prompt, *, tee_path=None, extra_args=None, progress_callback=None,
                     silence_timeout_sec=None, sandbox_prefix=None, timeout_sec=None, **kw):
        calls["streaming"] = dict(sandbox_prefix=sandbox_prefix, tee_path=tee_path,
                                  progress_callback=progress_callback, silence_timeout_sec=silence_timeout_sec,
                                  timeout_sec=timeout_sec)
        return _mk_result("ok")

    def _stub_bg(target, prompt, output_file, *, extra_args=None, sandbox_prefix=None):
        calls["background"] = dict(output_file=output_file, sandbox_prefix=sandbox_prefix)
        return "POPEN"

    def _stub_fg(*_args, **_kwargs):
        return _mk_result("fg")

    at.spawn_agent_streaming = _stub_stream
    at.spawn_agent_background = _stub_bg
    at.spawn_agent_foreground = _stub_fg
    sys.modules["agent_transport"] = at
    try:
        b = CCBackend()
        pfx = ["bwrap", "--"]
        env = b.dispatch("aog-kernel-worker", "p", kind="agent", mode="streaming", sandbox_prefix=pfx,
                         tee_path="t", silence_timeout=1200, progress_callback="cb", timeout=60)
        check("streaming → Envelope", isinstance(env, Envelope) and env.output_text == "ok")
        check("streaming sandbox_prefix verbatim", calls["streaming"]["sandbox_prefix"] == pfx)
        check("streaming threads tee/silence/cb/timeout", calls["streaming"]["tee_path"] == "t"
              and calls["streaming"]["silence_timeout_sec"] == 1200
              and calls["streaming"]["progress_callback"] == "cb" and calls["streaming"]["timeout_sec"] == 60)
        h = b.dispatch("x", "p", kind="agent", mode="background", output_file="/tmp/o", sandbox_prefix=pfx)
        check("background → raw Popen (not Envelope)", h == "POPEN")
        check("background sandbox_prefix + output_file", calls["background"]["sandbox_prefix"] == pfx
              and calls["background"]["output_file"] == "/tmp/o")
    finally:
        if _orig is not None:
            sys.modules["agent_transport"] = _orig
        else:
            sys.modules.pop("agent_transport", None)


if __name__ == "__main__":
    for t in [test_envelope_required_contract, test_normalize_agentresult, test_normalize_dict_envelope,
              test_identify_and_parse_cmd, test_is_a_backend, test_build_skill_cmd,
              test_build_skill_cmd_divergent_shape, test_resume_cmd_and_dispatch,
              test_dispatch_agent_modes_delegate_faithfully]:
        LOGGER.info("%s:", t.__name__)
        t()
    LOGGER.info("%s\n%s", "=" * 50,
                "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)

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

"""Unit tests for harness backend plugin resolution and non-default CLI backends.

No live LLM subprocesses are spawned here; the tests cover selection, command
shape, and prompt wrapping only.
"""
import logging
import os
import sys
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # orchestrator/

from backends import available_backends, get_backend  # noqa: E402
from backends.cc_backend import CCBackend  # noqa: E402
from backends.codex_backend import CodexBackend  # noqa: E402
from backends.opencode_backend import OpencodeBackend  # noqa: E402
from backends.registry import BackendResolutionError  # noqa: E402
from backends.skill_context import load_skill_context  # noqa: E402


FAILS = []
LOGGER = logging.getLogger(__name__)


def check(name, condition, detail=""):
    LOGGER.info("  [%s] %s%s", "PASS" if condition else "FAIL", name,
                f" — {detail}" if detail and not condition else "")
    if not condition:
        FAILS.append(name)


def test_default_and_alias_resolution():
    with patch.dict(os.environ, {}, clear=True):
        check("default backend is CCBackend", isinstance(get_backend(), CCBackend))
    for alias in ("claude", "cc", "claude_code"):
        check(f"alias {alias!r} resolves to CCBackend", isinstance(get_backend(alias), CCBackend))
    check("codex resolves to CodexBackend", isinstance(get_backend("codex"), CodexBackend))
    check("opencode resolves to OpencodeBackend", isinstance(get_backend("opencode"), OpencodeBackend))


def test_env_resolution_and_errors():
    with patch.dict(os.environ, {"AOG_HARNESS_BACKEND": "codex"}, clear=True):
        check("env selects CodexBackend", isinstance(get_backend(), CodexBackend))
    try:
        get_backend("missing-backend")
    except BackendResolutionError as e:
        msg = str(e)
        check("unknown backend raises useful error", "missing_backend" in msg and "claude_code" in msg)
    else:
        check("unknown backend raises useful error", False, "no exception raised")


def test_available_backends_are_canonical():
    names = available_backends()
    check("available includes claude_code", "claude_code" in names)
    check("available includes codex", "codex" in names)
    check("available includes opencode", "opencode" in names)


def test_codex_command_shape_and_prompt():
    b = CodexBackend(codex_bin="/tmp/codex")
    cmd = getattr(b, "_build_exec_cmd")("/tmp/last.txt", cwd="/repo", sandbox="read-only")
    check("codex command uses exec", cmd[:2] == ["/tmp/codex", "exec"])
    check("codex command requests JSONL events", "--json" in cmd)
    check("codex command writes last message", "-o" in cmd and "/tmp/last.txt" in cmd)
    check("codex command uses cwd", "-C" in cmd and "/repo" in cmd)
    check("codex command reads stdin prompt", cmd[-1] == "-")
    prompt = getattr(b, "_format_prompt")("aog-kernel-worker", "do work", kind="agent")
    check("codex prompt carries target", "aog-kernel-worker" in prompt)
    check("codex prompt carries original prompt", "do work" in prompt)
    skill_prompt = getattr(b, "_format_prompt")("aog-op-classify", "classify", kind="skill")
    check("codex skill prompt includes local skill instructions", "aog-op-classify/SKILL.md" in skill_prompt)


def test_opencode_command_shape_and_prompt():
    b = OpencodeBackend(opencode_bin="/tmp/opencode")
    prompt = getattr(b, "_format_prompt")("aog-op-classify", "classify", kind="skill")
    cmd = getattr(b, "_build_run_cmd")(prompt, session_id="ses_1", auto=True, cwd="/repo")
    check("opencode command uses run", cmd[:2] == ["/tmp/opencode", "run"])
    check("opencode command does not carry prompt as positional message", prompt not in cmd)
    check("opencode command can set project dir", "--dir" in cmd and "/repo" in cmd)
    check("opencode command can resume session", "--session" in cmd and "ses_1" in cmd)
    check("opencode command can auto approve", "--auto" in cmd)
    check("opencode skill prompt includes local skill instructions", "aog-op-classify/SKILL.md" in prompt)
    wiring = b.wire_safety([])
    check("opencode wire_safety points to plugin", wiring["kind"] == "host-hook-plugin")


def test_skill_context_loader():
    ctx = load_skill_context("aog-op-classify")
    check("skill context loader finds repo skill", ctx is not None and "aog-op-classify/SKILL.md" in ctx)
    check("skill context loader skips prompt-managed critic", load_skill_context("aog-self-critic") is None)
    check("skill context loader rejects unsafe names", load_skill_context("../aog-op-classify") is None)


if __name__ == "__main__":
    for test in [
        test_default_and_alias_resolution,
        test_env_resolution_and_errors,
        test_available_backends_are_canonical,
        test_codex_command_shape_and_prompt,
        test_opencode_command_shape_and_prompt,
        test_skill_context_loader,
    ]:
        LOGGER.info("%s:", test.__name__)
        test()
    LOGGER.info("%s", "=" * 50)
    if FAILS:
        LOGGER.info("FAILURES:")
        for f in FAILS:
            LOGGER.info(" - %s", f)
        sys.exit(1)
    LOGGER.info("ALL PASS")

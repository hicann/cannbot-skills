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

"""E2E smoke for non-default harness backend dispatch.

This uses temporary fake harness CLIs so the test is deterministic while still
exercising the full local path:

    env selector -> registry -> backend adapter -> skill_context -> subprocess -> Envelope
"""
from __future__ import annotations

import logging
import os
import stat
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # orchestrator/

from backends import get_backend  # noqa: E402


FAILS: list[str] = []
LOGGER = logging.getLogger(__name__)


def check(name: str, condition: bool, detail: str = "") -> None:
    LOGGER.info("  [%s] %s%s", "PASS" if condition else "FAIL", name,
                f" - {detail}" if detail and not condition else "")
    if not condition:
        FAILS.append(name)


def _write_exe(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_codex_fake_cli_skill_dispatch_e2e() -> None:
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "codex"
        _write_exe(
            fake,
            r'''
            import json
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            if args[:1] != ["exec"] or "-o" not in args or args[-1] != "-":
                print(f"bad argv: {args!r}", file=sys.stderr)
                raise SystemExit(2)
            last_path = Path(args[args.index("-o") + 1])
            prompt = sys.stdin.read()
            if "aog-op-classify/SKILL.md" not in prompt or "Return marker" not in prompt:
                print("missing skill context or prompt", file=sys.stderr)
                raise SystemExit(3)
            last_path.write_text("FAKE_CODEX_SKILL_OK")
            print(json.dumps({"type": "thread.started", "thread_id": "fake"}))
            ''',
        )
        with patch.dict(os.environ, {"AOG_HARNESS_BACKEND": "codex", "AOG_CODEX_BIN": str(fake)}, clear=False):
            env = get_backend().dispatch("aog-op-classify", "Return marker", kind="skill", timeout=10)
        check("codex fake CLI dispatch succeeds", not env.is_error, str(env.raw_envelope))
        check("codex fake CLI returns last-message output", env.output_text == "FAKE_CODEX_SKILL_OK")


def test_opencode_fake_cli_skill_dispatch_e2e() -> None:
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "opencode"
        _write_exe(
            fake,
            r'''
            import sys

            args = sys.argv[1:]
            if args[:1] != ["run"]:
                print(f"bad argv: {args!r}", file=sys.stderr)
                raise SystemExit(2)
            if "--format" in args:
                print("unexpected json format", file=sys.stderr)
                raise SystemExit(5)
            prompt = sys.stdin.read()
            if "aog-op-classify/SKILL.md" not in prompt or "Return marker" not in prompt:
                print("missing skill context or prompt", file=sys.stderr)
                raise SystemExit(3)
            if "Return marker" in args:
                print("prompt leaked into argv", file=sys.stderr)
                raise SystemExit(4)
            print("FAKE_OPENCODE_SKILL_OK")
            ''',
        )
        with patch.dict(os.environ, {"AOG_HARNESS_BACKEND": "opencode", "AOG_OPENCODE_BIN": str(fake)}, clear=False):
            env = get_backend().dispatch("aog-op-classify", "Return marker", kind="skill", timeout=10)
        check("opencode fake CLI dispatch succeeds", not env.is_error, str(env.raw_envelope))
        check("opencode fake CLI returns stdout output", env.output_text.strip() == "FAKE_OPENCODE_SKILL_OK")


if __name__ == "__main__":
    for test in [
        test_codex_fake_cli_skill_dispatch_e2e,
        test_opencode_fake_cli_skill_dispatch_e2e,
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

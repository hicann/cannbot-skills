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
"""CodeArts (华为云码道) CLI backend plugin.

The CodeArts Agent CLI is an OpenCode fork: it speaks the same ``run``
subcommand, the same ``OPENCODE_CONFIG_CONTENT`` config injection and the same
JSON event stream, so this backend reuses ``OpencodeBackend`` verbatim and
re-points only what differs:

  * binary: ``codearts`` (env override ``AOG_CODEARTS_BIN``);
  * orphan-process markers for recover.py: ``codearts run``;
  * hook identity label: ``codearts:<agent>`` (matches the
    ``{backend.name}:<agent>`` convention used by the stop-gate payload).

Runtime deps (node/bun + ripgrep) and the fail-closed safety-net probe are
inherited unchanged from the opencode backend.

**Same-environment rule (2026-09-08, interop probe finding):** the codearts
CLI must be native to the engine's environment — Linux CLI for a WSL/Linux
engine, Windows CLI for a Windows-native engine. Driving the Windows
``codearts.exe`` from a WSL engine through interop does NOT work: interop
does not forward per-dispatch env injection (``OPENCODE_CONFIG_CONTENT`` …)
without per-var ``WSLENV`` entries, and the WSL-baked ``/mnt/...`` paths in
the injected agent config are meaningless to a Windows binary.

Graybox note: the sandboxed user-config runtime copy still sources
``OPENCODE_USER_CONFIG`` (default ``~/.config/opencode``). A CodeArts install
that needs its own provider config inside the graybox must export
``OPENCODE_USER_CONFIG`` pointing at a directory holding that config.
"""
from __future__ import annotations

import os

from .opencode_backend import OpencodeBackend


class CodeartsBackend(OpencodeBackend):
    name = "codearts"

    # process markers for recover.py orphan identification (mirror of
    # _OPENCODE_RUN_MARKERS; the double-space form covers `codearts  run`)
    _RUN_MARKERS = ("codearts run", "codearts  run")

    def __init__(self, opencode_bin: str | None = None):
        super().__init__(opencode_bin or os.environ.get("AOG_CODEARTS_BIN", "codearts"))

    @staticmethod
    def build_env(target: str, prompt: str | None = None, *, kind: str = "agent",
                  cwd: str | None = None) -> dict:
        env = OpencodeBackend.build_env(target, prompt, kind=kind, cwd=cwd)
        if env.get("AOG_HOOK_AGENT_ID") == f"opencode:{target}":
            env["AOG_HOOK_AGENT_ID"] = f"codearts:{target}"
        return env

    def identify_cmd(self, cmd: str) -> bool:
        return any(marker in cmd for marker in self._RUN_MARKERS)

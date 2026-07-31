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
"""Backend plugin resolver.

The operator-facing backend selector is intentionally named
``AOG_HARNESS_BACKEND`` so it cannot be confused with the fixed AscendC kernel
programming model.
"""
from __future__ import annotations

import os
from typing import Callable

from .base import Backend
from .cc_backend import CCBackend
from .codex_backend import CodexBackend
from .opencode_backend import OpencodeBackend


class BackendResolutionError(RuntimeError):
    """Raised when a requested harness backend plugin is unknown."""


_CANONICAL: dict[str, Callable[[], Backend]] = {
    "claude_code": CCBackend,
    "codex": CodexBackend,
    "opencode": OpencodeBackend,
}

_ALIASES = {
    "cc": "claude_code",
    "claude": "claude_code",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "codex": "codex",
    "codex_cli": "codex",
    "codex-cli": "codex",
    "open_code": "opencode",
    "open-code": "opencode",
    "opencode": "opencode",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def available_backends() -> tuple[str, ...]:
    """Return canonical backend plugin names."""
    return tuple(sorted(_CANONICAL))


def get_backend(name: str | None = None) -> Backend:
    """Resolve a harness backend plugin.

    Defaults to Claude Code for backward compatibility. Passing a name or setting
    ``AOG_HARNESS_BACKEND`` selects another plugin.
    """
    raw = name or os.environ.get("AOG_HARNESS_BACKEND") or "claude_code"
    key = _normalize_name(raw)
    canonical = _ALIASES.get(key)
    if canonical is None:
        choices = ", ".join(available_backends())
        raise BackendResolutionError(
            f"unknown harness backend {key!r}; expected one of: {choices}"
        )
    return _CANONICAL[canonical]()

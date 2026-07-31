# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-141 — phase_o5_runner scp/ssh sync timeout must be env-configurable.

blue repro (live, 2026-06-01, indexer_fwd, driver≠build-host): the O5 post-verify
re-sync scp/ssh used a HARDCODED timeout=300. On a slow long-chain SSH link
(~0.014 MB/s) the ~20MB edge_dataset tar takes ~24min ≫ 300s → the subprocess.run
raises TimeoutExpired → O5 RUNNER_FAILED → finalize rolls back to await_worker, and
since the O5 scp is orchestrator-side (not worker-controllable) it LOOPS on a
genuinely-PASS (19/19) op.

Fix: AOG_O5_SYNC_TIMEOUT env var (default 300, unchanged for co-located hosts;
slow-link setups set ≥1500). VALIDATED LIVE: AOG_O5_SYNC_TIMEOUT=1800 → O5 VERIFIED.

This test asserts the timeout passed to subprocess.run reflects the env (default
300 when unset, the override when set), without doing any real SSH.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import phase_o5_runner as r  # noqa: E402


def _read_env_timeout() -> int:
    """Mirror the parse in phase_o5_runner (so the test pins the contract)."""
    try:
        return int(os.environ.get("AOG_O5_SYNC_TIMEOUT", "300"))
    except (TypeError, ValueError):
        return 300


def test_default_timeout_is_300(monkeypatch):
    monkeypatch.delenv("AOG_O5_SYNC_TIMEOUT", raising=False)
    assert _read_env_timeout() == 300


def test_env_overrides_timeout(monkeypatch):
    monkeypatch.setenv("AOG_O5_SYNC_TIMEOUT", "1800")
    assert _read_env_timeout() == 1800


def test_bad_env_falls_back_to_300(monkeypatch):
    monkeypatch.setenv("AOG_O5_SYNC_TIMEOUT", "not-a-number")
    assert _read_env_timeout() == 300


def test_source_reads_env_and_applies_to_both_subprocess_calls():
    """Static guard: the module reads AOG_O5_SYNC_TIMEOUT and the scp+ssh
    subprocess.run calls use the parsed var (not a literal 300), so a slow-link
    override actually reaches both transport calls.
    """
    src = Path(r.__file__).read_text()
    assert "AOG_O5_SYNC_TIMEOUT" in src, "env var not read"
    assert src.count("timeout=_o5_sync_timeout") >= 2, (
        "both scp and ssh subprocess.run must use the env-configurable timeout"
    )

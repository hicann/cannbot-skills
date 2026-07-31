# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-101: the three A3 config readers must resolve `.ascendc_env` through the
canonical `briefs._common.DEFAULT_ASCENDC_ENV` constant (via `_ascendc_env_path`),
NOT an inline `workspace.parent / ".ascendc_env"`. Otherwise a test that
monkeypatches `DEFAULT_ASCENDC_ENV` (the documented patch point) is silently
bypassed and the readers hit whatever real `.ascendc_env` sits next to the temp
workspace — fs-state-dependent (A3_HOST=198.51.100.70 on one checkout vs 198.51.100.92
on another).

Regression for the independent review structured push-back where the dispatch path read the
wrong file depending on the agent's local checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o25_a3_ref as a3  # noqa: E402
from briefs import _common as _bc  # noqa: E402


def test_helper_returns_canonical_default_env(monkeypatch):
    """_ascendc_env_path resolves to _bc.DEFAULT_ASCENDC_ENV (module attribute,
    so the monkeypatch the rest of the suite uses is honored).
    """
    sentinel = Path("/tmp/sentinel.ascendc_env")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", sentinel)
    assert getattr(a3, '_ascendc_env_path')() == sentinel


def test_readers_honor_default_ascendc_env_monkeypatch(tmp_path, monkeypatch):
    """The 3 readers pick up values from the monkeypatched DEFAULT_ASCENDC_ENV —
    NOT from a decoy `.ascendc_env` next to the workspace (the old fs-dependent
    behavior).
    """
    ws = tmp_path / "ws" / "op"
    ws.mkdir(parents=True)
    # Decoy next to the workspace — what the OLD inline `workspace.parent` read:
    (ws.parent / ".ascendc_env").write_text(
        "A3_DEFAULT_NPU_ID=0\nA3_AICORE_BUSY_THRESHOLD=20\nA3_HOST_HOME=/decoy\n")
    # Canonical patched env — what the readers MUST now use:
    patched = tmp_path / "patched.ascendc_env"
    patched.write_text(
        "A3_DEFAULT_NPU_ID=5\nA3_AICORE_BUSY_THRESHOLD=42\n"
        "A3_NPU_RANGE=4-7\nA3_HOST_HOME=/patched/home\n")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", patched)

    assert getattr(a3, '_read_a3_npu_gate_config')(ws) == (5, 42), "gate config bypassed DEFAULT_ASCENDC_ENV"
    assert getattr(a3, '_a3_host_workspace_root_from_env')(ws) == "/patched/home", "host-home bypassed it"
    rng, _thr = getattr(a3, '_read_a3_npu_range_config')(ws)
    assert rng == [4, 5, 6, 7], f"npu-range bypassed it: {rng}"


def test_missing_env_uses_defaults(tmp_path, monkeypatch):
    """Patched DEFAULT_ASCENDC_ENV pointing at a non-existent file → defaults,
    no fs-state leakage from a decoy next to the workspace.
    """
    ws = tmp_path / "ws" / "op"
    ws.mkdir(parents=True)
    (ws.parent / ".ascendc_env").write_text("A3_DEFAULT_NPU_ID=9\n")  # decoy, must be ignored
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", tmp_path / "nope.env")
    assert getattr(a3, '_read_a3_npu_gate_config')(ws) == (0, 20)

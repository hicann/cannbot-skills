# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression test — ARCH35_WRAP_CHEAT gate must scan ALL compiled build TUs.

OL-160 recurrence guard (task#6, 2026-05-29). The FA archive shipped a
`flash_attention_score_apt.cpp` that `#include`d `arch35/*` (upstream
Ascend950PR/V351 source) — a wrap-cheat, NOT a V220→V351 port. The
ORIGINAL `_check_arch35_wrap_cheat` only scanned `<op>_kernels.cpp` +
`pybind11.cpp`, so an arch35 include living in `*_apt.cpp` (or an
`op_host/*_tiling.cpp`) slipped the gate silently — the same name-coupling
failure class as the A5-verify-path-fraud incident (OL-160): a safety net
hard-coded to a name list returns "0 violations" when the cheat moves to a
TU outside that list.

The gate was widened (audit#4 2026-05-23) to `rglob("*.cpp")` + `rglob("*.h")`
across `op_kernel/`, `op_host/`, and the archive root. This test LOCKS that
widening: if a future refactor re-narrows the scan to a fixed name list, the
`*_apt.cpp` / `op_host` cases below go red.

Probe that motivated this test (2026-05-29): running the live gate against
`output/a3_to_a5_port/src/kernels/flash_attention_score` (which still carries
`flash_attention_score_apt.cpp` with 5 `#include "arch35/..."` lines, pre-#255
build-switch) returns FIRES — confirming the widened gate covers `*_apt.cpp`.
This test reproduces that coverage on a synthetic fixture so it survives the
archive's eventual cleanup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from finalize_pipeline import (  # type: ignore  # noqa: E402
    GateID,
    _check_arch35_wrap_cheat,
)

_ARCH35_INCLUDE = '#include "arch35/flash_attention_score_entry_regbase.h"\n'
_CLEAN_INCLUDE = '#include "flash_attention_score_cube.h"\n'


def _make_port_a3_ws(tmp_path: Path, *, mode: str = "port_a3_to_a5") -> Path:
    """Synthetic workspace that detect_plugin() resolves to port_a3 (or `mode`)."""
    ws = tmp_path / "workspace_op"
    (ws / "op_kernel").mkdir(parents=True, exist_ok=True)
    (ws / "op_host").mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(
        json.dumps({"op": "flash_attention_score", "opgen_mode": mode})
    )
    return ws


def _strip_prestage_optin(monkeypatch) -> None:
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)


def test_apt_cpp_include_fires_gate(tmp_path, monkeypatch):
    """THE regression: arch35 include in *_apt.cpp (not kernels.cpp) must fire.

    This is the exact file/name that slipped the original narrow gate.
    """
    _strip_prestage_optin(monkeypatch)
    ws = _make_port_a3_ws(tmp_path)
    # kernels.cpp + pybind11.cpp are CLEAN (the narrow gate would have passed) ...
    (ws / "op_kernel" / "flash_attention_score_kernels.cpp").write_text(_CLEAN_INCLUDE)
    (ws / "pybind11.cpp").write_text(_CLEAN_INCLUDE)
    # ... but the apt TU carries the arch35 wrap.
    (ws / "op_kernel" / "flash_attention_score_apt.cpp").write_text(_ARCH35_INCLUDE)

    rationale = _check_arch35_wrap_cheat(ws)
    assert rationale is not None, "apt.cpp arch35 include MUST trip the gate (OL-160 recurrence)"
    assert "flash_attention_score_apt.cpp" in rationale
    assert GateID.ARCH35_WRAP_CHEAT.value == "arch35_wrap_cheat"


def test_op_host_tiling_include_fires_gate(tmp_path, monkeypatch):
    """audit#4 case: arch35 include in op_host/*_tiling.cpp must fire."""
    _strip_prestage_optin(monkeypatch)
    ws = _make_port_a3_ws(tmp_path)
    (ws / "op_host" / "flash_attention_score_tiling.cpp").write_text(_ARCH35_INCLUDE)
    rationale = _check_arch35_wrap_cheat(ws)
    assert rationale is not None
    assert "flash_attention_score_tiling.cpp" in rationale


def test_traversal_form_include_fires_gate(tmp_path, monkeypatch):
    """Relative-traversal include (.../arch35/...) is the same cheat — must fire."""
    _strip_prestage_optin(monkeypatch)
    ws = _make_port_a3_ws(tmp_path)
    (ws / "op_kernel" / "flash_attention_score_apt.cpp").write_text(
        '#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"\n'
    )
    rationale = _check_arch35_wrap_cheat(ws)
    assert rationale is not None


def test_clean_workspace_passes(tmp_path, monkeypatch):
    """No arch35 include anywhere → gate returns None (no false positive)."""
    _strip_prestage_optin(monkeypatch)
    ws = _make_port_a3_ws(tmp_path)
    (ws / "op_kernel" / "flash_attention_score_kernels.cpp").write_text(_CLEAN_INCLUDE)
    (ws / "op_kernel" / "flash_attention_score_apt.cpp").write_text(_CLEAN_INCLUDE)
    (ws / "pybind11.cpp").write_text(_CLEAN_INCLUDE)
    assert _check_arch35_wrap_cheat(ws) is None


def test_non_port_a3_out_of_scope(tmp_path, monkeypatch):
    """Gate is migration-scoped: an unsupported-mode workspace is ignored."""
    _strip_prestage_optin(monkeypatch)
    ws = _make_port_a3_ws(tmp_path, mode="unsupported")
    (ws / "op_kernel" / "flash_attention_score_apt.cpp").write_text(_ARCH35_INCLUDE)
    assert _check_arch35_wrap_cheat(ws) is None


def test_prestage_optin_suspends_gate(tmp_path, monkeypatch):
    """OPGEN_PRESTAGE_ARCH35=1 = explicit opt-in → gate suspended even with includes."""
    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    ws = _make_port_a3_ws(tmp_path)
    (ws / "op_kernel" / "flash_attention_score_apt.cpp").write_text(_ARCH35_INCLUDE)
    assert _check_arch35_wrap_cheat(ws) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

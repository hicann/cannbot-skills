# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P79 (2026-05-15, user Discord 03:30Z) — scan_delegation_cheating cross-mode regression.

User question: "对于其他mode的regression检测也要做" — does sanity suite cover
non-port_a3 modes? Audit found:
- Universal patterns (PYTHON_WRAPPER_PATTERNS / CPP_PATTERNS) apply to ALL
  modes via scan_op_workspace's mandatory model_new_ascendc.py + kernel/
  scan paths — implicitly cross-mode.
- port_a3-specific patterns must not fire on backward workspaces; the
  _is_port_a3_workspace gate decides.

This test pins both invariants so future scanner edits can't silently
regress mode-coverage in either direction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent.parent.parent  # src/scripts/
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import scan_delegation_cheating as sdc  # noqa: E402
from plugins import all_plugins, get_plugin  # noqa: E402


# ---------------------------------------------------------------------------
# Universal pattern coverage — model_new_ascendc.py in every mode
# ---------------------------------------------------------------------------

def _make_workspace(tmp_path: Path, opgen_mode: str, *,
                    model_new_body: str = "x = 1\n") -> Path:
    ws = tmp_path / f"ws_{opgen_mode}"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "fixture", "opgen_mode": opgen_mode,
    }))
    (ws / "model_new_ascendc.py").write_text(model_new_body)
    return ws


@pytest.mark.parametrize("mode", ["port_a3_to_a5", "backward"])
def test_python_wrapper_torch_npu_caught_in_all_modes(tmp_path, mode):
    """torch_npu.<api>() is CANN delegation regardless of mode. Universal
    scan must catch it in both supported modes.
    """
    body = (
        "import torch_npu\n"
        "def f(x):\n"
        "    return torch_npu.npu_rms_norm(x)\n"
    )
    ws = _make_workspace(tmp_path, mode, model_new_body=body)
    result = sdc.scan_op_workspace(ws)
    assert not result["ok"], f"mode={mode}: universal scan should flag torch_npu.<api>()"
    descs = [v["desc"] for v in result["violations"]]
    assert any("torch_npu" in d or "CANN delegation" in d for d in descs), \
        f"mode={mode}: missing torch_npu violation in {descs}"


@pytest.mark.parametrize("mode", ["port_a3_to_a5", "backward"])
def test_python_wrapper_aclnn_caught_in_all_modes(tmp_path, mode):
    """aclnn* calls are CANN built-ins regardless of mode."""
    body = "ret = aclnnRmsNorm(x, gamma)\n"
    ws = _make_workspace(tmp_path, mode, model_new_body=body)
    result = sdc.scan_op_workspace(ws)
    assert not result["ok"], f"mode={mode}: aclnn* must be caught"


# ---------------------------------------------------------------------------
# port_a3-specific patterns must NOT fire on non-port_a3 modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["backward"])
def test_port_a3_verify_forbidden_does_not_fire_in_other_modes(tmp_path, mode):
    """PORT_A3_VERIFY_FORBIDDEN (`torch.nn.functional.<op>` / `F.<op>` /
    `torch._foreach_<op>` in A5 verify path) is fatal in port_a3 but
    Must not false-positive on backward workspaces.

    We seed the workspace with `run_a5_verify.py` containing the exact
    pattern that would be fatal in port_a3, then verify scan_op_workspace
    does NOT report a port_a3_verify_forbidden category violation for
    backward.
    """
    ws = _make_workspace(tmp_path, mode)
    # Seed verify-path file with port_a3-forbidden patterns
    (ws / "run_a5_verify.py").write_text(
        "import torch.nn.functional as F\n"
        "import torch\n"
        "def verify(x, dev='npu'):\n"
        "    out = F.softmax(x)\n"
        "    return torch._foreach_abs([out])\n"
    )
    result = sdc.scan_op_workspace(ws)
    # The port_a3-only category must be absent
    port_a3_cat_hits = [v for v in result["violations"]
                        if v.get("category") == "port_a3_verify_forbidden"]
    assert not port_a3_cat_hits, (
        f"mode={mode}: port_a3_verify_forbidden fired on non-port_a3 mode — "
        f"hits: {port_a3_cat_hits}"
    )


def test_port_a3_verify_forbidden_does_fire_in_port_a3_mode(tmp_path):
    """Sanity check: the same pattern that's allowed in non-port modes
    modes IS caught when opgen_mode=port_a3_to_a5. Without this assertion
    the negative tests above could pass by the gate being totally off.
    """
    ws = _make_workspace(tmp_path, "port_a3_to_a5")
    (ws / "run_a5_verify.py").write_text(
        "import torch.nn.functional as F\n"
        "def verify(x):\n"
        "    return F.softmax(x)\n"
    )
    result = sdc.scan_op_workspace(ws)
    port_a3_cat_hits = [v for v in result["violations"]
                        if v.get("category") == "port_a3_verify_forbidden"]
    assert port_a3_cat_hits, (
        "port_a3_verify_forbidden didn't fire on port_a3 workspace — "
        "either mode detection broken or pattern catalog missing F.<op>"
    )


# ---------------------------------------------------------------------------
# Mode detection by .opgen_state.json — primary truth source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected_port_a3", [
    ("port_a3_to_a5", True),
    ("backward", False),
    ("unsupported", False),
])
def test_is_port_a3_workspace_respects_state_file(tmp_path, mode, expected_port_a3):
    """_is_port_a3_workspace MUST read .opgen_state.json opgen_mode field
    first, before falling back to PORT_A3_VERIFY_FILES heuristic. Without
    this, a non-port workspace that happens to have a `pass_a_runner.py`
    could be misclassified.
    """
    ws = tmp_path / f"ws_{mode}"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "fixture", "opgen_mode": mode,
    }))
    assert getattr(sdc, '_is_port_a3_workspace')(ws) == expected_port_a3, (
        f"mode={mode}: _is_port_a3_workspace={getattr(sdc, '_is_port_a3_workspace')(ws)} "
        f"expected={expected_port_a3}"
    )


# ---------------------------------------------------------------------------
# CPP scanner reaches kernel/ in every mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["port_a3_to_a5", "backward"])
def test_cpp_aclnn_caught_in_kernel_dir_all_modes(tmp_path, mode):
    """Kernel-dir .cpp is scanned for aclnn* in every mode."""
    ws = _make_workspace(tmp_path, mode)
    (ws / "kernel").mkdir()
    (ws / "kernel" / "main.cpp").write_text(
        '#include "kernel.h"\n'
        'extern "C" void launch() {\n'
        '    aclnnRmsNorm(x, gamma);\n'
        '}\n'
    )
    result = sdc.scan_op_workspace(ws)
    assert not result["ok"], f"mode={mode}: kernel-dir aclnn* must be caught"


# ---------------------------------------------------------------------------
# A scan that traverses no declared source is a hard coverage failure
# ---------------------------------------------------------------------------

def test_every_supported_mode_declares_python_and_cpp_compute_surfaces():
    """The two RFC workflows must remain mechanically scannable."""
    assert {plugin.name for plugin in all_plugins()} == {
        "port_a3_to_a5",
        "backward",
    }
    for plugin in all_plugins():
        assert plugin.kernel_logic_files(), plugin.name
        assert plugin.kernel_cpp_dirs(), plugin.name


@pytest.mark.parametrize("mode", ["port_a3_to_a5", "backward"])
def test_missing_declared_python_logic_is_coverage_failure(tmp_path, mode):
    plugin = get_plugin(mode)
    ws = tmp_path / f"missing_python_{mode}"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "fixture",
        "opgen_mode": mode,
    }))
    for rel in plugin.kernel_cpp_dirs():
        (ws / rel).mkdir(parents=True)

    result = sdc.scan_op_workspace(ws)

    assert not result["ok"]
    assert any(
        violation.get("category") == sdc.SCANNER_COVERAGE_CATEGORY
        and "kernel_logic_files" in violation.get("desc", "")
        for violation in result["violations"]
    )


@pytest.mark.parametrize("mode", ["port_a3_to_a5", "backward"])
def test_missing_declared_cpp_surface_is_coverage_failure(tmp_path, mode):
    ws = _make_workspace(tmp_path, mode)

    result = sdc.scan_op_workspace(ws)

    assert not result["ok"]
    assert any(
        violation.get("category") == sdc.SCANNER_COVERAGE_CATEGORY
        and "kernel_cpp_dirs" in violation.get("desc", "")
        for violation in result["violations"]
    )

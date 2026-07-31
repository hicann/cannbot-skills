# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P96 follow-up — VERIFIER_USES_MODELNEW gate.

User catch 2026-05-15 23:00Z: "run_a5_verify.py 是不是根本没有使用
model_new_ascendc.py？" Audit revealed:
- adaptive_avg_pool3d archive: model_new_ascendc.py present (OL-160
  filename gate passing) but pass_a_runner.py did NOT import/instantiate/
  call ModelNew. The ModelNew wrapper was decorative bypass.
- OL-160 filename rule = 0 USAGE check.

Co-developed with independent review agent's `check_verifier_uses_modelnew` plugin
hook (commit 28fd7221 on feature/independent review-target-onboarding).

This file pins the port_a3 plugin override + the finalize gate wiring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp
from plugins.port_a3 import PortA3Plugin
from plugins.backward import BackwardPlugin
from plugins.base import BasePlugin


# ── BasePlugin neutral default ──────────────────────────────────────────


def test_base_plugin_no_enforcement():
    """BasePlugin: no enforcement (returns None)."""
    bp = BasePlugin()
    assert bp.check_verifier_uses_modelnew(Path("/tmp"), {}) is None


# ── port_a3 plugin override ─────────────────────────────────────────────


def _seed_port_a3(tmp_path: Path, pass_a_body: str | None) -> Path:
    """Seed a port_a3-mode workspace with the given pass_a_runner.py body."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "test_op", "opgen_mode": "port_a3_to_a5"})
    )
    if pass_a_body is not None:
        (ws / "pass_a_runner.py").write_text(pass_a_body)
    return ws


def test_port_a3_real_usage_passes(tmp_path):
    """All 3 markers in code → gate passes."""
    body = '''"""docstring stripped."""
from model_new_ascendc import ModelNew

def main():
    mn = ModelNew()
    out = mn(x)
'''
    ws = _seed_port_a3(tmp_path, body)
    assert PortA3Plugin().check_verifier_uses_modelnew(ws, {}) is None


def test_port_a3_no_pass_a_file_neutral(tmp_path):
    """No pass_a_runner.py → return None (other gates handle missing file)."""
    ws = _seed_port_a3(tmp_path, None)
    assert PortA3Plugin().check_verifier_uses_modelnew(ws, {}) is None


def test_port_a3_only_docstring_reference_rejected(tmp_path):
    """ModelNew mentioned ONLY in docstring → gate rejects (must be in code)."""
    body = '''"""This file uses ModelNew and ModelNew() and instance calls but only in docstring."""
import torch

def main():
    out = subprocess.run(["runner_binary"], capture_output=True)
'''
    ws = _seed_port_a3(tmp_path, body)
    r = PortA3Plugin().check_verifier_uses_modelnew(ws, {})
    assert r is not None
    assert "model_new_ascendc" in r or "ModelNew" in r


def test_port_a3_missing_import_rejected(tmp_path):
    """Has ModelNew() + call but no import reference → reject."""
    body = '''
import torch
def main():
    mn = ModelNew()
    mn(x)
'''
    ws = _seed_port_a3(tmp_path, body)
    r = PortA3Plugin().check_verifier_uses_modelnew(ws, {})
    assert r is not None
    assert "model_new_ascendc" in r


def test_port_a3_missing_instantiation_rejected(tmp_path):
    """Has import + invocation but no ModelNew() → reject."""
    body = '''
from model_new_ascendc import ModelNew
def main():
    mn(x)  # no `mn = ModelNew()` anywhere
'''
    ws = _seed_port_a3(tmp_path, body)
    r = PortA3Plugin().check_verifier_uses_modelnew(ws, {})
    assert r is not None
    assert "ModelNew" in r


def test_port_a3_missing_invocation_rejected(tmp_path):
    """Has import + instantiation but never calls the instance → reject."""
    body = '''
from model_new_ascendc import ModelNew
def main():
    mn = ModelNew()
    # never invokes mn(...) or mn.forward(...)
'''
    ws = _seed_port_a3(tmp_path, body)
    r = PortA3Plugin().check_verifier_uses_modelnew(ws, {})
    assert r is not None
    assert "mn" in r and "never invokes" in r.lower()


def test_port_a3_method_call_passes(tmp_path):
    """gather_elements_v2 pattern: `mn = ModelNew()` then `mn.forward(...)`."""
    body = '''
from model_new_ascendc import ModelNew
def main():
    mn = ModelNew()
    out = mn.forward(x, y)
'''
    ws = _seed_port_a3(tmp_path, body)
    assert PortA3Plugin().check_verifier_uses_modelnew(ws, {}) is None


def test_port_a3_decorative_bypass_pattern_rejected(tmp_path):
    """The adaptive_avg_pool3d-style decorative bypass: model_new_ascendc.py
    exists but pass_a_runner directly invokes runner binary, NOT ModelNew.
    """
    body = '''
"""pass_a_runner — bypasses ModelNew, invokes runner binary directly."""
import json
import subprocess
from pathlib import Path
import torch

def main():
    # Build runner if needed
    subprocess.run(["bash", "build_runner.sh"], check=True)
    # Run each case via runner binary
    for case in cases:
        result = subprocess.run(["./aclnn_runner", json.dumps(case)],
                                capture_output=True, text=True, check=True)
'''
    ws = _seed_port_a3(tmp_path, body)
    r = PortA3Plugin().check_verifier_uses_modelnew(ws, {})
    assert r is not None
    assert "OL-160" in r or "decorative-bypass" in r.lower()


def test_verifier_gate_exception_fails_closed(tmp_path, monkeypatch):
    ws = _seed_port_a3(tmp_path, "from model_new_ascendc import ModelNew\n")

    def _raise(*args, **kwargs):
        raise OSError("unreadable verifier")

    monkeypatch.setattr(PortA3Plugin, "check_verifier_uses_modelnew", _raise)
    reason = getattr(fp, '_check_verifier_uses_modelnew')(ws, {})
    assert reason is not None
    assert "failed to inspect" in reason


def _seed_backward(tmp_path: Path, body: str) -> tuple[Path, dict]:
    ws = tmp_path / "mul_grad"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "mul_grad", "opgen_mode": "backward",
    }))
    (ws / "verify_mul_grad.py").write_text(body)
    vj = {
        "op": "mul_grad",
        "harness_pristine": {
            "o5_verdict": "VERIFIED", "sampled_at": "o5_post_verify",
        },
    }
    return ws, vj


@pytest.mark.parametrize("body,needle", [
    ("candidate = ModelNew()\ncandidate(x)\n", "import"),
    ("from model_new_ascendc import ModelNew\n", "instantiate"),
    ("from model_new_ascendc import ModelNew\ncandidate = ModelNew()\n", "never calls"),
])
def test_backward_modelnew_usage_gaps_rejected(tmp_path, body, needle):
    ws, vj = _seed_backward(tmp_path, body)
    reason = BackwardPlugin().check_verifier_uses_modelnew(ws, vj)
    assert reason is not None and needle in reason


def test_backward_modelnew_real_call_passes(tmp_path):
    ws, vj = _seed_backward(
        tmp_path,
        "from model_new_ascendc import ModelNew\n"
        "candidate = ModelNew()\n"
        "output = candidate(inputs)\n",
    )
    assert BackwardPlugin().check_verifier_uses_modelnew(ws, vj) is None


# ── GateID + finalize wiring ────────────────────────────────────────────


def test_gate_id_stable():
    assert fp.GateID.VERIFIER_USES_MODELNEW.value == "verifier_uses_modelnew"

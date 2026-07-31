# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""B3.1: `--backward` CLI mode scaffolding (_cmd_backward).

Verifies the backward op-gen mode entry point scaffolds a workspace that
plugins.detect_plugin resolves to the BackwardPlugin, with a GRADIENT op_class
tag (so is_backward_class fires → the C2 OL-200 brief block activates).
No hardware; pure scaffolding + state.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import orchestrator as _orchestrator_import  # noqa: E402

orchestrator = (
    _orchestrator_import
    if hasattr(_orchestrator_import, "run_single_op")
    else importlib.import_module("orchestrator.orchestrator")
)
from plugins import detect_plugin  # noqa: E402
from plugins.base import is_backward_class  # noqa: E402
from schema_norm import detect_op_class  # noqa: E402


def _fwd_spec(tmp_path: Path, name: str = "mul") -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text("import torch\n\ndef forward(x, w):\n    return x * w\n")
    return p


def _seed(tmp_path, monkeypatch, name="mul", plan_only=True, timing=False):
    ws_root = tmp_path / "workspace"
    ws_root.mkdir()
    monkeypatch.setattr(orchestrator, "WORKSPACE_ROOT", ws_root)
    spec = _fwd_spec(tmp_path, name)
    rc = getattr(orchestrator, "_cmd_backward")(
        forward_spec=spec, lane=0, plan_only=plan_only, cold_start=False,
        timing=timing,
    )
    return ws_root, rc


# ── op-name derivation ───────────────────────────────────────────────────

def test_op_name_derivation_appends_grad(tmp_path, monkeypatch):
    ws_root, rc = _seed(tmp_path, monkeypatch, name="mul")
    assert rc == 0  # plan_only
    assert (ws_root / "mul_grad").is_dir()


def test_op_name_no_double_grad_suffix(tmp_path, monkeypatch):
    ws_root, rc = _seed(tmp_path, monkeypatch, name="lightning_indexer_grad")
    assert (ws_root / "lightning_indexer_grad").is_dir()
    assert not (ws_root / "lightning_indexer_grad_grad").exists()


# ── state seeding → plugin detection ─────────────────────────────────────

def test_backward_workspace_resolves_to_backward_plugin(tmp_path, monkeypatch):
    ws_root, _ = _seed(tmp_path, monkeypatch, name="mul")
    ws = ws_root / "mul_grad"
    state = json.loads((ws / ".opgen_state.json").read_text())
    assert state["opgen_mode"] == "backward"
    assert state["backward_forward_source"].endswith("mul.py")
    found = detect_plugin(ws)
    assert found is not None and found.name == "backward"


def test_gradient_tag_fires_is_backward_class(tmp_path, monkeypatch):
    ws_root, _ = _seed(tmp_path, monkeypatch, name="mul")
    ws = ws_root / "mul_grad"
    cls = json.loads((ws / "op_classification.json").read_text())
    assert "GRADIENT" in cls["op_class_tags"]
    # detect_op_class joins tags → is_backward_class fires → C2 brief block active
    assert is_backward_class(detect_op_class(ws, {})) is True


# ── non-plan run routing (B3.2: scaffold → run_single_op) ────────────────

def test_non_plan_invokes_run_single_op(tmp_path, monkeypatch):
    """B3.2 (2026-05-30): without --plan, _cmd_backward scaffolds then hands to
    run_single_op (mirrors _cmd_port_a3), whose O2.5 backward dispatch produces
    the self-contained reference. Here we assert the handoff with run_single_op
    stubbed (the dispatch itself is covered by test_backward_o25_b3_2.py).
    """
    captured = {}

    def fake_run(op, **kwargs):
        captured["op"] = op
        captured.update(kwargs)
        return 98  # B3.2 boundary
    monkeypatch.setattr(orchestrator, "run_single_op", fake_run)
    ws_root, rc = _seed(
        tmp_path, monkeypatch, name="mul", plan_only=False, timing=True
    )
    assert rc == 98
    assert captured["op"] == "mul_grad"
    assert captured["workspace"] == ws_root / "mul_grad"
    assert captured["timing"] is True
    assert (ws_root / "mul_grad" / ".opgen_state.json").is_file()  # scaffolded


# ── validation ───────────────────────────────────────────────────────────

def test_missing_spec_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "WORKSPACE_ROOT", tmp_path / "workspace")
    rc = getattr(orchestrator, "_cmd_backward")(
        forward_spec=tmp_path / "nope.py", lane=0, plan_only=False, cold_start=False,
    )
    assert rc == 2


def test_non_py_spec_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "WORKSPACE_ROOT", tmp_path / "workspace")
    bad = tmp_path / "spec.txt"
    bad.write_text("not python")
    rc = getattr(orchestrator, "_cmd_backward")(
        forward_spec=bad, lane=0, plan_only=False, cold_start=False,
    )
    assert rc == 2


def test_missing_ascendc_env_fails_fast(tmp_path, monkeypatch):
    """Backward mode fails fast with rc=2
    when .ascendc_env can't load — MIRROR _cmd_port_a3. Regression: previously it printed the
    plan + entered the pipeline (backward CPU-truth doesn't need the NPU immediately), only
    crashing LATER at build = a wasted run; and --plan with a missing env exited 0. The
    preflight runs BEFORE the plan branch, so the run stops early with a non-zero exit.
    """
    monkeypatch.setattr(orchestrator, "WORKSPACE_ROOT", tmp_path / "workspace")
    from briefs import _common as _bc

    def _raise(*a, **k):
        raise FileNotFoundError(".ascendc_env not found at /nonexistent")

    monkeypatch.setattr(_bc, "load_env", _raise)
    rc = getattr(orchestrator, "_cmd_backward")(
        forward_spec=_fwd_spec(tmp_path, "mul"), lane=0, plan_only=True, cold_start=False,
    )
    assert rc == 2  # fail-fast, non-zero even on the --plan path

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Finalize hard-fail FSM-transition tests (2026-08-30, PR13 WP-A / A.4.2).

Every finalize hard-fail must land on a recorded FSM transition instead of a
bare ``HandlerResult.ret(7)`` (the old exit killed the process with no
transition, so resume re-ran the same failing promotion with no recovery
path — 2026-08-29 2_FFN_evo).  Routing rules:

- worker-fixable delivery-contract violations (unrecognized delivery files)
  → ``await_worker``, with the filenames and repair guidance in the rationale;
- harness-side gates (static safety) and KB-merge failures
  → ``await_user_decision``.

Run: cd src/scripts && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
     orchestrator/tests/ut/test_finalize_hard_fail_transitions.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/

import orchestrator  # noqa: E402,F401  (force module identity for fsm_context read-through)
import events  # noqa: E402
import finalize_pipeline  # noqa: E402
import fsm_phase_finalize as F  # noqa: E402
from fsm_phase_finalize import (  # noqa: E402
    _check_delivery_static_safety,
    _merge_finalize_knowledge,
    _route_finalize_to_done,
)
import kb_invoke  # noqa: E402
import state_executor  # noqa: E402


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "op"
    ws.mkdir()
    return ws


def _report(workspace: Path, errors: list[str]):
    return finalize_pipeline.FinalizeReport(
        op="op", workspace=workspace, archive_dir=workspace / "archive",
        errors=list(errors),
    )


def _last_transition(workspace: Path) -> dict:
    import json

    lines = (workspace / "state_transitions.jsonl").read_text().splitlines()
    return json.loads(lines[-1])


def test_unrecognized_delivery_files_route_to_await_worker(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    report = _report(ws, [
        "promote op_host/op_x_mystery.xyz: TileLang2AscendC delivery contains "
        "an unrecognized file that would be omitted: op_host/op_x_mystery.xyz",
        "promote kernel/op_kernel/strange.xyz: TileLang2AscendC delivery "
        "contains an unrecognized file that would be omitted: "
        "kernel/op_kernel/strange.xyz",
    ])
    result = _route_finalize_to_done(ws, lane=0, runtime_kwargs={}, finalize_report=report)
    assert result.action == "continue"
    assert state_executor.current_state(ws) == "await_worker"
    entry = _last_transition(ws)
    assert entry["from_state"] == "finalize"
    assert entry["to_state"] == "await_worker"
    # The rationale must carry the filenames and actionable repair guidance.
    assert "op_host/op_x_mystery.xyz" in entry["rationale"]
    assert "kernel/op_kernel/strange.xyz" in entry["rationale"]
    assert "delete" in entry["rationale"]


def test_mixed_promotion_errors_route_to_await_user_decision(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    report = _report(ws, [
        "promote op_host/op_x_mystery.xyz: TileLang2AscendC delivery contains "
        "an unrecognized file that would be omitted: op_host/op_x_mystery.xyz",
        "promote kernel/op_x.cpp → op_kernel/op_x.cpp: permission denied",
    ])
    result = _route_finalize_to_done(ws, lane=0, runtime_kwargs={}, finalize_report=report)
    assert result.action == "continue"
    assert state_executor.current_state(ws) == "await_user_decision"


def test_non_delivery_promotion_error_routes_to_await_user_decision(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    report = _report(ws, ["copy failed"])
    result = _route_finalize_to_done(ws, lane=0, runtime_kwargs={}, finalize_report=report)
    assert result.action == "continue"
    assert state_executor.current_state(ws) == "await_user_decision"


def test_static_safety_failure_routes_to_await_user_decision(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(
        F,
        "_run_delivery_static_check",
        lambda _ws: {"passed": False, "reports": [], "error": "boom"},
    )
    result = _check_delivery_static_safety(ws, lane=0)
    assert result is not None and result.action == "continue"
    assert state_executor.current_state(ws) == "await_user_decision"
    assert "boom" in _last_transition(ws)["rationale"]


def test_kb_merge_exception_routes_to_await_user_decision(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    (ws / "knowledge_update.md").write_text("validated knowledge\n" * 20)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)

    def _raise(_ws):
        raise RuntimeError("kb manager exploded")

    monkeypatch.setattr(kb_invoke, "merge_one", _raise)
    result = _merge_finalize_knowledge(ws, lane=0)
    assert result is not None and result.action == "continue"
    assert state_executor.current_state(ws) == "await_user_decision"
    assert "kb manager exploded" in _last_transition(ws)["rationale"]


def test_kb_merge_failure_routes_to_await_user_decision(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    (ws / "knowledge_update.md").write_text("validated knowledge\n" * 20)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(
        kb_invoke,
        "merge_one",
        lambda _ws: {"success": False, "log_entry": {"exit_code": 3}},
    )
    result = _merge_finalize_knowledge(ws, lane=0)
    assert result is not None and result.action == "continue"
    assert state_executor.current_state(ws) == "await_user_decision"


def test_kb_merge_success_still_passes_through(monkeypatch, tmp_path):
    ws = _workspace(tmp_path)
    (ws / "knowledge_update.md").write_text("validated knowledge\n" * 20)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(kb_invoke, "merge_one", lambda _ws: {"success": True})
    assert _merge_finalize_knowledge(ws, lane=0) is None

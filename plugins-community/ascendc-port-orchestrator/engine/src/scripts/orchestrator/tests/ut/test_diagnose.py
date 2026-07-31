# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for diagnose.py fleet aggregator (V2 #1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import diagnose  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Workspace with PROGRESS.md."""
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


def _seed(ws, *, transitions=None, envelopes=None, criticisms=None,
           kb_merges=None, schema_norms=None, verification=None):
    if transitions is not None:
        with open(ws / "state_transitions.jsonl", "w") as f:
            for t in transitions:
                f.write(json.dumps(t) + "\n")
    if envelopes is not None:
        with open(ws / ".cc_envelope_log.jsonl", "w") as f:
            for e in envelopes:
                f.write(json.dumps(e) + "\n")
    if criticisms is not None:
        with open(ws / ".critic_invoke_log.jsonl", "w") as f:
            for c in criticisms:
                f.write(json.dumps(c) + "\n")
    if kb_merges is not None:
        with open(ws / ".kb_merge_log.jsonl", "w") as f:
            for k in kb_merges:
                f.write(json.dumps(k) + "\n")
    if schema_norms is not None:
        with open(ws / ".schema_normalizations.log", "w") as f:
            for n in schema_norms:
                f.write(json.dumps(n) + "\n")
    if verification is not None:
        (ws / "verification.json").write_text(json.dumps(verification))


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------
def test_build_report_empty_workspace(ws):
    rep = diagnose.build_report("test_op", workspace=ws)
    assert rep.op == "test_op"
    assert rep.n_spawns == 0
    assert rep.total_cost_usd == 0
    assert rep.transitions == []


def test_build_report_with_envelopes(ws):
    _seed(ws, envelopes=[
        {"agent_type": "kw", "cost_usd": 1.5, "duration_ms": 60000,
         "permission_denials": []},
        {"agent_type": "ko", "cost_usd": 2.0, "duration_ms": 120000,
         "permission_denials": [{"tool_name": "Bash"}]},
    ])
    rep = diagnose.build_report("test", workspace=ws)
    assert rep.n_spawns == 2
    assert rep.total_cost_usd == 3.5
    assert rep.total_duration_s == 180
    assert rep.n_envelope_denials == 1


def test_build_report_with_transitions(ws):
    _seed(ws, transitions=[
        {"from_state": "init", "to_state": "await_worker", "handoff": "x",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
        {"from_state": "await_worker", "to_state": "await_optimizer", "handoff": "y",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
        {"from_state": "await_optimizer", "to_state": "finalize", "handoff": "z",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
    ])
    rep = diagnose.build_report("test", workspace=ws)
    assert rep.transitions == [
        ("init", "await_worker"),
        ("await_worker", "await_optimizer"),
        ("await_optimizer", "finalize"),
    ]


def test_build_report_critic_timed_out(ws):
    _seed(ws, criticisms=[
        {"trigger": "pre_phase_o4_first_spawn", "success": True, "timed_out": False},
        {"trigger": "pre_finalize", "success": False, "timed_out": True},
    ])
    rep = diagnose.build_report("test", workspace=ws)
    assert rep.n_critic_fired == 2
    assert rep.n_critic_timed_out == 1


def test_build_report_schema_rejects(ws):
    _seed(ws, schema_norms=[
        {"category": "SAFE", "field_path": "from"},
        {"category": "TERMINAL_REJECT", "field_path": "to_state"},
        {"category": "TERMINAL_REJECT", "field_path": "to_state"},
    ])
    rep = diagnose.build_report("test", workspace=ws)
    assert rep.n_schema_rejects == 2


def test_build_report_verification(ws):
    _seed(ws, verification={
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS"},
                      "pass_b": {"status": "PASS"}},
        "performance": {"status": "PASS", "ratio": 1.15},
    })
    rep = diagnose.build_report("test", workspace=ws)
    assert rep.pass_a == "PASS"
    assert rep.pass_b == "PASS"
    assert rep.perf_ratio == 1.15


# ---------------------------------------------------------------------------
# fleet_status
# ---------------------------------------------------------------------------
def test_fleet_status_empty(tmp_path):
    assert diagnose.fleet_status(root=tmp_path) == []


def test_fleet_status_skips_dirs_without_progress(tmp_path):
    (tmp_path / "no_progress").mkdir()
    (tmp_path / "with_progress").mkdir()
    (tmp_path / "with_progress" / "PROGRESS.md").write_text("# x\n")
    out = diagnose.fleet_status(root=tmp_path)
    assert len(out) == 1
    assert out[0].op == "with_progress"


def test_fleet_status_aggregates_multiple_ops(tmp_path):
    for op in ["op1", "op2"]:
        (tmp_path / op).mkdir()
        (tmp_path / op / "PROGRESS.md").write_text("# x\n")
        _seed(tmp_path / op, envelopes=[
            {"agent_type": "kw", "cost_usd": 1.0, "duration_ms": 30000,
             "permission_denials": []},
        ])
    reports = diagnose.fleet_status(root=tmp_path)
    assert len(reports) == 2
    assert all(r.n_spawns == 1 for r in reports)


# ---------------------------------------------------------------------------
# path_coverage
# ---------------------------------------------------------------------------
def test_path_coverage_unique_pairs(tmp_path):
    (tmp_path / "op1").mkdir()
    (tmp_path / "op1" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "op1", transitions=[
        {"from_state": "await_worker", "to_state": "finalize", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
    ])
    (tmp_path / "op2").mkdir()
    (tmp_path / "op2" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "op2", transitions=[
        {"from_state": "await_worker", "to_state": "await_optimizer", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
        {"from_state": "await_optimizer", "to_state": "finalize", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
    ])
    reports = diagnose.fleet_status(root=tmp_path)
    pc = diagnose.path_coverage(reports)
    pairs = pc["unique_pairs"]
    assert ("await_worker", "finalize") in pairs
    assert ("await_worker", "await_optimizer") in pairs
    assert ("await_optimizer", "finalize") in pairs
    assert "op2" in pc["ops_with_optimizer_path"]
    assert "op1" not in pc["ops_with_optimizer_path"]


def test_path_coverage_probe_path(tmp_path):
    (tmp_path / "op_p").mkdir()
    (tmp_path / "op_p" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "op_p", transitions=[
        {"from_state": "await_worker", "to_state": "await_probe", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
        {"from_state": "await_probe", "to_state": "await_worker", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
        {"from_state": "await_worker", "to_state": "finalize", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
    ])
    reports = diagnose.fleet_status(root=tmp_path)
    pc = diagnose.path_coverage(reports)
    assert "op_p" in pc["ops_with_probe_path"]


# ---------------------------------------------------------------------------
# cost_rollup
# ---------------------------------------------------------------------------
def test_cost_rollup(tmp_path):
    for op, cost in [("op1", 1.5), ("op2", 2.5)]:
        (tmp_path / op).mkdir()
        (tmp_path / op / "PROGRESS.md").write_text("# x\n")
        _seed(tmp_path / op, envelopes=[
            {"agent_type": "kw", "cost_usd": cost, "duration_ms": 60000,
             "permission_denials": []},
        ])
    reports = diagnose.fleet_status(root=tmp_path)
    cr = diagnose.cost_rollup(reports)
    assert cr["n_ops"] == 2
    assert cr["total_cost_usd"] == 4.0
    assert cr["total_spawns"] == 2
    assert cr["avg_cost_per_op"] == 2.0


# ---------------------------------------------------------------------------
# denial_audit
# ---------------------------------------------------------------------------
def test_denial_audit_returns_only_ops_with_denials(tmp_path):
    (tmp_path / "clean_op").mkdir()
    (tmp_path / "clean_op" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "clean_op", envelopes=[
        {"agent_type": "kw", "cost_usd": 1.0, "duration_ms": 60000,
         "permission_denials": []},
    ])
    (tmp_path / "dirty_op").mkdir()
    (tmp_path / "dirty_op" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "dirty_op", envelopes=[
        {"agent_type": "kw", "cost_usd": 1.0, "duration_ms": 60000,
         "permission_denials": [{"tool_name": "Bash"}, {"tool_name": "Bash"}]},
    ])
    reports = diagnose.fleet_status(root=tmp_path)
    audit = diagnose.denial_audit(reports)
    assert len(audit) == 1
    assert audit[0]["op"] == "dirty_op"
    assert audit[0]["n_denials"] == 2


# ---------------------------------------------------------------------------
# Formatters smoke
# ---------------------------------------------------------------------------
def test_format_fleet_table_empty():
    assert "no ops" in diagnose.format_fleet_table([])


def test_format_fleet_table_with_data(tmp_path):
    (tmp_path / "op1").mkdir()
    (tmp_path / "op1" / "PROGRESS.md").write_text("# x\n")
    _seed(tmp_path / "op1", verification={
        "precision": {"pass_a": {"status": "PASS"}, "pass_b": {"status": "PASS"}},
        "performance": {"status": "PASS", "ratio": 1.15},
    })
    reports = diagnose.fleet_status(root=tmp_path)
    table = diagnose.format_fleet_table(reports)
    assert "op1" in table
    assert "PASS" in table


def test_format_op_detail(ws):
    _seed(ws, envelopes=[
        {"agent_type": "kw", "cost_usd": 1.5, "duration_ms": 60000,
         "permission_denials": []},
    ], transitions=[
        {"from_state": "await_worker", "to_state": "finalize", "handoff": "",
         "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {}},
    ])
    rep = diagnose.build_report("test", workspace=ws)
    detail = diagnose.format_op_detail(rep)
    assert "test" in detail
    assert "spawns:           1" in detail
    assert "await_worker → finalize" in detail

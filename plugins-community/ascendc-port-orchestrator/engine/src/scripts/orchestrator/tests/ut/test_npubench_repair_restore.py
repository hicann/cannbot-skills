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
"""P1-1 (2026-08-28, DSH+v4pro §4.a) — repair-reset branches on failure class.

`_prepare_npubench_candidate_repair` used to unconditionally archive the
workspace candidate out of the worker-visible tree, forcing a clean
re-author on every repair (the oc-line 3h idle / cc-line stream-log replay
root cause).  Now:

- engine/infra class (evaluate_report status=ERROR or latest infra rollback)
  -> the ENGINE restores the newest verified archive back into the workspace
- candidate class (real precision MISMATCH, status=FAIL / non-infra rollback)
  -> legacy move-out + re-author semantics, unchanged
- missing/unreadable/corrupt evidence -> fail-closed to the legacy move-out
- graybox_answer_gate_respawn -> always move-out (the seal retry needs an
  answer-free workspace)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import orchestrator_coldstart as cs


def _coldstart_attr(name):
    # Resolve protected cold-start helpers at call time for monkeypatch safety.
    return getattr(cs, name)


def _ws(tmp_path, *, rollback_kind="infra", evaluate_status=None):
    """Workspace with a candidate tree + optional failure evidence."""
    ws = tmp_path / "3_Add"
    ws.mkdir()
    (ws / "kernel").mkdir()
    (ws / "kernel" / "add_kernel.cpp").write_text("// candidate v1\n")
    (ws / "model_new_ascendc.py").write_text("# candidate v1\n")
    if rollback_kind is not None:
        (ws / "state_transitions.jsonl").write_text(
            json.dumps({
                "ts": "2026-08-28T00:00:00Z",
                "from_state": "finalize",
                "to_state": "await_worker",
                "rollback_kind": rollback_kind,
            })
            + "\n"
        )
    if evaluate_status is not None:
        evidence = ws / "npubench_evidence"
        evidence.mkdir()
        (evidence / "evaluate_report.json").write_text(
            json.dumps({"status": evaluate_status})
        )
    return ws


def _repair(ws, monkeypatch, backup_root, **kwargs):
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(backup_root))
    return _coldstart_attr("_prepare_npubench_candidate_repair")(ws, **kwargs)


def _read_record(ws):
    return json.loads((ws / ".npubench_candidate_repair.json").read_text())


def test_archive_carries_sha256_manifest(tmp_path, monkeypatch):
    """Every repair archive gets an engine-owned manifest (restore contract)."""
    ws = _ws(tmp_path, rollback_kind=None)
    backup_root = tmp_path / "backups"

    record = _repair(ws, monkeypatch, backup_root)

    assert record["action"] == "archived"
    archive = backup_root / ws.name / record["archive_id"]
    manifest = json.loads((archive / "repair_manifest.json").read_text())
    assert manifest["schema"] == "cannbot.npubench_candidate_repair_manifest/v1"
    assert set(manifest["entries"]) == {"kernel/", "model_new_ascendc.py"}
    assert manifest["symlinks"] == []
    for rel, sha in manifest["files"].items():
        assert hashlib.sha256((archive / rel).read_bytes()).hexdigest() == sha


def test_infra_failure_restores_prior_tree(tmp_path, monkeypatch):
    """Second infra-class repair restores the first archived tree (race case)."""
    ws = _ws(tmp_path, rollback_kind=None)  # no evidence yet: plain archive
    backup_root = tmp_path / "backups"

    first = _repair(ws, monkeypatch, backup_root, failure_kind="candidate_contract")
    assert first["action"] == "archived"
    assert not (ws / "kernel").exists()

    # The O5 evaluation then raced the empty window and ERRORed; the next
    # repair reset must bring the archived tree back instead of forcing a
    # clean re-author.
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-28T00:00:01Z",
            "from_state": "finalize",
            "to_state": "await_worker",
            "rollback_kind": "infra",
        })
        + "\n"
    )
    second = _repair(ws, monkeypatch, backup_root, failure_kind="candidate_contract")

    assert second["action"] == "restored_prior_tree"
    assert second["restored_from"] == first["archive_id"]
    assert second["restore_sha256_verified"] is True
    assert (ws / "kernel" / "add_kernel.cpp").read_text() == "// candidate v1\n"
    assert (ws / "model_new_ascendc.py").read_text() == "# candidate v1\n"
    record = _read_record(ws)
    assert record["action"] == "restored_prior_tree"
    assert record["failure_class"] == "engine_infra"


def test_infra_failure_with_current_tree_keeps_latest(tmp_path, monkeypatch):
    """Infra-class reset with an in-workspace tree keeps that tree in place.

    The current tree is archived first (audit), so the newest archive holds
    the latest tree and the restore brings THAT back — never an older one.
    """
    ws = _ws(tmp_path)
    backup_root = tmp_path / "backups"

    record = _repair(ws, monkeypatch, backup_root, failure_kind="candidate_contract")

    assert record["action"] == "restored_prior_tree"
    assert record["restored_from"] == record["archive_id"]
    assert (ws / "kernel" / "add_kernel.cpp").read_text() == "// candidate v1\n"
    # The tree round-tripped through the archive: audit trail exists.
    archive = backup_root / ws.name / record["archive_id"]
    assert not (archive / "kernel").exists()  # moved back into the workspace


def test_evaluate_error_alone_marks_engine_infra(tmp_path, monkeypatch):
    """status=ERROR in evaluate_report.json is engine/infra-class evidence."""
    ws = _ws(tmp_path, rollback_kind=None, evaluate_status="ERROR")
    backup_root = tmp_path / "backups"

    record = _repair(ws, monkeypatch, backup_root)

    assert record["failure_class"] == "engine_infra"
    assert record["action"] == "restored_prior_tree"


def test_candidate_failure_keeps_moveout(tmp_path, monkeypatch):
    """A real precision MISMATCH (status=FAIL) keeps move-out + re-author."""
    ws = _ws(tmp_path, rollback_kind="infra", evaluate_status="FAIL")
    backup_root = tmp_path / "backups"

    first = _repair(ws, monkeypatch, backup_root)
    assert first["action"] == "archived"
    assert first["failure_class"] == "candidate"
    # Even with a restorable archive present, a candidate-class signal wins.
    second = _repair(ws, monkeypatch, backup_root)
    assert second["action"] == "archived"
    assert not (ws / "kernel").exists()


def test_missing_evidence_fails_closed(tmp_path, monkeypatch):
    """No evaluate report and no rollback record -> legacy move-out."""
    ws = _ws(tmp_path, rollback_kind=None)
    backup_root = tmp_path / "backups"

    record = _repair(ws, monkeypatch, backup_root)

    assert record["failure_class"] == "unknown"
    assert record["action"] == "archived"
    assert not (ws / "kernel").exists()


def test_tampered_archive_fails_closed(tmp_path, monkeypatch):
    """An archive whose bytes no longer match its manifest is never restored."""
    ws = _ws(tmp_path, rollback_kind=None)  # no evidence: plain archive
    backup_root = tmp_path / "backups"

    first = _repair(ws, monkeypatch, backup_root)
    assert first["action"] == "archived"
    # Tamper with the archived tree after the manifest was written.
    archived = backup_root / ws.name / first["archive_id"] / "kernel" / "add_kernel.cpp"
    archived.write_text("// tampered\n")
    # Worker re-authored a fresh tree meanwhile; the failure is infra class.
    (ws / "kernel").mkdir()
    (ws / "kernel" / "add_kernel.cpp").write_text("// candidate v2\n")
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"rollback_kind": "infra", "to_state": "await_worker"}) + "\n"
    )

    second = _repair(ws, monkeypatch, backup_root)

    # The tampered archive is skipped; the newer just-archived tree (v2) is
    # the newest verified one and is restored instead.
    assert second["action"] == "restored_prior_tree"
    assert second["restored_from"] != first["archive_id"]
    assert (ws / "kernel" / "add_kernel.cpp").read_text() == "// candidate v2\n"


def test_graybox_respawn_never_restores(tmp_path, monkeypatch):
    """The answer-gate respawn archive-retry requires an answer-free workspace."""
    ws = _ws(tmp_path)  # infra evidence present
    backup_root = tmp_path / "backups"

    record = _repair(
        ws, monkeypatch, backup_root, failure_kind="graybox_answer_gate_respawn"
    )

    assert record["action"] == "archived"
    assert record["failure_kind"] == "graybox_answer_gate_respawn"
    assert not (ws / "kernel").exists()


def test_symlinked_archive_is_unrestorable(tmp_path, monkeypatch):
    """A manifest recording symlinks makes its archive unrestorable."""
    ws = _ws(tmp_path, rollback_kind=None)  # unknown class: plain archive
    backup_root = tmp_path / "backups"

    first = _repair(ws, monkeypatch, backup_root)
    assert first["action"] == "archived"
    manifest_path = (
        backup_root / ws.name / first["archive_id"] / "repair_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["symlinks"] = ["kernel/evil"]
    manifest_path.write_text(json.dumps(manifest))

    # Now flip the evidence to engine/infra: the symlink-flagged archive must
    # not be restored; with nothing else restorable the reset does nothing
    # (workspace already empty) and reports the legacy archived outcome.
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"rollback_kind": "infra", "to_state": "await_worker"}) + "\n"
    )
    second = _repair(ws, monkeypatch, backup_root)
    assert second["action"] == "archived"
    assert not (ws / "kernel").exists()

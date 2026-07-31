# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-168 regression: audit_doc_needs_refresh must fire on a STALE audit.

Before the fix, `_ensure_audit_artifacts` re-ran the post-worker self-critic
only when `audit_self_critic_post_worker.md` was ABSENT. After a worker
respawn CLOSED the audit's own findings (e.g. kw-4 adding bf16 + Q-path
coverage), the audit doc was never refreshed even though verification.json /
edge_dataset.pt / pass_a_runner.py were rewritten. The stale PARTIAL verdict
was re-read on every finalize attempt → rollback → respawn → LOOP-BREAK →
await_user_decision (observed on top_k_top_p_sample, 2026-06-24).

The producer now uses `audit_doc_needs_refresh`, with a BROADER trigger set
than deleg_marker_needs_refresh (DEBT-155): the post-worker audit evaluates
kw-output *claimed results* (verification.json / edge_dataset.pt /
pass_a_runner.py) and docs (knowledge_update.md / PROGRESS.md), not just
kernel code. A coverage-only worker respawn changes none of the kernel files
but DOES change those result files — a kernel-only check would miss it.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import orchestrator  # noqa: E402


def _touch(p: Path, mtime: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (mtime, mtime))


AUDIT = Path("audit_self_critic_post_worker.md")


def test_audit_absent_needs_refresh(tmp_path):
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_audit_fresh_no_refresh(tmp_path):
    # all kw-output artifacts OLDER than the audit → not stale → no refresh.
    _touch(tmp_path / "kernel" / "k_kernel.h", 1000.0)
    _touch(tmp_path / "model_new_ascendc.py", 1000.0)
    _touch(tmp_path / "verification.json", 1000.0)
    _touch(tmp_path / "edge_dataset.pt", 1000.0)
    _touch(tmp_path / AUDIT, 2000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is False


def test_audit_stale_after_verification_json_change(tmp_path):
    # THE BUG (top_k_top_p_sample): kw respawn fixed coverage, rewrote
    # verification.json (PASS count went 16->... re-measured) but did NOT
    # touch kernel code. Kernel-only freshness would miss this; the audit
    # trigger set MUST include verification.json.
    _touch(tmp_path / "kernel" / "k_kernel.h", 1000.0)        # older than audit
    _touch(tmp_path / AUDIT, 2000.0)
    _touch(tmp_path / "verification.json", 3000.0)            # rewritten AFTER audit
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_audit_stale_after_edge_dataset_change(tmp_path):
    _touch(tmp_path / AUDIT, 1000.0)
    _touch(tmp_path / "edge_dataset.pt", 3000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_audit_stale_after_pass_a_runner_change(tmp_path):
    _touch(tmp_path / AUDIT, 1000.0)
    _touch(tmp_path / "pass_a_runner.py", 3000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_audit_stale_after_progress_or_kb_change(tmp_path):
    _touch(tmp_path / AUDIT, 1000.0)
    _touch(tmp_path / "PROGRESS.md", 3000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True
    _touch(tmp_path / "PROGRESS.md", 500.0)  # reset
    _touch(tmp_path / "knowledge_update.md", 3000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_audit_stale_after_kernel_rebuild_needs_refresh(tmp_path):
    # mirrors DEBT-155: kernel rebuilt after the audit → stale.
    _touch(tmp_path / AUDIT, 1000.0)
    _touch(tmp_path / "kernel" / "k_kernel.cpp", 3000.0)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is True


def test_within_slack_not_stale(tmp_path):
    # 1s slack (same as the gate + DEBT-155): verification.json < audit + 1.0
    # is NOT stale.
    _touch(tmp_path / AUDIT, 1000.0)
    _touch(tmp_path / "verification.json", 1000.5)
    assert orchestrator.audit_doc_needs_refresh(tmp_path) is False


def _seed_ws_for_ensure_audit(ws: Path) -> None:
    """Workspace state when _ensure_audit_artifacts fires (step-1
    delegation-scan short-circuits via the marker; only step-2 self-critic
    fires)."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": {"status": "N/A"},
    }))
    (ws / ".delegation_scan_passed").write_text("stub: scan passed\n")


def test_stale_audit_archived_then_regenerated(tmp_path, monkeypatch):
    """Integration: a STALE audit doc must be (a) archived as
    audit_self_critic_post_worker.STALE_<ts>.md (audit trail), then
    (b) regenerated. Pre-fix: stale doc was left in place verbatim and the
    gate re-read its stale verdict forever.
    """
    ws = tmp_path / "stale_op"
    _seed_ws_for_ensure_audit(ws)
    # seed an OLD (stale) audit doc + a NEW verification.json. Re-seed
    # verification.json LAST so its mtime > audit mtime (the stale trigger).
    stale_body = "# STALE audit — partial verdict from before the coverage fix\nVerdict: PARTIAL\n"
    audit = ws / AUDIT
    audit.write_text(stale_body)
    os.utime(audit, (1000.0, 1000.0))
    # rewrite verification.json so it's newer than the stale audit
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": {"status": "N/A"},
    }))
    assert orchestrator.audit_doc_needs_refresh(ws) is True

    fresh_body = "# FRESH audit\nVerdict: PASS\nC13: clean.\n"

    def fake_run(cmd, **kwargs):
        # skill writes the gate filename directly (happy path)
        (ws / AUDIT).write_text(fresh_body)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    # (b) regenerated
    assert audit.read_text() == fresh_body
    # (a) stale doc archived
    archived = [p for p in ws.iterdir() if p.name.startswith("audit_self_critic_post_worker.STALE_")]
    assert len(archived) == 1, f"expected exactly one STALE archive, got {[p.name for p in ws.iterdir()]}"
    assert archived[0].read_text() == stale_body


def test_fresh_audit_not_touched(tmp_path, monkeypatch):
    """Regression guard (was the pre-DEBT-168 behavior): a FRESH audit doc
    must NOT be archived/regenerated and the subprocess must NOT fire.
    """
    ws = tmp_path / "fresh_op"
    _seed_ws_for_ensure_audit(ws)
    fresh_body = "# fresh PASS audit\nVerdict: PASS\n"
    audit = ws / AUDIT
    audit.write_text(fresh_body)
    # all kw artifacts older than the audit → fresh → no refresh
    os.utime(ws / "verification.json", (1000.0, 1000.0))
    os.utime(audit, (2000.0, 2000.0))
    assert orchestrator.audit_doc_needs_refresh(ws) is False

    subprocess_fired = [False]

    def fake_run(cmd, **kwargs):
        subprocess_fired[0] = True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    assert not subprocess_fired[0], "fresh audit must skip the self-critic subprocess"
    assert audit.read_text() == fresh_body
    archived = [p for p in ws.iterdir() if "STALE_" in p.name]
    assert archived == [], "fresh audit must not be archived"

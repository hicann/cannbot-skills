# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""2026-05-13 bug fix: orchestrator._ensure_audit_artifacts must always
produce `audit_self_critic_post_worker.md` so the finalize gate can advance.

Background:
    The /aog-self-critic skill writes `self_critic_report.md` (its canonical
    filename, per SKILL.md line 412). The orchestrator's post-worker audit
    gate looks for `audit_self_critic_post_worker.md`. Before this fix,
    when the skill ran successfully but wrote the wrong filename, the gate
    blocked forever — observed in the ctc_loss_v3 e2e port run (2026-05-13).

Fix: post-subprocess, the orchestrator now (a) promotes a freshly-written
`self_critic_report.md` to the gate filename, or (b) falls back to wrapping
the subprocess stdout as the audit doc so the gate has SOMETHING to inspect.

This test exercises three cases without spawning a real `claude` subprocess.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import orchestrator  # noqa: E402


def _seed_workspace_at_finalize_audit_gate(ws: Path) -> None:
    """Workspace state when _ensure_audit_artifacts fires: kw returned with
    PASS verification.json + delegation scan already passed (so step 1
    short-circuits and only step 2 — self-critic — fires in this test)."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"status": "PASS",
                                                    "tier1_pass": 50, "total": 50}},
        "performance": {"status": "PASS", "ratio": 1.5},
    }))
    # Short-circuit step 1 (delegation scan) so subprocess.run patches in
    # each test capture only the step 2 (self-critic) invocation.
    (ws / ".delegation_scan_passed").write_text("test stub: scan passed\n")


def test_audit_doc_promoted_from_self_critic_report(tmp_path, monkeypatch):
    """Skill writes self_critic_report.md (canonical filename) but not the
    gate filename. Orchestrator must promote it so the gate advances.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_at_finalize_audit_gate(ws)

    skill_body = (
        "# self-critic — test_op\n\nVerdict: PASS\n\n"
        "C13: clean. C18: clean.\n"
    )

    def fake_run(cmd, **kwargs):
        # Simulate: skill ran successfully and wrote self_critic_report.md
        # (its canonical filename), NOT the audit_self_critic_post_worker.md
        # filename the orchestrator gate expects.
        (ws / "self_critic_report.md").write_text(skill_body)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    # delegation_scan path also calls subprocess; harmless given fake_run above.

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    audit_doc = ws / "audit_self_critic_post_worker.md"
    assert audit_doc.exists(), (
        "post-fix invariant: even when skill writes the wrong filename, "
        "orchestrator must materialize the gate filename"
    )
    content = audit_doc.read_text()
    assert "Auto-promoted from `self_critic_report.md`" in content, (
        "promoted file must self-identify so a future reader knows the "
        "origin (audit-trail clarity)"
    )
    assert skill_body in content, "must include the skill's actual verdict body"


def test_audit_doc_synthesized_from_stdout_when_skill_silent(tmp_path, monkeypatch):
    """Skill ran but wrote NO file at all (neither canonical nor gate name).
    Orchestrator must fall back to wrapping subprocess stdout.
    """
    ws = tmp_path / "test_op_silent"
    _seed_workspace_at_finalize_audit_gate(ws)

    stdout_payload = '{"type":"result","subtype":"success","result":"verdict: PASS — no findings"}'

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=stdout_payload, stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    audit_doc = ws / "audit_self_critic_post_worker.md"
    assert audit_doc.exists(), (
        "fallback invariant: silent skill must still result in an audit doc"
    )
    content = audit_doc.read_text()
    assert "Synthesized from subprocess stdout" in content
    assert stdout_payload in content
    assert "returncode: 0" in content


def test_audit_doc_unchanged_when_skill_writes_correct_filename(tmp_path, monkeypatch):
    """Happy path: skill writes the gate filename directly. Orchestrator
    must NOT overwrite it with promoted/synthesized content.
    """
    ws = tmp_path / "test_op_happy"
    _seed_workspace_at_finalize_audit_gate(ws)

    correct_body = "# audit — test_op_happy\nVerdict: PASS\nC13: clean.\n"

    def fake_run(cmd, **kwargs):
        # Skill writes the orchestrator's expected filename directly
        (ws / "audit_self_critic_post_worker.md").write_text(correct_body)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    audit_doc = ws / "audit_self_critic_post_worker.md"
    assert audit_doc.exists()
    assert audit_doc.read_text() == correct_body, (
        "happy path: must NOT wrap/clobber the skill's correct write"
    )


def test_audit_doc_not_recreated_when_already_present(tmp_path, monkeypatch):
    """If audit_doc already exists before _ensure_audit_artifacts runs,
    the subprocess must NOT fire (no LLM cost on re-entry).
    """
    ws = tmp_path / "test_op_idempotent"
    _seed_workspace_at_finalize_audit_gate(ws)
    pre_existing = "# pre-existing audit doc from earlier orchestrator pass\n"
    (ws / "audit_self_critic_post_worker.md").write_text(pre_existing)

    subprocess_fired = [False]

    def fake_run(cmd, **kwargs):
        subprocess_fired[0] = True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    getattr(orchestrator, "_ensure_audit_artifacts")(ws, lane=0)

    assert not subprocess_fired[0], (
        "idempotence: audit_doc.exists() short-circuit must skip the "
        "$1-2 / 3min self-critic subprocess"
    )
    assert (ws / "audit_self_critic_post_worker.md").read_text() == pre_existing

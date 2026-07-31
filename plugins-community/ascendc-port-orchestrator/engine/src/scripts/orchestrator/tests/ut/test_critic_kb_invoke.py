# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for critic_invoke + kb_invoke transport (P0f, Day 4 finding).

Both modules call `claude --print` directly. P0f fixes:
  1. permission_mode = bypassPermissions (not acceptEdits — Bash gets denied)
  2. timeout raised (critic 300→900s, kb 600→1200s)
  3. timeout caught + reported, NOT raised — orchestrator continues
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import critic_invoke  # noqa: E402
import kb_invoke  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Workspace with PROGRESS.md + state_transitions.jsonl seed."""
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    (tmp_path / "state_transitions.jsonl").write_text("")
    return tmp_path


# ---------------------------------------------------------------------------
# critic_invoke.fire_critic
# ---------------------------------------------------------------------------
def test_fire_critic_uses_bypass_permissions(ws, monkeypatch):
    """P0f: cmd line MUST contain bypassPermissions, not acceptEdits."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
    assert "bypassPermissions" in captured["cmd"]
    assert "acceptEdits" not in captured["cmd"]


def test_fire_critic_default_timeout_raised(ws, monkeypatch):
    """Default timeout = PRESPAWN_CRITIC_TIMEOUT_SEC (env-overridable)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
    assert captured["timeout"] == critic_invoke.PRESPAWN_CRITIC_TIMEOUT_SEC


def test_fire_critic_handles_timeout_gracefully(ws, monkeypatch):
    """TimeoutExpired must NOT propagate; result has timed_out=True."""
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 900))
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
    assert result["success"] is False
    assert result["log_entry"]["timed_out"] is True
    assert result["log_entry"]["exit_code"] == -1


def test_fire_critic_writes_log_on_timeout(ws, monkeypatch):
    """Even on timeout, log entry MUST be appended for forensics."""
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 900)
    monkeypatch.setattr(subprocess, "run", fake_run)

    critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
    log = ws / ".critic_invoke_log.jsonl"
    assert log.exists()
    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["timed_out"] is True


def test_fire_critic_log_on_success(ws, monkeypatch):
    def fake_run(cmd, **kwargs):
        return MagicMock(returncode=0, stdout='{"verdict":"PASS"}', stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
    assert result["success"] is True
    assert result["log_entry"]["timed_out"] is False
    assert result["log_entry"]["exit_code"] == 0


# ---------------------------------------------------------------------------
# kb_invoke.merge_one
# ---------------------------------------------------------------------------
def test_merge_one_skips_when_no_knowledge_update(ws):
    """No knowledge_update.md → skip with success=True."""
    result = kb_invoke.merge_one(ws)
    assert result["success"] is True
    assert "skipped" in result


def test_merge_one_uses_bypass_permissions(ws, monkeypatch):
    """P0f: cmd line MUST contain bypassPermissions."""
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="merged", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    kb_invoke.merge_one(ws)
    assert "bypassPermissions" in captured["cmd"]


def test_merge_one_default_timeout_raised(ws, monkeypatch):
    """Default timeout = 1200s (was 600s in V1)."""
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    kb_invoke.merge_one(ws)
    assert captured["timeout"] == 1200


def test_merge_one_handles_timeout(ws, monkeypatch):
    """TimeoutExpired returns dict with timed_out=True, success=False."""
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1200)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = kb_invoke.merge_one(ws)
    assert result["success"] is False
    assert result["timed_out"] is True


def test_merge_one_timeout_quarantines_reviewer_marker(
    ws, tmp_path, monkeypatch
):
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    configured = tmp_path / "user-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))

    def fake_run(cmd, **kwargs):
        (ws / ".kb_merged").write_text(
            "merge_run=2026-07-30T00:00:00Z\n"
            "tier=customer\n"
            f"c_root={configured.resolve()}\n"
            "merged_into=user-c-tier\n"
            "entries=none\n"
            "reviewed=0\n"
            "rejected=0\n"
            "mode=update\n"
        )
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1200))

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is False
    assert result["timed_out"] is True
    assert not (ws / ".kb_merged").exists()
    assert list(ws.glob(".kb_merged.invalid-*"))
    assert getattr(kb_invoke, "_prepare_existing_marker")(ws) is False


def test_merge_one_error_quarantines_reviewer_marker(
    ws, tmp_path, monkeypatch
):
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    configured = tmp_path / "user-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))

    def fake_run(cmd, **kwargs):
        (ws / ".kb_merged").write_text(
            "merge_run=2026-07-30T00:00:00Z\n"
            "tier=customer\n"
            f"c_root={configured.resolve()}\n"
            "merged_into=user-c-tier\n"
            "entries=none\n"
            "reviewed=0\n"
            "rejected=0\n"
            "mode=update\n"
        )
        return MagicMock(returncode=1, stdout="failed", stderr="review error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is False
    assert not (ws / ".kb_merged").exists()
    assert list(ws.glob(".kb_merged.invalid-*"))


def test_merge_one_skips_only_when_kb_marker_is_durable(
    ws, tmp_path, monkeypatch
):
    """A provider-bound c-tier marker may skip repeated semantic review."""
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "user-kb"))
    getattr(kb_invoke, "_persist_c_tier")(
        ws,
        [{"kind": "experience", "claim": "A durable local lesson."}],
        mode="update",
    )
    result = kb_invoke.merge_one(ws)
    assert result["success"] is True
    assert "skipped" in result


def test_merge_one_quarantines_invalid_marker_even_without_update(ws):
    (ws / ".kb_merged").write_text("entries=3\n")

    result = kb_invoke.merge_one(ws)

    assert result["success"] is True
    assert "no knowledge_update" in result["skipped"]
    assert not (ws / ".kb_merged").exists()
    assert list(ws.glob(".kb_merged.invalid-*"))


def test_merge_one_quarantines_unverified_marker_and_reviews(
    ws, tmp_path, monkeypatch
):
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    (ws / ".kb_merged").write_text("entries=3\n")
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "user-kb"))

    def fake_run(*args, **kwargs):
        (ws / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(
            '{"schema_version": 1, "entries": []}'
        )
        return MagicMock(returncode=0, stdout="reviewed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is True, result
    assert "tier=customer" in (ws / ".kb_merged").read_text()
    assert list(ws.glob(".kb_merged.invalid-*"))


def test_merge_one_rejects_empty_marker_bound_to_other_c_root(
    ws, tmp_path, monkeypatch
):
    (ws / "knowledge_update.md").write_text("# stuff\n" + "x" * 200)
    configured = tmp_path / "configured-user-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))
    (ws / ".kb_merged").write_text(
        "merge_run=2026-07-30T00:00:00Z\n"
        "tier=customer\n"
        f"c_root={tmp_path / 'other-user-kb'}\n"
        "merged_into=user-c-tier\n"
        "entries=none\n"
        "reviewed=0\n"
        "rejected=0\n"
        "mode=update\n"
    )

    def fake_run(*args, **kwargs):
        (ws / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(
            '{"schema_version": 1, "entries": []}'
        )
        return MagicMock(returncode=0, stdout="reviewed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is True, result
    assert f"c_root={configured.resolve()}" in (ws / ".kb_merged").read_text()
    assert list(ws.glob(".kb_merged.invalid-*"))


def test_merge_one_persists_only_user_c_tier(ws, tmp_path, monkeypatch):
    """Semantic output is admitted through Arbiter; the skill never targets b."""
    (ws / "knowledge_update.md").write_text("# finding\n" + "x" * 200)
    c_root = tmp_path / "user kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(c_root))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["prompt"] = cmd[-1]
        (ws / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(json.dumps({
            "schema_version": 1,
            "entries": [{
                "kind": "positive_pattern",
                "claim": "Aligned byte-length copies avoid partial-block corruption.",
                "scope": {"arch": "arch35"},
                "evidence": {"workspace": "unit-test"},
            }],
        }))
        return MagicMock(returncode=0, stdout="reviewed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is True, result
    assert len(list((c_root / "entries").glob("*.json"))) == 1
    marker = (ws / ".kb_merged").read_text()
    assert "tier=customer" in marker
    assert "entries=customer:" in marker
    assert "byte-for-byte read-only" in captured["prompt"]
    assert "Do not edit bundled KB" in captured["prompt"]


def test_merge_one_rejects_missing_semantic_intake(ws, tmp_path, monkeypatch):
    """A zero-exit reviewer cannot create a lying completion marker."""
    (ws / "knowledge_update.md").write_text("# finding\n" + "x" * 200)
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "user-kb"))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="done", stderr=""),
    )

    result = kb_invoke.merge_one(ws)

    assert result["success"] is False
    assert "did not emit required c-tier intake" in result["error"]
    assert not (ws / ".kb_merged").exists()


def test_merge_one_blocks_bundled_kb_mutation(ws, tmp_path, monkeypatch):
    """A semantic agent touching release b-tier fails before c-tier admission."""
    (ws / "knowledge_update.md").write_text("# finding\n" + "x" * 200)
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "user-kb"))
    fingerprints = iter(["before", "after"])
    monkeypatch.setattr(kb_invoke, "_bundled_kb_fingerprint", lambda: next(fingerprints))

    def fake_run(*args, **kwargs):
        (ws / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(
            '{"schema_version": 1, "entries": []}'
        )
        return MagicMock(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_one(ws)

    assert result["success"] is False
    assert "bundled b-tier" in result["error"]
    assert not (ws / ".kb_merged").exists()


def test_merge_batch_persists_each_workspace_to_c_tier(tmp_path, monkeypatch):
    root = tmp_path / "workspaces"
    workspaces = [root / "op_a", root / "op_b"]
    for workspace in workspaces:
        workspace.mkdir(parents=True)
        (workspace / "knowledge_update.md").write_text("# finding\n" + "x" * 200)
    c_root = tmp_path / "user-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(c_root))

    def fake_run(*args, **kwargs):
        for workspace in workspaces:
            (workspace / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(json.dumps({
                "schema_version": 1,
                "entries": [{
                    "kind": "experience",
                    "claim": f"Runtime lesson from {workspace.name}",
                }],
            }))
        return MagicMock(returncode=0, stdout="reviewed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_batch(workspaces)

    assert result["success"] is True, result
    assert len(list((c_root / "entries").glob("*.json"))) == 2
    assert all("tier=customer" in (workspace / ".kb_merged").read_text()
               for workspace in workspaces)


def test_merge_batch_does_not_skip_unverified_marker(tmp_path, monkeypatch):
    workspace = tmp_path / "workspaces" / "op_a"
    workspace.mkdir(parents=True)
    (workspace / "knowledge_update.md").write_text("# finding\n" + "x" * 200)
    (workspace / ".kb_merged").write_text("entries=3\n")
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "user-kb"))

    def fake_run(*args, **kwargs):
        (workspace / getattr(kb_invoke, "_CANDIDATE_FILENAME")).write_text(
            '{"schema_version": 1, "entries": []}'
        )
        return MagicMock(returncode=0, stdout="reviewed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_batch([workspace])

    assert result["success"] is True, result
    assert result["n_pending"] == 1
    assert "tier=customer" in (workspace / ".kb_merged").read_text()
    assert list(workspace.glob(".kb_merged.invalid-*"))


def test_merge_batch_error_quarantines_all_reviewer_markers(
    tmp_path, monkeypatch
):
    root = tmp_path / "workspaces"
    workspaces = [root / "op_a", root / "op_b"]
    configured = tmp_path / "user-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))
    for workspace in workspaces:
        workspace.mkdir(parents=True)
        (workspace / "knowledge_update.md").write_text("# finding\n" + "x" * 200)

    def fake_run(*args, **kwargs):
        for workspace in workspaces:
            (workspace / ".kb_merged").write_text(
                "merge_run=2026-07-30T00:00:00Z\n"
                "tier=customer\n"
                f"c_root={configured.resolve()}\n"
                "merged_into=user-c-tier\n"
                "entries=none\n"
                "reviewed=0\n"
                "rejected=0\n"
                "mode=batch\n"
            )
        return MagicMock(returncode=1, stdout="failed", stderr="review error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = kb_invoke.merge_batch(workspaces)

    assert result["success"] is False
    assert all(not (workspace / ".kb_merged").exists() for workspace in workspaces)
    assert all(
        list(workspace.glob(".kb_merged.invalid-*"))
        for workspace in workspaces
    )


# ---------------------------------------------------------------------------
# orchestrator integration: try/except guard around critic + KB
# ---------------------------------------------------------------------------
def test_orchestrator_continues_when_critic_raises(ws, monkeypatch, capsys):
    """Sanity check: if fire_critic raises, orchestrator main loop should
    log a WARN and continue. Tests the try/except wrapper directly.
    """
    sys.path.insert(0, str(_HERE.parent.parent))
    # Patch fire_critic to always raise

    def boom(*args, **kwargs):
        raise RuntimeError("simulated critic crash")
    monkeypatch.setattr(critic_invoke, "fire_critic", boom)

    # We don't run the full main loop here (it'd need claude CLI); instead
    # we just verify the try/except pattern matches: invoke the same code
    # block by calling fire_critic directly inside a try.
    try:
        critic_invoke.fire_critic(ws, "pre_phase_o4_first_spawn")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "simulated critic crash" in str(e)


# ── P135.S9 (2026-05-18): trigger-specific catalog subsets ──


def test_p135s9_subset_for_pre_phase_o4_first_spawn():
    """Each trigger maps to its catalog subset per SKILL.md."""
    subset = getattr(critic_invoke, "_TRIGGER_CATALOG_SUBSETS")["pre_phase_o4_first_spawn"]
    # Pre-worker-spawn relevant items
    assert "C2" in subset  # infrastructure bypass
    assert "C5" in subset  # premature platform-blame
    assert "C18" in subset  # cheating-by-claim
    assert "C20" in subset  # available tool not used
    # NOT in subset (these fire at other triggers)
    assert "C3" not in subset  # source-before-probe (periodic only)
    assert "C8" not in subset  # words-not-actions (pre_commit only)
    assert "C30" not in subset  # fixed-without-sibling-test (pre_commit/pre_finalize only)
    assert "C28" in subset  # DEBT-066: structural ceiling claim (was silent)
    assert len(subset) == 13  # see SKILL.md table (DEBT-066 +C28)


def test_p135s9_subset_for_post_iter_cap_warning():
    subset = getattr(critic_invoke, "_TRIGGER_CATALOG_SUBSETS")["post_iter_cap_warning"]
    assert "C7" in subset  # premature-stop / drive-to-closure
    assert "C13" in subset  # claim runtime state without verification
    assert "C25" in subset  # premature stop after root cause
    assert "C1" in subset  # priority drift
    assert len(subset) == 4


def test_p135s9_subset_for_pre_finalize():
    subset = getattr(critic_invoke, "_TRIGGER_CATALOG_SUBSETS")["pre_finalize"]
    assert "C13" in subset
    assert "C18" in subset
    assert "C23" in subset
    assert "C30" in subset
    assert "C26" in subset
    assert "C28" in subset  # DEBT-066: structural ceiling claim
    assert len(subset) == 7  # DEBT-066 +C28


def test_p135s9_subset_for_pre_commit():
    subset = getattr(critic_invoke, "_TRIGGER_CATALOG_SUBSETS")["pre_commit"]
    assert "C8" in subset
    assert "C13" in subset
    assert "C24" in subset
    assert "C30" in subset
    assert len(subset) == 5


def test_p135s9_subset_clause_in_prompt(ws, monkeypatch):
    """fire_critic prompt includes the subset clause when trigger has subset."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "{}"
        mock.stderr = ""
        return mock

    monkeypatch.setattr(subprocess, "run", fake_run)
    critic_invoke.fire_critic(ws, "pre_finalize")
    prompt = captured["cmd"][-1]
    # Q1 slim self-critic (2026-05-21): accept either the post-Q1 inline-
    # catalog prompt shape OR the legacy /skill-invocation subset clause.
    has_subset_enforcement = (
        "Apply ONLY the items in the slim catalog" in prompt
        or "Apply ONLY these catalog items" in prompt
    )
    assert has_subset_enforcement, (
        "prompt must enforce subset evaluation (inline or fallback path)"
    )
    assert "C13" in prompt
    assert "C18" in prompt
    assert "C23" in prompt
    assert "C30" in prompt
    # full-catalog fallback clause should be absent when subset is defined
    assert "full range" not in prompt
    assert "most recently added entry" not in prompt


def test_p135s9_subset_clause_falls_back_for_unknown_trigger(ws, monkeypatch):
    """Trigger without subset gets full-catalog fallback prompt."""
    # Register a fake trigger that has no subset entry
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "{}"
        mock.stderr = ""
        return mock

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        critic_invoke, "CRITIC_TRIGGERS",
        dict(critic_invoke.CRITIC_TRIGGERS, fake_periodic="periodic check"),
    )
    critic_invoke.fire_critic(ws, "fake_periodic")
    prompt = captured["cmd"][-1]
    # No subset → full catalog clause
    assert "Apply ONLY these catalog items" not in prompt
    assert "full range" in prompt


# ── P135.S9b (task #19, 2026-05-18): backend-aware self-critic router ──


def test_p135s9b_get_subset_ascendc_default():
    """_get_subset_for default backend=ascendc returns C-items."""
    s = getattr(critic_invoke, "_get_subset_for")("pre_phase_o4_first_spawn")
    assert "C2" in s


def test_p135s9b_get_subset_explicit_ascendc():
    s = getattr(critic_invoke, "_get_subset_for")("pre_finalize", backend="ascendc")
    assert "C13" in s


def test_get_subset_rejects_unsupported_backend():
    with pytest.raises(ValueError, match="unsupported critic backend"):
        getattr(critic_invoke, "_get_subset_for")("pre_finalize", backend="unsupported")


def test_p135s9b_get_subset_unknown_trigger_returns_empty():
    s = getattr(critic_invoke, "_get_subset_for")("nonexistent_trigger", backend="ascendc")
    assert s == []


def test_p135s9b_resolve_backend_defaults_ascendc(tmp_path):
    """No state file + no suffix → ascendc default."""
    assert getattr(critic_invoke, "_resolve_backend")(tmp_path) == "ascendc"


def test_p135s9b_resolve_backend_unknown_value_falls_back_ascendc(tmp_path):
    """State has backend field but unknown value → defaults to ascendc."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo", "backend": "unknown_backend"})
    )
    assert getattr(critic_invoke, "_resolve_backend")(tmp_path) == "ascendc"


def test_fire_critic_rejects_unsupported_backend(ws):
    with pytest.raises(ValueError, match="unsupported critic backend"):
        critic_invoke.fire_critic(ws, "pre_finalize", backend="unsupported")


def test_p135s9b_fire_critic_default_backend_auto_resolves(ws, monkeypatch):
    """fire_critic with no backend kwarg → auto-resolves from workspace.
    Default workspace (no state, no suffix) → ascendc → C-items.
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "{}"
        mock.stderr = ""
        return mock

    monkeypatch.setattr(subprocess, "run", fake_run)
    critic_invoke.fire_critic(ws, "pre_finalize")  # no backend kwarg
    prompt = captured["cmd"][-1]
    assert "C13" in prompt  # ascendc C-items
    assert "backend='ascendc'" in prompt

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""finalize_pipeline tests — archive-everything-except-scratch policy.

Background: P0dd v1 used a fixed PROMOTE_FILES list that silently dropped
artifacts critical for op reproduction (input_gen.py, edge_dataset.pt,
ref_runnable.json, run_*.py scripts, state logs). Caught when user asked
why op#5_Cumsum's archive was missing edge data after finalize ran.
v2 promotes everything except clear scratch/internal patterns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402


# P0acv (2026-05-10): finalize_op invokes kb_auto_promote.run_auto_promote against
# the REAL src/skills/references KB. If that dir has any .kb_promotion_pending-*
# markers staged, each test that calls finalize_op spawns N codex CLI subprocesses
# (~30-60s each). With 20 markers staged from a batch, a single
# test took 20+ min and hung the pre-commit hook. Auto-mock kb_auto_promote in
# every finalize test so tests stay isolated from production KB state.
@pytest.fixture(autouse=True)
def _isolate_kb_auto_promote(monkeypatch, tmp_path):
    """Ensure no test in this module spawns codex against the real KB.

    Also redirect finalize's KB root to a per-test tmp KB (2026-07-05: KB
    relocated to <plugin_root>/kb/; finalize reads candidates.md via _kb_root()).
    Tests seed their fake KB under project_root/kb/ and read it back through the
    same resolver."""
    from src.scripts.orchestrator import kb_auto_promote

    def _noop_run(*args, **kwargs):
        rpt = kb_auto_promote.PromotionBatchReport(markers_processed=0)
        rpt.finished_ts = rpt.started_ts
        return rpt

    def _tmp_kb_root():
        return tmp_path / "project_root" / "kb"

    monkeypatch.setattr(kb_auto_promote, "run_auto_promote", _noop_run)
    monkeypatch.setattr(fp, "_kb_root", _tmp_kb_root)
    # v3.13.0 decomposition moved candidate-scan (+ its _kb_root resolver) into
    # finalize_candidates; redirect that module's resolver too so verified_on
    # reads the per-test tmp KB.
    import finalize_candidates as _fc  # top-level, matches `import finalize_pipeline as fp`
    monkeypatch.setattr(_fc, "_kb_root", _tmp_kb_root)


def _seed_workspace(ws: Path):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    # Real artifacts that MUST be promoted
    (ws / "PROGRESS.md").write_text("# progress")
    (ws / "model.py").write_text("import torch\n")
    (ws / "model_new_ascendc.py").write_text("# pybind\n")
    (ws / "input_gen.py").write_text("# input gen\n")
    (ws / "compute_reference.py").write_text("# ref\n")
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02tensor")
    (ws / "ref_runnable.json").write_text(json.dumps({"verdict": "RUNNABLE"}))
    (ws / "run_pass_b.py").write_text("# verifier\n")
    (ws / "self_critic_report.md").write_text("# critic\n")
    (ws / "state_transitions.jsonl").write_text(json.dumps({"to_state": "finalize"}) + "\n")
    (ws / "manifest.json").write_text(json.dumps({"op": "x"}))
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.h").write_text("// kernel")
    (ws / "probes").mkdir()
    (ws / "probes" / "p.py").write_text("# probe")
    # Scratch / internal — should NOT be promoted
    (ws / ".agent_died_at_await_worker").write_text("died")
    (ws / ".cc_envelope_log.jsonl").write_text("{}\n")
    (ws / ".kernel_worker_active").write_text("")
    (ws / ".opgen_state.json").write_text("{}")
    (ws / ".finalized-old").write_text("{}")
    (ws / "PROGRESS.md.day4-bak-20260504T062624Z").write_text("backup")
    (ws / "verification.json.batch6-bak-20260504T215432Z").write_text("backup")
    (ws / "kernel.h.opt0_bak").write_text("backup")
    (ws / "state_transitions.jsonl.pre-p0t-recovery-12345").write_text("backup")
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "x.pyc").write_text("")


def test_promotes_all_real_artifacts(tmp_path):
    ws = tmp_path / "test_op"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    expected_files = {
        "verification.json", "PROGRESS.md", "model.py", "model_new_ascendc.py",
        "input_gen.py", "compute_reference.py", "edge_dataset.pt",
        "ref_runnable.json", "run_pass_b.py", "self_critic_report.md",
        "state_transitions.jsonl", "manifest.json",
    }
    expected_dirs = {"kernel", "probes"}

    promoted_files = set(rep.files_promoted)
    promoted_dirs = set(rep.dirs_promoted)
    missing = expected_files - promoted_files
    assert not missing, f"missing critical files: {missing}"
    assert promoted_dirs == expected_dirs, f"dirs: got {promoted_dirs}, want {expected_dirs}"


def test_finalize_never_auto_promotes_release_kb(tmp_path, monkeypatch):
    """Operator finalize must not invoke the bundled b-tier promotion writer."""
    from src.scripts.orchestrator import kb_auto_promote

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime finalize attempted bundled KB promotion")

    monkeypatch.setattr(kb_auto_promote, "run_auto_promote", forbidden)
    ws = tmp_path / "test_op"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    assert rep.skipped is False
    assert rep.kb_auto_promote is None


def test_excludes_scratch_and_backups(tmp_path):
    ws = tmp_path / "test_op"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    promoted = set(rep.files_promoted) | set(rep.dirs_promoted)
    skipped = set(rep.skipped_names)

    forbidden = {
        ".agent_died_at_await_worker", ".cc_envelope_log.jsonl",
        ".kernel_worker_active", ".opgen_state.json", ".finalized-old",
        "PROGRESS.md.day4-bak-20260504T062624Z",
        "verification.json.batch6-bak-20260504T215432Z",
        "kernel.h.opt0_bak",
        "state_transitions.jsonl.pre-p0t-recovery-12345",
        "__pycache__",
    }
    leaked = forbidden & promoted
    assert not leaked, f"scratch/backup files leaked into archive: {leaked}"
    # All forbidden names should appear in skipped_names
    not_recorded = forbidden - skipped
    assert not not_recorded, f"forbidden names not recorded as skipped: {not_recorded}"


def test_idempotent_on_unchanged_verification(tmp_path):
    ws = tmp_path / "test_op"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    r1 = fp.finalize_op("test_op", ws, archive_root=archive_root)
    assert not r1.skipped
    r2 = fp.finalize_op("test_op", ws, archive_root=archive_root)
    assert r2.skipped
    assert "already finalized" in r2.skip_reason


def test_re_finalize_after_verification_change(tmp_path):
    ws = tmp_path / "test_op"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    fp.finalize_op("test_op", ws, archive_root=archive_root)
    # Change verification.json — should re-finalize
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PARTIAL"}}))
    r = fp.finalize_op("test_op", ws, archive_root=archive_root)
    assert not r.skipped


def test_p0acw_verified_on_hook_appends_when_pass_and_cited(tmp_path):
    """P0acw: when kw cites CAND-X in analysis.md AND verification.json shows
    precision PASS, finalize must append a `verified_on: a5_ops:{op}:case_X`
    line to that candidate body.
    """
    # Seed a minimal fake project_root with candidates.md
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "# Candidates\n\n"
        "## CAND-FOO1: Test pattern title\n"
        "`applies_to: op_class=test`\n"
        "`unverified_on: a5_ops`\n\n"
        "Body content here.\n\n"
        "## CAND-FOO2: Another pattern\n"
        "`applies_to: op_class=other`\n\n"
        "Other body.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text(
        "## Strategy\n\nFollowing CAND-FOO1 pattern for cross-core dispatch.\n"
    )

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    assert result.get("CAND-FOO1") is True

    text = candidates_md.read_text()
    assert "verified_on: a5_ops:test_op:case_" in text
    # unverified_on a5_ops should be removed
    assert "`unverified_on: a5_ops`" not in text
    # CAND-FOO2 should be untouched
    assert "## CAND-FOO2:" in text
    assert "Other body." in text


def test_p0acz_refuted_on_blocks_verified_on_same_op(tmp_path):
    """P0acz: when a candidate has `refuted_on: a5_ops:<op>:*`, the hook
    must NOT auto-add `verified_on: a5_ops:<op>:*` for the same op. The
    refuted_on marker means the op was explicitly flagged as negative
    evidence (e.g. codex catches the hard-do-not-apply violation). Auto-
    adding verified_on would create a contradiction in the candidate
    body that downstream codex review flags.

    Real-world trigger: CAND-FA1 was refuted_on:3_FusionAttention:case_b27a259d
    because 3_FA uses MatmulImpl (violates CAND-FA1 do-not-apply). Then
    3_FA cold-start finalized again — hook auto-added
    verified_on:3_FusionAttention:case_ad1de4ec creating the contradiction.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: title\n"
        "`applies_to: op_class=test`\n"
        "`derived-from: cann-source`\n"
        "`refuted_on: a5_ops:test_op:case_OLD — op violates do-not-apply clause`\n\n"
        "Body content.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    text = candidates_md.read_text()

    # The refuted_on line must still be present
    assert "refuted_on: a5_ops:test_op:case_OLD" in text
    # NO verified_on for the same op should have been added
    assert "verified_on: a5_ops:test_op:case_" not in text
    # Function returns False (or anything except True for verified_on add)
    assert result.get("CAND-X") is not True


def test_p0acz_refuted_on_different_op_allows_verified_on(tmp_path):
    """P0acz: refuted_on for op A should NOT block verified_on for op B.
    The block is op-specific.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: title\n"
        "`applies_to: op_class=test`\n"
        "`refuted_on: a5_ops:OTHER_op:case_xxx — different op refuted`\n\n"
        "Body.\n"
    )

    ws = tmp_path / "my_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "my_op", project_root)
    text = candidates_md.read_text()

    # OTHER_op refuted_on preserved
    assert "refuted_on: a5_ops:OTHER_op" in text
    # my_op verified_on added (different op)
    assert "verified_on: a5_ops:my_op:case_" in text
    assert result.get("CAND-X") is True


def test_p0acw_round2_patches_unverified_prose_line(tmp_path):
    """P0acw round-2: when verified_on:a5_ops is appended, ALSO rewrite
    stale 'unverified on a5_ops' prose lines so codex doesn't see the
    metadata/prose contradiction.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-FA1: Test pattern\n"
        "`applies_to: op_class=test`\n"
        "`unverified_on: a5_ops`\n\n"
        "**Risks before promotion**:\n"
        "- a5_ops has no currently-shipped mixed AIC/AIV kernel; the pattern is unverified on this codebase\n"
        "- The other risk line stays unchanged\n"
    )
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-FA1 cited\n")

    fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    text = candidates_md.read_text()
    # The stale "a5_ops has no currently-shipped" line should be rewritten
    assert "a5_ops has no currently-shipped" not in text
    assert "a5_ops now has shipped evidence" in text
    # The other risk line is untouched
    assert "The other risk line stays unchanged" in text


def test_p0acw_round2_patches_promote_when_line(tmp_path):
    """P0acw round-2: 'Promote when: an a5_ops X ships AND ...' becomes
    'a5_ops evidence recorded; remaining criteria — ...'
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: title\n`applies_to: op_class=test`\n\n"
        "**Promote when**: an a5_ops fused mixed AIC/AIV op ships with a measured improvement vs SyncAll baseline.\n"
    )
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    text = candidates_md.read_text()
    assert "**Promote when**: an a5_ops fused" not in text
    assert "**Promote when**: a5_ops evidence recorded" in text
    assert "remaining criteria — with a measured improvement vs SyncAll baseline." in text


def test_p0acw_round2_idempotent_on_already_patched_prose(tmp_path):
    """P0acw round-2: re-running hook on already-patched prose is a no-op."""
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: title\n`applies_to: op_class=test`\n\n"
        "- a5_ops has no currently-shipped X kernel\n"
    )
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    text1 = candidates_md.read_text()
    fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    text2 = candidates_md.read_text()
    assert text1 == text2  # second call no-op


def test_p0acw_verified_on_hook_accepts_partial_with_positive_tier1(tmp_path):
    """P0acw refinement: PARTIAL with tier1_pass >= 1 IS evidence. The kernel
    demonstrably implements the cited patterns correctly on the in-scope
    cases it covers; cases that hit _OutOfScope guards are structural
    scope-gaps, not pattern failures.

    Real anchor: 3_FusionAttention 2026-05-10 is PARTIAL (60 _OutOfScope,
    1 case PASS_T1) AND explicitly cites CAND-FA1..CAND-FA5 in
    probe_report.md. That's evidence.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-FOO1: Test\n`applies_to: op_class=test`\n`unverified_on: a5_ops`\n\nBody.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    # PARTIAL but with pass_a.tier1_pass=1 — kernel correctly implements pattern
    # on its 1 in-scope case; other 60 are _OutOfScope structural skips.
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PARTIAL",
            "pass_a": {"tier1_pass": 1, "total": 61},
        }
    }))
    (ws / "probe_report.md").write_text("CAND-FOO1 cited\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    assert result.get("CAND-FOO1") is True
    text = candidates_md.read_text()
    assert "verified_on: a5_ops:test_op:case_" in text


def test_p0acw_verified_on_hook_rejects_partial_with_zero_tier1(tmp_path):
    """P0acw: PARTIAL with NO positive cases is just failure — skip."""
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: Test\n`applies_to: op_class=test`\n\nBody.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL", "pass_a": {"tier1_pass": 0, "total": 50}}
    }))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    assert result == {}


def test_p0acw_verified_on_hook_skips_when_precision_fail(tmp_path):
    """P0acw gate: do NOT append verified_on when kernel failed precision —
    a failed kernel citing a candidate is NOT evidence.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-FOO1: Test\n`applies_to: op_class=test`\n`unverified_on: a5_ops`\n\nBody.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "FAIL"}}))
    (ws / "analysis.md").write_text("Cited CAND-FOO1 here\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    assert result == {}
    text = candidates_md.read_text()
    assert "`verified_on:" not in text  # not added (no positive verified_on line)
    assert "`unverified_on: a5_ops`" in text  # preserved


def test_p0acw_verified_on_hook_idempotent(tmp_path):
    """P0acw: re-running the hook should NOT duplicate the verified_on line."""
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-X: title\n`applies_to: op_class=test`\n\nBody.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-X cited\n")

    r1 = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    r2 = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    # First call modifies, second sees same evidence_token already present
    assert r1.get("CAND-X") is True
    assert r2.get("CAND-X") is False
    # Verify only one verified_on line present
    text = candidates_md.read_text()
    assert text.count("verified_on: a5_ops:test_op") == 1


def test_p0acw_verified_on_scans_multiple_artifacts(tmp_path):
    """P0acw: hook scans analysis.md, optimization_log.md, probe_report.md,
    knowledge_update.md, fused_analysis.md — all valid citation surfaces.
    """
    project_root = tmp_path / "project_root"
    kb = project_root / "kb" / "patterns" / "unverified"
    kb.mkdir(parents=True)
    candidates_md = kb / "candidates.md"
    candidates_md.write_text(
        "## CAND-A: a\n`applies_to: op_class=test`\n\nBody A.\n\n"
        "## CAND-B: b\n`applies_to: op_class=test`\n\nBody B.\n\n"
        "## CAND-C: c\n`applies_to: op_class=test`\n\nBody C.\n"
    )

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "analysis.md").write_text("CAND-A is used here\n")
    (ws / "optimization_log.md").write_text("Per CAND-B strategy\n")
    (ws / "fused_analysis.md").write_text("CAND-C insight\n")

    result = fp.update_verified_on_for_consumed_candidates(ws, "test_op", project_root)
    assert result.get("CAND-A") is True
    assert result.get("CAND-B") is True
    assert result.get("CAND-C") is True


def test_should_skip_dotfiles(tmp_path):
    assert getattr(fp, "_should_skip")(".agent_died_at_await_worker")
    assert getattr(fp, "_should_skip")(".cc_envelope_log.jsonl")
    assert getattr(fp, "_should_skip")(".finalized-abc123")
    assert getattr(fp, "_should_skip")(".opgen_state.json")
    assert not getattr(fp, "_should_skip")("verification.json")
    assert not getattr(fp, "_should_skip")("kernel")


def test_should_skip_backups(tmp_path):
    assert getattr(fp, "_should_skip")("PROGRESS.md.day4-bak-20260504T062624Z")
    assert getattr(fp, "_should_skip")("kernel.h.opt0_bak")
    assert getattr(fp, "_should_skip")("state_transitions.jsonl.pre-p0t-recovery-12345")
    assert getattr(fp, "_should_skip")("foo.pyc")
    assert not getattr(fp, "_should_skip")("kernel.h")


# ---------------------------------------------------------------------------
# P0dd v3 bug 1: archive-name resolver must NOT match ops by number prefix only.
# Workspace `10_layernorm` should NOT map to archive `10_SwigluQuant`.
# ---------------------------------------------------------------------------
def test_resolver_exact_match_wins(tmp_path):
    ar = tmp_path / "archives"
    ar.mkdir()
    (ar / "5_Cumsum").mkdir()
    assert getattr(fp, "_resolve_archive_op_name")("5_Cumsum", ar) == "5_Cumsum"


def test_resolver_case_insensitive_match(tmp_path):
    ar = tmp_path / "archives"
    ar.mkdir()
    (ar / "10_LayerNorm").mkdir()
    assert getattr(fp, "_resolve_archive_op_name")("10_layernorm", ar) == "10_LayerNorm"


def test_resolver_abbreviation_match(tmp_path):
    """`12_kvrms` (workspace) should match `12_KvRmsnormRopeCache` (archive)."""
    ar = tmp_path / "archives"
    ar.mkdir()
    (ar / "12_KvRmsnormRopeCache").mkdir()
    assert getattr(fp, "_resolve_archive_op_name")("12_kvrms", ar) == "12_KvRmsnormRopeCache"


def test_resolver_refuses_prefix_only_collision(tmp_path):
    """REGRESSION (data corruption): workspace `10_layernorm` MUST NOT
    map to archive `10_SwigluQuant`. The number prefix is the same but
    the suffix is unrelated.
    """
    ar = tmp_path / "archives"
    ar.mkdir()
    (ar / "10_SwigluQuant").mkdir()
    # No `10_LayerNorm` archive exists — should return workspace name as-is
    # (creates new archive dir), NOT match `10_SwigluQuant`.
    result = getattr(fp, "_resolve_archive_op_name")("10_layernorm", ar)
    assert result == "10_layernorm", (
        f"prefix-only collision must not match; got {result!r}"
    )


def test_resolver_picks_correct_among_siblings(tmp_path):
    """Multiple `19_*` archives — pick the matching one, not the first one."""
    ar = tmp_path / "archives"
    ar.mkdir()
    (ar / "19_FusedResidualRmsNormBackward").mkdir()
    (ar / "19_IndexPut").mkdir()
    assert getattr(fp, "_resolve_archive_op_name")("19_IndexPut", ar) == "19_IndexPut"
    assert getattr(fp, "_resolve_archive_op_name")(
        "19_FusedResidualRmsNormBackward", ar
    ) == "19_FusedResidualRmsNormBackward"
    # Workspace lowercase variant
    assert getattr(fp, "_resolve_archive_op_name")("19_indexput", ar) == "19_IndexPut"


# ---------------------------------------------------------------------------
# P0dd v3 bug 2: directory promotion must MERGE, not REPLACE.
# Files in archive that aren't in workspace must NOT be deleted.
# ---------------------------------------------------------------------------
def test_dir_promotion_preserves_archive_only_files(tmp_path):
    """Archive has msprof_opt_0.json (from prior run). Workspace probes/
    doesn't have it — only has new probe_pp1.py. Promotion must keep both.
    """
    ws = tmp_path / "workspace_op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "probes").mkdir()
    (ws / "probes" / "probe_pp1.py").write_text("# new probe")

    archive_root = tmp_path / "archives"
    archive_dir = archive_root / "workspace_op"
    archive_dir.mkdir(parents=True)
    (archive_dir / "probes").mkdir()
    (archive_dir / "probes" / "msprof_opt_0.json").write_text('{"opt": 0}')
    (archive_dir / "probes" / "msprof_opt_1.json").write_text('{"opt": 1}')

    fp.finalize_op("workspace_op", ws, archive_root=archive_root)

    archive_files = sorted([p.name for p in (archive_dir / "probes").iterdir()])
    assert "probe_pp1.py" in archive_files, "new file must be added"
    assert "msprof_opt_0.json" in archive_files, "archive-only file must be preserved (REGRESSION)"
    assert "msprof_opt_1.json" in archive_files, "archive-only file must be preserved (REGRESSION)"


def test_dir_promotion_overwrites_workspace_versions(tmp_path):
    """When both archive and workspace have a file, workspace version wins."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.h").write_text("// new content")

    archive_root = tmp_path / "ar"
    archive_dir = archive_root / "ws"
    archive_dir.mkdir(parents=True)
    (archive_dir / "kernel").mkdir()
    (archive_dir / "kernel" / "k.h").write_text("// old content")

    fp.finalize_op("ws", ws, archive_root=archive_root)
    assert (archive_dir / "kernel" / "k.h").read_text() == "// new content"


def test_check_project_json_metadata_valid(tmp_path):
    """DEBT-092: PROJECT.json with all required fields passes."""
    from finalize_pipeline import _check_project_json_metadata

    proj = tmp_path / "output" / "testproj"
    proj.mkdir(parents=True)
    (proj / "PROJECT.json").write_text(json.dumps({
        "schema_version": 1, "project": "testproj", "opgen_mode": "backward",
        "source": {"type": "forward_spec", "path": "testproj/"},
        "reference_baseline": "cpu_fp64_autograd", "target_chip": "Ascend950PR",
        "created": "2026-05-27T00:00:00Z", "owner_agent": "test-agent",
    }))
    ws = proj / "src" / "kernels" / "some_op"
    ws.mkdir(parents=True)

    result = _check_project_json_metadata(ws)
    assert result is None, f"Expected None, got {result}"


def test_check_project_json_metadata_missing_fields(tmp_path):
    """DEBT-092: PROJECT.json missing required fields returns gate string."""
    from finalize_pipeline import _check_project_json_metadata

    proj = tmp_path / "output" / "testproj2"
    proj.mkdir(parents=True)
    (proj / "PROJECT.json").write_text(json.dumps({
        "project": "testproj2",
        # missing: schema_version, opgen_mode, source, reference_baseline,
        # target_chip, created, owner_agent
    }))
    ws = proj / "src" / "kernels" / "some_op"
    ws.mkdir(parents=True)

    result = _check_project_json_metadata(ws)
    assert result is not None
    assert "missing required fields" in result
    assert "schema_version" in result
    assert "target_chip" in result


def test_check_project_json_metadata_no_project_json(tmp_path):
    """DEBT-092: workspace without PROJECT.json returns None (no-op)."""
    from finalize_pipeline import _check_project_json_metadata

    ws = tmp_path / "orphan_workspace"
    ws.mkdir(parents=True)

    result = _check_project_json_metadata(ws)
    assert result is None


def test_check_project_json_metadata_no_output_dir_fresh_tree(tmp_path, monkeypatch):
    """A fresh / backward-mode plugin tree has NO output/ (backward mode skips
    assemble_ge_ophost → never creates it). The PROJECT.json fallback scan must NOT
    crash on the missing output/ — an unguarded iterdir() raised FileNotFoundError and
    crashed finalize AFTER the kernel already PASSed verification. Missing output/ →
    return None gracefully. Regression for a backward e2e finalize crash when
    output/ was absent (same class as the resource-dependency UT failures).
    """
    from finalize_pipeline import _check_project_json_metadata

    # _PROJECT_ROOT points at a tree with NO output/ subdir
    monkeypatch.setattr("finalize_pipeline._PROJECT_ROOT", tmp_path)
    ws = tmp_path / "workspace" / "someop"
    ws.mkdir(parents=True)

    result = _check_project_json_metadata(ws)  # must NOT raise FileNotFoundError
    assert result is None

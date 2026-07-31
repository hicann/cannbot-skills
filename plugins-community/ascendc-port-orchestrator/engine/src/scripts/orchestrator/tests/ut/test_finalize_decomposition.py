# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""UT for the DEBT-201 finalize_pipeline decomposition (2026-07-06).

finalize_pipeline.py was split into cohesive sibling modules:
  * finalize_shared.py      — pure leaf helpers + marker constants cross-imported
                              by finalize_pipeline AND the finalize_checks_*
                              siblings (housing them here breaks the pre-existing
                              finalize_pipeline<->finalize_checks_* import cycle).
  * finalize_readme.py      — archive README rendering.
  * finalize_ge_ophost.py   — GE op_host template assembly.
  * finalize_candidates.py  — KB-candidate verified-on tracking + archive op-name.

This file:
  1. Locks the split as BEHAVIOR-NEUTRAL (re-exported same-object; the two
     monkeypatched names stay defined in finalize_pipeline).
  2. Fills the direct-coverage gaps for pure helpers that had no dedicated UT
     before the split.
  3. Verifies the finalize_checks_* siblings now source the shared helpers from
     finalize_shared (cycle-break), not from finalize_pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # orchestrator/

import finalize_pipeline as fp  # noqa: E402
import finalize_shared as fs  # noqa: E402
import finalize_readme as fr  # noqa: E402
import finalize_ge_ophost as fg  # noqa: E402
import finalize_candidates as fc  # noqa: E402
import finalize_dispatch as fd  # noqa: E402


_REEXPORTS = {
    fs: ("_verification_hash", "_is_negative_assertion", "_is_negative_assertion_window",
         "_is_kernel_caused_context_window", "_benchmark_case_count", "_kb_writeup_body_len",
         "_is_harness_internal", "_is_v220_ec41_output_pad_exempt", "_has_profiler_csv_method",
         "_HARNESS_INTERNAL_FILES"),
    fr: ("_render_verification_conclusion", "_assemble_readme", "_write_archive_readme"),
    fg: ("assemble_ge_ophost",),
    fc: ("_resolve_archive_op_name", "_scan_workspace_for_candidate_refs",
         "_patch_evidence_prose", "_append_verified_on",
         "update_verified_on_for_consumed_candidates"),
    fd: ("_get_active_plugin", "_run_plugin_extra_finalize_checks",
         "_check_op_host_completeness", "_pass_branch_gate_specs",
         "_PLUGIN_EXTRAS_SENTINEL", "check_finalize_eligibility", "batch_precheck",
         "_precheck_blocked", "format_batch_precheck_report",
         "_finalize_with_plugin_layout", "finalize_op", "FinalizeReport"),
}


# ── split-neutrality locks ───────────────────────────────────────────────────
def test_reexports_are_same_object() -> None:
    for mod, names in _REEXPORTS.items():
        for name in names:
            assert hasattr(fp, name), f"finalize_pipeline lost re-export {name}"
            assert getattr(fp, name) is getattr(mod, name), (
                f"{name}: finalize_pipeline re-export differs from {mod.__name__} "
                "definition — the split copied instead of re-importing"
            )


def test_monkeypatched_names_live_in_finalize_dispatch() -> None:
    """DEBT-201 batch5 (2026-07-06): _get_active_plugin + finalize_op are the
    two monkeypatched entry points of the plugin-dispatch cluster. To push
    finalize_pipeline under 1000 lines the WHOLE cluster moved to
    finalize_dispatch.py — and it had to move together, because
    _run_plugin_extra_finalize_checks / _check_op_host_completeness /
    check_finalize_eligibility / finalize_op call _get_active_plugin by BARE
    NAME. For monkeypatch.setattr(..., "_get_active_plugin", ...) to bite those
    intra-cluster callers, the patch target is finalize_dispatch (the real
    home), NOT the finalize_pipeline re-export.

    Contract:
      * both names are DEFINED in finalize_dispatch (__module__ check);
      * finalize_pipeline still RE-EXPORTS the identical object so
        `finalize_pipeline.finalize_op` / `._get_active_plugin` resolve for any
        qualified-attribute caller (e.g. orchestrator.py, whose
        `finalize_pipeline.finalize_op` monkeypatch still bites because it reads
        the attribute at call time).
    """
    for name in ("_get_active_plugin", "finalize_op"):
        fn = getattr(fd, name)
        assert fn.__module__ == "finalize_dispatch", (
            f"{name} is not defined in finalize_dispatch "
            f"(__module__={fn.__module__})"
        )
        assert getattr(fp, name) is fn, (
            f"finalize_pipeline.{name} is not the same object as "
            f"finalize_dispatch.{name} — the re-export drifted"
        )


def test_cycle_break_siblings_source_shared_from_finalize_shared() -> None:
    """The finalize_checks_* siblings must import the relocated shared helpers
    from finalize_shared (NOT finalize_pipeline) so the import cycle is broken
    for those symbols.
    """
    orch = Path(__file__).resolve().parents[2]
    infra = (orch / "finalize_checks_infra.py").read_text()
    prec = (orch / "finalize_checks_precision.py").read_text()
    struct = (orch / "finalize_checks_structural.py").read_text()
    assert "from finalize_shared import" in infra
    assert "_is_negative_assertion_window" in infra.split("from finalize_shared import", 1)[1][:200]
    assert "from finalize_shared import" in prec
    assert "_benchmark_case_count" in prec.split("from finalize_shared import", 1)[1][:120]
    assert "from finalize_shared import _is_v220_ec41_output_pad_exempt" in struct


# ── finalize_readme coverage gaps ────────────────────────────────────────────
def test_assemble_readme_writes_op_and_verdict(tmp_path: Path) -> None:
    # _assemble_readme writes archive_dir/README.md (returns None) and lists the
    # customer-facing root files.
    arch = tmp_path / "7_MyOp"
    arch.mkdir()
    (arch / "model.py").write_text("x")
    assert getattr(fr, '_assemble_readme')("7_MyOp", arch, "VERDICT: PASS") is None
    body = (arch / "README.md").read_text()
    assert "7_MyOp" in body
    assert "VERDICT: PASS" in body
    assert "`model.py`" in body


def test_write_archive_readme_writes_file(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "verification.json").write_text('{"precision": {"status": "PASS"}}')
    arch = tmp_path / "arch"
    arch.mkdir()
    getattr(fr, '_write_archive_readme')(arch, "7_MyOp", ws)
    readme = arch / "README.md"
    assert readme.is_file()
    assert "7_MyOp" in readme.read_text()


def test_render_verification_conclusion_pass() -> None:
    txt = getattr(fr, '_render_verification_conclusion')({"precision": {"status": "PASS"}})
    assert isinstance(txt, str) and txt.strip()


# ── finalize_ge_ophost coverage gaps ─────────────────────────────────────────
def test_ge_ophost_project_root_matches_pipeline() -> None:
    # The locally-computed _PROJECT_ROOT must equal finalize_pipeline's.
    assert getattr(fg, '_PROJECT_ROOT') == getattr(fp, '_PROJECT_ROOT')


# ── finalize_candidates coverage gaps ────────────────────────────────────────
def test_scan_workspace_for_candidate_refs_finds_tokens(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "knowledge_update.md").write_text("References CAND-123 and CAND-0007 here.")
    refs = getattr(fc, '_scan_workspace_for_candidate_refs')(ws)
    assert "CAND-123" in refs and "CAND-0007" in refs


def test_scan_workspace_for_candidate_refs_empty(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    assert getattr(fc, '_scan_workspace_for_candidate_refs')(ws) == set()


def test_resolve_archive_op_name_bare(tmp_path: Path) -> None:
    # With an empty archive root, the op name resolves to itself.
    arch = tmp_path / "arch"
    arch.mkdir()
    assert getattr(fc, '_resolve_archive_op_name')("7_MyOp", arch) == "7_MyOp"


def test_candidates_verification_hash_is_shared_object() -> None:
    # finalize_candidates imports _verification_hash from finalize_shared (not
    # from finalize_pipeline) — the cycle-safe path.
    assert getattr(fc, '_verification_hash') is getattr(fs, '_verification_hash')


# ── finalize_shared coverage gaps ────────────────────────────────────────────
def test_is_harness_internal_matches_known_and_markers() -> None:
    assert getattr(fs, '_is_harness_internal')("state_transitions.jsonl") is True
    assert getattr(fs, '_is_harness_internal')("sub/dir/knowledge_update.md") is True
    assert getattr(fs, '_is_harness_internal')(".finalized-deadbeef") is True
    assert getattr(fs, '_is_harness_internal')(".cc_stream_log_kw_3.jsonl") is True
    assert getattr(fs, '_is_harness_internal')("model_new_ascendc.py") is False


def test_kb_writeup_body_len_counts_body(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "knowledge_update.md").write_text("# Title\n\nsome real body text here\n")
    assert getattr(fs, '_kb_writeup_body_len')(ws) > 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert getattr(fs, '_kb_writeup_body_len')(empty) == 0


# ── finalize_dispatch coverage + patch-bite proof (DEBT-201 batch5) ──────────
def test_dispatch_patch_bites_run_plugin_extra_checks(tmp_path, monkeypatch):
    """LOAD-BEARING: patching finalize_dispatch._get_active_plugin MUST reach
    the bare-name caller _run_plugin_extra_finalize_checks (which lives in the
    same module). This is the exact contract the split preserves — if the patch
    stopped biting, every DEBT-124 test would silently pass against the real
    plugin layer. Also prove the negative: patching the finalize_pipeline
    re-export does NOT reach the bare-name caller (why the tests target fd).
    """
    calls = {"n": 0}

    class _StrictPlugin:
        name = "strict"

        def extra_finalize_checks(self):
            calls["n"] += 1
            return [("bite_gate", lambda ws, v: "patched plugin fired")]

    # Patch at the real home → the bare-name call inside finalize_dispatch sees it.
    monkeypatch.setattr(fd, "_get_active_plugin", lambda ws: _StrictPlugin())
    result = getattr(fd, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert calls["n"] == 1, "patch did NOT bite the bare-name _get_active_plugin call"
    assert result is not None and result["gate"] == "bite_gate"


def test_dispatch_patch_on_parent_does_not_bite_bare_name(tmp_path, monkeypatch):
    """Documents WHY the patch target is finalize_dispatch, not finalize_pipeline:
    rebinding the parent's re-exported attribute leaves the intra-module
    bare-name reference pointing at the original function object. Patching fp
    here must NOT change what _run_plugin_extra_finalize_checks resolves — it
    keeps calling the REAL _get_active_plugin (returns None with no plugins dir),
    so no synthetic 'sentinel' plugin fires.
    """
    sentinel = {"fired": False}

    class _SentinelPlugin:
        name = "sentinel"

        def extra_finalize_checks(self):
            sentinel["fired"] = True
            return [("never", lambda ws, v: "should not fire")]

    # Patch only the PARENT re-export attribute.
    monkeypatch.setattr(fp, "_get_active_plugin", lambda ws: _SentinelPlugin())
    getattr(fd, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert sentinel["fired"] is False, (
        "patching finalize_pipeline._get_active_plugin unexpectedly reached the "
        "finalize_dispatch bare-name caller — the module-boundary assumption the "
        "test repoint relies on is wrong"
    )


def test_dispatch_finalize_op_smoke(tmp_path) -> None:
    """finalize_op end-to-end on a minimal fixture promotes a workspace
    and drops the .finalized marker — exercises the moved promote path directly
    on finalize_dispatch.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "verification.json").write_text('{"precision": {"status": "PASS"}}')
    (ws / "model.py").write_text("x = 1\n")
    archive_root = tmp_path / "archive"
    rep = fd.finalize_op("7_MyOp", ws, archive_root=archive_root)
    assert rep.skipped is False
    assert rep.archive_dir is not None and rep.archive_dir.exists()
    assert (rep.archive_dir / "model.py").exists()
    assert rep.finalized_marker is not None and rep.finalized_marker.exists()


def test_dispatch_check_eligibility_missing_verification(tmp_path) -> None:
    """check_finalize_eligibility on an empty workspace → not eligible, rolls
    back to await_worker with the VERIFICATION_FILE_MISSING gate.
    """
    r = fd.check_finalize_eligibility(tmp_path)
    assert r["eligible"] is False
    assert r["gate"] == fp.GateID.VERIFICATION_FILE_MISSING.value


def test_dispatch_batch_precheck_not_applicable_for_non_pass(tmp_path) -> None:
    (tmp_path / "verification.json").write_text('{"precision": {"status": "FAIL"}}')
    r = fd.batch_precheck(tmp_path)
    assert r["applicable"] is False
    assert r["ok"] is True


def test_dispatch_precheck_blocked_shape() -> None:
    r = getattr(fd, '_precheck_blocked')("some_gate", "some reason")
    assert r["ok"] is False
    assert r["precondition_block"] == {"gate": "some_gate", "reason": "some reason"}


def test_dispatch_format_batch_precheck_report_lists_failures() -> None:
    result = {
        "summary": "2 failures",
        "precondition_block": None,
        "failures": [
            {"gate": "gate_a", "reason": "reason a"},
            {"gate": "gate_b", "reason": "reason b"},
        ],
    }
    md = fd.format_batch_precheck_report(result)
    assert "gate_a" in md and "gate_b" in md and "reason a" in md

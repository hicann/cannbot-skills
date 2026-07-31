# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for kb_marker_verifier.

The verifier reads a `.kb_merged` marker and confirms its `entries=`
claims actually appear in the KB files listed under `merged_into=`.

These tests use a synthetic mini-project layout per case (tempdir with
`src/skills/references/target/arch35/{OPERATIONAL,ERROR,PLATFORM}.md`
+ `workspace/<op>/.kb_merged`) so the verifier walks real files.

Anchor cases:

1. Marker claims TR-OL-22, file has `### TR-OL-22:` → OK.
2. Marker claims TR-OL-22, file does NOT have it → MISSING_ENTRIES,
   stderr names the missing entry — the empirical 8_Sort failure mode.
3. Marker claims `TR-EC-4(refinement)` → that token is treated as
   unchecked (refinement / evidence-append), so verdict OK if no other
   new entries are missing.
4. Marker missing entirely → NO_MARKER (exit 0; existing SC2 gate
   handles the "marker absent" case separately).
5. Mixed: TR-OL-22 found, TR-OL-23 missing → MISSING_ENTRIES naming
   only TR-OL-23.
6. `merged_into=` lists a file that doesn't exist → diagnostic surfaced.
7. Bare-filename `merged_into=` → resolved via rglob, found case works.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
_SCRIPT = _HERE.parent.parent / "kb_marker_verifier.py"

# Make the script directory importable for unit tests
sys.path.insert(0, str(_SCRIPT.parent))
import kb_marker_verifier as kmv  # noqa: E402
import orchestrator_sweep as sweep  # noqa: E402


def _make_project(tmp: pathlib.Path,
                  kb_body_ol: str = "",
                  kb_body_ec: str = "",
                  kb_body_pb: str = "") -> pathlib.Path:
    """Build a minimal repo with KB layout under tmp."""
    refs = tmp / "src" / "skills" / "references" / "target" / "arch35"
    refs.mkdir(parents=True)
    (refs / "OPERATIONAL_KNOWLEDGE.md").write_text(kb_body_ol or "# OL\n")
    (refs / "ERROR_CORRECTIONS.md").write_text(kb_body_ec or "# EC\n")
    (refs / "PLATFORM_BUGS.md").write_text(kb_body_pb or "# PB\n")
    (tmp / ".claude").mkdir()
    return tmp


def _write_marker(workspace: pathlib.Path, content: str) -> pathlib.Path:
    workspace.mkdir(parents=True, exist_ok=True)
    p = workspace / ".kb_merged"
    p.write_text(content)
    return p


def _write_customer_marker(
    workspace: pathlib.Path,
    c_root: pathlib.Path,
    entries: str,
    *,
    reviewed: int,
    rejected: int = 0,
    mode: str = "update",
) -> pathlib.Path:
    return _write_marker(workspace, (
        "merge_run=2026-07-30T00:00:00Z\n"
        "tier=customer\n"
        f"c_root={c_root}\n"
        "merged_into=user-c-tier\n"
        f"entries={entries}\n"
        f"reviewed={reviewed}\n"
        f"rejected={rejected}\n"
        f"mode={mode}\n"
    ))


def test_marker_claims_entry_actually_in_file_ok(tmp_path) -> None:
    proj = _make_project(
        tmp_path,
        kb_body_OL="# OL\n\n### TR-OL-22: tl.sort BLOCK ceiling\nbody\n",
    )
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-22\n"
        "mode=update\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "OK", rep
    assert "TR-OL-22" in rep.found


def test_marker_claims_entry_not_in_file_blocks(tmp_path) -> None:
    """The 8_Sort tonight failure: marker says TR-OL-22 merged, file has no header."""
    proj = _make_project(tmp_path)  # empty KB files
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T22:08:11Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-22\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "MISSING_ENTRIES"
    assert "TR-OL-22" in rep.missing


def test_refinement_id_binds_to_existing_entry(tmp_path) -> None:
    """`TR-EC-4(refinement)` must bind to the canonical entry it refines."""
    proj = _make_project(tmp_path, kb_body_EC="### TR-EC-4: existing\n")
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T22:08:11Z\n"
        "merged_into=target/arch35/ERROR_CORRECTIONS.md\n"
        "entries=TR-EC-4(refinement)\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "OK", rep
    assert "TR-EC-4(refinement)" in rep.refinement_ids


def test_refinement_without_existing_entry_blocks(tmp_path) -> None:
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merged_into=target/arch35/ERROR_CORRECTIONS.md\n"
        "entries=TR-EC-4(refinement)\n"
    ))

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert rep.missing == ["TR-EC-4(refinement)"]


@pytest.mark.parametrize("entries", ["", "3", "none"])
def test_legacy_marker_rejects_empty_or_unknown_entries(
    tmp_path, entries: str
) -> None:
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "invalid"
    _write_marker(ws, (
        "merged_into=target/arch35/ERROR_CORRECTIONS.md\n"
        f"entries={entries}\n"
    ))

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"


def test_legacy_refinement_requires_merged_into_file(tmp_path) -> None:
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "invalid"
    _write_marker(ws, "entries=TR-EC-4(refinement)\n")

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"


def test_legacy_marker_rejects_absolute_path_outside_kb_roots(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    outside = tmp_path / "outside.md"
    outside.write_text("### TR-OL-22: forged\n")
    ws = proj / "workspace" / "invalid"
    _write_marker(ws, f"merged_into={outside}\nentries=TR-OL-22\n")

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"


def test_missing_marker_returns_no_marker(tmp_path) -> None:
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "1_GELU__arch35"
    ws.mkdir(parents=True)
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "NO_MARKER"


def test_marker_symlink_is_rejected_without_reading_target(tmp_path) -> None:
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "invalid"
    ws.mkdir(parents=True)
    target = tmp_path / "outside-marker"
    target.write_text(
        "merge_run=2026-07-30T00:00:00Z\n"
        "tier=customer\n"
        f"c_root={tmp_path / 'user-kb'}\n"
        "merged_into=user-c-tier\n"
        "entries=none\n"
        "reviewed=0\n"
        "rejected=0\n"
        "mode=update\n"
    )
    (ws / ".kb_merged").symlink_to(target)

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert any("regular file" in item for item in rep.diagnostics)


def test_mixed_found_and_missing_blocks(tmp_path) -> None:
    """Found entries reported, missing entries surfaced. Verdict is MISSING_ENTRIES."""
    proj = _make_project(
        tmp_path,
        kb_body_OL="### TR-OL-22: tl.sort BLOCK ceiling\nbody\n",
    )
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-22,TR-OL-23\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "MISSING_ENTRIES"
    assert "TR-OL-22" in rep.found
    assert rep.missing == ["TR-OL-23"]


def test_merged_into_lists_nonexistent_file_surfaces_diagnostic(tmp_path) -> None:
    """patterns/unverified/candidates.md doesn't exist — the 8_Sort case."""
    proj = _make_project(tmp_path)
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=target/arch35/patterns/unverified/candidates.md\n"
        "entries=TR-CAND-1\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    # TR-CAND-1 matches the new-entry regex; no candidates file → missing
    assert rep.verdict == "MISSING_ENTRIES"
    assert any("patterns/unverified" in d for d in rep.diagnostics)


def test_bare_filename_resolves_via_rglob(tmp_path) -> None:
    """`OPERATIONAL_KNOWLEDGE.md` alone should resolve under refs/."""
    proj = _make_project(
        tmp_path,
        kb_body_OL="### TR-OL-7: foo\nbody\n",
    )
    ws = proj / "workspace" / "x"
    _write_marker(ws, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-7\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "OK"


def test_customer_marker_binds_to_c_tier_entry(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    c_root = tmp_path / "user kb"
    claim = "lesson"
    content_hash = hashlib.sha1("lesson||[]".encode()).hexdigest()[:12]
    entry_id = f"customer:{content_hash}"
    entry_file = c_root / "entries" / f"{content_hash}.json"
    entry_file.parent.mkdir(parents=True)
    entry_file.write_text(json.dumps({
        "id": entry_id,
        "tier": "customer",
        "role": "user-local",
        "kind": "experience",
        "scope": {},
        "key": "",
        "claim": claim,
    }))
    _write_customer_marker(ws, c_root, entry_id, reviewed=1)

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "OK", rep
    assert rep.found[entry_id] == str(entry_file)


def test_customer_marker_rejects_content_hash_mismatch(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    c_root = tmp_path / "user kb"
    entry_id = "customer:0123456789ab"
    entry_file = c_root / "entries" / "0123456789ab.json"
    entry_file.parent.mkdir(parents=True)
    entry_file.write_text(json.dumps({
        "id": entry_id,
        "tier": "customer",
        "role": "user-local",
        "kind": "experience",
        "scope": {},
        "key": "",
        "claim": "forged payload whose content does not match its id",
    }))
    _write_customer_marker(ws, c_root, entry_id, reviewed=1)

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert rep.missing == [entry_id]


def test_customer_marker_can_be_bound_to_configured_c_root(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    _write_customer_marker(
        ws, tmp_path / "attacker-root", "none", reviewed=0
    )

    rep = kmv.verify_marker(
        ws,
        project_root=proj,
        expected_c_root=tmp_path / "configured-root",
    )

    assert rep.verdict == "MISSING_ENTRIES"
    assert any("configured provider root" in item for item in rep.diagnostics)


def test_customer_marker_requires_orchestrator_completion_schema(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    _write_marker(ws, (
        "tier=customer\n"
        f"c_root={tmp_path / 'user kb'}\n"
        "entries=none\n"
    ))

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert any("completion schema" in item for item in rep.diagnostics)


def test_sweep_only_skips_provider_bound_customer_marker(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "project"
    workspace = repo / "workspace" / "op"
    archive = repo / "output" / "a3_to_a5_port" / "src" / "kernels" / "op"
    workspace.mkdir(parents=True)
    archive.mkdir(parents=True)
    (workspace / "knowledge_update.md").write_text("lesson\n" * 20)
    configured = tmp_path / "configured-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))
    monkeypatch.setattr(sweep, "REPO", repo)
    monkeypatch.setattr(sweep, "WORKSPACE", repo / "workspace")

    _write_customer_marker(
        archive, tmp_path / "unrelated-kb", "none", reviewed=0
    )
    assert getattr(sweep, '_has_knowledge_update_unmerged')("op") is True

    _write_customer_marker(archive, configured.resolve(), "none", reviewed=0)
    assert getattr(sweep, '_has_knowledge_update_unmerged')("op") is False


def test_sweep_rejects_forged_legacy_marker(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "project"
    workspace = repo / "workspace" / "op"
    archive = repo / "output" / "a3_to_a5_port" / "src" / "kernels" / "op"
    workspace.mkdir(parents=True)
    archive.mkdir(parents=True)
    (workspace / "knowledge_update.md").write_text("lesson\n" * 20)
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(tmp_path / "configured-kb"))
    monkeypatch.setattr(sweep, "REPO", repo)
    monkeypatch.setattr(sweep, "WORKSPACE", repo / "workspace")
    _write_marker(archive, "merged_into=x\nentries=3\n")

    assert getattr(sweep, '_has_knowledge_update_unmerged')("op") is True


def test_customer_marker_missing_entry_blocks(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    entry_id = "customer:fedcba987654"
    _write_customer_marker(
        ws, tmp_path / "missing user kb", entry_id, reviewed=1
    )

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert rep.missing == [entry_id]


def test_customer_marker_rejects_malformed_entry_id(tmp_path) -> None:
    proj = _make_project(tmp_path / "project")
    ws = proj / "workspace" / "x"
    _write_customer_marker(
        ws,
        tmp_path / "user kb",
        "customer:not-a-content-hash",
        reviewed=1,
    )

    rep = kmv.verify_marker(ws, project_root=proj)

    assert rep.verdict == "MISSING_ENTRIES"
    assert rep.missing == ["customer:not-a-content-hash"]


def test_cli_exit_codes(tmp_path) -> None:
    """CLI must return 0 on OK / NO_MARKER, 2 on MISSING_ENTRIES."""
    proj = _make_project(
        tmp_path,
        kb_body_OL="### TR-OL-9: bar\n",
    )
    ws_ok = proj / "workspace" / "ok"
    _write_marker(ws_ok, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-9\n"
    ))
    proc_ok = subprocess.run(
        [sys.executable, str(_SCRIPT), str(ws_ok), "--project-root", str(proj)],
        capture_output=True, text=True, check=False,
    )
    assert proc_ok.returncode == 0, proc_ok.stderr

    ws_bad = proj / "workspace" / "bad"
    _write_marker(ws_bad, (
        "merge_run=2026-05-17T00:00:00Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-OL-99\n"
    ))
    proc_bad = subprocess.run(
        [sys.executable, str(_SCRIPT), str(ws_bad), "--project-root", str(proj)],
        capture_output=True, text=True, check=False,
    )
    assert proc_bad.returncode == 2, proc_bad.stderr
    assert "TR-OL-99" in proc_bad.stderr
    assert "MISSING" in proc_bad.stderr

    ws_none = proj / "workspace" / "none"
    ws_none.mkdir(parents=True)
    proc_none = subprocess.run(
        [sys.executable, str(_SCRIPT), str(ws_none), "--project-root", str(proj)],
        capture_output=True, text=True, check=False,
    )
    assert proc_none.returncode == 0, proc_none.stderr


def test_8_sort_post_salvage_audit_fixture(tmp_path) -> None:
    """DEBT-105: deterministic fixture replacing the former fs-state-dependent
    `test_real_repo_8_Sort_audit` (which hardcoded `/home/npu_user/workspace/a5_ops_arch35`
    and depended on the live `workspace/8_Sort__arch35/.kb_merged` snapshot — so it
    skipped/passed/failed per checkout + worktree drift, the same isolation class as
    DEBT-101).

    Reproduces the post-2026-05-17-salvage 8_Sort scenario without touching any real
    workspace: TR-OL-22 was hand-merged into `OPERATIONAL_KNOWLEDGE.md` (resolves →
    `found`); TR-CAND-1 is claimed merged into `patterns/unverified/candidates.md`
    which does not exist as a real path (→ `missing` + unresolvable-file diagnostic).
    Verdict MISSING_ENTRIES. Runs everywhere, deterministically. The full audit trail
    for the original 3-of-4 failure lives in docs/design/KB_DESIGN_NOTES.md#kb-marker-verifier-design.
    """
    proj = _make_project(
        tmp_path,
        kb_body_OL="# OL\n\n### TR-OL-22: tl.sort BLOCK ceiling\nbody\n",
    )
    ws = proj / "workspace" / "8_Sort__arch35"
    _write_marker(ws, (
        "merge_run=2026-05-17T23:50:00Z\n"
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md,"
        "target/arch35/patterns/unverified/candidates.md\n"
        "entries=TR-OL-22,TR-CAND-1\n"
        "mode=update\n"
    ))
    rep = kmv.verify_marker(ws, project_root=proj)
    assert rep.verdict == "MISSING_ENTRIES", rep
    assert "TR-OL-22" in rep.found, rep
    assert "TR-CAND-1" in rep.missing, rep
    assert any("patterns/unverified" in d for d in rep.diagnostics), rep.diagnostics


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

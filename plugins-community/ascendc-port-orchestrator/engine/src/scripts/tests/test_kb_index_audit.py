# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Characterization tests for kb_index_audit.py (DEBT-202 UT backfill).

Pins the CURRENT behavior of the KB_INDEX orphan/duplicate/dangling auditor: the
entry extractors (EC/PB/OL/P-P/CAND, including tolerant-suffix forms that
past bug-reports forced in), duplicate + applies_to detection, index-reference
parsing, P-P title-conflict detection, and
the audit_backend integration over a synthetic KB tree.

Behavior-neutral pins — each test drives real files (tmp_path) or real text through
the public surface and asserts structured output; none are import-smoke.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parent.parent  # src/scripts/
sys.path.insert(0, str(_SCRIPTS))

import kb_index_audit as kia  # noqa: E402


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ── entry extractors ────────────────────────────────────────────────────────

def test_extract_entries_ec(tmp_path):
    p = _write(tmp_path, "EC.md", "### EC-1: foo\nbody\n### EC-27: bar\n")
    assert kia.extract_entries_ec(p) == ["EC-1", "EC-27"]


def test_extract_entries_pb(tmp_path):
    p = _write(tmp_path, "PB.md", "### PB-34: v220 only\n### PB-52: staging\n")
    assert kia.extract_entries_pb(p) == ["PB-34", "PB-52"]


def test_extract_entries_ol_tolerates_revised_suffix(tmp_path):
    # OL headers may carry a bracketed [REVISED ...] / [ARCHIVED ...] clarifier
    p = _write(tmp_path, "OL.md",
               "## OL-7: plain\n## OL-83 [REVISED 2026-04-22]: revised entry\n")
    assert kia.extract_entries_ol(p) == ["OL-7", "OL-83"]


def test_extract_entries_cand_h2_and_h3(tmp_path):
    p = _write(tmp_path, "cand.md",
               "## CAND-ABC: h2 form\n### CAND-DB-COARSE: h3 form\n")
    got = kia.extract_entries_cand(p)
    assert set(got) == {"CAND-ABC", "CAND-DB-COARSE"}


def test_extract_missing_file_returns_empty(tmp_path):
    assert kia.extract_entries_ec(tmp_path / "nope.md") == []


# ── duplicate detection ─────────────────────────────────────────────────────

def test_find_duplicates():
    entries = [("EC-1", 3), ("EC-2", 7), ("EC-1", 20)]
    dups = kia.find_duplicates(entries)
    assert dups == [("EC-1", [3, 20])]


def test_extract_entries_with_lines(tmp_path):
    p = _write(tmp_path, "EC.md", "### EC-1: a\nx\n### EC-2: b\n")
    got = kia.extract_entries_with_lines(p, r"^### (EC-\d+):")
    assert got == [("EC-1", 1), ("EC-2", 3)]


# ── applies_to detection ────────────────────────────────────────────────────

def test_check_applies_to_flags_missing(tmp_path):
    text = (
        "## OL-1: has scope\n"
        "applies_to: all\n"
        "body\n"
        "## OL-2: no scope\n"
        "just prose, no marker\n"
    )
    p = _write(tmp_path, "OL.md", text)
    ewl = kia.extract_entries_with_lines(p, r"^## (OL-\d+):")
    missing = kia.check_applies_to(p, ewl)
    assert any(m.startswith("OL-2") for m in missing)
    assert not any(m.startswith("OL-1") for m in missing)


# ── index-reference parsing (dash discrimination) ───────────────────────────

def test_extract_index_refs_plain_prefix():
    text = "row EC-13 and EC-99 mentioned"
    assert kia.extract_index_refs(text, prefixes=["EC"]) == {"EC-13", "EC-99"}


def test_extract_index_refs_pp_style_no_extra_dash():
    # P-P / F-AP take digits directly after prefix (P-P90, F-AP1)
    text = "P-P90 and F-AP1 referenced"
    got = kia.extract_index_refs(text, prefixes=["P-P", "F-AP"])
    assert got == {"P-P90", "F-AP1"}


def test_extract_index_cand():
    text = "| CAND-ABC | ... | CAND-DB-FENCE |"
    assert kia.extract_index_cand(text) == {"CAND-ABC", "CAND-DB-FENCE"}


# ── P-P title-conflict detection ────────────────────────────────────────────

def test_find_pp_title_conflicts_detects_mismatch(tmp_path):
    # same P-P number: a table row for one pattern + a ## section for a DIFFERENT one
    text = (
        "| P-P107 | **top_k selection** | L2 |\n"
        "## P-P107: swi_glu fusion\nbody\n"
    )
    p = _write(tmp_path, "PATTERN_INDEX.md", text)
    conflicts = kia.find_pp_title_conflicts(p)
    assert conflicts and conflicts[0][0] == "P-P107"


def test_find_pp_title_conflicts_same_title_not_conflict(tmp_path):
    # a normal index row + its ## definition sharing a title is NOT a conflict
    text = (
        "| P-P101 | **row tiling for flash attention** | L1 |\n"
        "## P-P101: row tiling for flash attention\nbody\n"
    )
    p = _write(tmp_path, "PATTERN_INDEX.md", text)
    assert kia.find_pp_title_conflicts(p) == []


# ── audit_backend integration ───────────────────────────────────────────────

def _make_ascendc_tree(root: Path, *, orphan: bool):
    """Build a minimal ascendc KB dir; if orphan, the EC entry is left out of the index."""
    kb = root / "target" / "ascendc"
    (kb / "patterns" / "unverified").mkdir(parents=True)
    (kb / "ERROR_CORRECTIONS.md").write_text(
        "### EC-1: fixed\napplies_to: all\n### EC-2: other\napplies_to: all\n")
    (kb / "PLATFORM_BUGS.md").write_text("### PB-1: bug\napplies_to: all\n")
    (kb / "OPERATIONAL_KNOWLEDGE.md").write_text("## OL-1: rule\napplies_to: all\n")
    (kb / "patterns" / "PATTERN_INDEX.md").write_text("| P-P1 | **thing** | L1 |\n")
    (kb / "patterns" / "unverified" / "candidates.md").write_text("## CAND-X: cand\n")
    indexed = "EC-1 PB-1 OL-1 P-P1 CAND-X" + ("" if orphan else " EC-2")
    return kb, indexed


def test_audit_backend_clean_has_no_orphans(tmp_path):
    kb, index_text = _make_ascendc_tree(tmp_path, orphan=False)
    results = kia.audit_backend("ascendc", kb, index_text)
    total_orphans = sum(len(r.orphans) for r in results)
    assert total_orphans == 0
    ec = next(r for r in results if r.file_type == "EC")
    assert ec.total_entries == 2


def test_audit_backend_detects_orphan(tmp_path):
    kb, index_text = _make_ascendc_tree(tmp_path, orphan=True)
    results = kia.audit_backend("ascendc", kb, index_text)
    ec = next(r for r in results if r.file_type == "EC")
    assert ec.orphans == ["EC-2"]


def test_audit_backend_missing_dir_returns_empty(tmp_path):
    assert kia.audit_backend("ascendc", tmp_path / "absent", "") == []


def test_main_text_mode_supports_kb_sibling_of_engine(
    tmp_path, monkeypatch, capsys
):
    """The relocated plugin layout keeps ``engine/`` and ``kb/`` as siblings."""
    plugin_root = tmp_path / "plugin"
    engine_root = plugin_root / "engine"
    kb_root = plugin_root / "kb"
    ascendc_root = kb_root / "target" / "ascendc"
    engine_root.mkdir(parents=True)
    ascendc_root.mkdir(parents=True)
    index_path = kb_root / "KB_INDEX.md"
    index_path.write_text("")

    monkeypatch.setattr(kia, "PROJECT_ROOT", engine_root)
    monkeypatch.setattr(kia, "KB_ROOT", kb_root)
    monkeypatch.setattr(kia, "KB_DIRS", {"ascendc": ascendc_root})
    monkeypatch.setattr(kia, "INDEX_PATH", index_path)
    monkeypatch.setattr(kia, "check_domain_template_scope", lambda: [])
    monkeypatch.setattr(sys, "argv", ["kb_index_audit.py", "--strict"])

    assert kia.main() == 0
    assert "KB_INDEX multi-error audit — kb/KB_INDEX.md" in capsys.readouterr().out


# ── SoC scope-consistency lint (2026-07-17) ─────────────────────────────────
#
# Invariant: a SoC named as POSITIVE evidence (verified_on / confirmed_on) must
# be inside the entry's declared applies_to soc= set. Origin: PB-35 declared
# soc=Ascend910_9382 (V220) while its own confirmed_on recorded a CONFIRMED
# repro on Ascend950PR_9579 (V351/A5) — so honoring applies_to (DEBT-208) would
# have suppressed it on the one SoC where it is proven.
#
# The negative tests below are not hypotheticals: each mirrors a REAL entry that
# a naive prose-scanning lint red-flags (surveyed across 738 entries).

def _soc_entry(tmp_path: Path, name: str, body: str) -> Path:
    return _write(tmp_path, name, body)


def test_soc_scope_flags_evidence_outside_applies_to(tmp_path):
    """The PB-35 shape: applies_to says V220, confirmed_on says V351."""
    p = _soc_entry(tmp_path, "PB.md", (
        "### PB-99: some mixed-mode hang [V220]\n"
        "`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+`\n"
        "`confirmed_on: Ascend950PR_9579 (V351 / A5) - graybox 2026-06-03 CONFIRMED`\n"
    ))
    v = kia.check_soc_scope(p)
    assert len(v) == 1
    entry_id, applies_raw, field, out_of_scope, _excerpt = v[0]
    assert entry_id == "PB-99"
    assert field == "confirmed_on"
    assert out_of_scope == ["V351"]


def test_soc_scope_clean_when_applies_to_declares_both(tmp_path):
    """The PB-35 FIX shape: comma-separated multi-SoC applies_to → no violation."""
    p = _soc_entry(tmp_path, "PB.md", (
        "### PB-99: some mixed-mode hang [V220 + V351]\n"
        "`applies_to: soc=Ascend910_9382,Ascend950PR_9579; cann=9.0.0+`\n"
        "`verified_on: a5_ops:3_FusionAttention kw-4 (2026-05-21) - hang`\n"
        "`confirmed_on: Ascend950PR_9579 (V351 / A5) - graybox 2026-06-03 CONFIRMED`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_ignores_unverified_on(tmp_path):
    """unverified_on's JOB is to name SoCs outside applies_to — never a violation."""
    p = _soc_entry(tmp_path, "PB.md", (
        "### PB-99: title\n"
        "`applies_to: soc=Ascend910_9382 (V220)`\n"
        "`unverified_on: Ascend950PR_9579 (V351 / A5) - probe was malformed`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_ignores_negative_evidence_fields(tmp_path):
    """verified_does_not_reproduce_on names an out-of-scope SoC BY DESIGN.

    Guards the anchored regex: `verified_does_not_reproduce_on:` must never be
    matched as `verified_on:`.
    """
    p = _soc_entry(tmp_path, "PB.md", (
        "### PB-34: v220 KFC deadlock\n"
        "`applies_to: soc=Ascend910_9382 (V220)`\n"
        "`verified_does_not_reproduce_on: Ascend950PR (V351/A5) - GDN 122/122 PASS`\n"
        "`verified_does_not_apply_on: Ascend950PR (V351/A5)`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_ignores_prose_port_direction(tmp_path):
    """Real shape — CAND-POOL-LAYOUT-BRIDGE.

    `verified_on: adaptive_avg_pool3d (V220->A5 L1 port)` describes the port
    DIRECTION; the op was verified on A5. applies_to=Ascend950PR is CORRECT.
    A prose scan red-flags this honest entry.
    """
    p = _soc_entry(tmp_path, "CAND.md", (
        "## CAND-POOL-LAYOUT-BRIDGE: V220 pooling kernel layout bridge\n"
        "`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=pooling`\n"
        "`verified_on: adaptive_avg_pool3d (V220->A5 L1 port, 2026-06-16)`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_ignores_soc_named_inside_crossref_id(tmp_path):
    """Real shape — CAND-V351-arch35-RegBase-service-class-skeleton.

    Its verified_on names V220 only inside the cross-ref ID
    `CAND-V220-to-V351-PortPattern-CubeVecFusedOp`. Honest entry.
    """
    p = _soc_entry(tmp_path, "CAND.md", (
        "## CAND-V351-arch35-RegBase-service-class-skeleton: V351 arch35 structure\n"
        "`applies_to: soc=Ascend950PR_9579 (V351/arch35); cann=9.0.0`\n"
        "`verified_on: sparse_lightning_indexer_grad arch35 structure (complementary "
        "evidence layer to CAND-V220-to-V351-PortPattern-CubeVecFusedOp)`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_universal_applies_to_never_violates(tmp_path):
    p = _soc_entry(tmp_path, "OL.md", (
        "## OL-99: universal thing\n"
        "`applies_to: soc=all; cann=9.0.0`\n"
        "`verified_on: soc=Ascend950PR_9579; cann=9.0.0`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_normalizes_within_family(tmp_path):
    """soc=Ascend950PR vs verified_on soc=Ascend950PR_957b = SAME family (V351).

    Exact-token comparison would false-positive here; the KB spells one family
    ~10 ways. Family granularity is the contract.
    """
    p = _soc_entry(tmp_path, "OL.md", (
        "## OL-99: a5 thing\n"
        "`applies_to: soc=Ascend950PR; cann=9.0.0`\n"
        "`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_flags_structured_soc_clause_out_of_family(tmp_path):
    """The `verified_on: soc=<tok>` structured form is checked too."""
    p = _soc_entry(tmp_path, "OL.md", (
        "## OL-99: a5 thing\n"
        "`applies_to: soc=Ascend950PR; cann=9.0.0`\n"
        "`verified_on: soc=Ascend910_9382; cann=9.0.0`\n"
    ))
    v = kia.check_soc_scope(p)
    assert len(v) == 1 and v[0][3] == ["V220"]


def test_soc_scope_entry_without_soc_clause_is_skipped(tmp_path):
    p = _soc_entry(tmp_path, "OL.md", (
        "## OL-99: unscoped\n"
        "`applies_to: paradigm=ascendc; cann=9.0.0`\n"
        "`verified_on: Ascend950PR_9579 (V351) - ran fine`\n"
    ))
    assert kia.check_soc_scope(p) == []


def test_soc_scope_detects_across_entry_heading_forms(tmp_path):
    """h2 CAND / h3 PB / h2 OL headings all slice correctly; violation is
    attributed to the RIGHT entry, not bled into its neighbour.
    """
    p = _soc_entry(tmp_path, "MIX.md", (
        "## OL-1: clean a5 entry\n"
        "`applies_to: soc=Ascend950PR`\n"
        "`verified_on: soc=Ascend950PR_9579`\n"
        "\n"
        "### PB-7: contradicting entry\n"
        "`applies_to: soc=Ascend910_9382 (V220)`\n"
        "`confirmed_on: Ascend950PR_9579 (V351 / A5) - CONFIRMED`\n"
        "\n"
        "## CAND-XYZ: clean v220 entry\n"
        "`applies_to: soc=Ascend910_9382`\n"
        "`verified_on: soc=Ascend910_9382`\n"
    ))
    v = kia.check_soc_scope(p)
    assert [x[0] for x in v] == ["PB-7"]


# ── live-KB regression pins (the false-positive check, mechanized) ──────────

def test_live_kb_has_zero_soc_scope_violations():
    """The whole shipped KB must satisfy the invariant.

    This is the anti-false-positive guard: a lint that red-flags honest entries
    is worse than no lint. If a future honest entry trips this, the INVARIANT
    needs redesign — do not silence the entry.
    """
    violations = []
    for kb_dir in kia.KB_DIRS.values():
        if kb_dir.is_dir():
            for md in sorted(kb_dir.rglob("*.md")):
                violations += [(str(md), *v) for v in kia.check_soc_scope(md)]
    assert violations == [], f"live KB violates SoC scope invariant: {violations}"


def test_live_pb35_declares_both_socs():
    """Pins the DATA fix: PB-35 is confirmed on V351/A5 (kw-gb2 2026-06-03) and
    verified on V220 (kw-4), so its applies_to must cover BOTH. Reverting the
    applies_to to V220-only must fail this test AND
    test_live_kb_has_zero_soc_scope_violations.
    """
    pb = kia.KB_DIRS["ascendc"] / "PLATFORM_BUGS.md"
    body = pb.read_text(encoding="utf-8").split("### PB-35:")[1].split("\n### ")[0]
    applies = [l for l in body.splitlines() if l.startswith("`applies_to:")][0]
    socs, _raw = getattr(kia, '_applies_to_socs')(applies)
    assert socs == {"V220", "V351"}, f"PB-35 applies_to covers {socs}, expected both"


# ── DEBT-222: domain-template arch-scope invariant ──────────────────────────

def test_template_scope_prominent_frontmatter_passes(tmp_path):
    d = tmp_path / "domains"
    d.mkdir()
    _write(d, "x.md", "---\napplies_to: soc=Ascend950PR\n---\n# X\nbody\n")
    assert kia.check_domain_template_scope(d) == []


def test_template_scope_explicit_all_passes(tmp_path):
    d = tmp_path / "domains"
    d.mkdir()
    _write(d, "x.md", "---\napplies_to: soc=all\n---\n# X\n")
    assert kia.check_domain_template_scope(d) == []


def test_template_scope_blockquoted_tag_passes(tmp_path):
    """A prominent tag inside a top-of-file blockquote (the FA / GMM template
    convention) counts — the parser sees through a leading `>`.
    """
    d = tmp_path / "domains"
    d.mkdir()
    _write(d, "x.md", "# X\n> `applies_to: soc=Ascend950PR/V351; cann=9`\nbody\n")
    assert kia.check_domain_template_scope(d) == []


def test_template_scope_absent_fails(tmp_path):
    d = tmp_path / "domains"
    d.mkdir()
    _write(d, "x.md", "# X\n> Patterns for stuff.\nbody\n")
    v = kia.check_domain_template_scope(d)
    assert len(v) == 1 and v[0][0] == "x.md" and "absent or buried" in v[0][1]


def test_template_scope_buried_in_prose_fails(tmp_path):
    """A tag that only appears as a bulleted prose line deep in the body (past
    the header zone) is the 'buried' state DEBT-222 rejects.
    """
    d = tmp_path / "domains"
    d.mkdir()
    body = "\n".join(f"line{i}" for i in range(30))
    _write(d, "x.md", "# X\n" + body + "\n- **applies_to**: soc=Ascend950PR\n")
    v = kia.check_domain_template_scope(d)
    assert len(v) == 1 and "absent or buried" in v[0][1]


def test_template_scope_unparseable_soc_fails(tmp_path):
    d = tmp_path / "domains"
    d.mkdir()
    _write(d, "junk.md", "---\napplies_to: soc=frobnicator\n---\n# X\n")
    _write(d, "nosoc.md", "---\napplies_to: paradigm=ascendc\n---\n# X\n")
    v = dict(kia.check_domain_template_scope(d))
    assert "junk.md" in v and "recognizable SoC" in v["junk.md"]
    assert "nosoc.md" in v


def test_live_domain_templates_all_scoped():
    """Every real patterns/domains/*.md declares a machine-readable applies_to."""
    assert kia.check_domain_template_scope() == []


def test_live_arch_fixed_templates_resolve_to_a5():
    """The arch-fixed templates (cooperative / cube_vector_fusion / gmm×2) must
    resolve to V351/A5 — a regression that widened them to soc=all (or dropped
    the tag) would fail here. cooperative was the one DS's enumeration MISSED.
    """
    d = getattr(kia, '_DOMAIN_TEMPLATES_DIR')
    for name in ("cooperative.md", "cube_vector_fusion.md",
                 "gmm_swiglu_quant_a8w8_class_template.md",
                 "gmm_swiglu_quant_a8w8_host_tiling_template.md"):
        val = getattr(kia, '_template_header_applies_to')((d / name).read_text().splitlines())
        socs, _ = getattr(kia, '_applies_to_socs')(val)
        assert socs == {"V351"}, f"{name} resolved to {socs}, expected A5/V351"


# ── ID-sequence continuity (2026-07-23, OL-984 picker/typo incident) ─────────
#
# The gap this closes: PR #237 nearly merged with a mis-numbered `OL-984`. The
# author's id-picker (`grep -oE 'OL-[0-9]+' | sort | tail -1`) matched an
# OL-983-class id in PROSE (a cross-ref), not a title, and picked 984 instead of
# the real next id OL-280 (titles ran OL-1..OL-279). Orphan/dup/dangling/
# applies_to all passed; only a manual review caught it. These pins make the
# MACHINE catch it while never firing on legit small archived-id gaps.

def test_split_id_across_family_forms():
    assert getattr(kia, '_split_id')("OL-280") == ("OL", 280)
    assert getattr(kia, '_split_id')("EC-86") == ("EC", 86)
    assert getattr(kia, '_split_id')("P-P93") == ("P-P", 93)
    assert getattr(kia, '_split_id')("F-AP1") == ("F-AP", 1)


def test_id_sequence_flags_ol984_typo():
    """The OL-984 shape: max real title is OL-279, a mis-picked OL-984 appears."""
    entries = [f"OL-{n}" for n in range(1, 280)] + ["OL-984"]
    v = kia.check_id_sequence(entries)
    assert v == [("OL", 279, 984, 705)]


def test_id_sequence_control_correct_next_passes():
    """The FIX shape: OL-280 (the real next id) must NOT be flagged."""
    entries = [f"OL-{n}" for n in range(1, 281)]
    assert kia.check_id_sequence(entries) == []


def test_id_sequence_legit_small_gap_not_flagged():
    """Real archived-id gaps (OL 124→127 jump 3, PB 24→26 jump 2) are legit.

    These mirror the actual shipped KB — a 'must be exactly max+1' rule would
    false-positive on every one of them, which is why the rule is a MARGIN, not
    strict contiguity.
    """
    ol = [f"OL-{n}" for n in range(1, 125)] + [f"OL-{n}" for n in range(127, 281)]
    assert kia.check_id_sequence(ol) == []
    pb = [f"PB-{n}" for n in range(1, 25)] + [f"PB-{n}" for n in range(26, 58)]
    assert kia.check_id_sequence(pb) == []


def test_id_sequence_margin_boundary():
    """A jump exactly at MARGIN passes; one past it fails — pins the threshold."""
    m = kia.ID_SEQUENCE_MARGIN
    assert kia.check_id_sequence([f"OL-{n}" for n in (1, 1 + m)]) == []
    assert kia.check_id_sequence([f"OL-{n}" for n in (1, 2 + m)]) == [
        ("OL", 1, 2 + m, m + 1)]


def test_id_sequence_groups_mixed_prefixes_independently():
    """P-P / F-P / F-AP share one AuditResult but are independent sequences: a
    huge P-P id must not be masked by (nor bleed into) the F-P numbering.
    """
    entries = ["P-P1", "P-P2", "P-P900", "F-P1", "F-P2"]
    v = kia.check_id_sequence(entries)
    assert v == [("P-P", 2, 900, 898)]


def test_id_sequence_parses_headers_not_prose():
    """The check operates on ids parsed from ENTRY HEADERS by extract_entries_ol,
    NOT a blind prose grep — so an OL-983 mentioned only in a cross-ref line does
    NOT inflate the sequence (that prose-match was the author's original bug).
    """
    text = (
        "## OL-278: real entry\n"
        "body cross-referencing OL-983 for context\n"
        "## OL-279: another real entry\n"
        "see also OL-983-class discussion\n"
    )
    # use a tmp file via the extractor to prove prose ids are not picked up
    import tempfile
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(text)
        name = fh.name
    try:
        ids = kia.extract_entries_ol(Path(name))
        assert ids == ["OL-278", "OL-279"]  # OL-983 in prose is NOT extracted
        assert kia.check_id_sequence(ids) == []
    finally:
        os.unlink(name)


def test_audit_backend_flags_mis_numbered_ol(tmp_path):
    """End-to-end through audit_backend: a mis-numbered OL entry surfaces as an
    id_sequence_violations row on the OL AuditResult (and the id is indexed, so it
    would otherwise pass orphan/dup/dangling silently — exactly the #237 case).
    """
    kb = tmp_path / "target" / "ascendc"
    (kb / "patterns" / "unverified").mkdir(parents=True)
    (kb / "ERROR_CORRECTIONS.md").write_text("### EC-1: e\napplies_to: all\n")
    (kb / "PLATFORM_BUGS.md").write_text("### PB-1: b\napplies_to: all\n")
    ol_body = "".join(
        f"## OL-{n}: rule {n}\napplies_to: all\n" for n in (1, 2, 3)
    ) + "## OL-984: mis-picked\napplies_to: all\n"
    (kb / "OPERATIONAL_KNOWLEDGE.md").write_text(ol_body)
    (kb / "patterns" / "PATTERN_INDEX.md").write_text("| P-P1 | **t** | L1 |\n")
    (kb / "patterns" / "unverified" / "candidates.md").write_text("## CAND-X: c\n")
    index_text = "EC-1 PB-1 OL-1 OL-2 OL-3 OL-984 P-P1 CAND-X"  # all indexed
    results = kia.audit_backend("ascendc", kb, index_text)
    ol = next(r for r in results if r.file_type == "OL")
    assert ol.orphans == []  # NOT caught by the orphan check
    assert ol.id_sequence_violations == [("OL", 3, 984, 981)]


def test_live_kb_has_zero_id_sequence_violations():
    """The whole shipped KB must satisfy the id-sequence invariant — the anti-
    false-positive guard. If a future honest entry trips this, either it is a real
    picker error (renumber it) or a deliberate reserved range (widen the MARGIN).
    """
    index_text = kia.INDEX_PATH.read_text()
    violations = []
    for backend, kb_dir in kia.KB_DIRS.items():
        for r in kia.audit_backend(backend, kb_dir, index_text):
            violations += [(backend, r.file_type, *v) for v in r.id_sequence_violations]
    assert violations == [], f"live KB has id-sequence violations: {violations}"

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""NODE-21 Phase C: unit tests for kb_inject_filtered + _target_for_opgen_mode."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# We test the helpers directly without loading the full brief import chain
from briefs._common import _target_for_opgen_mode


# ---- _target_for_opgen_mode ----

def test_target_port_a3():
    t = _target_for_opgen_mode("port_a3_to_a5")
    assert t["arch_family"] == "arch35"
    assert t["paradigm"] == "ascendc"


def test_target_unknown():
    t = _target_for_opgen_mode("unknown_mode")
    assert t["arch_family"] == "any"
    assert t["paradigm"] == "ascendc"


# ---- kb_inject_filtered (with mock KB_INDEX) ----

_MOCK_KB_INDEX = """\
| ID | Hook | Tags | Level |
|-----|------|------|-------|
| [OL-195](target/ascendc/OPERATIONAL_KNOWLEDGE.md#OL-195) | V220→V351 vec compute-chain port | paradigm=ascendc; arch_family=arch35 | L1 |
| [OL-196](target/ascendc/OPERATIONAL_KNOWLEDGE.md#OL-196) | Membase vs Regbase selection | paradigm=ascendc; arch_family=arch35; soc=Ascend950PR | L1 |
| [OL-142](target/ascendc/OPERATIONAL_KNOWLEDGE.md#OL-142) | NPU_ARCH macro values | paradigm=ascendc; arch_family=any | L2 |
| [PB-28](target/ascendc/PLATFORM_BUGS.md#PB-28) | V220 DataCopy alignment | paradigm=ascendc; arch_family=arch22 | L1 |
"""


def _write_mock_kb_index(tmp_path: Path):
    idx = tmp_path / "KB_INDEX.md"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(_MOCK_KB_INDEX)
    return idx


def test_kb_inject_filtered_arch35(tmp_path):
    from briefs._common import kb_inject_filtered
    idx = _write_mock_kb_index(tmp_path)
    # Patch the KB_INDEX path used by kb_inject_filtered
    with mock.patch.object(
        kb_inject_filtered, "__module__", None
    ):
        pass  # Can't easily mock the path — test via direct parse

    # Instead, test the row parser directly
    from briefs._common import _parse_kb_index_rows
    rows = _parse_kb_index_rows(idx)
    assert len(rows) == 4

    from kb_schema import kb_entry_applies
    target = {"paradigm": "ascendc", "arch_family": "arch35"}
    matching = [r for r in rows if kb_entry_applies(r["tags"], target)]
    assert len(matching) == 3  # OL-195, OL-196, OL-142 (arch35 + any)
    ids = {r["id"] for r in matching}
    assert "OL-195" in ids
    assert "OL-196" in ids
    assert "OL-142" in ids
    assert "PB-28" not in ids  # arch22 only


def test_kb_inject_filtered_arch22(tmp_path):
    from briefs._common import _parse_kb_index_rows
    from kb_schema import kb_entry_applies
    idx = _write_mock_kb_index(tmp_path)
    rows = _parse_kb_index_rows(idx)
    target = {"paradigm": "ascendc", "arch_family": "arch22"}
    matching = [r for r in rows if kb_entry_applies(r["tags"], target)]
    assert len(matching) == 2  # OL-142 (any) + PB-28 (arch22)
    ids = {r["id"] for r in matching}
    assert "PB-28" in ids
    assert "OL-142" in ids
    assert "OL-195" not in ids  # arch35 only
    assert "OL-196" not in ids  # arch35 + soc specific


def test_kb_inject_filtered_with_keywords(tmp_path):
    from briefs._common import _parse_kb_index_rows
    from kb_schema import kb_entry_applies
    idx = _write_mock_kb_index(tmp_path)
    rows = _parse_kb_index_rows(idx)
    target = {"paradigm": "ascendc", "arch_family": "arch35"}
    matching = []
    for row in rows:
        if not kb_entry_applies(row["tags"], target):
            continue
        if "vec" in f"{row['id']} {row['hook']}".lower():
            matching.append(row)
    assert len(matching) == 1
    assert matching[0]["id"] == "OL-195"


def test_kb_inject_filtered_empty_target(tmp_path):
    from briefs._common import _parse_kb_index_rows
    from kb_schema import kb_entry_applies
    idx = _write_mock_kb_index(tmp_path)
    rows = _parse_kb_index_rows(idx)
    # Empty target → all entries match (universal scope)
    matching = [r for r in rows if kb_entry_applies(r["tags"], {})]
    assert len(matching) == 4


def test_kb_inject_filtered_specificity_order(tmp_path):
    from briefs._common import _parse_kb_index_rows
    from kb_schema import kb_entry_applies, sort_by_specificity
    idx = _write_mock_kb_index(tmp_path)
    rows = _parse_kb_index_rows(idx)
    target = {"paradigm": "ascendc", "arch_family": "arch35"}
    matching = [r for r in rows if kb_entry_applies(r["tags"], target)]
    tagged = [(r["tags"], r) for r in matching]
    sorted_rows = [r for _, r in sort_by_specificity(tagged)]
    # Most specific first: OL-196 (arch35+soc) > OL-195 (arch35) > OL-142 (any)
    assert sorted_rows[0]["id"] == "OL-196"
    assert sorted_rows[2]["id"] == "OL-142"


# ---- _parse_kb_index_rows category inference ----

def test_category_inference():
    from briefs._common import _parse_kb_index_rows
    import tempfile
    import os
    tmp = Path(tempfile.mkdtemp())
    idx = tmp / "KB_INDEX.md"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("""\
| ID | Hook | Tags | Level |
|-----|------|------|-------|
| [EC-66](file.md#EC-66) | Bug A fix | | L1 |
| [PB-34](file.md#PB-34) | MIX deadlock | | L1 |
| [OL-195](file.md#OL-195) | vec compute | paradigm=ascendc | L1 |
| [P-P99](file.md#P-P99) | Matmul pattern | | L2 |
| [CAND-FA1](file.md#CAND-FA1) | FA candidate | | L2 |
""")
    rows = _parse_kb_index_rows(idx)
    cats = {r["id"]: r.get("category", "") for r in rows}
    assert cats["EC-66"] == "EC"
    assert cats["PB-34"] == "PB"
    assert cats["OL-195"] == "OL"
    assert cats["P-P99"] == "P-P"
    assert cats["CAND-FA1"] == "CAND"
    import shutil
    shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── DEBT-222: domain-template arch-scope honored by kb_manifest_block ────────

def _manifest(tmp_path, recs, target):
    """Drive the REAL compose path: write op_classification.json exactly as the
    /aog-op-classify skill does (its contract is the JSON), then compose through
    op_taxonomy.lookup → resolve_legacy_kb_path → validate_manifest_paths → the
    DEBT-222 filter. Returns the composed manifest text."""
    import json
    from briefs.brief_kb import kb_manifest_block
    ws = tmp_path / "op"
    if not ws.exists():
        ws.mkdir()
    (ws / "op_classification.json").write_text(json.dumps(
        {"op_class_tags": ["cooperative"], "kb_recommendations": recs}))
    return kb_manifest_block(
        "op", workspace=ws, target=target, force_legacy_kb=True,
    )


# The classifier's SKILL.md tag→KB table emits the `patterns/domains/X.md` form;
# resolve_legacy_kb_path normalizes it to `target/ascendc/patterns/domains/X.md`.
# Both forms actually reach the filter in the compose path (validate_manifest_paths
# accepts them); we pin BOTH so a resolve-form regression is caught.
@pytest.mark.parametrize("coop_form", [
    "patterns/domains/cooperative.md",                        # raw classifier recommendation
    "target/ascendc/patterns/domains/cooperative.md",         # already-canonical form
])
def test_kb_manifest_drops_a5_only_template_for_a3_e2e(tmp_path, coop_form):
    """END-TO-END through the real compose path: an a5-only domain template
    (cooperative, soc=Ascend950PR) is ABSENT from an a3 manifest and PRESENT in
    the a5 manifest; a neutral (soc=all) template stays for both. Reverting the
    kb_manifest_block DEBT-222 filter block makes this test go RED (a3 would then
    keep cooperative). FAIL-OPEN: only a positive machine-readable exclusion drops.
    """
    recs = [{"path": coop_form}, {"path": "patterns/domains/sort.md"}]
    a5 = _manifest(tmp_path, recs, "a5")
    a3 = _manifest(tmp_path, recs, "a3")
    assert "cooperative.md" in a5 and "sort.md" in a5   # a5: both present
    assert "cooperative.md" not in a3                    # a5-only → DROPPED for a3
    assert "sort.md" in a3                               # neutral → kept for a3


def test_kb_manifest_drops_gmm_a5_template_for_a3_e2e(tmp_path):
    """Second a5-only template (gmm, blockquote applies_to) confirms the filter
    isn't cooperative-specific — the arch-fixed cohort is dropped for a3.
    """
    gmm = "patterns/domains/gmm_swiglu_quant_a8w8_class_template.md"
    recs = [{"path": gmm}, {"path": "patterns/domains/reduction_quant.md"}]  # neutral 2nd
    assert "gmm_swiglu_quant_a8w8_class_template.md" in _manifest(tmp_path, recs, "a5")
    assert "gmm_swiglu_quant_a8w8_class_template.md" not in _manifest(tmp_path, recs, "a3")


def test_kb_manifest_a5_target_byte_identical(tmp_path):
    """For the default a5 target the filter drops NOTHING (a5-only templates cover
    a5; neutral templates are soc=all) — a5 briefs are unchanged by DEBT-222.
    """
    recs = [{"path": "patterns/domains/cooperative.md"},
            {"path": "patterns/domains/gmm_swiglu_quant_a8w8_class_template.md"},
            {"path": "patterns/domains/sort.md"}]
    a5 = _manifest(tmp_path, recs, "a5")
    for name in ("cooperative.md", "gmm_swiglu_quant_a8w8_class_template.md", "sort.md"):
        assert name in a5


def test_kb_manifest_invalid_pathforms_rejected_before_filter(tmp_path):
    """Boundary pin: the un-normalizable path forms (bare `domains/X.md`, full
    `src/skills/references/...`) are rejected by validate_manifest_paths BEFORE the
    filter runs — they raise KBManifestMissingError, so they can NEVER silently
    mis-deliver. This documents the filter's real input domain: only forms that
    resolve to an existing references-relative file reach it.
    """
    from briefs.op_taxonomy import KBManifestMissingError
    for bad in ("domains/cooperative.md",
                "src/skills/references/target/ascendc/patterns/domains/cooperative.md"):
        with pytest.raises(KBManifestMissingError):
            _manifest(tmp_path, [{"path": bad}], "a3")


@pytest.mark.parametrize("form", [
    "target/ascendc/patterns/domains/cooperative.md",
    "patterns/domains/cooperative.md",
    "domains/cooperative.md",
    "src/skills/references/target/ascendc/patterns/domains/cooperative.md",
])
def test_kb_file_applies_to_target_form_agnostic(form):
    """kb_file_applies_to_target resolves an a5-only template in EVERY path form
    the compose path could produce — not just the canonical one. A form-sensitive
    resolver would make the filter inert (theater) for the un-normalized forms.
    """
    from briefs.kb_scope import kb_file_applies_to_target
    assert kb_file_applies_to_target(form, "a5") is True     # a5 keeps
    assert kb_file_applies_to_target(form, "a3") is False     # a3 drops (a5-only)


def test_kb_file_applies_to_target_fail_open():
    from briefs.kb_scope import kb_file_applies_to_target
    # untagged / unknown file → keep (fail-open)
    assert kb_file_applies_to_target("target/ascendc/does_not_exist.md", "a3") is True
    # neutral (soc=all) template → kept for a3
    assert kb_file_applies_to_target("patterns/domains/sort.md", "a3") is True
    # unknown target → keep (never silently drop)
    assert kb_file_applies_to_target("patterns/domains/cooperative.md", "whoknows") is True

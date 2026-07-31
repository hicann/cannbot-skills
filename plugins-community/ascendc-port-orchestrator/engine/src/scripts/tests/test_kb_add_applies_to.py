# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for NODE-21 Phase B KB applies_to migration script."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src" / "scripts" / "migrations"))

import kb_add_applies_to as mod  # noqa: E402


def _entry(body: str, file_name: str = "OPERATIONAL_KNOWLEDGE.md", id_str: str = "OL-99") -> mod.Entry:
    lines = body.splitlines()
    return mod.Entry(
        file=Path(f"src/skills/references/target/ascendc/{file_name}"),
        id=id_str,
        start_line=0,
        end_line=len(lines),
        header=lines[0] if lines else "",
        body_lines=lines,
    )


def test_infer_paradigm_from_path():
    e = _entry("## OL-99: foo\nbody")
    assert getattr(mod, '_infer_tags')(e).get("paradigm") == "ascendc"


def test_infer_arch_family_loose_v300_prose_does_not_tag():
    """Task #41 mis-tag fix: loose 'V300' prose alone must NOT set arch_family
    (it appears in comparative asides inside generic entries — OL-7/OL-88
    mis-tag class). Only deterministic signals (macro / literal arch token)
    set it.
    """
    e = _entry("## OL-99: foo\nunlike V300 this is a generic rule\n")
    assert getattr(mod, '_infer_tags')(e).get("arch_family") is None


def test_infer_arch_family_from_npu_arch_macro():
    """Deterministic: __NPU_ARCH__ macro value maps to family."""
    e = _entry("## OL-99: foo\n`__NPU_ARCH__ == 3510` guard\n")
    assert getattr(mod, '_infer_tags')(e).get("arch_family") == "arch35"
    e2 = _entry("## OL-99: foo\n`__NPU_ARCH__ == 3003` V220 guard\n")
    assert getattr(mod, '_infer_tags')(e2).get("arch_family") == "arch22"


def test_infer_arch_family_from_literal_arch_token():
    """Literal arch35/ directory-style token (not 'V300' prose) sets it."""
    e = _entry("## OL-99: foo\nwrite the `arch35/foo.h` variant\n")
    assert getattr(mod, '_infer_tags')(e).get("arch_family") == "arch35"


def test_infer_arch_family_ambiguous_both_tokens_skips():
    """When both arch35 and arch22 literals appear, don't guess — leave unset."""
    e = _entry("## OL-99: foo\nport `arch22/x.h` to `arch35/x.h`\n")
    # macro absent + both literals present → neither branch fires
    assert getattr(mod, '_infer_tags')(e).get("arch_family") is None


def test_infer_soc_explicit():
    e = _entry("## OL-99: foo\nverified on Ascend950PR_9579\n")
    assert getattr(mod, '_infer_tags')(e).get("soc") == "Ascend950PR_9579"


def test_infer_npu_arch_macro():
    e = _entry("## OL-99: foo\n`__NPU_ARCH__ == 3510` guard\n")
    assert getattr(mod, '_infer_tags')(e).get("npu_arch") == "3510"


def test_infer_cann_version():
    e = _entry("## OL-99: foo\nFix lands in CANN 9.0.0 release\n")
    assert getattr(mod, '_infer_tags')(e).get("cann") == "9.0.0"


def test_has_applies_to_single_line():
    body = "## OL-99: foo\n`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=attention`\nbody"
    e = _entry(body)
    assert getattr(mod, '_has_applies_to')(e) is True


def test_has_applies_to_yaml_block():
    body = "## OL-99: foo\n\n```yaml\napplies_to:\n  paradigm: ascendc\n```\nbody"
    e = _entry(body)
    assert getattr(mod, '_has_applies_to')(e) is True


def test_has_applies_to_no_line():
    e = _entry("## OL-99: foo\nplain body\n")
    assert getattr(mod, '_has_applies_to')(e) is False


def test_process_entry_skips_existing_single_line():
    """Conservative policy: never touch an entry that already has applies_to."""
    body = "## OL-99: foo\n`applies_to: arch_family=arch22`\nV300 mentioned\n"
    e = _entry(body)
    out = getattr(mod, '_process_entry')(e)
    assert out is None  # skipped — already has a scope tag


def test_process_entry_skips_existing_yaml_block():
    """Conservative policy: no duplicate YAML block insertion (task #41 bug 1)."""
    body = (
        "## OL-99: foo\n\n```yaml\n"
        "applies_to:\n"
        "  paradigm: ascendc\n"
        "  arch_family: arch35\n"
        "```\nbody about Ascend950PR\n"
    )
    e = _entry(body)
    out = getattr(mod, '_process_entry')(e)
    assert out is None  # skipped — already has a YAML block, no duplicate


def test_yaml_block_format():
    text = getattr(mod, '_yaml_block')({"paradigm": "ascendc", "arch_family": "arch35"})
    assert text == "```yaml\napplies_to:\n  paradigm: ascendc\n  arch_family: arch35\n```"


def test_yaml_block_empty():
    assert getattr(mod, '_yaml_block')({}) == ""


def test_process_entry_inserts_paradigm_only_by_default():
    """Default bulk policy: paradigm only — discriminating tags stripped
    (OL-173 catalog mis-scope lesson).
    """
    e = _entry("## OL-99: foo\nplain body about Ascend950PR\n")
    out = getattr(mod, '_process_entry')(e)
    assert out is not None
    assert "paradigm: ascendc" in out
    assert "soc:" not in out          # discriminating tag stripped in default mode


def test_process_entry_full_infer_keeps_discriminating():
    """Reviewed path (paradigm_only=False) keeps high-precision discriminating tags."""
    e = _entry("## OL-99: foo\nverified on Ascend950PR_9579\n")
    out = getattr(mod, '_process_entry')(e, paradigm_only=False)
    assert out is not None
    assert "paradigm: ascendc" in out
    assert "soc: Ascend950PR_9579" in out


def test_process_entry_returns_none_when_no_tags():
    e = _entry("## OL-99: foo\n")
    # file_name still ascendc → paradigm=ascendc inferred
    out = getattr(mod, '_process_entry')(e)
    assert out is not None  # paradigm inference produces a tag
    assert "paradigm: ascendc" in out


def test_process_entry_skips_entry_with_no_inference():
    # Override file path so paradigm inference fails (unknown directory)
    e = mod.Entry(
        file=Path("/tmp/unknown/OPERATIONAL_KNOWLEDGE.md"),
        id="OL-99",
        start_line=0,
        end_line=1,
        header="## OL-99: foo",
        body_lines=["## OL-99: foo", "no scope keywords here"],
    )
    out = getattr(mod, '_process_entry')(e)
    assert out is None

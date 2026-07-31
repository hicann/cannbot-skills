# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Guard tests for terminology_gate.py — the V220/V300 arch-naming gate.

The four mutation cases the design requires (bare V220 -> fail; 220x -> pass;
V300x -> pass; rule-doc text -> pass) are asserted here, plus the false-positive
sources the predicate must survive (CAND ids, lowercase, the allow-marker, and the
"documenting the rule must not trip the rule" guarantee).
"""
import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
_MOD_PATH = _HERE.parent.parent / "terminology_gate.py"
_spec = importlib.util.spec_from_file_location("terminology_gate", _MOD_PATH)
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)


# --- The four required mutation cases -----------------------------------------

def test_bare_v220_fails():
    assert tg.scan_line("target the V220 kernel path") == ["V220"]


def test_bare_v300_fails():
    assert tg.scan_line("inspect the V300 arch35 source") == ["V300"]


def test_real_arch_token_220x_passes():
    assert tg.scan_line("generate for arch 220x and 351x") == []


def test_v300x_trailing_x_passes():
    # V300x / V351x / V220x are the real hiascend arch tokens (trailing x).
    assert tg.scan_line("see NPU架构版本V300x and V351x and V220x") == []


def test_claude_md_rule_text_passes():
    # The rule's own documentation must quote the forbidden token yet not trip.
    # It carries the allow-marker; this is the "documenting the rule must not
    # trip the rule" guarantee.
    rule = ('Never write "V220"/"V300": use arch 220x (a3) / 351x (a5). '
            "<!-- terminology-ok -->")
    assert tg.scan_line(rule) == []


# --- False-positive sources the predicate must survive ------------------------

def test_lowercase_v220_v300_fire():
    assert tg.scan_line("the v220 and v300 targets") == ["v220", "v300"]


def test_cand_id_v300sync_not_flagged():
    # V300 followed by a word char (S) is already not a whole word.
    assert tg.scan_line("regression from CAND-V300SYNC-1 applies here") == []


def test_cand_id_hyphen_bounded_not_flagged():
    # CAND-V220-V300-FA-DIFF-1: V220/V300 ARE hyphen-bounded whole words, but the
    # whole CAND-... id is stripped before scanning, so the line is clean.
    assert tg.scan_line("cross-ref CAND-V220-V300-FA-DIFF-1 for the diff") == []


def test_bare_token_on_same_line_as_cand_id_still_fires():
    # Stripping the id must NOT blind us to a genuine bare token elsewhere.
    line = "CAND-V300SYNC-1 documents the V220 path"
    assert tg.scan_line(line) == ["V220"]


def test_removing_marker_makes_rule_text_fire():
    # Proves the marker is load-bearing (the gate WOULD catch a stray V220 even in
    # CLAUDE.md), not that the token is silently ignored.
    rule = 'Never write "V220"/"V300": use arch 220x (a3) / 351x (a5).'
    assert tg.scan_line(rule) == ["V220", "V300"]


def test_v351_bare_is_not_policed():
    # 351 is not a colliding product token; only 220/300 are policed.
    assert tg.scan_line("the V351 alias") == []


def test_word_embedded_not_flagged():
    # DEVV220X / xV300y etc. — embedded, not a bare token.
    assert tg.scan_line("myV220var and REVV300") == []


# --- scan_added_lines aggregation + env escape hatch --------------------------

def test_scan_added_lines_reports_path_and_token():
    added = [("a.md", "clean 220x line"), ("b.py", "# V300 here")]
    out = tg.scan_added_lines(added)
    assert out == [("b.py", "V300", "# V300 here")]


def test_self_machinery_files_are_exempt():
    # The detector, its test, and the hook carry the literal tokens as their own
    # definition; scanning them would self-flag the rule. Exact-path exempt.
    added = [
        ("src/scripts/terminology_gate.py", "block bare V220 here"),
        ("src/scripts/tests/test_terminology_gate.py", 'assert x == ["V300"]'),
        (".githooks/pre-commit", "# V220/V300 gate"),
    ]
    assert tg.scan_added_lines(added) == []


def test_non_machinery_file_still_scanned():
    # The exemption is path-exact, not a free pass to any file mentioning the token.
    added = [("docs/whatever.md", "new bare V220 in a doc")]
    assert tg.scan_added_lines(added) == [("docs/whatever.md", "V220", "new bare V220 in a doc")]


def test_env_off_disables_gate(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(getattr(tg, '_ENV_OFF'), "1")
    assert tg.main([]) == 0
    assert "disabled" in capsys.readouterr().out

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-98 regression test: P96 infra_baseline_paper_over gate must NOT
fire on negative-assertion text (worker documenting they DID NOT perform
paper-over per P9 discipline) OR backtick-cited keyword lists.

Caught 2026-05-24 GMSQ_v2 cold-start: gate fired 80+ times in infinite loop
on PROGRESS.md text like:
  `replaced libophost` / `replacing libopapi` / ... — **NONE present**.
or:
  did NOT replace .so manually, did NOT skip --clean, did NOT bypass --pkg.
"""
import logging
import sys
import tempfile
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from finalize_pipeline import (
    _check_infra_paper_over,
    _is_negative_assertion,
    _is_negative_assertion_window,
)


def test_negation_marker_detection():
    """Each canonical negation marker is detected."""
    assert _is_negative_assertion("did not attempt to ")
    assert _is_negative_assertion("did not replace ")
    assert _is_negative_assertion("we declined to perform ")
    assert _is_negative_assertion("none present in workspace ")
    assert _is_negative_assertion("we did not bypass ")
    # negative case
    assert not _is_negative_assertion("we performed the ")
    assert not _is_negative_assertion("attempting to ")


def test_backtick_cited_keyword_not_flagged():
    """When keyword is in `backticks`, treat as citation not action."""
    text = "audit checks for `replace libophost` / `bypass --pkg` keywords"
    idx = text.lower().find("replace libophost")
    assert _is_negative_assertion_window(text.lower(), idx, idx + len("replace libophost"))


def test_post_match_negation_window():
    """Negation marker AFTER the keyword (e.g. `...` — **NONE present**)."""
    text = "we checked `replace libophost` and `bypass --pkg` — none present"
    idx = text.lower().find("replace libophost")
    assert _is_negative_assertion_window(text.lower(), idx, idx + len("replace libophost"))


def test_real_world_p9_affirmation_not_flagged():
    """Real GMSQ_v2 PROGRESS.md negative-assertion lines."""
    test_lines = [
        '**P9 (Infra paper-over)**: did NOT replace any .so, manually merge binary_info_config, or bypass --pkg.',
        '**P9 (Infra paper-over)**: did NOT replace .so manually, did NOT skip --clean, did NOT bypass --pkg.',
    ]
    for line in test_lines:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            (ws / "PROGRESS.md").write_text(line)
            result = _check_infra_paper_over(ws)
            assert result is None, f"False-positive on negative-assertion: {line[:80]}... → {result}"


def test_real_positive_paper_over_still_flagged():
    """Real positive paper-over (worker DID the paper-over) MUST still trip."""
    text = "kw-5: I replaced libophost.so in the install tree to work around the env issue."
    with tempfile.TemporaryDirectory() as tmp:
        ws = pathlib.Path(tmp)
        (ws / "PROGRESS.md").write_text(text)
        result = _check_infra_paper_over(ws)
        # NOTE: real positive may also trip via npu_error_present check;
        # this test asserts the paper_over keyword detection IS triggered
        # when there's no negation context — exact result format may
        # vary but it must NOT be None.
        assert result is not None, "True positive missed: real paper-over text not flagged"


def test_debt194_anti_pattern_citation_list_not_flagged():
    """DEBT-194 (2026-07-03): the REAL gate-#2 gelu-run line that P96
    false-positived + looped on — the worker CITING the anti-patterns to
    distinguish a legitimate container-restart from them (documenting what was
    NOT done), which the protocol asks for. Must PASS.
    """
    line = (
        "Resolved by DISTINGUISHING legitimate recovery (start an "
        "expected-running, SIGKILLed container) from the actual P9 anti-patterns "
        "(replace .so / bypass --pkg / merge binary / retry on error codes). "
        "Confirmed the kill was a host-level batch event, not a per-op defect."
    )
    with tempfile.TemporaryDirectory() as tmp:
        ws = pathlib.Path(tmp)
        (ws / "PROGRESS.md").write_text(line)
        result = _check_infra_paper_over(ws)
        assert result is None, f"DEBT-194 false-positive on anti-pattern citation: {result}"


def test_debt194_slash_enumeration_in_parens_not_flagged():
    """DEBT-194: structural — a phrase inside a parenthetical slash-list of
    >=2 items is an ENUMERATION (cite), not an action.
    """
    text = "the anti-patterns to avoid are (manual install / bypass --pkg / hand-edit binary_info_config)"
    idx = text.lower().find("bypass --pkg")
    assert _is_negative_assertion_window(text.lower(), idx, idx + len("bypass --pkg"))


def test_debt194_single_action_with_one_slash_still_flagged():
    """DEBT-194 hole-check: a REAL paper-over dressed with a single slash
    (not a >=2-item cite-list) MUST still fire — no evasion hole.
    """
    text = "Step: I performed bypass --pkg / then rebuilt to get past the env error."
    with tempfile.TemporaryDirectory() as tmp:
        ws = pathlib.Path(tmp)
        (ws / "PROGRESS.md").write_text(text)
        result = _check_infra_paper_over(ws)
        assert result is not None, "DEBT-194 hole: single-action paper-over with one slash was exempted"


if __name__ == "__main__":
    test_negation_marker_detection()
    test_backtick_cited_keyword_not_flagged()
    test_post_match_negation_window()
    test_real_world_p9_affirmation_not_flagged()
    test_real_positive_paper_over_still_flagged()
    test_debt194_anti_pattern_citation_list_not_flagged()
    test_debt194_slash_enumeration_in_parens_not_flagged()
    test_debt194_single_action_with_one_slash_still_flagged()
    logging.info("all 8 DEBT-98/194 negative-assertion tests pass")

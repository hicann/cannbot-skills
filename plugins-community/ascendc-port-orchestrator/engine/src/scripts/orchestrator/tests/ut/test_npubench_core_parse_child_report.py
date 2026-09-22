# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.opensource.org/licenses/MIT
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------------------------------------
"""Glued-noise recovery for the child report wire protocol.

A cold GE/tbe initialization prints progress dots to stdout without a
newline, so the precision child's single JSON report line can arrive as
``..{...}`` — without recovery the parent would classify a completed,
measured FAIL child as "no machine-readable report" and O5 would route a
real kernel defect into the infra-retry lane.
"""
from __future__ import annotations

import json

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)

from npubench.npubench_core import _parse_child_report

REPORT = json.dumps(
    {
        "schema": "cannbot.npubench.precision/v1",
        "status": "FAIL",
        "nested": {"case": 0, "metrics": {"max_abs": 1.5}},
    }
)


def test_clean_trailing_line_still_parses():
    assert _parse_child_report(REPORT + "\n") == json.loads(REPORT)


def test_glued_progress_dots_are_recovered():
    stdout = ".." + REPORT + "\n"
    assert _parse_child_report(stdout) == json.loads(REPORT)


def test_multiline_noise_before_glued_report_is_recovered():
    stdout = "ge init noise\n.......W914 padding" + REPORT
    assert _parse_child_report(stdout) == json.loads(REPORT)


def test_prefix_noise_containing_braces_does_not_win():
    stdout = "stage {warmup}" + REPORT
    assert _parse_child_report(stdout) == json.loads(REPORT)


def test_trailing_garbage_fails_closed():
    stdout = ".." + REPORT + "\ntail teardown noise"
    assert _parse_child_report(stdout) is None


def test_plain_noise_without_json_returns_none():
    assert _parse_child_report("....\n") is None
    assert _parse_child_report("") is None
    assert _parse_child_report(None) is None


def test_inner_object_of_the_document_does_not_win():
    # The rightmost '{' is the nested object; it parses but leaves trailing
    # text, so the scan must fall back to the document start.
    stdout = ".." + REPORT
    parsed = _parse_child_report(stdout)
    assert parsed == json.loads(REPORT)
    assert parsed["status"] == "FAIL"

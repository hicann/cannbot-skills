# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression test for report_not_log_lint.py — the deterministic "Report != Log" lint.

Dual-fixture (per main's no-false-positive requirement, 2026-06-20):
  - a LEGIT report (clean comparison-data front section) MUST pass (exit 0, 0 violations)
  - a VIOLATING report (log narrative + event phrases in the front core-data section) MUST be
    caught (exit 2, >0 violations)

The legit fixture deliberately includes the patterns a too-aggressive lint would false-positive on
(a "✅ 达标" pass-marker table cell, a `>` blockquote that mentions "40×-NO-GO 翻案 ... 见 §三", an
intro line summarizing the conclusion, history in §三) — so this test guards against the lint
regressing into false-positives.
"""
import logging
import subprocess
import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

_REPO = _reorg_paths.REPO_ROOT

_LINT = _REPO / "src" / "scripts" / "report_not_log_lint.py"
_FIX = _REPO / "src" / "scripts" / "tests" / "fixtures"


def _run(fixture):
    r = subprocess.run([sys.executable, str(_LINT), str(_FIX / fixture)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def test_legit_report_passes_no_false_positive():
    code, out = _run("report_not_log_legit.md")
    assert code == 0, f"LEGIT fixture wrongly flagged (false-positive):\n{out}"
    assert "report_not_log_violations=0" in out, out


def test_violating_report_is_caught():
    code, out = _run("report_not_log_violating.md")
    assert code == 2, f"VIOLATING fixture not caught (lint too weak):\n{out}"
    assert "report_not_log_violations=0" not in out, out
    assert "C1 log-event-phrase" in out, out


if __name__ == "__main__":
    for fn in (test_legit_report_passes_no_false_positive, test_violating_report_is_caught):
        fn()
        logging.info(f"PASS {fn.__name__}")

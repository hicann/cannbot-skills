# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test the three post-slim report-quality lints.

The owner-requested items from 2026-06-19 are:
  1. report_jargon_lint.py        — internal-harness jargon in the FRONT (核心数据) prose.
  2. report_archive_link_lint.py  — every front core-results table carries its archive link.
  3. report_concision_advisory.py — front-prose-length ADVISORY (never blocks).

Lineage: mirrors PR #473's report_not_log_lint dual-fixture pattern. Per main's HARD no-false-positive
requirement, each blocking lint (jargon, archive-link) has a LEGIT fixture (passes clean, exit 0) AND a
VIOLATING fixture (caught, exit 2). The legit fixtures deliberately carry the patterns a too-aggressive
lint would false-positive on:
  - jargon legit: jargon (cannbot / OL-/DEBT-/P0cc / a bare SHA) living in §二 history (OUT of front) +
    backticked technical tokens (`verification.json`, `compare_cv`) in front blockquotes → must NOT fire.
  - archive legit: a `归档:` link line under each core-results table + a small key=value table in §二
    (not a core-results table → must NOT be required to carry a link).

Also asserts the REAL current reports stay CLEAN on all three lints (no-false-positive on production
reports, after the archive-links were added to selective_scan + BabelStream).
"""
import logging
import subprocess
import sys

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

_REPO = _reorg_paths.REPO_ROOT
_SCRIPTS = _REPO / "src" / "scripts"
_FIX = _SCRIPTS / "tests" / "fixtures"
_LOG = logging.getLogger(__name__)


def _run(script, target):
    r = subprocess.run([sys.executable, str(_SCRIPTS / script), str(target)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


# ---- item 1: jargon lint ----------------------------------------------------

def test_jargon_legit_passes_no_false_positive():
    code, out = _run("report_jargon_lint.py", _FIX / "report_jargon_legit.md")
    assert code == 0, f"LEGIT jargon fixture wrongly flagged (false-positive):\n{out}"
    assert "report_jargon_violations=0" in out, out


def test_jargon_violating_is_caught():
    code, out = _run("report_jargon_lint.py", _FIX / "report_jargon_violating.md")
    assert code == 2, f"VIOLATING jargon fixture not caught (lint too weak):\n{out}"
    assert "report_jargon_violations=0" not in out, out
    assert "J:cannbot" in out, out


# ---- item 2: archive-link lint ---------------------------------------------

def test_archive_link_legit_passes_no_false_positive():
    code, out = _run("report_archive_link_lint.py", _FIX / "report_archive_link_legit.md")
    assert code == 0, f"LEGIT archive fixture wrongly flagged (false-positive):\n{out}"
    assert "report_archive_link_violations=0" in out, out


def test_archive_link_violating_is_caught():
    code, out = _run("report_archive_link_lint.py", _FIX / "report_archive_link_violating.md")
    assert code == 2, f"VIOLATING archive fixture not caught (lint too weak):\n{out}"
    assert "report_archive_link_violations=0" not in out, out
    assert "A:table-missing-archive-link" in out, out


# ---- item 3: concision advisory (never blocks) -----------------------------

def test_concision_advisory_never_blocks(tmp_path):
    # Even a report with over-long front prose must exit 0 — it's advisory, not a gate.
    report = tmp_path / "REPORT.md"
    report.write_text(
        "# Report\n\n## 核心数据\n\n" + "这是一段需要精简的说明。" * 50 + "\n",
        encoding="utf-8",
    )
    code, out = _run("report_concision_advisory.py", report)
    assert code == 0, f"concision advisory must never block (exit 0), got {code}:\n{out}"
    assert "report_concision_advisories=" in out, out


# ---- no-false-positive on the REAL current reports -------------------------

_REAL_REPORTS = [
    _REPO / "output" / "BabelStream" / "docs" / "REPORT.md",
]


def test_real_reports_clean_on_jargon():
    for rep in _REAL_REPORTS:
        if not rep.exists():
            continue
        code, out = _run("report_jargon_lint.py", rep)
        assert code == 0, f"real report false-positive on jargon lint: {rep}\n{out}"


def test_real_reports_clean_on_archive_link():
    for rep in _REAL_REPORTS:
        if not rep.exists():
            continue
        code, out = _run("report_archive_link_lint.py", rep)
        assert code == 0, f"real report false-positive on archive-link lint: {rep}\n{out}"


def test_real_reports_clean_on_report_not_log():
    # the archive-link lines added to the real reports must not regress the existing Report≠Log lint
    for rep in _REAL_REPORTS:
        if not rep.exists():
            continue
        code, out = _run("report_not_log_lint.py", rep)
        assert code == 0, f"real report regressed report_not_log lint after archive-link add: {rep}\n{out}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            _LOG.info("PASS %s", name)

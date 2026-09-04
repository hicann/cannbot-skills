# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-168 regression: _check_post_worker_audit must recognize the self-critic's
actual PARTIAL+waiver verdict formats.

Before the fix, the gate's waiver grant check was
    `"waiver:" in audit_lower or "waiver " in audit_lower`
which required "waiver" to be followed by `:` or a literal space. The
/aog-self-critic skill actually writes the grant in formats like

    `**Overall verdict**: ⚠ **PARTIAL + waiver** — …`
    `## Verdict: PARTIAL + waiver`
    `waiver <reason>`
    `waiver: <reason>`

i.e. "waiver" followed by `**`, a newline, or `.`. None of those satisfy the
old substring test, so a legitimate PARTIAL+waiver verdict was falsely
blocked (observed on top_k_top_p_sample, 2026-06-24 — the doc said
"PARTIAL + waiver**" but the gate reported "verdict is not PASS (and not
PARTIAL+waiver)").

The fix matches "waiver" as a whole word (with negative lookbehind so a
*denied* waiver — "no waiver" / "without waiver" — does NOT grant passage).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import finalize_pipeline as fp  # noqa: E402

_VERDICT_BLOCK = "not PASS (and not PARTIAL+waiver)"


def _vj():
    return {"precision": {"status": "PASS",
                          "pass_a": {"status": "PASS", "tier1_pass": 16, "total": 16}}}


def _write_audit(ws: Path, body: str) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "audit_self_critic_post_worker.md").write_text(body)


def test_unparseable_verdict_is_non_blocking(tmp_path):
    """Audit M4 (2026-08-22): an unparseable verdict wording is an
    ambiguous harness-authored condition, not fraud — the gate warns and
    continues instead of hard-failing a precision-PASS candidate.
    """
    ws = tmp_path / "op_m4"
    _write_audit(ws, (
        "# Post-Worker Self-Critic Audit\n\n"
        "## Verdict\n\n**PASSED** — all checks green.\n\n"
        "## Findings\n\n- none\n"
    ))
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert _VERDICT_BLOCK not in (err or "")


def test_partial_plus_waiver_bolded_passes(tmp_path):
    """THE OBSERVED BUG: the self-critic's real format — bolded
    "PARTIAL + waiver**". Pre-fix this was blocked; post-fix recognized.
    """
    ws = tmp_path / "op_b"
    _write_audit(ws, (
        "# Post-Worker Self-Critic Audit\n\n"
        "- **Overall verdict**: ⚠ **PARTIAL + waiver** — kernel correct; "
        "perf N/A is structural (no A3 hardware).\n\n"
        "## Verdict: PARTIAL + waiver\n"
    ))
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    # verdict stage must NOT block (it may fail a *later* check — marker/
    # pass_b/perf — but must not return the verdict-block message)
    assert err is None or _VERDICT_BLOCK not in err, (
        f"PARTIAL+waiver (bolded) must be recognized as waived; got: {err}"
    )


def test_partial_waiver_colon_passes(tmp_path):
    ws = tmp_path / "op_c"
    _write_audit(ws, "# audit\nVerdict: PARTIAL + waiver: perf-NA structural\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is None or _VERDICT_BLOCK not in err


def test_partial_waiver_inline_space_passes(tmp_path):
    ws = tmp_path / "op_d"
    _write_audit(ws, "# audit\nVerdict: PARTIAL. waiver granted (structural).\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is None or _VERDICT_BLOCK not in err


def test_partial_no_waiver_blocked(tmp_path):
    """PARTIAL verdict WITHOUT any waiver grant must block (the gate's
    safety property: unwaived issues are not shippable).
    """
    ws = tmp_path / "op_e"
    _write_audit(ws, "# audit\nVerdict: PARTIAL — coverage gap in bf16 Q-path remains.\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is not None and _VERDICT_BLOCK in err


def test_partial_denied_waiver_blocked(tmp_path):
    """Negative-lookbehind guard: a PARTIAL verdict that mentions "no waiver"
    must NOT be treated as a granted waiver.
    """
    ws = tmp_path / "op_f"
    _write_audit(ws, "# audit\nVerdict: PARTIAL — no waiver granted; must fix.\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is not None and _VERDICT_BLOCK in err, (
        "'no waiver granted' must NOT count as a waiver grant"
    )


def test_fail_verdict_blocked(tmp_path):
    ws = tmp_path / "op_g"
    _write_audit(ws, "# audit\nVerdict: FAIL — delegation cheating detected.\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is not None and _VERDICT_BLOCK in err


def test_pass_verdict_recognized(tmp_path):
    """Sanity: a clean PASS verdict proceeds past the verdict stage."""
    ws = tmp_path / "op_h"
    _write_audit(ws, "# audit\nVerdict: PASS — all catalog items clean.\n")
    err = getattr(fp, '_check_post_worker_audit')(ws, _vj())
    assert err is None or _VERDICT_BLOCK not in err

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0w (2026-05-05): _extract_handoff_from_progress scopes to last EXIT marker.

Origin: op#28 multimodal_rope 2026-05-05 quota-resume. After --cold-start
cleared workspace state, kw-1 ran and wrote PROGRESS.md with full Phase A
activity but ALSO a historical "→ orchestrator: done" line in the middle
(template scaffold or older session content preserved). Worker died at
quota mid-iter without writing a final exit handoff. P0r resume
re-invoked orchestrator. P0o read PROGRESS.md, matched the historical
"done" via last-line-of-canonical-prefix logic, routed to finalize
incorrectly.

Fix: scope candidate search to lines AFTER the last "### EXIT" / "## EXIT"
marker. If no EXIT marker, take only last 30 lines as recency heuristic.
This catches the common worker-finalize pattern where workers append a
trailing "### EXIT" section AND ignores mid-document historical content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402


def test_p0w_picks_handoff_after_exit_marker(tmp_path):
    """Historical 'done' line in middle + EXIT marker + new handoff at end →
    extract picks the handoff AFTER the EXIT marker, ignoring the historical.
    """
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op
Mode: backward

### [10:30] kw-1 (Phase A)
Some early activity.

→ orchestrator: done

### [10:35] kw-1 (Phase D verify)
Build PASS, precision PASS.

### EXIT
@aog-kernel-optimizer — perf 0.19× < 0.6× threshold; needs tuning
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h is not None
    assert h.startswith("@aog-kernel-optimizer"), \
        f"Expected handoff after EXIT marker, got {h!r}"


def test_p0w_ignores_mid_document_done_with_no_exit_marker(tmp_path):
    """No EXIT marker but mid-document historical 'done' present →
    last 30 lines window catches end-of-doc handoff if it's the actually-recent one.
    """
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op
Mode: backward

→ orchestrator: done    (historical from old session)

### [10:30] kw-1 (Phase A)
""" + "Some scaffolded content here.\n" * 50 + """
### [11:00] kw-1 (Phase D verify)
Some new fresh worker entry but ended without writing exit handoff.
Worker died on quota. No new canonical line.
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    # No canonical line in last 30 lines → returns None (correct behavior:
    # don't pick the mid-document historical line)
    assert h is None, \
        f"Should NOT extract historical mid-document handoff outside scope, got {h!r}"


def test_p0w_exit_marker_variants(tmp_path):
    """Various EXIT marker forms recognized."""
    cases = [
        "### EXIT",
        "## EXIT",
        "# EXIT",
        "EXIT",
        "**EXIT**",
        "**Exit**",
    ]
    for marker in cases:
        (tmp_path / "PROGRESS.md").write_text(f"""\
# Op
→ orchestrator: done   (historical)

{marker}
@aog-precision-probe — fresh handoff
""")
        h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
        assert h is not None and h.startswith("@aog-precision-probe"), \
            f"marker={marker!r} should scope to fresh handoff, got {h!r}"


def test_p0w_canonical_handoff_in_exit_section(tmp_path):
    """Standard worker-finalize pattern: PROGRESS.md ends with EXIT section
    containing canonical handoff.
    """
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS

### [10:30] kw-1 (Phase A)
Analysis done.

### [10:35] kw-1 (Phase D)
Built and verified. Pass A 60/60.

### EXIT
→ orchestrator: done — Pass A 60/60 + Pass B 16/16 + Det 60/60, perf 0.65×
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h.startswith("→ orchestrator: done")
    assert "0.65" in h


def test_p0w_no_exit_marker_canonical_last_line_picked(tmp_path):
    """No EXIT marker but last non-empty line IS canonical → pick it.
    This is the standard worker pattern: append handoff as last line of
    PROGRESS.md per brief.
    """
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op
""" + "stuff\n" * 100 + """\

### [11:00] kw-1 (final exit)
Wrote artifacts.

@aog-kernel-optimizer — perf 0.4× needs tuning
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h is not None and h.startswith("@aog-kernel-optimizer")


def test_p0w_no_exit_marker_last_line_not_canonical_returns_none(tmp_path):
    """No EXIT marker AND last line is not canonical → return None.
    Worker brief says handoff goes on PROGRESS.md tail; if it's not on
    the tail, don't speculatively grab a mid-document line.
    """
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op

→ orchestrator: done    (historical mid-document)

### [10:30] kw-1 doing work
Just finished a build, no handoff written yet.
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h is None, \
        f"Last non-empty line is not canonical, should return None, got {h!r}"


def test_p0w_multiple_exit_markers_picks_last(tmp_path):
    """If file has multiple EXIT markers (rare — multi-iter PROGRESS), use the LAST one."""
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op

### [10:30] kw-1
work

### EXIT
→ orchestrator: done   (kw-1 first iter exit, historical)

### [11:00] kw-2 (respawn)
new work

### EXIT
@aog-precision-probe — kw-2 found stuck
""")
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h is not None and h.startswith("@aog-precision-probe"), \
        f"Should scope to LAST EXIT marker, got {h!r}"


def test_p0w_op28_scenario_exact_replay(tmp_path):
    """Exact reproduction of op#28 quota-resume failure mode."""
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — 28_MultimodalRopePositionComputationWithGridBasedIndexing
Mode: backward
Started: 2026-05-04 20:30 UTC

## Strategy
Strategy notes (template scaffold containing example handoff):
Environment fix applied: workspace/.ascendc_env CANN_PATH=/data/cann_b103/cann-9.0.0.

→ orchestrator: done

### [10:27] aog-kernel-worker (Phase A — cold-start re-validation)
Started cold-start re-validation. Found verifier tightened.

### [10:30] aog-kernel-worker (Phase B — kernel rewrite)
Applied fp16 native-precision fix to model_new_ascendc.py.
Build started.
""")
    # No EXIT marker, fresh Phase B entry doesn't include canonical handoff line
    # → P0w should NOT pick the historical mid-document "done"
    h = getattr(sm, '_extract_handoff_from_progress')(tmp_path)
    assert h is None, \
        f"op#28 scenario: historical 'done' must NOT be picked, got {h!r}"

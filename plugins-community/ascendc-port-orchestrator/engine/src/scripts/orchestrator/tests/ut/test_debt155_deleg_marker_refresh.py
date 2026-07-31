# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-155 regression: deleg_marker_needs_refresh must fire on a STALE marker.

Before the fix, `_ensure_audit_artifacts` re-ran scan_delegation_cheating.py only
when `.delegation_scan_passed` was ABSENT. After a worker rebuilt the kernel, the
marker went STALE (older than the new kernel files) but was never refreshed → the
finalize freshness gate kept rejecting → infinite rollback → LOOP-BREAK →
await_user_decision (observed twice on mul_grad, 2026-06-13).

The producer now uses `deleg_marker_needs_refresh`, which mirrors the gate's
freshness check (finalize_pipeline.py ~2825): ABSENT or STALE → refresh.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import orchestrator  # noqa: E402


def _touch(p: Path, mtime: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (mtime, mtime))


def test_marker_absent_needs_refresh(tmp_path):
    assert orchestrator.deleg_marker_needs_refresh(tmp_path) is True


def test_marker_fresh_no_refresh(tmp_path):
    # kernel files OLDER than the marker → not stale → no refresh.
    _touch(tmp_path / "kernel" / "k_kernel.h", 1000.0)
    _touch(tmp_path / "model_new_ascendc.py", 1000.0)
    _touch(tmp_path / ".delegation_scan_passed", 2000.0)
    assert orchestrator.deleg_marker_needs_refresh(tmp_path) is False


def test_marker_stale_after_kernel_rebuild_needs_refresh(tmp_path):
    # THE BUG: kernel rebuilt (newer) than the existing marker → STALE → must refresh.
    _touch(tmp_path / ".delegation_scan_passed", 1000.0)
    _touch(tmp_path / "kernel" / "k_kernel.h", 3000.0)  # rebuilt after the scan
    assert orchestrator.deleg_marker_needs_refresh(tmp_path) is True


def test_stale_via_model_new_ascendc(tmp_path):
    # model_new_ascendc.py rewritten after the marker also counts as stale.
    _touch(tmp_path / ".delegation_scan_passed", 1000.0)
    _touch(tmp_path / "model_new_ascendc.py", 3000.0)
    assert orchestrator.deleg_marker_needs_refresh(tmp_path) is True


def test_within_slack_not_stale(tmp_path):
    # 1s slack (same as the gate): kernel < marker + 1.0 is NOT stale.
    _touch(tmp_path / ".delegation_scan_passed", 1000.0)
    _touch(tmp_path / "kernel" / "k_kernel.h", 1000.5)
    assert orchestrator.deleg_marker_needs_refresh(tmp_path) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

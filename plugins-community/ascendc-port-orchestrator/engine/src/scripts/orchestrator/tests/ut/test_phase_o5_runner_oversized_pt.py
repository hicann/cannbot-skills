# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression test for phase_o5_runner oversized .pt SCP refusal (P135.TS).

An operator workspace ballooned to 23GB because stress sweeps wrote a
20GB+ edge_inputs.pt.
phase_o5_runner built the tar in-memory via `io.BytesIO` then scp'd —
either OOM'd the host or hit the 300s scp timeout.

Two safety changes locked in here:

1. **Oversized .pt refusal**: any single payload .pt file >100 MiB causes
   `_resync_workspace_to_container` to return a clear error string
   BEFORE attempting tar/scp. This forces input_gen.py to regenerate
   edge cases at sane sizes rather than silently dropping them
   (silent drop = pass_b coverage fraud, banned by
   `feedback_input_requirements_immutable`).

2. **Disk-backed tar**: tar is now written directly to a temp file via
   `tarfile.open(host_tar, mode="w")` rather than buffered in
   `io.BytesIO` first. Prevents OOM on multi-GB tars that the prior
   path could not survive even if scp succeeded.

We mirror the size-check logic here (matching the pattern in
`test_phase_o5_runner_target_cann.py`) so the test is independent of
phase_o5_runner's heavy import graph. If the in-file logic drifts
from this mirror, the contract pin still locks the intended behavior.
"""
from __future__ import annotations

import logging

from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

OVERSIZED_PT_THRESHOLD = 100 * 1024 * 1024  # 100 MiB


def _check_oversized_pt(push_files: list[tuple[str, int]]) -> str | None:
    """Mirror of the size-check block now in
    phase_o5_runner._resync_workspace_to_container (P135.TS, 2026-05-18).

    Takes [(filename, size_bytes), ...] of files that would be tar'd.
    Returns None if all .pt files are within the threshold; returns an
    error string naming the offenders otherwise.
    """
    oversized = [(name, sz) for name, sz in push_files
                 if name.endswith(".pt") and sz > OVERSIZED_PT_THRESHOLD]
    if not oversized:
        return None
    details = ", ".join(f"{name} ({sz / (1024*1024):.1f} MiB)"
                        for name, sz in oversized)
    return (f"phase_o5 SCP aborted: oversized payload .pt files exceed "
            f"100 MiB threshold — {details}. Regenerate edge_inputs.pt / "
            f"edge_dataset.pt with smaller LB stress cases (input_gen.py "
            f"size cap) before re-running. See task #23 (2026-05-18).")


def test_oversized_edge_inputs_refused() -> None:
    """200 MiB edge_inputs.pt → refusal string naming the offender."""
    result = _check_oversized_pt([
        ("model.py", 1024),
        ("edge_inputs.pt", 200 * 1024 * 1024),
        ("verification.json", 4096),
    ])
    assert result is not None
    assert "oversized payload" in result.lower()
    assert "edge_inputs.pt" in result
    assert "200.0 MiB" in result


def test_oversized_edge_dataset_refused() -> None:
    """500 MiB edge_dataset.pt → refusal string naming the offender."""
    result = _check_oversized_pt([
        ("edge_dataset.pt", 500 * 1024 * 1024),
    ])
    assert result is not None
    assert "edge_dataset.pt" in result
    assert "500.0 MiB" in result


def test_multiple_oversized_both_reported() -> None:
    """Both edge_inputs.pt and edge_dataset.pt oversized → both named."""
    result = _check_oversized_pt([
        ("edge_inputs.pt", 200 * 1024 * 1024),
        ("edge_dataset.pt", 300 * 1024 * 1024),
    ])
    assert result is not None
    assert "edge_inputs.pt" in result
    assert "edge_dataset.pt" in result


def test_sane_pt_passes() -> None:
    """1 MiB edge_inputs.pt → no refusal."""
    result = _check_oversized_pt([
        ("model.py", 1024),
        ("edge_inputs.pt", 1024 * 1024),
        ("verification.json", 4096),
    ])
    assert result is None


def test_threshold_boundary_exact() -> None:
    """Exactly 100 MiB is NOT oversized (strictly > threshold)."""
    result = _check_oversized_pt([
        ("edge_inputs.pt", 100 * 1024 * 1024),
    ])
    assert result is None


def test_threshold_boundary_just_over() -> None:
    """100 MiB + 1 byte IS oversized."""
    result = _check_oversized_pt([
        ("edge_inputs.pt", 100 * 1024 * 1024 + 1),
    ])
    assert result is not None
    assert "edge_inputs.pt" in result


def test_non_pt_files_exempt() -> None:
    """Large non-.pt file (e.g., model.py with embedded data) is NOT refused.

    The threshold targets edge_inputs.pt / edge_dataset.pt / a5_capture.pt
    which carry stress-test payloads. A large .py is unusual but not
    the SCP-timeout pattern we're protecting against.
    """
    result = _check_oversized_pt([
        ("model.py", 500 * 1024 * 1024),
    ])
    assert result is None


def test_in_file_logic_matches_mirror() -> None:
    """Read the actual phase_o5_runner.py and verify the threshold
    constant matches this test's mirror. If the source drifts, this
    test fails loudly so the pin can be re-aligned.
    """
    runner = _reorg_paths.ORCH_DIR / "phase_o5_runner.py"
    source = runner.read_text()
    assert "OVERSIZED_PT_THRESHOLD = 100 * 1024 * 1024" in source, (
        "phase_o5_runner.py threshold constant drifted or was renamed — "
        "update this test's mirror to match."
    )
    assert 'p.suffix == ".pt"' in source, (
        "phase_o5_runner.py size-check loop no longer filters by .pt suffix — "
        "the protective scope may have widened or broken."
    )


if __name__ == "__main__":
    test_oversized_edge_inputs_refused()
    test_oversized_edge_dataset_refused()
    test_multiple_oversized_both_reported()
    test_sane_pt_passes()
    test_threshold_boundary_exact()
    test_threshold_boundary_just_over()
    test_non_pt_files_exempt()
    test_in_file_logic_matches_mirror()
    logging.info("OK")

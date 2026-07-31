# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression pin for the backward verify TRUTH-FILE sync gap (2026-07-15).

Incident (selective_scan_full_grad kw-4): phase_o5's `backward_verify_runner`
runs the self-contained `verify_<op>.py` on the NPU. That verifier loads EVERY
case's inputs / grad_outputs / fp64-grads directly from `backward_cpu_truth.pt`
(records used AS-IS, no re-seed). But `_resync_workspace_to_container`:

  1. never listed `backward_cpu_truth.pt` in its push set, AND
  2. would have refused it anyway via the P135.TS oversized-.pt guard
     (the truth file is legitimately >100 MiB — fp64 grads over multi-dim
     records, ~183 MiB for selective_scan).

Net effect: O5 ran verify against a `current_task/` with NO truth file →
`FileNotFoundError: backward_cpu_truth.pt not found` → RUNNER_FAILED →
endless `finalize → await_worker` rollback on a kernel that was already at
its verified 9/48 Tier-2 floor.

A THIRD coupled gap: the remote branch called `_run_verifier(...)` WITHOUT
`raw=True`, so the parsed summary was folded into the normalized
{tier1_pass,total,status} pass_a shape and the `performance` block was
dropped → phase_o5's `independent_re_measure` went missing → finalize perf
gate rollback (the abs_nocase_grad 2026-06-17 gap). The local-container
branch already did it right; the remote branch now mirrors it.

These are source-anchor pins (the live function needs an SSH+docker env);
if the source drifts, the pins fail loudly so the intent is re-aligned.
"""
from __future__ import annotations

import logging

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors


def _runner_source() -> str:
    return (_reorg_paths.ORCH_DIR / "phase_o5_runner.py").read_text()


def test_backward_truth_in_push_list() -> None:
    """backward_cpu_truth.pt must be in the resync push_files name tuple so
    O5's backward verify finds it in current_task/.
    """
    src = _runner_source()
    assert '"backward_cpu_truth.pt"' in src, (
        "backward_cpu_truth.pt dropped from _resync_workspace_to_container "
        "push list — backward verify will FileNotFound-crash on the NPU."
    )


def test_backward_truth_exempt_from_oversized_guard() -> None:
    """The truth file is legitimately >100 MiB and REQUIRED; it must be
    exempt from the P135.TS oversized-.pt refusal.
    """
    src = _runner_source()
    assert 'OVERSIZED_EXEMPT = {"backward_cpu_truth.pt"}' in src, (
        "backward_cpu_truth.pt oversized-guard exemption removed — the O5 "
        "resync will abort finalize on the >100 MiB backward truth file."
    )
    # The guard loop must actually consult the exemption set.
    assert 'p.name not in OVERSIZED_EXEMPT' in src, (
        "oversized-guard loop no longer honors OVERSIZED_EXEMPT — the "
        "backward truth exemption is dead code."
    )


def test_backward_verify_runner_uses_raw_for_perf() -> None:
    """The remote backward_verify_runner must call _run_verifier(..., raw=True)
    so the verify summary's `performance` block survives → phase_o5's
    independent_re_measure is populated (no false perf-gate rollback).
    """
    src = _runner_source()
    assert 'raw=True' in src, (
        "backward_verify_runner remote branch dropped raw=True — _run_verifier "
        "normalizes away the performance block → independent_re_measure missing "
        "→ finalize perf-gate rollback (abs_nocase_grad 2026-06-17 gap)."
    )


if __name__ == "__main__":
    test_backward_truth_in_push_list()
    test_backward_truth_exempt_from_oversized_guard()
    test_backward_verify_runner_uses_raw_for_perf()
    logging.info("OK")

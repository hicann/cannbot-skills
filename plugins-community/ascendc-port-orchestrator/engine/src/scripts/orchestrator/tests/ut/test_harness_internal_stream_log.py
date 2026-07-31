# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression test for `_is_harness_internal` stream-log inclusion
(2026-05-19, ROADMAP §F7 Meta-Harness data preservation).

Empirical anchor: foreach_sqrt + foreach_neg + 9_TopK archives shipped
2026-05-18..19 did NOT include `.cc_stream_log_<agent>_<idx>.jsonl`
files even though those are the richest stream-json traces the harness
produces. Without archive-time copy, those traces get lost on next
cold-start (workspace/ is gitignored + reset).

Meta-Harness (arxiv 2603.28052) ablation: filesystem-based execution
traces are decisive — without them, accuracy collapses 50.0 → 34.9.
So preserving stream-log traces in every shipped archive turns each
finalize into a Meta-Harness training-data candidate.
"""
from __future__ import annotations

import logging

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _is_harness_internal  # noqa: E402


def test_stream_log_files_routed_to_harness():
    """`.cc_stream_log_<agent>_<idx>.jsonl` per-agent stream-json traces
    MUST be classified as harness-internal so they land in archive's
    .harness/ during finalize.
    """
    cases = [
        ".cc_stream_log_aog-kernel-worker_1.jsonl",
        ".cc_stream_log_aog-kernel-worker_4.jsonl",
        ".cc_stream_log_aog-precision-probe_2.jsonl",
        ".cc_stream_log_aog-kernel-optimizer_2.jsonl",
        ".cc_stream_log_aog-researcher_2.jsonl",
        ".cc_stream_log_aog-fused-optimizer_1.jsonl",
        ".cc_stream_log_aog-determinism-analyzer_1.jsonl",
    ]
    for name in cases:
        assert _is_harness_internal(name), (
            f"{name} should be harness-internal so it lands in archive .harness/ "
            f"for Meta-Harness training-data preservation (ROADMAP §F7)"
        )


def test_other_dotted_files_not_routed():
    """Defensive: non-stream-log dotfiles starting with `.cc_*` should
    NOT auto-route — only the explicit `.cc_stream_log_*` prefix.
    """
    cases = [
        ".cc_other_thing.jsonl",
        ".cc_stream_log_no_extension",  # missing .jsonl
        "cc_stream_log_aog-kernel-worker_1.jsonl",  # missing leading dot
        ".cc_stream_log_aog.jsonl.bak",  # wrong extension
    ]
    for name in cases:
        assert not _is_harness_internal(name), (
            f"{name} should NOT be classified as harness-internal; "
            f"only .cc_stream_log_*.jsonl exact pattern matches"
        )


def test_existing_harness_internal_still_routed():
    """Regression: pre-existing harness-internal files still routed."""
    cases = [
        "state_transitions.jsonl",
        "orchestrator_events.jsonl",
        "audit_self_critic_post_worker.md",
        "audit_self_critic_post_worker.STALE_kw3.md",
        "self_critic_report.md",
        "knowledge_update.md",
        ".finalized-abc123",
    ]
    for name in cases:
        assert _is_harness_internal(name), (
            f"{name} was harness-internal pre-2026-05-19 and must remain so"
        )


def test_user_facing_files_not_routed():
    """User-facing files MUST stay at archive root, not .harness/."""
    cases = [
        "verification.json",
        "PROGRESS.md",
        "model.py",
        "model_new_ascendc.py",
        "manifest.json",
        "kernel/some_kernel.h",
    ]
    for name in cases:
        assert not _is_harness_internal(name), (
            f"{name} is user-facing and must stay at archive root, "
            f"not be hidden in .harness/"
        )


if __name__ == "__main__":
    test_stream_log_files_routed_to_harness()
    test_other_dotted_files_NOT_routed()
    test_existing_harness_internal_still_routed()
    test_user_facing_files_NOT_routed()
    logging.info("OK")

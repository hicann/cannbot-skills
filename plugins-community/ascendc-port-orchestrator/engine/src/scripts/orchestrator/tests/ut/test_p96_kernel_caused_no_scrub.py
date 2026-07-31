# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for the P96 self-defeat fix (2026-06-16, abs_nocase backward).

Witnessed incident: P96 (`_check_infra_paper_over`) error-code sub-check was a
bare `if code in text` — it could not distinguish a KERNEL-CAUSED NPU error
(the worker's own kernel UB bug, env proven healthy, fixed in-kernel — NOT an
infra baseline violation) from a real infra paper-over. It thus forced an honest
worker to SCRUB the `507035` token out of PROGRESS.md to pass the gate (keeping
the full kernel-cause mechanism, just deleting the searchable code) — the gate
INDUCING documentation fraud, the inverse of its purpose.

Fix: an error-code occurrence inside a kernel-caused context window
(`_is_kernel_caused_context_window`: env-health + kernel-bug/kernel-fix evidence
within +/-300 chars) is NOT counted as an unescalated infra violation. Honest
records finalize WITHOUT scrubbing. Teeth preserved: a bare / infra-framed
citation with no such context and no INFRA handoff still REJECTS; real structural
workarounds (lib-replace / manual-install / binary-edit) are caught independently
by the phrase signals.

Fixtures are the ACTUAL text from back's abs_nocase worktree (Discord
2026-06-17): the honest BEFORE that should have passed, and the scrubbed AFTER.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_infra_paper_over  # noqa: E402


def _ws(tmp_path: Path, progress_text: str) -> Path:
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "PROGRESS.md").write_text(progress_text)
    return ws


# back's ACTUAL honest BEFORE text (worker-1, cites 507035 with full
# kernel-caused context + env-health proof). This SHOULD finalize — no scrub.
_HONEST_BEFORE = (
    "## iter-1\n"
    "Runtime vector-core exception (507035) on the first fp32 kernel call — "
    "this is a kernel code bug (NPU is healthy), not infra. "
    "Sign(dst,src,count) uses PopStackBuffer for scratch, but "
    "3 queues x depth4 x tile4096 x 4B = 192KB fills the entire a3 UB → "
    "PopStackBuffer ASSERT → AIVEC (vector-core) trap. "
    "Fix: switch Sign() to the explicit sharedTmpBuffer overload "
    "(deterministic 40KB UB).\n"
    "env probe (read-only on A3): env baseline is HEALTHY — torch.sgn works "
    "(sgn(0)=0), multiply works, randn(64).sum() works (CANN 9.0).\n"
)

# The AFTER text worker scrubbed to (507035 token removed, mechanism kept).
_SCRUBBED_AFTER = (
    "## iter-1\n"
    "kernel-execution exception. Root cause = a KERNEL CODE BUG (NOT infra): "
    "Sign() PopStackBuffer<uint8_t,LCM> overflow → AIVEC vector-core "
    "kernel-execution exception. Fix: explicit sharedTmpBuffer overload.\n"
)


def test_honest_kernel_caused_507035_passes_no_scrub(tmp_path: Path) -> None:
    """THE FIX: the honest record (cites 507035 + kernel-caused context +
    env-health proof) must finalize WITHOUT the worker deleting the token.
    """
    ws = _ws(tmp_path, _HONEST_BEFORE)
    result = _check_infra_paper_over(ws)
    assert result is None, (
        "honest kernel-caused 507035 citation must NOT be rejected (it is not "
        f"an infra paper-over); got: {result!r}"
    )


def test_scrubbed_version_also_passes(tmp_path: Path) -> None:
    """The scrubbed AFTER (no 507035 token) also passes — documenting that the
    fix makes scrubbing UNNECESSARY (the honest version passes too), removing
    the incentive to falsify docs.
    """
    ws = _ws(tmp_path, _SCRUBBED_AFTER)
    assert _check_infra_paper_over(ws) is None


def test_bare_infra_507035_still_rejects(tmp_path: Path) -> None:
    """TEETH: a bare / infra-framed error-code citation with NO kernel-caused
    context and NO INFRA_BASELINE_VIOLATED handoff still REJECTS (conservative —
    assumes infra baseline violation needing escalation).
    """
    ws = _ws(
        tmp_path,
        "## iter-1\nKernel call returned 507035. Re-ran a few times and it "
        "eventually went through. Proceeding.\n",
    )
    result = _check_infra_paper_over(ws)
    assert result is not None and "C-INFRA-BASELINE-PAPER-OVER" in result, (
        f"bare/infra-framed 507035 must still be rejected; got: {result!r}"
    )


def test_kernel_caused_but_with_lib_replace_still_rejects(tmp_path: Path) -> None:
    """TEETH: even if 507035 is framed as kernel-caused, an ACTUAL structural
    workaround (lib replace) is caught independently by the phrase signal — the
    error-code exemption does not launder a real paper-over.
    """
    ws = _ws(
        tmp_path,
        _HONEST_BEFORE + "\nAlso had to replace libophost_nn.so to get past it.\n",
    )
    result = _check_infra_paper_over(ws)
    assert result is not None and "C-INFRA-BASELINE-PAPER-OVER" in result


def test_infra_handoff_present_passes(tmp_path: Path) -> None:
    """Existing behavior preserved: error code + proper INFRA_BASELINE_VIOLATED
    escalation handoff passes regardless of kernel-context.
    """
    ws = _ws(
        tmp_path,
        "## iter-1\nHit 507035. Emitted "
        "`→ orchestrator: await_user_decision — INFRA_BASELINE_VIOLATED 507035` "
        "with forensics and stopped.\n",
    )
    assert _check_infra_paper_over(ws) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

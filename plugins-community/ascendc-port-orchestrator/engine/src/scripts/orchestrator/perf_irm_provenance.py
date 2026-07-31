# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Provenance stamping for `performance.independent_re_measure` (DEBT-217).

## The defect this closes

`performance.independent_re_measure` (irm) exists to record ONE property:
**author != measurer** — the perf ratio in the archive was re-measured by the
orchestrator, not self-reported by the worker that wrote the kernel.

Before DEBT-217 that property was asserted but never checked:

- The only writer that stamped a `source` was
  `phase_o5._write_perf_independent_re_measure`, reachable only when a runner
  populates `MeasuredResult.perf` — i.e. **backward** (`backward_verify_runner`)
  and **port_a3 A3-golden** (`_maybe_port_a3_perf_remeasure`).
- **The legacy generic route had no orchestrator-side IRM writer.** `kw_brief.py`
  asked the WORKER to author the field, and the finalize gate only checked that
  it existed with `ran=true` + a `ratio` — never who measured it. So the field
  whose entire purpose is author!=measurer was written by
  the author.

Nobody cheated: workers complied with the brief exactly as written. The label
was the thing that lied.

## The contract this module defines

`ran: true` is a claim that an INDEPENDENT re-measure happened. That claim is
only truthful when the orchestrator measured it, so:

    ran=true  REQUIRES an orchestrator `source` stamp.

When the orchestrator could NOT re-measure, the honest encoding is the one the
DEBT-192 worker-perf-report contract already established for port_a3:
`{ran: false, status: "N/A", reason: <why>, source: <who>}` — never a bare
self-report promoted to `ran: true`. The worker's numbers are preserved
alongside (as `worker_authored_aux` / `self_reported_*`), so nothing is lost —
only relabelled truthfully.

`finalize_checks_provenance._check_post_worker_audit` enforces this via
`is_orchestrator_measured()`. Because the orchestrator stamps the irm at
finalize BEFORE the gate reads it (`fsm_phase_finalize._run_perf_capture` runs
at `:69`, eligibility at `:72`), a live finalize always presents a stamped irm.
A `ran=true` block with no orchestrator source means no orchestrator writer ran
— which is exactly the state worth failing on.

## Why a local-only capture can report N/A

`phase_o5_perf_capture._measure_local` runs the profiler harness via a bare
local `python3` subprocess. That works only when the orchestrator is
co-located with the NPU (`CONTAINER=local`). On the SSH+docker fleet the
orchestrator host has no `torch_npu`, so the harness fails on import and
`measure_op_perf` honestly returns `status="N/A"` — the module's own v1 scope
note says the SSH+docker-exec runner is "migration plan step 3", still unwritten.

So on the SSH fleet a local-only perf attempt remains worker-self-reported — but now it is
LABELLED as such (`ran=false` + reason) instead of masquerading as independent.
That is a `scanner_coverage_gap` ("we never measured"), not misconduct. When
step 3 lands, `measure_op_perf` starts returning a real ratio and this module
promotes it to a genuine `ran=true` irm with no further change here.
"""
from __future__ import annotations

from typing import Any, Optional


# Canonical `source` value for an irm the WORKER authored. Honest label for the
# pre-DEBT-217 state: the block exists, but it is a self-report, not an
# independent measure. NEVER pair this with ran=true — see module docstring.
SOURCE_WORKER_AUTHORED = "worker_authored"

# Every orchestrator-side irm writer stamps a `source` beginning with one of
# these. Prefix-matched rather than exact-matched because the existing stamps
# are human-readable prose that names the mechanism. One form records a phase
# O5 post-verification rerun on NPU for backward and A3-golden flows; the other
# records the symmetric profiler run performed by this module's
# `orchestrator_irm_from_perf_result` helper.
# Keeping the prose is deliberate: the archive should say HOW it was measured,
# not just that a flag was set. The prefix is the machine-checkable part.
ORCHESTRATOR_SOURCE_PREFIXES: tuple = ("phase_o5",)

# Stamp written when the orchestrator attempted a capture and it was not
# measurable in this environment (the SSH-fleet case described above).
SOURCE_CAPTURE_ATTEMPTED = (
    "phase_o5_perf_capture: orchestrator attempted an independent perf "
    "re-measure and it was not measurable in this environment"
)

SOURCE_CAPTURE_MEASURED = (
    "phase_o5_perf_capture: orchestrator ran the symmetric profiler harness "
    "(model.py vs model_new_<backend>.py) in ITS OWN context — independent of "
    "the worker's verification.json self-report"
)


def is_orchestrator_source(source: Any) -> bool:
    """True iff `source` names an orchestrator-side measurer.

    Anything else — `worker_authored`, an empty string, None, or a stamp this
    codebase does not recognise — is NOT independent. Unknown stamps are
    rejected rather than trusted: a forged/typo'd source must not buy
    independence it did not earn.
    """
    if not isinstance(source, str):
        return False
    stripped = source.strip()
    if not stripped:
        return False
    return any(stripped.startswith(p) for p in ORCHESTRATOR_SOURCE_PREFIXES)


def is_orchestrator_measured(irm: Any) -> bool:
    """True iff `irm` is an independent re-measure the ORCHESTRATOR performed.

    This is the predicate the finalize gate uses to decide whether a `ran=true`
    claim is truthful.
    """
    if not isinstance(irm, dict):
        return False
    return is_orchestrator_source(irm.get("source"))


def orchestrator_irm_from_perf_result(
    perf_result: dict,
    *,
    worker_ratio: Optional[float] = None,
) -> dict:
    """Build a source-stamped irm from a `phase_o5_perf_capture.measure_op_perf`
    result.

    `measure_op_perf` returns either a real measurement (status
    PASS/PASS_WITHIN_TOLERANCE/BELOW_THRESHOLD + numeric `ratio`) or an honest
    `status="N/A"` + `reason` when it could not measure. Map those to the two
    truthful irm shapes — a measured `ran=true` or an unmeasured `ran=false`.
    Never invent a ratio for the N/A case.
    """
    ratio = perf_result.get("ratio")
    status = perf_result.get("status")

    if ratio is None or status == "N/A":
        reason = (
            perf_result.get("reason")
            or perf_result.get("error")
            or "phase_o5_perf_capture returned no ratio"
        )
        return {
            "ran": False,
            "status": "N/A",
            "reason": reason,
            "source": SOURCE_CAPTURE_ATTEMPTED,
        }

    irm = {
        "ran": True,
        "ratio": ratio,
        "method": perf_result.get("method"),
        "source": SOURCE_CAPTURE_MEASURED,
    }
    if worker_ratio is not None:
        try:
            irm["delta_vs_kw_self_report"] = round(
                float(ratio) - float(worker_ratio), 4
            )
        except (TypeError, ValueError):
            pass
    return irm


def worker_authored_irm(existing: Any, reason: str) -> dict:
    """Relabel a worker-authored irm truthfully, preserving its numbers.

    The worker's self-reported figures are kept under `self_reported_*` so an
    auditor still sees exactly what the worker claimed — this downgrades the
    CLAIM, not the DATA. `ran` becomes False because no independent re-measure
    happened; `reason` records why the orchestrator could not supply one.
    """
    irm: dict = {
        "ran": False,
        "status": "N/A",
        "reason": reason,
        "source": SOURCE_WORKER_AUTHORED,
    }
    if isinstance(existing, dict):
        if existing.get("ratio") is not None:
            irm["self_reported_ratio"] = existing.get("ratio")
        if existing.get("method") is not None:
            irm["self_reported_method"] = existing.get("method")
    return irm


def merge_perf_preserving_irm(worker_perf: Any, perf_result: dict) -> dict:
    """Merge an orchestrator capture into the worker's `performance` block.

    DEBT-218: `fsm_phase_finalize` previously did `vj["performance"] =
    perf_result` — a WHOLESALE REPLACEMENT. `perf_result` carries no
    `independent_re_measure` and no `ratio_baseline`, so the replacement
    dropped both. The very next gate read `perf["independent_re_measure"]` →
    None → finalize ROLLBACK. That made `AOG_PERF_CAPTURE_OVERRIDE_WORKER=1`
    — the ONE documented way to force an honest independent capture — destroy
    the archive it was meant to strengthen.

    Fix: overlay, don't replace. Orchestrator-measured keys win (that is the
    point of the capture), but worker keys the capture does not own —
    `ratio_baseline` (which the finalize gate reads for its Path-A carve-out),
    `independent_re_measure`, and any mode-specific annotations — survive. The
    caller then stamps the irm truthfully on top.
    """
    merged = dict(worker_perf) if isinstance(worker_perf, dict) else {}
    merged.update(perf_result)
    return merged

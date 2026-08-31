# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 preflight for a frozen old-format NPUKernelBench provider.

This module is intentionally independent from every live-source provider.  It
consumes only the immutable staged bundle named by ``reference`` and delegates
task/API checks to the target transport.  It must never load a source-runtime
environment, source-stage artifact, or historical capture as functional truth.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import events
from logging_config import get_logger
from npubench.npubench_inputs import NPUBENCH_SOURCE, verify_npubench_stage


log = get_logger(__name__)


def _report_status(report: object) -> str:
    """Normalize a runner preflight status without trusting arbitrary values."""
    if not isinstance(report, Mapping):
        return "ERROR"
    status = report.get("status")
    return status if isinstance(status, str) else "ERROR"


def _report_reason(report: object) -> str:
    """Extract a bounded diagnostic without serializing task-owned objects."""
    if not isinstance(report, Mapping):
        return "runner returned a non-object preflight report"
    reason = report.get("reason") or report.get("error") or "NPUKernelBench preflight failed"
    return str(reason)[:500]


def provision_npubench_reference(
    *,
    workspace: Path,
    reference: Mapping[str, Any],
    lane: int,
) -> Optional[int]:
    """Verify one immutable task bundle and its executable preflight.

    ``None`` lets the FSM proceed to the worker.  ``7`` is the existing O2.5
    reference-invalid exit class.  Expected input/task failures are reported as
    structured events; unexpected runner exceptions also fail closed and never
    fall back to a different provider.
    """
    if not isinstance(reference, Mapping) or reference.get("source") != NPUBENCH_SOURCE:
        reason = "durable reference does not select npubench"
        _emit_block(workspace, lane, "STAGED_INPUTS_INVALID", reason)
        return 7

    valid, reason, manifest = verify_npubench_stage(workspace, reference)
    if not valid:
        _emit_block(workspace, lane, "STAGED_INPUTS_INVALID", reason)
        return 7

    try:
        # A remote A5 configuration must execute this old-format Python task
        # on A5, not on the controller. ``npubench_target`` retains the
        # explicit A5_CONTAINER=local path for direct-NPU deployments while
        # using a fresh tokenized target workspace for every remote preflight.
        # It is deliberately separate from the legacy ``current_task`` sync
        # route, which can retain a stale candidate or runner closure.
        npubench_target = importlib.import_module("npubench.npubench_target")

        report = npubench_target.preflight_npubench_on_target(
            workspace=Path(workspace), reference=reference, lane=lane
        )
    except Exception as exc:  # Defensive: provider code must not weaken O2.5.
        reason = f"npubench target preflight raised {type(exc).__name__}: {exc}"
        _emit_block(workspace, lane, "PREFLIGHT_ERROR", reason)
        return 7

    status = _report_status(report)
    if status not in {"PASS", "READY"}:
        _emit_block(workspace, lane, status or "PREFLIGHT_ERROR", _report_reason(report))
        return 7

    case_encoding = manifest.get("sidecar_encoding") if isinstance(manifest, Mapping) else None
    binding = report.get("binding_sha256") if isinstance(report, Mapping) else None
    events.emit(
        workspace,
        "orchestrator.phase_o25_npubench_ready",
        lane=lane,
        data={
            "status": status,
            "sidecar_encoding": case_encoding,
            "binding_sha256": binding,
        },
    )
    log.info("phase O2.5 (npubench): immutable task preflight ready")
    return None


def _emit_block(workspace: Path, lane: int, verdict: str, reason: str) -> None:
    """Emit a compact fail-closed O2.5 event for the NPUKernelBench provider."""
    log.info("phase O2.5 (npubench): %s: %s", verdict, reason)
    events.emit(
        workspace,
        "orchestrator.phase_o25_npubench_block",
        lane=lane,
        data={"verdict": verdict, "errors": [str(reason)[:500]]},
    )

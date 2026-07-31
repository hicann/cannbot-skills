# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""fsm_phase_o25_dispatch.py — Phase O2.5 reference-provider dispatch (DEBT-201).

Extracted VERBATIM from run_single_op's Phase O2.5 preamble block
(orchestrator.py) as part of the god-function decomposition. This is the
mode-dispatched reference-provider gate that runs BEFORE the FSM spawn loop:

  - port_a3_to_a5 → phase_o25_a3_ref.provision_a3_reference (live A3 truth)
  - backward      → phase_o25_backward.provision_backward_reference
    (BACKWARD_E2E opt-out returns 98)

DEPENDENCY CONTRACT: this slice references NO orchestrator-module-level names
(no read-through needed) — only per-run inputs and the two mode-specific
reference modules imported lazily. Sibling module objects are the SAME ones tests patch, so
`monkeypatch.setattr(<sibling>, ...)` bites directly.

Returns an int EXIT CODE when the run must abort (missing source → 7, ref
capture unrecoverable → 7, BACKWARD_E2E=0 opt-out → 98) and None when the O2.5
gate passed and run_single_op should proceed to Phase O3. Body is byte-identical
to the original modulo each `return N` staying `return N` and the block's
`o25` / `_*_handled` locals living inside this function.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import events
from logging_config import get_logger
from source_arch import load_port_a3_build_source, verify_source_stage

log = get_logger(__name__)

_PORT_A3_BUILD_SOURCE_ENV = "CANNBOT_PORT_A3_BUILD_SOURCE"


def provision_reference(
    op: str,
    workspace: Path,
    *,
    lane: int,
    extra_lanes: Optional[list[int]] = None,
    plan_only: bool = False,
) -> Optional[int]:
    """Run the mode-dispatched Phase O2.5 reference gate.

    Returns an exit code to `return` from run_single_op, or None to proceed.
    """
    # The original ops-nn checkout is trusted harness context needed only if
    # live A3 dispatch proves the op is not shipped.  Consume it before any
    # worker can spawn; durable state intentionally exposes only the fixed
    # source-only snapshot to preserve graybox isolation.
    _port_a3_build_source = os.environ.pop(_PORT_A3_BUILD_SOURCE_ENV, None)
    # W15 (2026-05-12, ROADMAP §1.5): Phase O2.5 dispatch on opgen_mode.
    # port_a3_to_a5 mode uses the A3-CANN reference variant (phase_o25_a3_ref),
    # which runs the existing A3 kernel via aclnn on real A3 hardware to
    # capture ground-truth outputs.
    _opgen_state_path = workspace / ".opgen_state.json"
    _opgen_mode_for_o25 = None
    _port_a3_source = None
    _backward_forward_source = None
    if _opgen_state_path.exists():
        try:
            _opst = json.loads(_opgen_state_path.read_text())
            _opgen_mode_for_o25 = _opst.get("opgen_mode")
            _port_a3_source = _opst.get("port_a3_source")
            _backward_forward_source = _opst.get("backward_forward_source")
        except Exception as exc:
            log.info("phase O2.5: durable state is unreadable: %r", exc)
            return 2

    # Only the two customer modes are accepted.  Unknown or missing state fails
    # closed even when this helper is called outside ``run_single_op``.
    if _opgen_mode_for_o25 == "port_a3_to_a5":
        valid_stage, stage_reason, _stage_manifest = verify_source_stage(
            workspace, _opst
        )
        if not valid_stage:
            log.info(
                "phase O2.5: migration source-only snapshot rejected: %s",
                stage_reason,
            )
            return 7
        try:
            registered_build_source = load_port_a3_build_source(
                workspace,
                source_stage_digest=_opst.get("source_stage_digest"),
            )
        except (OSError, ValueError) as exc:
            log.info(
                "phase O2.5: private migration build-source binding rejected: %s",
                exc,
            )
            return 7
        if _port_a3_build_source is not None:
            try:
                transient_build_source = Path(
                    _port_a3_build_source
                ).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                log.info(
                    "phase O2.5: transient migration build source is invalid: %r",
                    exc,
                )
                return 7
            if transient_build_source != registered_build_source:
                log.info(
                    "phase O2.5: transient/private migration build-source mismatch"
                )
                return 7
        _port_a3_build_source = str(registered_build_source)
        return _provision_port_a3(
            op,
            workspace,
            lane,
            _port_a3_source,
            _port_a3_build_source,
        )
    elif _opgen_mode_for_o25 == "backward":
        return _provision_backward(op, workspace, lane, _backward_forward_source)
    log.info(
        "phase O2.5: unsupported or missing opgen_mode=%r; expected one of "
        "port_a3_to_a5/backward",
        _opgen_mode_for_o25,
    )
    return 2


def _provision_port_a3(
    op: str,
    workspace: Path,
    lane: int,
    _port_a3_source,
    _port_a3_build_source=None,
) -> Optional[int]:
    """Require a complete live A3 reference capture before proceeding."""
    if not _port_a3_source:
        log.info(
            "phase O2.5 (port_a3_to_a5): port_a3_source missing "
            "from .opgen_state.json — cannot proceed. Re-invoke with "
            "`python3 -m orchestrator --port-a3 <ops-nn-op-dir>`."
        )
        events.emit(workspace, "orchestrator.phase_o25_port_a3_block", lane=lane,
                    data={"verdict": "MISSING_PORT_A3_SOURCE"})
        return 7
    import phase_o25_a3_ref
    try:
        from briefs._common import load_env as __load_env_a3
        __env_a3 = __load_env_a3()
    except Exception as _e:
        log.info(f"phase O2.5 (port_a3_to_a5): failed to load env: {_e!r}")
        return 7
    o25_a3 = phase_o25_a3_ref.provision_a3_reference(
        op_dir=Path(_port_a3_source),
        source_build_op_dir=(
            Path(_port_a3_build_source) if _port_a3_build_source else None
        ),
        workspace=workspace,
        a3_host=__env_a3.a3_host,
        a3_user=__env_a3.a3_user or "root",
        a3_container=__env_a3.a3_container,
        a3_cann_path=__env_a3.a3_cann_path,
        probe_only=False,  # W14b live exec
    )
    # A cross-generation run may proceed only with a complete live source-device
    # capture.  Probe-only or synthetic truth is not an acceptable substitute.
    if o25_a3.verdict != "READY":
        print(phase_o25_a3_ref.format_block_message(op, o25_a3))
        events.emit(
            workspace,
            "orchestrator.phase_o25_port_a3_block",
            lane=lane,
            data={"verdict": o25_a3.verdict, "errors": o25_a3.errors},
        )
        return 7
    log.info(f"phase O2.5 (port_a3_to_a5): {o25_a3.summary}")
    # port_a3 provisioned its own A3-CANN reference (a3_reference_runnable.json
    # + edge_dataset.pt with a3_outputs); the stock CPU-truth check would fail
    # looking for ref_runnable.json, so we skip it and proceed to Phase O3.
    return None


def _provision_backward(
    op: str, workspace: Path, lane: int, _backward_forward_source,
) -> Optional[int]:
    """backward O2.5: self-contained autograd reference. Returns 7 on missing
    forward-spec / provision failure, 98 on BACKWARD_E2E=0 opt-out, None to
    proceed."""
    # B3.2 (BACKWARD_PLUGIN_DESIGN §5.5): backward (gradient) mode O2.5.
    # Produce the backward reference truth SELF-CONTAINED (autograd oracle
    # over the forward spec — NO edge_dataset, NO NPU/A3/CANN). The owner-
    # fixed decision (§5.5) is the self-contained verify path that mul_grad
    # hardware-validated, not the benchmark edge_dataset path (which depends
    # on the github-unreachable benchmark submodule).
    if not _backward_forward_source:
        log.info(
            "phase O2.5 (backward): backward_forward_source missing from "
            ".opgen_state.json — re-invoke with `--backward <forward_spec.py>`."
        )
        events.emit(workspace, "orchestrator.phase_o25_backward_block", lane=lane,
                    data={"verdict": "FORWARD_SPEC_MISSING"})
        return 7
    import phase_o25_backward
    bref = phase_o25_backward.provision_backward_reference(
        workspace=workspace,
        forward_spec=Path(_backward_forward_source),
        op=op,
    )
    if bref.verdict != "READY":
        print(phase_o25_backward.format_block_message(op, bref))
        events.emit(workspace, "orchestrator.phase_o25_backward_block", lane=lane,
                    data={"verdict": bref.verdict, "errors": bref.errors})
        return 7
    log.info(f"phase O2.5 (backward): {bref.summary}")
    events.emit(workspace, "orchestrator.phase_o25_backward_ready", lane=lane,
                data={"n_ok": bref.n_ok, "n_skipped": bref.n_skipped,
                      "wrt": bref.wrt, "artifacts": bref.artifacts})
    if os.environ.get("BACKWARD_E2E") != "0":
        # B3.3b DEFAULT (2026-05-31): `orch --backward` runs END-TO-END.
        # Flow into the worker loop; the backward worker brief
        # (BackwardPlugin.kw_brief_phase_block, B3.3a) drives self-contained
        # generation + verify vs this autograd reference. Synthesize a READY
        # O25Report so the stock-path artifact check is skipped (mirrors the
        # port_a3 _port_a3_handled flow).
        #
        # Default flipped ON after increment-2 (#313, e1841476) landed the
        # finalize/phase_o5 backward-awareness (BackwardPlugin op_host override
        # + phase_o5_runner.backward_verify_runner + the pass_a/pass_b-N/A
        # schema contract) AND a full cold-start `BACKWARD_E2E=1 orch --backward`
        # e2e ran end-to-end to a clean archive (rms_norm_grad, 6/6 precision
        # PASS, O5 VERIFIED, finalize→done, zero manual spawn/finalize —
        # 2026-05-31). Opt-OUT with `BACKWARD_E2E=0` to stop at the B3.2
        # reference boundary (reference-only, no worker; for ref-truth debugging).
        log.info("phase O2.5 (backward): flowing to worker loop "
                 "(B3.3b end-to-end default; set BACKWARD_E2E=0 to stop at reference)")
        # backward provisioned its own self-contained autograd reference;
        # skip the stock CPU-truth check and proceed to Phase O3.
        return None
    else:
        # BACKWARD_E2E=0 opt-out: stop at the B3.2 reference boundary.
        # Reference truth + canonical model.py / forward_spec.py / backward_ref.json
        # / backward_cpu_truth.pt are on disk; no worker spawn. For reference-truth
        # debugging / inspecting the autograd oracle without a full generation run.
        print("=" * 72)
        print(f"B3.2 (BACKWARD_E2E=0): backward reference truth READY for {op} — "
              f"{bref.n_ok} scoreable cases, grads for {bref.wrt}.")
        print(f"  artifacts: {', '.join(bref.artifacts)}")
        print("  Worker kernel-gen + on-hardware verify SKIPPED (opt-out). "
              "Unset BACKWARD_E2E to run end-to-end.")
        print("=" * 72)
        return 98

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""pipeline_exhaustion — legitimate-exhaustion judgement, extracted from
orchestrator.py (behavior-neutral god-function decomposition, DEBT-201,
2026-07-06).

Pure extraction: byte-identical logic. `_is_legitimate_pipeline_exhaustion`
(P0y / P0aa / DEBT-108 / DEBT-112) decides whether an iter_cap hit represents
a LEGITIMATE pipeline exhaustion (PARTIAL_PERSIST terminal) vs a real stuck
loop. It is a self-contained predicate over workspace artifacts
(verification.json, probe_result.json, .rollback_history.jsonl,
cann_strategy_inference.md). It is NOT monkeypatched and calls no
orchestrator-local function (only `schema_norm._resolve_perf_threshold`, a
module ref). orchestrator re-imports it (bottom import) so the call site
inside run_single_op + the `orchestrator._is_legitimate_pipeline_exhaustion`
external/test access path are unaffected."""
from __future__ import annotations
import logging

import json
from pathlib import Path

import schema_norm


def _is_legitimate_pipeline_exhaustion(workspace: Path, state: str) -> bool:
    """P0y: detect that an iter_cap hit represents legitimate pipeline
    exhaustion (PARTIAL_PERSIST is the terminal state) vs a real error.

    Conditions for legitimate exhaustion (ALL must hold):
    - state is await_researcher (the V3.8.8 final-stage gate)
    - cann_strategy_inference.md exists (researcher actually ran with output)
    - probe_result.json classification == "requirement" (probe confirmed
      no candidate-side fix exists)
    OR:
    - state is await_optimizer / await_fused_optimizer AND
      researcher already ran AND probe verdict=requirement AND
      verification.json shows perf below threshold (full perf path
      exhausted with no actionable gains)

    All other iter_cap hits are real errors (workflow stuck in a loop).
    """
    # The customer workflow has one researcher artifact name. Do not accept
    # arbitrary backend-prefixed files: doing so would silently reopen a removed
    # customer route.
    def _has_researcher_output(ws: Path) -> bool:
        return (ws / "cann_strategy_inference.md").exists()

    if state == "await_researcher":
        if not _has_researcher_output(workspace):
            return False
        probe_result_path = workspace / "probe_result.json"
        if probe_result_path.exists():
            try:
                pr = json.loads(probe_result_path.read_text())
                if pr.get("classification") == "requirement":
                    return True
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
        # Even without probe_result.json, if researcher exhausted with output,
        # that IS a legitimate exhaustion (researcher tried alternate vendor
        # strategies and found none).
        return True

    # P0aa (2026-05-05): await_optimizer / await_fused_optimizer iter_cap is
    # legitimate exhaustion when the full pipeline has run AND perf is still
    # below threshold. Conditions:
    #   - researcher already ran (<backend>_strategy_inference.md exists)
    #   - probe verdict was `requirement` (no precision-side fix possible)
    #   - verification.json.performance.ratio < 0.6 (perf still below floor)
    # Origin: op#9 TopKTopP 2026-05-05 — ko-5 emitted KO_PERF_PLATEAU
    # 0.385x after researcher (ar-2) + probe (pp-3 requirement) full cycle.
    # Without this branch, orchestrator returned error 2 instead of cleanly
    # finalizing PARTIAL_PERSIST. P135.X 2026-05-18: backend-prefix-agnostic.
    if state in ("await_optimizer", "await_fused_optimizer"):
        if not _has_researcher_output(workspace):
            return False
        verif_path = workspace / "verification.json"
        if not verif_path.exists():
            return False
        try:
            v = json.loads(verif_path.read_text())
            ratio = v.get("performance", {}).get("ratio")
            # Delegate to the canonical profile-aware resolver so the
            # exhaustion and finalize paths cannot diverge on their threshold.
            perf_floor = getattr(schema_norm, "_resolve_perf_threshold")(workspace, v)
            if ratio is None or ratio >= perf_floor:
                return False
        except Exception:
            return False
        # P135.PP (2026-05-18): probe_result.json with classification=requirement
        # is sufficient evidence (precision-side fix attempted + failed).
        # But it's NOT necessary — if precision is ALREADY FULLY PASS,
        # probe was correctly never fired (no precision issue to bisect).
        # 6_Histc tonight: 15/15 PASS bit-exact + perf 0.184× confirmed
        # ceiling by 5 ko iters → legitimate exhaustion but old code
        # required probe_result.json which never existed → exit 2 loop.
        probe_result_path = workspace / "probe_result.json"
        if probe_result_path.exists():
            try:
                pr = json.loads(probe_result_path.read_text())
                if pr.get("classification") != "requirement":
                    return False
            except Exception:
                return False
            return True
        # No probe → only legitimate if precision is already fully PASS.
        # Schema (V3.8.x verification.json):
        #   - precision.pass_a.status (nested form)
        #   - precision.status (rolled-up form)
        try:
            precision = v.get("precision", {}) or {}
            pass_a_status = (
                precision.get("pass_a", {}).get("status")
                or precision.get("status")
                or ""
            )
            if pass_a_status in ("PASS", "PASS_WITHIN_TOLERANCE"):
                return True
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        return False

    # DEBT-108 (2026-05-20): await_worker iter_cap is legitimate exhaustion
    # when the artifacts in verification.json show pass_a PASS AND the
    # remaining rollback signature is an "infra" gate (build_evidence,
    # P96 paper-over, delegation_scan, kb_writeup, etc.) NOT an algorithm-
    # rework gate (precision_status_not_in, perf_below_threshold). The
    # iter_cap was burning on infrastructure debt unrelated to the kernel
    # logic — kernel was correct since kw-1.
    #
    # Origin: gather_elements_v2 task #51 (2026-05-20) — kw-1 produced pass_a
    # PASS 8/8 byte-identical-preserved through 10 iters, kw-2..10 fixed only
    # binary_provenance / P96 / delegation_scan / post_worker_audit gates.
    # iter_cap=9 penalized infra debt. Required manual --bump-cap worker:5
    # to drive through. Trigger: when await_worker iter_cap hits AND
    # verification.json has pass_a PASS AND last rollback gate is in the
    # infra set, return True (route to finalize PARTIAL_PERSIST). Algorithm
    # gates (verification_file_missing, pass_a_coverage, model_py_shape,
    # pass_count) NEVER qualify — those reflect real algorithm work needed.
    if state == "await_worker":
        # Read verification.json: must show pass_a PASS
        verif_path = workspace / "verification.json"
        if not verif_path.exists():
            return False
        try:
            v = json.loads(verif_path.read_text())
        except Exception:
            return False
        precision = (v.get("precision") or {})
        prec_status = precision.get("status", "")
        if prec_status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return False
        # Read last rollback signature from .rollback_history.jsonl
        rh_path = workspace / ".rollback_history.jsonl"
        if not rh_path.exists():
            return False
        try:
            lines = [
                json.loads(ln) for ln in rh_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except (json.JSONDecodeError, OSError):
            return False
        if not lines:
            return False
        last_gate = lines[-1].get("gate", "")
        # Infra-debt gate set: rollbacks NOT reflecting kernel-logic gaps.
        # See finalize_pipeline.GateID for canonical set. Algorithm-gates
        # (verification_file_missing, model_py_shape, pass_a_coverage,
        # pass_count, persist_evidence) are EXCLUDED from base set
        # because they usually indicate real issues the worker should fix.
        _INFRA_GATES = {
            "binary_provenance",
            "infra_baseline_paper_over",
            "infra_retry_without_cap",
            "port_a3_pass_b_schema",
            "stale_orchestrator",
            "kb_writeup",
            "post_worker_audit",
            "phase_o5_runner_failed",
            "phase_o5_mismatch",
            "perf_methodology_asymmetry",
            "verifier_uses_modelnew",
            "pass_b_coverage_silent_skip",
        }
        # DEBT-112 (2026-05-27, DS): when persist_verdict is explicitly
        # PARTIAL_PERSIST_INFRA_BLOCKED, algorithm gates that fired as a
        # consequence of infrastructure failure are ALSO legitimate
        # exhaustion. Without this, every infra-blocked op burns through
        # algorithm-gate iter budget (3_FusionAttention kw-1..5 burned 9
        # spawns across model_py_shape/pass_a_coverage/pass_count gates
        # when the root cause was infra, not algorithm).
        try:
            vj = json.loads((workspace / "verification.json").read_text())
            persist_v = vj.get("precision", {}).get("persist_verdict", "")
            if persist_v == "PARTIAL_PERSIST_INFRA_BLOCKED":
                _INFRA_GATES.update({
                    "model_py_shape",
                    "pass_a_coverage",
                    "pass_count",
                    "persist_evidence",
                })
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        if last_gate not in _INFRA_GATES:
            return False
        return True

    return False

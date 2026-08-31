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
"""finalize_dispatch — the plugin-dispatch + finalize-eligibility + promote cluster.

Behavior-neutral extraction from finalize_pipeline.py (DEBT-201 god-file
decomposition, 2026-07-06). The function bodies are byte-identical to their
former inline definitions; only relocated. finalize_pipeline re-imports the
public names below (its own bottom import) so every call site + import path
(`finalize_pipeline.finalize_op`, `from finalize_pipeline import
check_finalize_eligibility`, `fp.batch_precheck`, …) stays byte-for-byte stable.

Why the cluster moved TOGETHER (and not one function at a time): the two
monkeypatched entry points — `_get_active_plugin` and `finalize_op` — are
called by BARE name from their sibling functions in this cluster
(`_run_plugin_extra_finalize_checks`, `_check_op_host_completeness`,
`_pass_branch_gate_specs`, `_finalize_with_plugin_layout`, `finalize_op`). For
`monkeypatch.setattr(<module>, "_get_active_plugin", ...)` to still bite those
call sites, the callers must live in the SAME module as the patched function.
So the whole cluster relocates as a unit and the tests repoint their patch
target from `finalize_pipeline` to `finalize_dispatch` (see
test_finalize_decomposition.py + test_debt_124 + the phase_o5 / p0ww
integration tests). finalize_pipeline re-exports the same objects, so any
OTHER caller that patched or read `finalize_pipeline.finalize_op` /
`finalize_pipeline._get_active_plugin` by attribute continues to see the moved
object (the re-export is `is`-identical), while patching the parent attribute
would NOT reach these intra-cluster bare-name callers — hence the tests patch
at this module (the real home) per the DEBT-201 repoint recipe.

Cycle-avoidance: this module top-imports ONLY from finalize_pipeline (symbols
defined at parent top, before parent bottom-imports this module) and from the
already-stable leaf siblings (finalize_shared / finalize_readme /
finalize_candidates). The `_check_*` gate functions live in finalize_checks,
which finalize_pipeline bottom-imports AFTER this module; to keep the import
graph acyclic they are imported LAZILY inside the two functions that need them
(`_pass_branch_gate_specs`, `check_finalize_eligibility`, `batch_precheck`) so
the reference resolves at call time (finalize_checks fully loaded by then).
"""
from __future__ import annotations
import logging

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Symbols defined at finalize_pipeline top (available before parent's bottom
# import of this module). GateID is the gate-ID enum; the rest are the promote
# helpers + project-root anchor + the stale-orchestrator precondition gate.
from finalize_pipeline import (  # noqa: E402
    GateID, _check_stale_orchestrator, _merge_copy_dir, _should_skip,
    is_finalized, _PROJECT_ROOT,
)
from finalize_shared import (  # noqa: E402
    _verification_hash, _is_harness_internal, _kb_writeup_body_len,
)
from finalize_readme import _write_archive_readme  # noqa: E402
from finalize_candidates import _resolve_archive_op_name  # noqa: E402
import provenance_node as _provenance_node  # noqa: E402  (DEBT-203 S1)


def _inject_migration_metadata(workspace: Path) -> None:
    """Copy durable migration-source evidence into ``verification.json``.

    The legacy route is an arch22→arch35 migration.
    ``port-aclnn-tilelang2ascendc`` is already a target-format implementation
    context, so archived evidence must not claim an arch22 migration that
    never happened.
    """
    state_path = workspace / ".opgen_state.json"
    verification_path = workspace / "verification.json"
    try:
        state = json.loads(state_path.read_text())
        if state.get("opgen_mode") != "port_a3_to_a5":
            return
        source = state.get("port_source")
        source_kind = (
            source.get("kind") if isinstance(source, dict)
            else state.get("source_kind")
        )
        if source_kind == "port-aclnn-tilelang2ascendc":
            if state.get("source_arch") != "arch35" or state.get("target_arch") != "arch35":
                return
            migration = {
                "source_kind": source_kind,
                "source_arch": "arch35",
                "target_arch": "arch35",
                "source_arch_detection": state.get("source_arch_detection", {}),
                "semantic": "tilelang2ascendc_project_context",
            }
        else:
            if state.get("source_arch") != "arch22" or state.get("target_arch") != "arch35":
                return
            migration = {
                "source_arch": "arch22",
                "target_arch": "arch35",
                "source_arch_detection": state.get("source_arch_detection", {}),
            }
        verification = json.loads(verification_path.read_text())
        if not isinstance(verification, dict):
            return
        verification["migration"] = migration
        verification_path.write_text(json.dumps(verification, indent=2))
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )


def _inject_provenance_node(op: str, workspace: Path) -> None:
    """DEBT-203 S1 (additive-only): augment workspace/verification.json with a
    `provenance_node` block (fitness/is_buggy/lineage), DERIVED from the op's
    own verification + op_classification signature. Best-effort + fail-open:
    any error is swallowed so finalize behavior is otherwise unchanged.

    Idempotent: the injector preserves an existing created_ts, so identical
    verification content yields an identical block → the .finalized-<hash>
    idempotency key stays stable across re-finalize."""
    import datetime as _dt
    vp = workspace / "verification.json"
    try:
        verification = json.loads(vp.read_text())
        if not isinstance(verification, dict):
            return
        sig, source_sha = _provenance_signature(workspace)
        created_ts = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        branched, parent_id = _provenance_lineage(workspace)
        _provenance_node.inject_provenance_node(
            verification, op, source_sha, sig, created_ts=created_ts,
            branched=branched, parent_id=parent_id)
        vp.write_text(json.dumps(verification, indent=2))
    except Exception:
        # fail-open: provenance metadata is additive; never block finalize on it
        return


def _provenance_signature(workspace: Path) -> tuple[dict, str]:
    """Read the optional classification signature used by provenance nodes."""
    cls_path = workspace / "op_classification.json"
    if not cls_path.exists():
        return {}, ""
    classification = json.loads(cls_path.read_text())
    if not isinstance(classification, dict):
        return {}, ""
    return {
        "op_class_tags": classification.get("op_class_tags", []),
        "algorithm_classification": classification.get("algorithm_classification"),
    }, classification.get("source_sha256", "") or ""


def _provenance_lineage(workspace: Path) -> tuple[bool, Optional[str]]:
    """Read optional branch lineage, preserving its fail-open contract."""
    marker = workspace / ".branched_from.json"
    if not marker.is_file():
        return False, None
    try:
        lineage = json.loads(marker.read_text())
    except Exception:
        return False, None
    if not isinstance(lineage, dict) or not lineage.get("branched"):
        return False, None
    return True, lineage.get("parent_id")


def _inject_a_tier_loaded(op: str, workspace: Path) -> None:
    """§5.2 C2 (additive-only): merge the objective tier-a LOAD record
    (workspace/a_tier_manifest.json, harness-written at brief-time) into
    verification.json as `a_tier_loaded` — a compact, queryable list of the
    community tier-a skills SURFACED to the worker for this op, on the canonical
    per-op record (like KB provenance). Additive + fail-open: any error is swallowed
    so finalize behavior is otherwise unchanged. Absent manifest => no-op (default
    a5_ops runs write no route file => no manifest => verification.json untouched)."""
    vp = workspace / "verification.json"
    mf = workspace / "a_tier_manifest.json"
    try:
        if not mf.exists():
            return
        manifest = json.loads(mf.read_text())
        if not isinstance(manifest, dict):
            return
        surfaced = manifest.get("surfaced") or []
        verification = json.loads(vp.read_text())
        if not isinstance(verification, dict):
            return
        verification["a_tier_loaded"] = [
            {"topic": s.get("topic"), "skill": s.get("skill"), "kind": s.get("kind", "REQUIRED")}
            for s in surfaced if isinstance(s, dict)
        ]
        vp.write_text(json.dumps(verification, indent=2))
    except Exception:
        # fail-open: a_tier_loaded is additive metadata; never block finalize on it
        return


def _inject_a_tier_cross_check(op: str, workspace: Path) -> None:
    """§5.2 C4 (additive + fail-open): cross-check tier-a LOAD∧USE and record the result
    in verification.json.a_tier_cross_check (queryable evidence). Reuses
    cba_route_finalize_check (which reuses cba_route_gate for the USE parse). GENERAL
    policy is warn-not-block: a surfaced-but-unused REQUIRED route is recorded in
    `missing_use` but does NOT block finalize; the C6 proof case hard-asserts missing_use
    is empty on the recorded result. No manifest (config-gated off) => no-op."""
    vp = workspace / "verification.json"
    try:
        import cba_route_finalize_check as _cc
        result = _cc.check_a_tier_load_use(workspace)
        if result.get("checked", 0) == 0:
            return  # config-gated off / nothing surfaced
        verification = json.loads(vp.read_text())
        if not isinstance(verification, dict):
            return
        verification["a_tier_cross_check"] = {
            "checked": result["checked"],
            "results": result["results"],
            "missing_use": result["missing_use"],
        }
        vp.write_text(json.dumps(verification, indent=2))
    except Exception:
        # fail-open: cross-check evidence is additive; never block finalize on it
        return


def _inject_perf_deferred_partial_verdict(op: str, workspace: Path) -> None:
    """route-a first e2e (2026-07-14, main directive "c"): additive + fail-open.

    Under the PERF_DEFERRED perf-gate profile (--defer-perf-opt), an op whose
    correctness (precision PASS/PASS_WITHIN_TOLERANCE) + determinism pass but whose
    perf is sub-floor is a GENUINE PARTIAL_PERSIST — correctness is delivered, perf
    is honestly recorded as sub-floor, perf-optimization is a later rung. Stamp
    `precision.persist_verdict = "PARTIAL_PERSIST"` (basis = perf_deferred_subfloor,
    with the measured ratio) so the archived vj + REPORT read PARTIAL_PERSIST rather
    than a clean PASS that would imply perf met the bar. This is a harness-computed
    verdict from the declared mode + measured perf — NOT a lowered precision bar
    (precision detail is preserved intact) and NOT a hand-authored claim.

    No-op unless: profile is PERF_DEFERRED (allow_partial_on_subfloor) AND precision
    is PASS/PASS_WITHIN_TOLERANCE AND performance is sub-floor AND no persist_verdict
    already set (never clobber a precision-residual PARTIAL_PERSIST)."""
    vp = workspace / "verification.json"
    try:
        from perf_gate import resolve_profile
        profile = resolve_profile(workspace)
        if not getattr(profile, "allow_partial_on_subfloor", False):
            return
        v = json.loads(vp.read_text())
        if not isinstance(v, dict):
            return
        prec = v.get("precision", {}) or {}
        if prec.get("status") not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return  # not a correctness-PASS op — leave existing verdict logic alone
        if prec.get("persist_verdict"):
            return  # already carries a verdict (e.g. precision-residual PARTIAL) — don't clobber
        perf = v.get("performance", {}) or {}
        perf_status = perf.get("status")
        ratio = perf.get("ratio")
        # Only stamp when perf is genuinely sub-floor (BELOW_THRESHOLD, or a numeric
        # ratio under the profile/plugin floor). If perf met the bar, it's a clean PASS.
        is_subfloor = (perf_status == "BELOW_THRESHOLD") or (
            isinstance(ratio, (int, float)) and float(ratio) < (profile.finalize_threshold or 1.0)
        )
        if not is_subfloor:
            return
        prec["persist_verdict"] = "PARTIAL_PERSIST"
        prec["persist_basis"] = "perf_deferred_subfloor"
        prec["persist_note"] = (
            f"correctness PASS + determinism satisfied; perf sub-floor "
            f"(ratio={ratio}, status={perf_status}) with perf-optimization DEFERRED "
            f"(--defer-perf-opt / PERF_DEFERRED). Genuine PARTIAL_PERSIST: perf-opt is a later rung."
        )
        v["precision"] = prec
        vp.write_text(json.dumps(v, indent=2))
    except Exception:
        # fail-open: never block finalize on the verdict stamp
        return


def _check_op_host_completeness(workspace: Path) -> Optional[str]:
    """PB-33 op_host completeness gate (DEBT-094 phase 3 dispatcher).

    Body lives in each scoped plugin's ``check_op_host_completeness`` method;
    the pipeline knows nothing about mode-specific details.

    Pre-DEBT-094-phase-3 history: this function held the AscendC body
    inline with two boolean-flag plugin probes guarding it (the old
    PB-33 capability flag + the shared-utils detection flag). User catch
    2026-05-18 22:08Z: "防止当前的这个脚本变成一堆 if else 的这种缝合
    怪式的组合，我们还是要保证我们 plugin 的这种架构的干净" — flipped
    to full-delegate so finalize_pipeline has no plugin if/else.

    Original PB-33 background (2026-05-14): 5 archived ops shipped with
    op_host/ empty or only patches. PR4778 spec requires complete
    <op>_def.cpp + <op>_tiling.cpp + <op>_tiling.h + CMakeLists.txt
    (plus optional op_api/, <op>_infershape.cpp). Enforced across all
    supported AscendC modes (port_a3 and backward).

    Defensive fallback: when no plugin matches (shouldn't happen in
    practice, all op-gen modes have a registered plugin), fall back to
    BasePlugin body so the AscendC gate still fires.
    """
    plug = _get_active_plugin(workspace)
    if plug is not None:
        return plug.check_op_host_completeness(workspace)
    # No-plugin defensive path — use BasePlugin behavior so AscendC
    # contract is preserved if the dispatch layer ever returns None.
    try:
        from plugins.base import BasePlugin
    except ImportError:
        return None
    return BasePlugin().check_op_host_completeness(workspace)


# Sentinel marking the plugin-extras entry in the gate-spec list. Its
# predicate returns a full rejection dict (plugin-supplied gate id), not a
# reason string, so both iterators handle it specially.
_PLUGIN_EXTRAS_SENTINEL = "plugin_extras"


def _pass_branch_gate_specs() -> "list[tuple]":
    """Return the ordered [(GateID, predicate), ...] for the PASS branch.

    Predicate signature: (workspace: Path, v: dict) -> Optional[str].
    Order MUST match the historical inline PASS-branch order so the
    early-return finalize path fires the SAME first gate it always has.
    """
    from finalize_checks import (
        _check_kb_writeup, _check_pp88_compliance, _check_universal_entrypoints,
        _check_binary_provenance, _check_platform_blame_backed, _check_infra_paper_over,
        _check_infra_retry_budget, _check_port_a3_pass_b_schema, _check_pass_b_coverage,
        _check_arch35_wrap_cheat, _check_ge_ophost_raw_cann_copy,
        _check_architecture_class, _check_project_json_metadata,
        _check_verifier_uses_modelnew, _check_perf_methodology, _check_methodology_declaration,
        _check_pybind_host_logic,
        _check_post_worker_audit, _check_pass_count_concrete,
    )
    return [
        (GateID.KB_WRITEUP, _check_kb_writeup),
        (GateID.SIGMOID_FORM_REMEDIATION, lambda ws, v: _check_pp88_compliance(ws)),
        (GateID.ACLNN_VERIFY_PATH_FRAUD, _check_universal_entrypoints),
        (GateID.BINARY_PROVENANCE, _check_binary_provenance),
        (GateID.OP_HOST_COMPLETENESS, lambda ws, v: _check_op_host_completeness(ws)),
        (GateID.PLATFORM_BLAME_UNBACKED, lambda ws, v: _check_platform_blame_backed(ws)),
        (GateID.INFRA_BASELINE_PAPER_OVER, lambda ws, v: _check_infra_paper_over(ws)),
        (GateID.INFRA_RETRY_WITHOUT_CAP, lambda ws, v: _check_infra_retry_budget(ws)),
        (GateID.PORT_A3_PASS_B_SCHEMA, _check_port_a3_pass_b_schema),
        (GateID.PASS_B_COVERAGE_SILENT_SKIP, _check_pass_b_coverage),
        (GateID.ARCH35_WRAP_CHEAT, lambda ws, v: _check_arch35_wrap_cheat(ws)),
        (GateID.GE_OPHOST_RAW_CANN_COPY, lambda ws, v: _check_ge_ophost_raw_cann_copy(ws)),
        (GateID.ARCHITECTURAL_HACK, lambda ws, v: _check_architecture_class(ws)),
        (GateID.PROJECT_JSON_METADATA, lambda ws, v: _check_project_json_metadata(ws)),
        (GateID.VERIFIER_USES_MODELNEW, _check_verifier_uses_modelnew),
        (GateID.PERF_METHODOLOGY_ASYMMETRY, _check_perf_methodology),
        (GateID.METHODOLOGY_DECLARATION, _check_methodology_declaration),
        (GateID.PYBIND_HOST_BUSINESS_LOGIC, _check_pybind_host_logic),
        # Plugin-registered extras (P87 / DEBT-124) run HERE, before the
        # post-worker audit (matches the historical inline order). This entry
        # is special: _run_plugin_extra_finalize_checks returns a full
        # rejection DICT (gate id is plugin-supplied), not a reason string.
        # Both iterators recognise the PLUGIN_EXTRAS sentinel and adapt.
        (_PLUGIN_EXTRAS_SENTINEL, _run_plugin_extra_finalize_checks),
        (GateID.POST_WORKER_AUDIT, _check_post_worker_audit),
        (GateID.PASS_COUNT, _check_pass_count_concrete),
    ]


# ---------------------------------------------------------------------------
# P0ff (2026-05-05): rollback gate — refuse to finalize if precision is
# non-PASS and the pipeline isn't legitimately exhausted.
# ---------------------------------------------------------------------------
def _eligibility_rejection(
    gate: GateID, reason: str, rollback_state: str = "await_worker",
) -> dict:
    """Build the canonical failed-finalize response."""
    return {
        "eligible": False,
        "rollback_state": rollback_state,
        "gate": gate.value,
        "reason": reason,
    }


def _check_eligibility_structure(
    workspace: Path, prec: dict, shape_check, coverage_check,
) -> Optional[dict]:
    """Run the structural root-cause gates before branching by status."""
    shape_violation = shape_check(workspace)
    if shape_violation:
        return _eligibility_rejection(GateID.MODEL_PY_SHAPE, shape_violation)
    coverage_violation = coverage_check(workspace, prec)
    if coverage_violation:
        return _eligibility_rejection(GateID.PASS_A_COVERAGE, coverage_violation)
    return None


def _pass_branch_eligibility(
    workspace: Path, verification: dict, prec: dict, status: str,
) -> dict:
    """Return the first failed PASS-branch gate, preserving gate order."""
    for gate_id, predicate in _pass_branch_gate_specs():
        if gate_id == _PLUGIN_EXTRAS_SENTINEL:
            extras_rejection = predicate(workspace, verification)
            if extras_rejection is not None:
                return extras_rejection
            continue
        reason = predicate(workspace, verification)
        if reason:
            return _eligibility_rejection(gate_id, reason)
    pass_b = prec.get("pass_b", {}) or {}
    perf = verification.get("performance", {}) or {}
    return {
        "eligible": True,
        "rollback_state": None,
        "reason": (
            f"precision.status={status}; KB writeup "
            f"({_kb_writeup_body_len(workspace)}B) + "
            f"self-critic audit + delegation scan + pass_b={pass_b.get('status')} + "
            f"perf={perf.get('status')} all gated"
        ),
    }


def _partial_persist_eligibility(workspace: Path, verification: dict) -> dict:
    """Evaluate the evidence and plugin gates for PARTIAL_PERSIST."""
    if not (workspace / "probe_report.md").exists():
        return _eligibility_rejection(
            GateID.PERSIST_EVIDENCE,
            "PARTIAL_PERSIST claimed but no probe_report.md",
            "await_probe",
        )
    extras_rejection = _run_plugin_extra_finalize_checks(workspace, verification)
    if extras_rejection is not None:
        return extras_rejection
    from finalize_checks import _check_delegation_scan_marker
    delegation_reason = _check_delegation_scan_marker(workspace, verification)
    if delegation_reason:
        return _eligibility_rejection(GateID.POST_WORKER_AUDIT, delegation_reason)
    return {
        "eligible": True,
        "rollback_state": None,
        "reason": "PARTIAL_PERSIST with probe_report.md evidence",
    }


def _incomplete_precision_rollback(workspace: Path, status: str) -> dict:
    """Route incomplete non-PASS work to the first missing pipeline stage."""
    if not (workspace / "probe_report.md").exists():
        reason = f"precision.status={status} and no probe_report.md → probe must run"
        return _eligibility_rejection(GateID.PERSIST_EVIDENCE, reason, "await_probe")
    if not (workspace / "cann_strategy_inference.md").exists():
        reason = f"precision.status={status}, probe done, but no researcher report → researcher must run"
        return _eligibility_rejection(GateID.PERSIST_EVIDENCE, reason, "await_researcher")
    if not (workspace / "optimization_log.md").exists():
        reason = f"precision.status={status}, probe + researcher done, but no optimizer attempt"
        return _eligibility_rejection(GateID.PERSIST_EVIDENCE, reason, "await_optimizer")
    reason = (
        f"precision.status={status} with full pipeline artifacts but "
        "no persist_verdict — worker must explicitly emit PARTIAL_PERSIST "
        "or address remaining failures"
    )
    return _eligibility_rejection(GateID.PERSIST_EVIDENCE, reason)


def _npubench_pending_rejection(workspace: Path, verification: dict) -> Optional[dict]:
    """Return the provider-pending rejection when an O5 report is absent."""
    try:
        from npubench.npubench_finalize_contract import resolve_npubench_workspace

        is_npubench, npu_err = resolve_npubench_workspace(workspace)
        if is_npubench and not verification.get("npubench_evidence"):
            return _eligibility_rejection(
                GateID.NPUBENCH_EVALUATION_PENDING,
                npu_err or (
                    "NPUBENCH_EVALUATION_PENDING: O5 provider evaluation has not "
                    "written npubench_evidence yet (O5 runs before eligibility; "
                    "this line only fires on a reordered/invoked-early path)"
                ),
            )
    except Exception as error:
        # Non-NPUBench workspaces and optional-module friction keep the legacy
        # eligibility path; retain the diagnostic without changing that path.
        logging.getLogger(__name__).debug(
            "NPUBench eligibility probe unavailable; using legacy checks: %s",
            error,
        )
    return None


def check_finalize_eligibility(workspace: Path) -> dict:
    """Determine whether `workspace` should finalize or return a rollback.

    PASS work takes the ordered completeness gates. PARTIAL_PERSIST requires
    probe and plugin evidence; other non-PASS work returns to the first
    unfinished probe, researcher, optimizer, or worker stage.
    """
    from finalize_checks import _check_model_py_shape, _check_pass_a_coverage
    stale_msg = _check_stale_orchestrator()
    if stale_msg:
        return _eligibility_rejection(GateID.STALE_ORCHESTRATOR, stale_msg)
    verification_path = workspace / "verification.json"
    if not verification_path.exists():
        return _eligibility_rejection(
            GateID.VERIFICATION_FILE_MISSING,
            "verification.json missing — worker hasn't produced output",
        )
    try:
        verification = json.loads(verification_path.read_text())
    except Exception as error:
        return _eligibility_rejection(
            GateID.VERIFICATION_MALFORMED,
            f"verification.json malformed: {error}",
        )
    prec = verification.get("precision", {}) or {}
    pending_rejection = _npubench_pending_rejection(workspace, verification)
    if pending_rejection is not None:
        return pending_rejection
    structural_rejection = _check_eligibility_structure(
        workspace, prec, _check_model_py_shape, _check_pass_a_coverage,
    )
    if structural_rejection is not None:
        return structural_rejection
    status = prec.get("status")
    if status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return _pass_branch_eligibility(workspace, verification, prec, status)
    if status in ("PARTIAL", "PARTIAL_PASS", "FAIL"):
        if prec.get("persist_verdict") == "PARTIAL_PERSIST":
            return _partial_persist_eligibility(workspace, verification)
        return _incomplete_precision_rollback(workspace, status)
    return _eligibility_rejection(
        GateID.UNKNOWN_PRECISION_STATUS,
        f"unknown precision.status={status!r}",
    )


# ---------------------------------------------------------------------------
# BATCH PRECHECK (fix/port-a3-finalize-batch-precheck, 2026-06-16).
#
# Problem it solves: `check_finalize_eligibility` early-returns on the FIRST
# failed gate. The orchestrator then rolls back to await_worker for that one
# gate, the worker fixes it + respawns, finalize re-runs, catches the NEXT
# gate, rolls back again — serial churn (one expensive worker respawn per
# gate). Owner priority: "customer needs a complete usable op, not
# serial-rollback churn".
#
# `batch_precheck` runs the SAME gate set (single source of truth:
# `_pass_branch_gate_specs()`) in COLLECT-ALL mode and reports EVERY failure
# at once, so the worker fixes everything in ONE respawn. It does NOT alter
# any gate's pass/fail logic — finalize_pipeline's gates remain the
# authoritative backstop. This is purely an earlier + aggregated VIEW of the
# same checks.
#
# DRY guarantee: there is no hardcoded gate list here. Both this precheck and
# check_finalize_eligibility iterate `_pass_branch_gate_specs()`. Add a new
# PASS-branch gate → one new spec entry → both pick it up. (This is the exact
# drift that let P146 slip past the dde89e7e brief-contract approach, which
# had to re-enumerate gates in prose.)
# ---------------------------------------------------------------------------
def _non_pass_precheck_result(status: str) -> dict:
    """Build the not-applicable response for a non-PASS workspace."""
    return {
        "ok": True,
        "applicable": False,
        "failures": [],
        "precondition_block": None,
        "summary": (
            f"batch_precheck not applicable: precision.status={status!r} "
            "is not a PASS-class workspace (no PASS-branch gates to "
            "aggregate)"
        ),
    }


def _collect_pass_precheck_failures(workspace: Path, verification: dict) -> list[dict]:
    """Collect PASS-branch rejections in the same order as the backstop."""
    failures: list[dict] = []
    for gate_id, predicate in _pass_branch_gate_specs():
        if gate_id == _PLUGIN_EXTRAS_SENTINEL:
            extras_rejection = predicate(workspace, verification)
            if extras_rejection is not None:
                failures.append({
                    "gate": extras_rejection.get("gate", _PLUGIN_EXTRAS_SENTINEL),
                    "reason": extras_rejection.get("reason", ""),
                })
            continue
        reason = predicate(workspace, verification)
        if reason:
            failures.append({"gate": gate_id.value, "reason": reason})
    return failures


def _pass_precheck_result(status: str, failures: list[dict]) -> dict:
    """Build the aggregate PASS-branch precheck response."""
    if not failures:
        summary = (
            f"batch_precheck PASS: precision.status={status}; all "
            f"{len(_pass_branch_gate_specs())} PASS-branch finalize gates "
            "clear — finalize will promote in one pass"
        )
    else:
        gate_list = ", ".join(failure["gate"] for failure in failures)
        summary = (
            f"batch_precheck found {len(failures)} finalize-gate failure(s) "
            f"AT ONCE — fix ALL in one respawn (no serial rollback): "
            f"{gate_list}"
        )
    return {
        "ok": not failures,
        "applicable": True,
        "failures": failures,
        "precondition_block": None,
        "summary": summary,
    }


def batch_precheck(workspace: Path) -> dict:
    """Aggregate all PASS-branch finalize failures in gate-spec order.

    Hard preconditions stop aggregation; non-PASS work is not applicable.
    """
    from finalize_checks import _check_model_py_shape, _check_pass_a_coverage
    stale_msg = _check_stale_orchestrator()
    if stale_msg:
        return _precheck_blocked(GateID.STALE_ORCHESTRATOR.value, stale_msg)
    verification_path = workspace / "verification.json"
    if not verification_path.exists():
        return _precheck_blocked(
            GateID.VERIFICATION_FILE_MISSING.value,
            "verification.json missing — worker hasn't produced output",
        )
    try:
        verification = json.loads(verification_path.read_text())
    except Exception as error:
        return _precheck_blocked(
            GateID.VERIFICATION_MALFORMED.value,
            f"verification.json malformed: {error}",
        )
    prec = verification.get("precision", {}) or {}
    status = prec.get("status")
    shape_violation = _check_model_py_shape(workspace)
    if shape_violation:
        return _precheck_blocked(GateID.MODEL_PY_SHAPE.value, shape_violation)
    coverage_violation = _check_pass_a_coverage(workspace, prec)
    if coverage_violation:
        return _precheck_blocked(GateID.PASS_A_COVERAGE.value, coverage_violation)

    if status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return _non_pass_precheck_result(status)
    failures = _collect_pass_precheck_failures(workspace, verification)
    return _pass_precheck_result(status, failures)


def _precheck_blocked(gate: str, reason: str) -> dict:
    """Build a batch_precheck result for a hard precondition that prevented
    PASS-branch gate aggregation."""
    return {
        "ok": False,
        "applicable": True,
        "failures": [],
        "precondition_block": {"gate": gate, "reason": reason},
        "summary": (
            f"batch_precheck blocked by precondition gate {gate}: {reason}"
        ),
    }


def format_batch_precheck_report(result: dict) -> str:
    """Render a batch_precheck result as a worker-facing markdown block. The
    orchestrator can hand this to the respawned worker so it fixes every
    listed deliverable in one pass.
    """
    lines = ["## Finalize batch precheck", "", result.get("summary", "")]
    pcb = result.get("precondition_block")
    if pcb:
        lines += [
            "",
            f"### Precondition block — {pcb['gate']}",
            "",
            pcb["reason"],
            "",
            "(PASS-branch completeness gates were NOT evaluated — resolve the "
            "precondition first, then re-run the precheck.)",
        ]
        return "\n".join(lines)
    failures = result.get("failures", [])
    if not failures:
        lines += ["", "All finalize gates clear."]
        return "\n".join(lines)
    lines += ["", f"### {len(failures)} gate(s) to fix (ALL in one respawn):", ""]
    for i, f in enumerate(failures, 1):
        lines += [f"{i}. **{f['gate']}**", "", f"   {f['reason']}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
@dataclass
class FinalizeReport:
    op: str
    workspace: Path
    archive_dir: Optional[Path]
    skipped: bool = False
    skip_reason: str = ""
    files_promoted: list[str] = field(default_factory=list)
    dirs_promoted: list[str] = field(default_factory=list)
    skipped_names: list[str] = field(default_factory=list)
    finalized_marker: Optional[Path] = None
    errors: list[str] = field(default_factory=list)
    # P0acp 2026-05-10: KB auto-promote pipeline result (None if no pending
    # markers were found at finalize time)
    kb_auto_promote: Optional[dict] = None
    # DEBT-100 (2026-05-20): True if finalize successfully ran `git add` on the
    # archive directory; False if git-add was skipped (e.g. not a repo) or
    # failed; None if not attempted yet (e.g. early-skip path).
    git_staged: Optional[bool] = None


def _get_active_plugin(workspace: Path):
    """Plugin-dispatch helper (DEBT-094 phase 2). Lazy-imports plugins
    layer to avoid bootstrap issues. Returns the unique plugin matching
    `workspace`, or None when no plugin claims it.

    DEBT-216: this used to end `except Exception: return None`, which fed
    every finalize gate the same `None` for "no mode applies" and for "detect
    blew up". The gates read `None` as the former and skipped — silently, and
    reporting clean. A detect that cannot answer must not be laundered into an
    answer, so nothing is caught here now: ambiguity and a genuinely broken
    detect both surface. Only the plugins-package import is degraded (below),
    which is a real environment condition rather than a lost verdict.
    """
    try:
        from plugins import detect_plugin
    except ImportError:
        return None
    return detect_plugin(workspace)


def _run_plugin_extra_finalize_checks(workspace: Path, v: dict):
    """Iterate plugin-registered extra finalize checks. Returns None on
    pass (all hooks clean OR no plugin OR no hooks), else returns the
    canonical eligibility-rejection dict the caller should `return`
    directly (matching `check_finalize_eligibility`'s schema).

    DEBT-124 (2026-05-27): extracted from `check_finalize_eligibility`
    PASS-path inline (L2143-2162 pre-fix). Pre-fix the loop only ran on
    PASS path, so PARTIAL_PERSIST archives bypassed all plugin hooks.
    Now called from BOTH PASS and PARTIAL_PERSIST paths.

    Failure-mode contract:
    - Plugin layer absent (no plugins/ module) → None (default-pass)
    - `extra_finalize_checks()` raises → None (defensive; logged-only)
    - Individual gate callable raises → return rejection with
      synthesized "raised {type}: {msg}" reason so the rollback diagnostic
      surfaces the plugin author's bug instead of silent skip
    - Gate returns truthy string → return rejection with gate_name in
      schema (rollback_state="await_worker" matches pre-fix dispatch)
    """
    plugin = _get_active_plugin(workspace)
    if plugin is None:
        return None
    try:
        extras = plugin.extra_finalize_checks()
    except Exception:
        return None
    for gate_name, gate_fn in extras:
        try:
            violation = gate_fn(workspace, v)
        except Exception as e:
            violation = (
                f"plugin extra_finalize_checks[{gate_name!r}] "
                f"raised {type(e).__name__}: {e}"
            )
        if violation:
            return {
                "eligible": False,
                "rollback_state": "await_worker",
                "gate": gate_name,
                "reason": violation,
            }
    return None


def _freeze_archive_view(plugin, workspace: Path, op: str):
    """Capture a plugin-owned archive policy once per promotion.

    Old plugins expose only per-path mapping hooks and retain that behavior.
    A profile-sensitive plugin may opt into a frozen view so a mutable state
    file cannot choose different archive rules while files are being copied.
    """
    freezer = getattr(plugin, "freeze_archive_view", None) if plugin else None
    return freezer(workspace, op) if callable(freezer) else None


def _archive_view_rejection(archive_view) -> Optional[str]:
    if archive_view is None:
        return None
    reason = getattr(archive_view, "rejection_reason", None)
    return reason if isinstance(reason, str) and reason else None


def _archive_path_allowed(plugin, archive_view, workspace: Path, rel: str) -> bool:
    if archive_view is not None:
        return bool(archive_view.should_archive_path(rel))
    archive_policy = getattr(plugin, "should_archive_path", None) if plugin else None
    return not callable(archive_policy) or bool(archive_policy(workspace, rel))


def _archive_path_rejection(archive_view, rel: str) -> Optional[str]:
    """Return a profile-owned hard error for a path omitted by promotion.

    Most plugins historically expose only a bool archive filter, where a
    skipped runtime log is normal.  The direct-launch product has a stricter
    source-delivery contract: an unknown file must fail rather than vanish.
    Keep that richer result optional so every established plugin remains on
    the prior bool-only behavior.
    """
    if archive_view is None:
        return None
    classifier = getattr(archive_view, "rejected_archive_path_reason", None)
    if not callable(classifier):
        return None
    reason = classifier(rel)
    return reason if isinstance(reason, str) and reason else None


def _archive_target_rel(plugin, archive_view, workspace: Path, rel: str, op: str) -> str:
    if archive_view is not None:
        return archive_view.resolve_archive_target(rel, op)
    workspace_mapper = getattr(plugin, "resolve_archive_target_for_workspace", None)
    return (
        workspace_mapper(workspace, rel, op)
        if callable(workspace_mapper)
        else plugin.resolve_archive_target(rel, op) if plugin else rel
    )


def _archive_requires_regular_files(archive_view) -> bool:
    return bool(archive_view and getattr(archive_view, "requires_regular_files", False))


def _regular_non_symlink_file(path: Path) -> bool:
    """Check the delivery path itself, without following a task symlink."""
    import stat

    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def _finalize_with_plugin_layout(
    workspace: Path, archive_dir: Path, rep: "FinalizeReport", op: str,
    archive_view=None,
) -> list[str]:
    """Promote workspace artifacts to archive_dir using the active
    plugin's archive layout (plugin.resolve_archive_target() per file).

    Mode-agnostic: works for benchmark (identity mapping = flat copy)
    and port_a3 (ops-nn mirror) without core knowing which is active.

    User directive 2026-05-16: harness-internal files (state_transitions,
    self_critic, knowledge_update, user_decision, etc.) routed under
    archive/.harness/. Customer-facing files (op_kernel/, op_host/,
    runner.cpp, model.py/model_new_ascendc.py, verification.json,
    PROGRESS.md, analysis.md, edge_dataset.pt, etc.) stay at root.

    Updates rep.files_promoted / errors in place; returns skipped names.
    """
    plug = _get_active_plugin(workspace)
    archive_view = archive_view or _freeze_archive_view(plug, workspace, op)
    skipped_names: list[str] = []
    rejection = _archive_view_rejection(archive_view)
    if rejection:
        rep.errors.append(rejection)
        return skipped_names
    require_regular = _archive_requires_regular_files(archive_view)
    for entry in sorted(workspace.rglob("*")):
        if entry.is_dir():
            continue
        rel = entry.relative_to(workspace).as_posix()
        rejection = _archive_path_rejection(archive_view, rel)
        if rejection:
            rep.errors.append(f"promote {rel}: {rejection}")
            skipped_names.append(rel)
            continue
        if _should_skip(entry.name):
            skipped_names.append(rel)
            continue
        if not _archive_path_allowed(plug, archive_view, workspace, rel):
            skipped_names.append(rel)
            continue
        if any(part.startswith(".") for part in entry.relative_to(workspace).parts):
            skipped_names.append(rel)
            continue
        if require_regular and not _regular_non_symlink_file(entry):
            rep.errors.append(
                f"promote {rel}: direct delivery permits only regular non-symlink files"
            )
            continue
        # Harness-internal → .harness/ subdir (preserves filename)
        if _is_harness_internal(rel):
            target_rel = f".harness/{rel}"
        else:
            target_rel = _archive_target_rel(plug, archive_view, workspace, rel, op)
        dst = archive_dir / target_rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dst)
            rep.files_promoted.append(target_rel)
        except Exception as e:
            rep.errors.append(f"promote {rel} → {target_rel}: {e}")
    return skipped_names


def _archive_root_for_plugin(plugin, archive_root: Optional[Path]) -> Path:
    """Resolve the default archive root without changing plugin ownership."""
    if archive_root is not None:
        return archive_root
    subdir = (plugin.archive_project_subdir() if plugin else None) or "generated_ops"
    return _PROJECT_ROOT / "output" / subdir / "src" / "kernels"


def _can_finalize_workspace(workspace: Path, rep: FinalizeReport) -> bool:
    """Apply idempotency and verification-file preconditions to ``rep``."""
    if is_finalized(workspace):
        rep.skipped = True
        rep.skip_reason = "already finalized with current verification.json hash"
        return False
    if not (workspace / "verification.json").exists():
        rep.errors.append("verification.json missing — cannot finalize")
        return False
    return True


def _inject_finalize_metadata(op: str, workspace: Path) -> None:
    """Run additive verification metadata injectors in historical order."""
    _inject_migration_metadata(workspace)
    _inject_provenance_node(op, workspace)
    _inject_a_tier_loaded(op, workspace)
    _inject_a_tier_cross_check(op, workspace)
    _inject_perf_deferred_partial_verdict(op, workspace)


def _create_archive_dir(op: str, archive_root: Path) -> Path:
    """Resolve and create the archive directory for one operator."""
    archive_dir = archive_root / _resolve_archive_op_name(op, archive_root)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


@dataclass(frozen=True)
class _PromotionRequest:
    """Grouped workspace-to-archive promotion inputs.

    The five values always travel together (they describe one promotion of one
    workspace), so they are passed as a single request object rather than as a
    long positional parameter list.
    """

    workspace: Path
    archive_dir: Path
    op: str
    is_port_mode: bool
    archive_view: Optional[object] = None


def _flat_promote_direct_entries(
    request: _PromotionRequest, rep: FinalizeReport, plugin, archive_view,
) -> list[str]:
    """Promote every nested entry under the strict direct-delivery policy.

    Strict direct policies must apply to every nested item.  The legacy
    _merge_copy_dir helper intentionally only understands scratch-name
    filtering, so using it here would leak kernel/build/*.so and other
    runtime artefacts after a top-level kernel/ decision passed.
    """
    workspace = request.workspace
    skipped_names: list[str] = []
    for entry in sorted(workspace.rglob("*")):
        if entry.is_dir():
            continue
        rel = entry.relative_to(workspace).as_posix()
        path_rejection = _archive_path_rejection(archive_view, rel)
        if path_rejection:
            rep.errors.append(f"promote {rel}: {path_rejection}")
            skipped_names.append(rel)
            continue
        if _should_skip(entry.name) or not _archive_path_allowed(
            plugin, archive_view, workspace, rel
        ):
            skipped_names.append(rel)
            continue
        if any(part.startswith(".") for part in entry.relative_to(workspace).parts):
            skipped_names.append(rel)
            continue
        if not _regular_non_symlink_file(entry):
            rep.errors.append(
                f"promote {rel}: direct delivery permits only regular non-symlink files"
            )
            continue
        try:
            destination = request.archive_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, destination)
            rep.files_promoted.append(rel)
        except Exception as error:
            rep.errors.append(f"promote {rel}: {error}")
    return skipped_names


def _flat_promote_top_level_entries(
    request: _PromotionRequest, rep: FinalizeReport, plugin, archive_view,
) -> list[str]:
    """Merge the workspace's top-level entries with legacy best-effort copies."""
    workspace = request.workspace
    archive_dir = request.archive_dir
    skipped_names: list[str] = []
    for entry in sorted(workspace.iterdir()):
        name = entry.name
        if _should_skip(name):
            skipped_names.append(name)
            continue
        if not _archive_path_allowed(plugin, archive_view, workspace, name):
            skipped_names.append(name)
            continue
        try:
            if entry.is_dir():
                _merge_copy_dir(entry, archive_dir / name)
                rep.dirs_promoted.append(name)
            else:
                shutil.copy2(entry, archive_dir / name)
                rep.files_promoted.append(name)
        except Exception as error:
            rep.errors.append(f"promote {name}: {error}")
    return skipped_names


def _finalize_with_flat_layout(
    request: _PromotionRequest, rep: FinalizeReport,
) -> list[str]:
    """Merge non-port workspace entries into the archive directory."""
    workspace = request.workspace
    plugin = _get_active_plugin(workspace)
    archive_view = request.archive_view or _freeze_archive_view(
        plugin, workspace, workspace.name
    )
    rejection = _archive_view_rejection(archive_view)
    if rejection:
        rep.errors.append(rejection)
        return []
    if _archive_requires_regular_files(archive_view):
        return _flat_promote_direct_entries(request, rep, plugin, archive_view)
    return _flat_promote_top_level_entries(request, rep, plugin, archive_view)


def _promote_workspace(request: _PromotionRequest, rep: FinalizeReport) -> None:
    """Promote workspace files through the port or flat layout writer."""
    if request.is_port_mode:
        rep.skipped_names = _finalize_with_plugin_layout(
            request.workspace, request.archive_dir, rep, request.op, request.archive_view
        )
        return
    rep.skipped_names = _finalize_with_flat_layout(request, rep)


def _copy_plugin_docs(workspace: Path, archive_dir: Path, rep: FinalizeReport) -> None:
    """Copy plugin-owned documentation files into the archive."""
    plugin = _get_active_plugin(workspace)
    if plugin is None:
        return
    docs_target = archive_dir / "docs"
    for dest_name, src_path in plugin.docs_source_files(workspace):
        docs_target.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src_path, docs_target / dest_name)
            rep.files_promoted.append(f"docs/{dest_name}")
        except Exception as error:
            rep.errors.append(f"docs/{dest_name}: {error}")


def _ensure_archive_readme(op: str, workspace: Path, archive_dir: Path, rep: FinalizeReport) -> None:
    """Write the generated archive README when the plugin did not supply one."""
    if (archive_dir / "README.md").exists():
        return
    try:
        _write_archive_readme(archive_dir, op, workspace)
        rep.files_promoted.append("README.md")
    except Exception as error:
        rep.errors.append(f"README.md auto-gen: {error}")


def _promote_docs_and_readme(
    op: str, workspace: Path, archive_dir: Path, rep: FinalizeReport,
) -> None:
    """Apply the universal docs and README hook without blocking promotion."""
    try:
        _copy_plugin_docs(workspace, archive_dir, rep)
        _ensure_archive_readme(op, workspace, archive_dir, rep)
    except Exception as error:
        rep.errors.append(f"docs/README hook (non-blocking): {error}")


def _write_finalized_marker(op: str, workspace: Path, archive_dir: Path, rep: FinalizeReport) -> None:
    """Write the workspace idempotency marker after a successful promotion."""
    verification_hash = _verification_hash(workspace)
    marker = workspace / f".finalized-{verification_hash}"
    marker.write_text(json.dumps({
        "op": op,
        "archive_dir": str(archive_dir),
        "verification_hash": verification_hash,
        "files_promoted": rep.files_promoted,
        "dirs_promoted": rep.dirs_promoted,
    }))
    rep.finalized_marker = marker


def _project_archive_relative_path(archive_dir: Path) -> Optional[Path]:
    """Return the project-relative archive path, if staging is applicable."""
    try:
        return archive_dir.relative_to(_PROJECT_ROOT)
    except ValueError:
        return None


def _auto_stage_archive(archive_dir: Path, rep: FinalizeReport) -> None:
    """Best-effort stage a project-local finalized archive without committing."""
    try:
        import subprocess as _subproc
        relative_path = _project_archive_relative_path(archive_dir)
        if relative_path is None:
            rep.git_staged = False
            return
        git_executable = str(Path(shutil.which("git") or "git").resolve())
        result = _subproc.run(
            [git_executable, "-C", str(_PROJECT_ROOT), "add", "--", str(relative_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            rep.git_staged = False
            rep.errors.append(
                f"git-add auto-stage (non-blocking, DEBT-100): rc={result.returncode} "
                f"stderr={result.stderr.strip()[:200]}"
            )
        else:
            rep.git_staged = True
    except Exception as error:
        rep.git_staged = False
        rep.errors.append(f"git-add auto-stage (non-blocking, DEBT-100): {error}")


def finalize_op(
    op: str,
    workspace: Path,
    *,
    archive_root: Optional[Path] = None,
) -> FinalizeReport:
    """Finalize an eligible workspace using its plugin-owned archive layout."""
    plugin = _get_active_plugin(workspace)
    is_port_mode = bool(plugin and plugin.archive_layout_mapping(workspace))
    archive_view = _freeze_archive_view(plugin, workspace, op)
    archive_root = _archive_root_for_plugin(plugin, archive_root)
    rep = FinalizeReport(op=op, workspace=workspace, archive_dir=None)
    if not _can_finalize_workspace(workspace, rep):
        return rep
    rejection = _archive_view_rejection(archive_view)
    if rejection:
        rep.errors.append(rejection)
        return rep
    _inject_finalize_metadata(op, workspace)
    archive_dir = _create_archive_dir(op, archive_root)
    rep.archive_dir = archive_dir
    _promote_workspace(
        _PromotionRequest(
            workspace=workspace,
            archive_dir=archive_dir,
            op=op,
            is_port_mode=is_port_mode,
            archive_view=archive_view,
        ),
        rep,
    )
    # The direct profile is an allow-listed source product.  A failed copy of
    # any permitted entry must not be hidden by a finalized marker or a
    # partially promoted archive; legacy profiles retain their historical
    # best-effort promotion behavior.
    if archive_view is not None and getattr(archive_view, "strict_delivery", False) and rep.errors:
        return rep
    _promote_docs_and_readme(op, workspace, archive_dir, rep)
    _write_finalized_marker(op, workspace, archive_dir, rep)
    _auto_stage_archive(archive_dir, rep)
    return rep

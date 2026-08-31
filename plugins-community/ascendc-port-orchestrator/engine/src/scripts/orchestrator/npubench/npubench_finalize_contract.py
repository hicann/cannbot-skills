# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalization contract for runner-owned NPUKernelBench evidence.

This module is intentionally small and provider-specific.  It lets the broad
legacy finalize gates ask one question -- are the immutable bundle, candidate
binding, precision report, and quick-profile artifact still a single valid
evidence unit? -- without falling through to a live-source performance rule.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from npubench.npubench_inputs import NPUBENCH_SOURCE, verify_npubench_stage
from reference_source import (
    ReferenceSourceError,
    load_durable_state,
    resolve_reference_binding,
)


EVIDENCE_DIR = "npubench_evidence"
PERF_BACKFILL_ENV = "CANNBOT_NPUBENCH_PERF_BACKFILL"
_REPORT_FILENAMES = {
    "precision": "precision_report.json",
    "performance": "performance_report.json",
    "evaluate": "evaluate_report.json",
}
# source kind -> (provider constant name, fallback receipt path)
_CONTROLLED_BUILD_RECEIPTS = {
    "port-aclnn-tilelang2ascendc": (
        "TILELANG2ASCENDC_BUILD_RECEIPT_PATH",
        "npubench_evidence/tilelang2ascendc_build_receipt.json",
    ),
}


def resolve_npubench_workspace(workspace: Path | None) -> tuple[bool, Optional[str]]:
    """Return whether this is an NPUBench workspace and any binding error.

    Missing state remains non-applicable for generic legacy gate callers.  If
    a state explicitly names npubench, however, every malformed/incomplete
    binding is a provider error and must block finalization.
    """
    if workspace is None:
        return False, None
    try:
        state = load_durable_state(Path(workspace))
    except ReferenceSourceError:
        # Generic legacy finalize callers use this helper as a provider probe.
        # The main O5 source resolver still fails closed for an unreadable
        # migration state; do not follow a state symlink merely to classify it.
        return False, None
    if not isinstance(state, dict):
        return False, None
    reference = state.get("reference")
    if not isinstance(reference, Mapping) or reference.get("source") != NPUBENCH_SOURCE:
        return False, None
    try:
        resolve_reference_binding(state)
    except ReferenceSourceError as exc:
        return True, f"NPUBENCH_SOURCE_INVALID: {exc}"
    return True, None


def validate_npubench_finalize_evidence(workspace: Path, verification: Mapping[str, Any]) -> Optional[str]:
    """Return an actionable error unless all final evidence is one PASS unit."""
    is_npubench, source_error = resolve_npubench_workspace(workspace)
    if not is_npubench:
        return None
    if source_error:
        return source_error
    try:
        state = load_durable_state(Path(workspace))
        reference = resolve_reference_binding(state)
    except Exception as exc:
        return f"NPUBENCH_SOURCE_INVALID: {type(exc).__name__}: {exc}"
    valid_stage, reason, _manifest = verify_npubench_stage(Path(workspace), reference)
    if not valid_stage:
        return f"NPUBENCH_STAGE_INVALID: {reason}"
    if verification.get("truth_source") != NPUBENCH_SOURCE:
        return "NPUBENCH_EVIDENCE_INVALID: verification.truth_source is not npubench"
    evidence = verification.get("npubench_evidence")
    if not isinstance(evidence, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: missing harness-owned npubench_evidence block"
    reports, error = _load_reports(Path(workspace), evidence)
    if error:
        return error
    binding_error = _validate_binding_identity(evidence, reports)
    if binding_error:
        return binding_error
    evidence_error = _recompute_runner_bindings(Path(workspace), reports)
    if evidence_error:
        return evidence_error
    build_error = _validate_controlled_build_receipt(
        Path(workspace), state, reports["precision"].get("evaluation_binding")
    )
    if build_error:
        return build_error
    target_error = _validate_target_execution_receipt(
        Path(workspace), reference, evidence, reports
    )
    if target_error:
        return target_error
    inline_error = _validate_inline_agreement(verification, reports)
    if inline_error:
        return inline_error
    return _validate_report_contents(
        reports["precision"], reports["performance"], reports["evaluate"], Path(workspace)
    )


def _validate_binding_identity(
    evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Require one shared binding digest across the inline block and reports."""
    bindings = [
        evidence.get("binding_sha256"),
        reports["precision"].get("binding_sha256"),
        reports["performance"].get("binding_sha256"),
        reports["evaluate"].get("binding_sha256"),
    ]
    if not all(_is_sha256(item) for item in bindings):
        return "NPUBENCH_EVIDENCE_INVALID: a binding_sha256 is missing or malformed"
    if len(set(bindings)) != 1:
        return "NPUBENCH_EVIDENCE_INVALID: inline/report binding_sha256 values differ"
    return None


def _validate_inline_agreement(
    verification: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Require the inline verification block to mirror the runner reports."""
    precision = reports["precision"]
    if verification.get("performance") != reports["performance"]:
        return "NPUBENCH_EVIDENCE_INVALID: inline performance differs from runner report"
    inline_precision = verification.get("precision")
    if not isinstance(inline_precision, Mapping) or inline_precision.get("pass_a") != precision.get("pass_a"):
        return "NPUBENCH_EVIDENCE_INVALID: inline pass_a differs from runner report"
    return None


def _validate_controlled_build_receipt(
    workspace: Path,
    state: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any] | None,
) -> Optional[str]:
    """Bind final NPUKernelBench evidence to the preceding target build.

    O5 normally invokes the controlled builder directly before snapshotting,
    but finalization must not trust that call graph alone.  Recheck the
    authenticated route receipt and recompute its authored-candidate digest
    against the exact immutable snapshot used by the reports.  This closes a
    build→snapshot TOCTOU window and blocks hand-assembled evidence.
    """
    source = state.get("port_source")
    source_kind = source.get("kind") if isinstance(source, Mapping) else None
    if source_kind not in _CONTROLLED_BUILD_RECEIPTS:
        return None
    if not isinstance(evaluation_binding, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: missing evaluation binding for controlled build"
    candidate_digest = evaluation_binding.get("candidate_tree_sha256")
    if not _is_sha256(candidate_digest):
        return "NPUBENCH_EVIDENCE_INVALID: controlled-build candidate snapshot digest is malformed"
    source_stage_digest = state.get("source_stage_digest")
    if not _is_sha256(source_stage_digest):
        return "NPUBENCH_EVIDENCE_INVALID: controlled-build source stage digest is malformed"
    stage_error = _validate_source_stage_binding(source, source_stage_digest)
    if stage_error:
        return stage_error
    try:
        return _check_controlled_build_receipt(
            workspace, source_kind, str(source_stage_digest), str(candidate_digest)
        )
    # json.JSONDecodeError and UnicodeError are both ValueError subclasses, so
    # ValueError alone covers the decode failures this read can raise.
    except (OSError, ValueError, TypeError) as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: controlled build receipt is unreadable: {exc}"
    except Exception as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: controlled build receipt check failed: {type(exc).__name__}: {exc}"


def _validate_source_stage_binding(source: object, source_stage_digest: object) -> Optional[str]:
    """Require every digest the port source declares to name one stage tree."""
    if not isinstance(source, Mapping):
        return None
    for key in ("tree_sha256", "digest", "source_stage_digest"):
        value = source.get(key)
        if value is None:
            continue
        if not _is_sha256(value) or value != source_stage_digest:
            return "NPUBENCH_EVIDENCE_INVALID: controlled-build source stage binding differs"
    return None


def _check_controlled_build_receipt(
    workspace: Path,
    source_kind: str,
    source_stage_digest: str,
    candidate_digest: str,
) -> Optional[str]:
    """Authenticate the route receipt and rebind it to the frozen snapshot."""
    npubench_target = importlib.import_module("npubench.npubench_target")

    # Bind the provider's protected helpers by import instead of attribute
    # access.  The import stays function-local so the deliberate late import
    # and the test suite's sys.modules substitution both keep working.
    from npubench.npubench_target import _candidate_source_digest, _receipt_payload_valid

    receipt_attrs = _CONTROLLED_BUILD_RECEIPTS[source_kind]
    receipt_rel = getattr(npubench_target, receipt_attrs[0], receipt_attrs[1])
    receipt_path = workspace / receipt_rel
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return f"NPUBENCH_EVIDENCE_INVALID: missing controlled build receipt {receipt_rel}"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: controlled build receipt is not an object"
    if not _receipt_payload_valid(receipt, workspace):
        return "NPUBENCH_EVIDENCE_INVALID: controlled build receipt authentication failed"
    if not _is_route_pass_receipt(receipt, source_kind, source_stage_digest):
        return "NPUBENCH_EVIDENCE_INVALID: controlled build receipt is not a PASS for this route"
    if source_kind == "port-aclnn-tilelang2ascendc" and not _has_independence_proof(
        receipt, npubench_target.TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA
    ):
        return "NPUBENCH_EVIDENCE_INVALID: TileLang2AscendC candidate independence proof is missing"
    snapshot = workspace / ".npubench_candidate" / candidate_digest
    if not snapshot.is_dir() or snapshot.is_symlink():
        return "NPUBENCH_EVIDENCE_INVALID: controlled-build candidate snapshot is missing"
    snapshot_candidate_digest = _candidate_source_digest(snapshot)
    if receipt.get("candidate_source_sha256") != snapshot_candidate_digest:
        return (
            "NPUBENCH_EVIDENCE_INVALID: controlled build receipt candidate differs "
            "from the evaluated immutable snapshot"
        )
    return None


def _is_route_pass_receipt(
    receipt: Mapping[str, Any],
    source_kind: str,
    source_stage_digest: str,
) -> bool:
    """Check the fixed PASS fields a controlled build receipt must carry."""
    if receipt.get("status") != "PASS" or receipt.get("source_kind") != source_kind:
        return False
    if receipt.get("source_stage_digest") != source_stage_digest:
        return False
    if receipt.get("build_mode") != "controlled_authored_cmake":
        return False
    return receipt.get("returncode") == 0


def _has_independence_proof(receipt: Mapping[str, Any], expected_schema: object) -> bool:
    """Require a schema-tagged candidate independence proof on the receipt."""
    if receipt.get("candidate_independence_gate") != "PASS":
        return False
    if receipt.get("candidate_independence_schema") != expected_schema:
        return False
    return isinstance(receipt.get("candidate_independence_proof"), Mapping)


def _validate_target_execution_receipt(
    workspace: Path,
    reference: Mapping[str, Any],
    evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Require the one controller-imported A5 evaluation receipt.

    The provider owns target transport because it knows the exact staged
    runner/profile closure.  Finalization owns the policy boundary: every
    NPUKernelBench PASS must name the fixed receipt path and digest, and the
    transport verifier must bind those imported bytes to the local immutable
    bundle and candidate snapshot.  Keeping the receipt in harness evidence
    instead of durable state prevents an old candidate's target run being
    reused after a resume.
    """
    receipt_path = evidence.get("target_execution_receipt")
    if receipt_path != f"{EVIDENCE_DIR}/target_receipt.json":
        return "NPUBENCH_EVIDENCE_INVALID: target execution receipt path is not canonical"
    receipt_sha256 = evidence.get("target_execution_receipt_sha256")
    if not _is_sha256(receipt_sha256):
        return "NPUBENCH_EVIDENCE_INVALID: target execution receipt digest is missing or malformed"
    try:
        npubench_target = importlib.import_module("npubench.npubench_target")
    except Exception as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: cannot import target receipt verifier: {exc}"
    try:
        valid, reason = npubench_target.validate_target_evidence_receipt(
            workspace=workspace,
            reference=reference,
            evidence=evidence,
            reports=reports,
        )
    except Exception as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: target execution receipt check raised {type(exc).__name__}: {exc}"
    if valid is not True:
        return "NPUBENCH_EVIDENCE_INVALID: target execution receipt rejected: " + str(reason)
    return None


def _recompute_runner_bindings(
    workspace: Path,
    reports: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Rebind every evidence file to the immutable candidate snapshot.

    A matching hash written by a worker is not evidence on its own.  The
    snapshot path is deliberately derived from the reported content digest,
    rather than accepted from a report field, and each report is then checked
    through the runner's canonical binding builder.
    """
    precision = reports["precision"]
    binding = precision.get("evaluation_binding")
    if not isinstance(binding, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: precision report has no evaluation binding"
    try:
        npubench_runner = importlib.import_module("npubench.npubench_runner")
    except Exception as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: cannot import runner to verify evidence: {exc}"
    scheme_error = _validate_candidate_digest_scheme(
        binding, npubench_runner.CANDIDATE_DIGEST_SCHEME
    )
    if scheme_error:
        return scheme_error
    digest = binding.get("candidate_tree_sha256")
    if not _is_sha256(digest):
        return "NPUBENCH_EVIDENCE_INVALID: candidate snapshot digest is missing or malformed"
    snapshot = workspace / ".npubench_candidate" / str(digest)
    snapshot_error = _validate_candidate_snapshot(snapshot, str(digest), workspace)
    if snapshot_error:
        return snapshot_error
    try:
        current = npubench_runner.build_evaluation_binding(workspace, workspace)
    except Exception as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: cannot bind current candidate scope: {exc}"
    for field in ("candidate_tree_sha256", "candidate_entry_sha256", "candidate_entry"):
        if current.get(field) != binding.get(field):
            return (
                "NPUBENCH_EVIDENCE_INVALID: current candidate scope differs "
                f"from the frozen evaluation snapshot ({field})"
            )
    for verb, report in reports.items():
        valid, reason = npubench_runner.verify_evidence_report(
            workspace,
            report,
            expected_verb=verb,
            candidate_dir=snapshot,
        )
        if not valid:
            return f"NPUBENCH_EVIDENCE_INVALID: {verb} report failed binding recomputation: {reason}"
    return _validate_evaluate_payload_identity(reports)


def _validate_evaluate_payload_identity(reports: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """Require the evaluate report to embed the two fixed sibling reports."""
    evaluate = reports["evaluate"]
    if evaluate.get("precision") != reports["precision"]:
        return "NPUBENCH_EVIDENCE_INVALID: evaluate precision payload differs from fixed report"
    if evaluate.get("performance") != reports["performance"]:
        return "NPUBENCH_EVIDENCE_INVALID: evaluate performance payload differs from fixed report"
    return None


def _validate_candidate_digest_scheme(binding: Mapping[str, Any], current_scheme: object) -> Optional[str]:
    """Refuse evidence frozen under an older candidate-scope digest scheme.

    Digest-scheme drift: the evidence was frozen under an OLDER candidate
    scope exclusion list.  That is harness state evolution, not candidate
    drift — re-computing the old digest with the current exclusion list can
    only mismatch, and routing to await_worker cannot repair it (audit H1).
    The finalize self-heal path keys off this exact marker string.
    """
    binding_scheme = binding.get("candidate_digest_scheme")
    if binding_scheme == current_scheme:
        return None
    return (
        "NPUBENCH_EVIDENCE_INVALID: candidate digest scheme drift "
        f"(evidence frozen under scheme {binding_scheme!r}, current harness "
        f"scheme {current_scheme!r}) — harness must "
        "re-freeze the snapshot with the current exclusion list and "
        "re-evaluate; do not route to await_worker"
    )


def _validate_candidate_snapshot(snapshot: Path, digest: str, workspace: Path) -> Optional[str]:
    """Check the content-addressed snapshot is real, frozen, and in scope."""
    try:
        expected_parent = (workspace / ".npubench_candidate").resolve(strict=True)
        resolved = snapshot.resolve(strict=True)
        resolved.relative_to(expected_parent)
    except (OSError, RuntimeError, ValueError):
        return "NPUBENCH_EVIDENCE_INVALID: immutable candidate snapshot is missing or escapes workspace"
    if snapshot.is_symlink() or not resolved.is_dir() or resolved.name != digest:
        return "NPUBENCH_EVIDENCE_INVALID: immutable candidate snapshot is unsafe"
    try:
        npubench_runner = importlib.import_module("npubench.npubench_runner")

        # The snapshot also carries the built extension modules under
        # kernel/build (excluded from the scope digest — see npubench_runner.
        # _candidate_excluded); its content address is the scope digest, so
        # verify with the same exclusion.
        if npubench_runner.candidate_tree_sha256(resolved) != digest:
            return "NPUBENCH_EVIDENCE_INVALID: candidate snapshot content digest differs from evidence"
        for item in [resolved, *resolved.rglob("*")]:
            if item.is_symlink() or item.stat().st_mode & 0o222:
                return "NPUBENCH_EVIDENCE_INVALID: candidate snapshot is writable or contains a symlink"
    except (OSError, ValueError, RuntimeError) as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: cannot validate candidate snapshot: {exc}"
    return None


def _load_reports(
    workspace: Path, evidence: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], Optional[str]]:
    reports: dict[str, dict[str, Any]] = {}
    for name, filename in _REPORT_FILENAMES.items():
        declared = evidence.get(f"{name}_report")
        if declared != f"{EVIDENCE_DIR}/{filename}":
            return {}, f"NPUBENCH_EVIDENCE_INVALID: {name}_report path is not canonical"
        path = workspace / EVIDENCE_DIR / filename
        if path.is_symlink() or not path.is_file():
            return {}, f"NPUBENCH_EVIDENCE_INVALID: missing regular {name} report"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"NPUBENCH_EVIDENCE_INVALID: unreadable {name} report: {exc}"
        if not isinstance(payload, dict):
            return {}, f"NPUBENCH_EVIDENCE_INVALID: {name} report is not an object"
        reports[name] = payload
    return reports, None


def _validate_report_contents(
    precision: Mapping[str, Any],
    performance: Mapping[str, Any],
    evaluate: Mapping[str, Any],
    workspace: Path,
) -> Optional[str]:
    """Check the published report payloads are one complete PASS result."""
    if precision.get("status") != "PASS" or evaluate.get("status") != "PASS":
        return "NPUBENCH_EVIDENCE_INVALID: precision/evaluate status is not PASS"
    input_adapter = _sidecar_input_adapter(precision)
    expected_case_count: int | None = None
    if input_adapter is not None:
        value = input_adapter.get("case_count")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "NPUBENCH_EVIDENCE_INVALID: sidecar input adapter case_count is invalid"
        expected_case_count = value
        sidecar_error = _validate_sidecar_precision_coverage(precision, input_adapter, value)
        if sidecar_error:
            return sidecar_error
    pass_a_error = _validate_pass_a_coverage(precision, expected_case_count)
    if pass_a_error:
        return pass_a_error
    if performance.get("status") == "DEFERRED":
        return _validate_deferred_performance(performance, precision)
    adapter_error = _validate_performance_adapter_binding(
        performance, evaluate, input_adapter, expected_case_count
    )
    if adapter_error:
        return adapter_error
    return _validate_measured_performance(performance, precision, workspace, expected_case_count)


def _sidecar_input_adapter(precision: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the sidecar-descriptor input adapter named by the binding."""
    evaluation_binding = precision.get("evaluation_binding")
    if not isinstance(evaluation_binding, Mapping):
        return None
    input_adapter = evaluation_binding.get("input_adapter")
    if not isinstance(input_adapter, Mapping):
        return None
    if input_adapter.get("kind") != "sidecar_descriptor/v1":
        return None
    return input_adapter


def _validate_sidecar_precision_coverage(
    precision: Mapping[str, Any],
    input_adapter: Mapping[str, Any],
    expected_case_count: int,
) -> Optional[str]:
    """Require complete per-case precision evidence for a sidecar adapter."""
    if precision.get("input_adapter") != dict(input_adapter):
        return "NPUBENCH_EVIDENCE_INVALID: precision input adapter identity differs from binding"
    reported_count = precision.get("case_count")
    if reported_count != expected_case_count:
        return "NPUBENCH_EVIDENCE_INVALID: precision case_count differs from sidecar binding"
    cases = precision.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        return "NPUBENCH_EVIDENCE_INVALID: precision per-case coverage is incomplete"
    indices = [case.get("case") for case in cases if isinstance(case, Mapping)]
    if set(indices) != set(range(expected_case_count)) or len(indices) != expected_case_count:
        return "NPUBENCH_EVIDENCE_INVALID: precision case indices are incomplete"
    return None


def _validate_pass_a_coverage(
    precision: Mapping[str, Any], expected_case_count: int | None
) -> Optional[str]:
    """Require pass_a to be a complete positive tier-1 coverage result."""
    pass_a = precision.get("pass_a")
    if not isinstance(pass_a, Mapping) or pass_a.get("status") != "PASS":
        return "NPUBENCH_EVIDENCE_INVALID: pass_a status is not PASS"
    passed, total = pass_a.get("tier1_pass"), pass_a.get("total")
    incomplete = "NPUBENCH_EVIDENCE_INVALID: pass_a is not a complete positive coverage result"
    if not _is_plain_int(passed) or not _is_plain_int(total):
        return incomplete
    if total <= 0 or passed != total:
        return incomplete
    if expected_case_count is not None and total != expected_case_count:
        return "NPUBENCH_EVIDENCE_INVALID: pass_a total differs from sidecar binding case_count"
    return None


def _validate_deferred_performance(
    performance: Mapping[str, Any], precision: Mapping[str, Any]
) -> Optional[str]:
    """Accept a DEFERRED perf placeholder only outside a backfill run."""
    if os.environ.get(PERF_BACKFILL_ENV, "0") == "1":
        # Perf backfill (2026-08-25): a backfill run must REPLACE the
        # DEFERRED placeholder with a measured report.  Accepting another
        # DEFERRED here would route the op to done with perf never
        # measured (the SKIP_PERF-still-set spin codex review caught).
        return (
            "NPUBENCH_EVIDENCE_INVALID: performance is still DEFERRED "
            "under CANNBOT_NPUBENCH_PERF_BACKFILL=1; the backfill run "
            "must unset CANNBOT_NPUBENCH_SKIP_PERF and produce a measured "
            "performance report before finalize may route to done"
        )
    if performance.get("perf_deferred") is not True:
        return "NPUBENCH_EVIDENCE_INVALID: performance DEFERRED without perf_deferred marker"
    if performance.get("binding_sha256") != precision.get("binding_sha256"):
        return "NPUBENCH_EVIDENCE_INVALID: deferred performance binding differs from precision"
    if performance.get("evaluation_binding") != precision.get("evaluation_binding"):
        return "NPUBENCH_EVIDENCE_INVALID: deferred performance binding payload differs from precision"
    return None


def _validate_performance_adapter_binding(
    performance: Mapping[str, Any],
    evaluate: Mapping[str, Any],
    input_adapter: Optional[Mapping[str, Any]],
    expected_case_count: int | None,
) -> Optional[str]:
    """Bind the perf and evaluate reports to the same sidecar input adapter."""
    if expected_case_count is None:
        return None
    native_fixture = performance.get("native_fixture")
    if not isinstance(native_fixture, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: performance native fixture evidence is malformed"
    native_adapter = native_fixture.get("input_adapter")
    if native_adapter != input_adapter:
        return "NPUBENCH_EVIDENCE_INVALID: performance native input adapter differs from sidecar binding"
    evaluate_binding = evaluate.get("evaluation_binding")
    if not isinstance(evaluate_binding, Mapping) or evaluate_binding.get("input_adapter") != input_adapter:
        return "NPUBENCH_EVIDENCE_INVALID: evaluate input adapter differs from sidecar binding"
    return None


def _validate_measured_performance(
    performance: Mapping[str, Any],
    precision: Mapping[str, Any],
    workspace: Path,
    expected_case_count: int | None,
) -> Optional[str]:
    """Require a measured quick-mode PASS bound to a real profile archive."""
    if performance.get("status") != "PASS":
        return "NPUBENCH_EVIDENCE_INVALID: performance status is not PASS"
    if (
        os.environ.get(PERF_BACKFILL_ENV, "0") == "1"
        and performance.get("measurement_completed") is not True
    ):
        return (
            "NPUBENCH_EVIDENCE_INVALID: perf backfill "
            "(CANNBOT_NPUBENCH_PERF_BACKFILL=1) requires a measured "
            "performance report (status=PASS with measurement_completed=true)"
        )
    integrity_error = _validate_performance_integrity(precision, performance)
    if integrity_error:
        return integrity_error
    contract_error = _validate_quick_profile_contract(performance)
    if contract_error:
        return contract_error
    coverage_error = _validate_native_performance_coverage(
        performance, expected_case_count=expected_case_count
    )
    if coverage_error:
        return coverage_error
    archive = _resolve_archive(workspace, performance.get("profile_archive"))
    if archive is None:
        return "NPUBENCH_EVIDENCE_INVALID: profile archive is missing, outside evidence, or unsafe"
    try:
        actual_tree_digest = profile_tree_sha256(archive)
    except (OSError, ValueError) as exc:
        return f"NPUBENCH_EVIDENCE_INVALID: profile archive is unsafe: {exc}"
    if actual_tree_digest != performance.get("profile_tree_sha256"):
        return "NPUBENCH_EVIDENCE_INVALID: profile archive tree digest differs from report"
    return None


def _validate_performance_integrity(
    precision: Mapping[str, Any], performance: Mapping[str, Any]
) -> Optional[str]:
    """Recheck process/hash integrity for both reports and the fixture stage."""
    integrity_error = _validate_execution_integrity(precision, "precision")
    if integrity_error:
        return integrity_error
    integrity_error = _validate_execution_integrity(performance, "performance")
    if integrity_error:
        return integrity_error
    return _validate_execution_integrity(
        performance, "performance fixture", prefix="fixture_", require_returncode=False
    )


def _validate_quick_profile_contract(performance: Mapping[str, Any]) -> Optional[str]:
    """Require the fixed W3/R5/keep-prof quick-mode profiling contract."""
    warmup = performance.get("warm_up", performance.get("warmup"))
    if warmup != 3 or performance.get("repeats") != 5 or performance.get("keep_prof") is not True:
        return "NPUBENCH_EVIDENCE_INVALID: performance did not use the fixed W3/R5/keep-prof contract"
    if performance.get("profiling_mode") != "quick":
        return "NPUBENCH_EVIDENCE_INVALID: performance is not quick-mode gate evidence"
    command = performance.get("command")
    if not isinstance(command, list) or not _has_required_quick_command_flags(command):
        return "NPUBENCH_EVIDENCE_INVALID: performance command lacks required quick profiler flags"
    return None


def _is_plain_int(value: object) -> bool:
    """True only for a real int: bool is an int subclass and is rejected."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_execution_integrity(
    report: Mapping[str, Any],
    role: str,
    *,
    prefix: str = "",
    require_returncode: bool = True,
) -> Optional[str]:
    """Require the practical process/hash evidence emitted by this runner.

    These fields deliberately do **not** claim an OS-level adversarial
    sandbox.  They say only that benchmark code ran in a fresh child process
    and that the parent rechecked the frozen input/candidate/fixture hashes
    after it returned.  That is the useful integrity contract for ordinary
    agent iterations; malicious same-UID task code remains out of scope.
    """
    isolation = report.get(f"{prefix}execution_isolation")
    if isolation != "process_boundary":
        return (
            f"NPUBENCH_EVIDENCE_INVALID: {role} did not report the required "
            "process_boundary"
        )
    protection = report.get(f"{prefix}tamper_protection")
    if protection != "post_run_hash_check":
        return (
            f"NPUBENCH_EVIDENCE_INVALID: {role} did not report "
            "post_run_hash_check"
        )
    if require_returncode and report.get("child_returncode") != 0:
        return f"NPUBENCH_EVIDENCE_INVALID: {role} child did not exit cleanly"
    return None


def _has_required_quick_command_flags(command: list[Any]) -> bool:
    values = [str(item) for item in command]
    if values[:2] != ["python3", "ops/ops-profiling/scripts/msprof_perf_summary.py"]:
        return False
    return (
        "--quick" in values
        and "--keep-prof" in values
        and _has_flag_value(values, "--warmup", "3")
        and _has_flag_value(values, "--repeats", "5")
    )


def _has_flag_value(values: list[str], flag: str, expected: str) -> bool:
    """Check a split CLI option without accepting a value from another flag."""
    try:
        index = values.index(flag)
    except ValueError:
        return False
    return index + 1 < len(values) and values[index + 1] == expected


def _validate_native_performance_coverage(
    performance: Mapping[str, Any], *, expected_case_count: int | None = None
) -> Optional[str]:
    """Recheck the full frozen-fixture coverage after report publication.

    The runner independently validates raw profiler output before it writes the
    report.  Finalization repeats the small structural portion so a later
    report substitution cannot turn one convenient case into a PASS while
    retaining a still-valid candidate binding and profile archive.
    """
    native = performance.get("native_fixture")
    summary = performance.get("profiler_summary")
    if not isinstance(native, Mapping) or not isinstance(summary, Mapping):
        return "NPUBENCH_EVIDENCE_INVALID: performance lacks native fixture/summary evidence"
    digest_error = _validate_native_fixture_digests(native, summary)
    if digest_error:
        return digest_error
    count = native.get("case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        return "NPUBENCH_EVIDENCE_INVALID: native fixture has no positive case count"
    if expected_case_count is not None and count != expected_case_count:
        return "NPUBENCH_EVIDENCE_INVALID: native fixture case_count differs from sidecar binding"
    valid = _case_index_set(native.get("valid_case_indices"), count)
    empty = _case_index_set(native.get("empty_case_indices"), count)
    if valid is None or empty is None:
        return "NPUBENCH_EVIDENCE_INVALID: native fixture case partition is incomplete"
    if valid & empty or valid | empty != set(range(count)):
        return "NPUBENCH_EVIDENCE_INVALID: native fixture case partition is incomplete"
    if not valid:
        return "NPUBENCH_EVIDENCE_INVALID: native fixture has no non-empty case to profile"
    cases, records_error = _collect_native_case_records(native, count, empty)
    if records_error:
        return records_error
    if summary.get("n_cases_total") != count or summary.get("n_cases_valid") != len(valid):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary case counts are incomplete"
    return _validate_profiler_summary_rows(summary, cases, empty, count)


def _validate_native_fixture_digests(
    native: Mapping[str, Any], summary: Mapping[str, Any]
) -> Optional[str]:
    """Require the profiler summary to echo the frozen fixture digests."""
    manifest_sha = native.get("manifest_sha256")
    fixture_sha = native.get("fixture_sha256")
    if not _is_sha256(manifest_sha) or not _is_sha256(fixture_sha):
        return "NPUBENCH_EVIDENCE_INVALID: native fixture digest evidence is malformed"
    if summary.get("native_input_manifest_sha256") != manifest_sha:
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary manifest digest differs from native fixture"
    if summary.get("native_fixture_sha256") != fixture_sha:
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary common fixture digest differs from native fixture"
    return None


def _collect_native_case_records(
    native: Mapping[str, Any], count: int, empty: set[int]
) -> tuple[dict[int, Mapping[str, Any]], Optional[str]]:
    """Index the frozen per-case fixture records after checking each one."""
    records = native.get("case_fixtures")
    if not isinstance(records, list) or len(records) != count:
        return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture case records are incomplete"
    cases: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture case record is malformed"
        index = record.get("case")
        if not _is_case_index(index, count) or index in cases:
            return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture case record index is malformed"
        expected_path = f"native_perf_cases/case_{index:06d}.pt"
        if record.get("path") != expected_path or not _is_sha256(record.get("sha256")):
            return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture case record path/digest is malformed"
        if record.get("empty_tensor") is not (index in empty):
            return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture case empty marker is malformed"
        cases[index] = record
    if set(cases) != set(range(count)):
        return {}, "NPUBENCH_EVIDENCE_INVALID: native fixture omitted a case record"
    return cases, None


def _validate_profiler_summary_rows(
    summary: Mapping[str, Any],
    cases: Mapping[int, Mapping[str, Any]],
    empty: set[int],
    count: int,
) -> Optional[str]:
    """Require one profiler row per frozen case, bound to that case fixture."""
    rows = summary.get("per_case")
    if not isinstance(rows, list) or len(rows) != count:
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary per-case coverage is incomplete"
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return "NPUBENCH_EVIDENCE_INVALID: profiler summary case row is malformed"
        index = row.get("case")
        if not _is_case_index(index, count) or index not in cases or index in seen:
            return "NPUBENCH_EVIDENCE_INVALID: profiler summary case index is malformed"
        seen.add(index)
        row_error = _validate_profiler_case_row(row, cases[index], index in empty)
        if row_error:
            return row_error
    if seen != set(range(count)):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary omitted a frozen case"
    return None


def _validate_profiler_case_row(
    row: Mapping[str, Any], record: Mapping[str, Any], is_empty: bool
) -> Optional[str]:
    """Check one profiler row against its frozen per-case fixture record."""
    if row.get("native_case_fixture_sha256") != record.get("sha256"):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary case fixture digest differs"
    if row.get("native_case_path") != record.get("path"):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary case fixture path differs"
    if is_empty:
        return _validate_empty_profiler_row(row)
    if row.get("skipped") not in (None, ""):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary skipped a required case"
    for field in ("ref_us", "asc_us", "speedup"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return f"NPUBENCH_EVIDENCE_INVALID: profiler summary {field} is invalid"
    ref_us, asc_us, speedup = float(row["ref_us"]), float(row["asc_us"]), float(row["speedup"])
    if abs(speedup - ref_us / asc_us) > max(1.0e-12, abs(speedup) * 1.0e-9):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary speedup differs from paired timings"
    if not isinstance(row.get("ref_prof_dir"), str) or not isinstance(row.get("asc_prof_dir"), str):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary lacks paired retained profile paths"
    return None


def _validate_empty_profiler_row(row: Mapping[str, Any]) -> Optional[str]:
    """An empty case must be explicitly skipped and carry no timing evidence."""
    if row.get("skipped") != "empty_tensor":
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary did not explicitly skip an empty case"
    timing_fields = ("ref_us", "asc_us", "speedup", "ref_prof_dir", "asc_prof_dir")
    if any(row.get(field) is not None for field in timing_fields):
        return "NPUBENCH_EVIDENCE_INVALID: profiler summary timed an empty case"
    return None


def _is_case_index(value: object, count: int) -> bool:
    """True only for a real int index inside range(count); bool is rejected."""
    return isinstance(value, int) and not isinstance(value, bool) and value in range(count)


def _case_index_set(value: object, count: int) -> Optional[set[int]]:
    if not isinstance(value, list):
        return None
    result: set[int] = set()
    for item in value:
        if not _is_case_index(item, count) or item in result:
            return None
        result.add(item)
    return result


def _resolve_archive(workspace: Path, value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
        evidence_root = (workspace / EVIDENCE_DIR).resolve(strict=True)
        resolved.relative_to(evidence_root)
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate.is_symlink() or not resolved.is_dir():
        return None
    return resolved


def profile_tree_sha256(root: Path) -> str:
    """Digest regular profile files in stable relative-path/bytes order."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("profile root must be a real directory")
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"profile archive contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"profile archive contains non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((relative, digest))
    encoded = "".join(f"{relative}\0{digest}\n" for relative, digest in entries)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)

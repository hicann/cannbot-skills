# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Engine-owned O5 bridge for frozen NPUKernelBench evaluation.

The runner accepts only lanes assigned by the surrounding orchestrator.  It
does not scan for or select a device itself.  A small flock-protected lease
registry turns those assigned lanes into an auditable precision/performance
manifest, lets the evaluator run both lanes when two are safely available, and
releases every lease in ``finally``.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


LEASE_SCHEMA = "cannbot.npubench_lease/v1"
# Must exceed the largest isolated phase cap (CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC,
# up to 3h in batch campaigns) so a LIVE owner's lease cannot expire mid-phase.
# Dead owners are reclaimed via the owner_pid liveness check in
# _lease_is_reclaimable, so this TTL only bounds legacy/unknown owners and
# pid-reuse races.
LEASE_TTL_SECONDS = 4 * 60 * 60
_TARGET_RECEIPT_PATH = "npubench_evidence/target_receipt.json"


class NpubenchLeaseError(RuntimeError):
    """The engine cannot safely reserve an assigned evaluation lane."""


def _authenticated_build_failure_kind(
    workspace: Path,
    source_kind: str,
    build: Mapping[str, Any],
    *,
    lane: int,
    expected_attempt_id: str,
) -> str:
    """Read a current, authenticated routing category from the error receipt.

    The receipt is accepted only when its digest, attempt token, durable
    source binding, candidate/CMake digests, target identity, and build
    identity match both the current build return and the current workspace.
    A missing, malformed, stale, or tampered receipt therefore fails closed as
    ``target_build`` and can never send a worker back into the authoring loop.
    """
    # Deliberately late, function-local import: it keeps the module-level import
    # graph acyclic and lets a test monkeypatch on ``npubench_target`` be
    # honoured, because the binding is re-resolved on every call.
    npubench_target = importlib.import_module("npubench.npubench_target")

    if not isinstance(build, Mapping) or build.get("status") != "ERROR":
        return "target_build"
    try:
        # Resolved by name rather than re-imported: the module is already bound
        # above, and a test double may not carry this helper at all - that
        # AttributeError is a legitimate fail-closed path handled below.
        contract = getattr(npubench_target, "_build_contract")(source_kind)
    except Exception:
        return "target_build"
    try:
        receipt = _authenticated_receipt(
            workspace, source_kind, build, contract, expected_attempt_id
        )
        if receipt is None:
            return "target_build"
        kind = receipt.get("failure_kind")
        if kind not in {"candidate_contract", "target_build"}:
            return "target_build"
        if not _receipt_source_bindings_match(workspace, receipt, source_kind):
            return "target_build"
        target_state = _receipt_target_state(workspace, receipt, lane)
        if target_state is None:
            return "target_build"
        if not _receipt_build_identity_matches(
            receipt, kind, contract=contract, target_state=target_state
        ):
            return "target_build"
        return kind
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        npubench_target.TargetTransportError,
    ):
        # UnicodeError and json.JSONDecodeError are ValueError subclasses and
        # are already covered by the ValueError entry above.
        return "target_build"


def _authenticated_receipt(
    workspace: Path,
    source_kind: str,
    build: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_attempt_id: str,
) -> Optional[Mapping[str, Any]]:
    """Return the persisted receipt only when it authenticates against ``build``.

    ``None`` means "not authentic"; the caller must then fail closed.
    """
    from npubench.npubench_target import _receipt_payload_valid

    receipt_relative = contract.get("receipt_path")
    if not isinstance(receipt_relative, str):
        return None
    receipt_path = workspace / receipt_relative
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return None
    if _sha256_file(receipt_path) != build.get("receipt_sha256"):
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        return None
    if receipt.get("schema") != contract.get("schema"):
        return None
    if receipt.get("status") != "ERROR":
        return None
    if receipt.get("source_kind") != source_kind:
        return None
    if not _receipt_payload_valid(receipt, workspace):
        return None
    # The production writer returns the exact persisted payload plus its
    # path/digest.  This blocks an in-memory failure label from being
    # paired with a different signed receipt.
    if any(build.get(key) != value for key, value in receipt.items()):
        return None
    if build.get("receipt_path") != receipt_relative:
        return None
    if build.get("build_attempt_id") != expected_attempt_id:
        return None
    if receipt.get("build_attempt_id") != expected_attempt_id:
        return None
    return receipt


def _receipt_source_bindings_match(
    workspace: Path, receipt: Mapping[str, Any], source_kind: str
) -> bool:
    """Bind the receipt to the workspace's durable source and authored inputs."""
    npubench_target = importlib.import_module("npubench.npubench_target")
    from npubench.npubench_target import (
        _authored_cmake_sha256,
        _candidate_source_digest,
        _source_stage_digest,
    )
    from reference_source import load_durable_state

    state = load_durable_state(workspace)
    expected_stage_digest = _source_stage_digest(workspace, state, source_kind)
    if receipt.get("source_stage_digest") != expected_stage_digest:
        return False
    try:
        candidate_digest = _candidate_source_digest(workspace)
    except (OSError, npubench_target.TargetTransportError):
        candidate_digest = None
    if receipt.get("candidate_source_sha256") != candidate_digest:
        return False
    try:
        authored_cmake_digest = _authored_cmake_sha256(workspace)
    except (OSError, npubench_target.TargetTransportError):
        authored_cmake_digest = None
    return receipt.get("authored_cmake_sha256") == authored_cmake_digest


def _receipt_target_state(
    workspace: Path, receipt: Mapping[str, Any], lane: int
) -> Optional[tuple[Any, str, Optional[str]]]:
    """Re-resolve the live target and confirm the receipt still describes it.

    Returns ``(target, configured_soc, observed_soc)``, or ``None`` when the
    receipt is stale with respect to the current target configuration.
    """
    npubench_target = importlib.import_module("npubench.npubench_target")
    from npubench.npubench_target import (
        _probe_target_soc,
        _resolve_cann_set_env,
        _target,
        _target_identity,
    )

    current_target = _target(workspace, lane)
    # Re-resolve the configured SoC from the current target.  Never use
    # the receipt's own SoC to reconstruct the expected identity: a signed
    # receipt remains stale when the target configuration changes between
    # build and O5 classification (for example, Ascend950PR -> Ascend910).
    current_soc = npubench_target.a5_soc_version(current_target.env)
    if receipt.get("soc") != current_soc:
        return None
    current_observed_soc = None
    receipt_observed_soc = receipt.get("observed_soc")
    if receipt_observed_soc is not None:
        try:
            current_observed_soc = _probe_target_soc(
                current_target,
                _resolve_cann_set_env(Path(current_target.cann_path).expanduser()),
            )
        except (OSError, npubench_target.TargetTransportError):
            return None
        if receipt_observed_soc != current_observed_soc:
            return None
    if receipt.get("target") != _target_identity(current_target):
        return None
    return current_target, current_soc, current_observed_soc


def _receipt_build_identity_matches(
    receipt: Mapping[str, Any],
    kind: str,
    *,
    contract: Mapping[str, Any],
    target_state: tuple[Any, str, Optional[str]],
) -> bool:
    """Confirm a signed candidate identity still matches the live toolchain."""
    npubench_target = importlib.import_module("npubench.npubench_target")
    from npubench.npubench_target import (
        _candidate_build_identity,
        _resolve_cann_set_env,
        _target_python,
    )

    build_identity = receipt.get("build_identity")
    if kind == "candidate_contract" and not isinstance(build_identity, Mapping):
        return False
    if build_identity is None:
        return True
    if not isinstance(build_identity, Mapping):
        return False
    current_target, current_soc, current_observed_soc = target_state
    script = (
        Path(npubench_target.__file__).resolve().parents[2]
        / "patches"
        / "build_tilelang2ascendc.py"
    )
    expected_identity = _candidate_build_identity(
        current_target,
        current_soc,
        current_observed_soc,
        _target_python(current_target.env),
        script,
        _resolve_cann_set_env(Path(current_target.cann_path).expanduser()),
        identity_schema=contract.get("identity_schema", ""),
    )
    return dict(build_identity) == expected_identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


_EXEC_CANARY_ENV = "CANNBOT_NPUBENCH_EXEC_CANARY"
# 2026-08-22 A5: raised 20s→240s.  A degraded driver (dead cards 2/4 on the
# host) makes torch_npu init + first copy take 80-100s per fresh process —
# a 20s cap misjudges "slow" as "wedged" and wrongly fails the gate.
_EXEC_CANARY_TIMEOUT_SECONDS = 240


def _exec_canary_error(device: int) -> str | None:
    """Execution-plane sentinel: bare H2D copy in a fresh child process.

    The management-plane query can stay healthy while the device execution
    plane is wedged (2026-08-22 A5: health=OK yet every torch copy hung).
    The sentinel catches that state BEFORE precision/perf spend a full task
    timeout inside it.  It runs as a child (the controller must never import
    torch / touch the NPU itself — device init can hang in a bad window), with
    start_new_session + killpg so a wedged child cannot outlive the gate.

    Fail-open: a child that exits non-zero for any reason OTHER than 507035
    or a hang (no torch, no NPU in this environment, import errors) does not
    block the evaluation.

    Opt-in via CANNBOT_NPUBENCH_EXEC_CANARY=1: the campaign launchers export
    it, while plain harness use (and the unit-test suite) keeps the gate
    management-only by default — a real 20s device probe must never fire
    inside tests.
    """
    if os.environ.get(_EXEC_CANARY_ENV, "0") != "1":
        return None
    code = (
        "import torch\n"
        "t = torch.randn(1024 * 1024, dtype=torch.float16)\n"
        "d = t.to('npu:0')\n"
        "torch.npu.synchronize()\n"
        "print('CANARY_OK')\n"
    )
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    command = [sys.executable, "-c", code]
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=_EXEC_CANARY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # The wedged child may already be gone, in which case killpg has
            # nothing left to signal.  Draining the pipes afterwards reaps it;
            # a second timeout or an already-closed pipe is not actionable in
            # a fail-open gate, so both suppressions are deliberate and typed.
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
                proc.communicate(timeout=10)
            return (
                f"device {device} execution canary hung "
                f"({_EXEC_CANARY_TIMEOUT_SECONDS}s)"
            )
    except OSError:
        return None  # fail-open: cannot even spawn the sentinel
    if "CANARY_OK" in (out or ""):
        return None
    if "507035" in (out or "") or "507035" in (err or ""):
        return f"device {device} execution canary faulted (507035)"
    return None  # fail-open: any other child failure is an env issue, not the gate's call


def _lane_health_gate_error(devices: Iterable[int]) -> str | None:
    """Pre-flight health gate for the assigned evaluation lanes.

    Returns an infrastructure-failure message when an assigned device is
    reported Critical/Unknown or its per-device health query hangs.  A hang
    here is itself the box-bad-state signature (2026-08-22 A5: a dead lane
    wedges the npu-smi management path so badly the query survives even
    SIGKILL - the eval would hang for the full task timeout otherwise).

    Fail-open: no npu-smi binary, a missing device entry, or unparsable
    output does not block the evaluation - only explicit bad health or a
    stuck query does.  Per-device queries are used instead of the full
    listing because the full listing iterates every device and hangs as soon
    as one device is wedged.
    """
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return None
    problems: list[str] = []
    for device in sorted({int(d) for d in devices}):
        command = [
            "timeout",
            "-k",
            "2",
            "20",
            npu_smi,
            "info",
            "-t",
            "health",
            "-i",
            str(device),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=27,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            # A hung query is the box-bad-state signature; report it.
            problems.append(f"device {device} health query failed ({type(exc).__name__})")
            continue
        output = (completed.stdout or "") + (completed.stderr or "")
        match = re.search(r"Health\s*Status\s*:\s*(\S+)", output)
        health = match.group(1).strip().lower() if match is not None else ""
        if health in ("critical", "unknown", "fail", "alarm"):
            problems.append(f"device {device} reported health '{health}'")
            continue
        # Non-zero exit with no parseable bad-health line is fail-open: the
        # query tooling itself may be unavailable (e.g. exit 127 in a
        # CANN-less environment); only an explicit bad health or a hung
        # query blocks the evaluation.
        # Management plane responsive: also check the execution plane.  A
        # wedged execution plane hangs every precision/perf attempt for a
        # full task timeout; the 20s canary converts that into a fast infra
        # failure with the O5 backoff retry as recovery.
        canary_error = _exec_canary_error(device)
        if canary_error is not None:
            problems.append(canary_error)
    if problems:
        return "lane health gate failed: " + "; ".join(problems)
    return None


# Source kinds whose target-side build the engine itself drives, mapped to the
# ``npubench_target`` entry point and the human-facing build label.
_CONTROLLED_TARGET_BUILDS = {
    "port-aclnn-tilelang2ascendc": (
        "build_tilelang2ascendc_candidate_on_target",
        "TileLang2AscendC",
    ),
}


def _controlled_build_precheck(workspace: Path, lane: int):
    """Run the engine-controlled target build before any lane is leased.

    Returns a terminal ``MeasuredResult`` when the run must stop here (the
    target build did not PASS, or a limited SoC only permits the
    code-generation smoke check) and ``None`` when the evaluation may proceed
    to lease acquisition.
    """
    from a5_target_capability import (
        is_limited_a5_soc,
        limited_a5_validation_error,
        a5_soc_version,
    )
    from phase_o5 import MeasuredResult
    from phase_o5_runner import _read_ascendc_env
    from reference_source import load_durable_state

    durable_state = load_durable_state(workspace)
    port_source = durable_state.get("port_source")
    source_kind = port_source.get("kind") if isinstance(port_source, Mapping) else None
    controlled_build = _CONTROLLED_TARGET_BUILDS.get(source_kind)
    if controlled_build is None:
        return None
    limited_target_soc: str | None = None
    target_env = _read_ascendc_env(workspace)
    target_soc = a5_soc_version(target_env)
    if is_limited_a5_soc(target_soc):
        # Ascend910 may compile a candidate as a code-generation smoke
        # check.  Stop only after that target-side build and before
        # snapshot/precision/performance acceptance evaluation.
        limited_target_soc = target_soc
    # Compilation does not consume a precision/performance evaluator
    # lease.  Keep it before lease acquisition so a 910 codegen smoke
    # can terminate at the documented capability gate without
    # reserving an acceptance lane.
    npubench_target = importlib.import_module("npubench.npubench_target")

    build_attempt_id = secrets.token_hex(16)
    build = getattr(npubench_target, controlled_build[0])(
        workspace, lane, build_attempt_id=build_attempt_id
    )
    if build.get("status") != "PASS":
        reason = build.get("reason") or f"{controlled_build[1]} target build did not PASS"
        return MeasuredResult(
            runner_error=str(reason),
            rollback_kind="infra",
            failure_kind=_authenticated_build_failure_kind(
                workspace,
                source_kind,
                build,
                lane=lane,
                expected_attempt_id=build_attempt_id,
            ),
        )
    if limited_target_soc is None:
        return None
    return MeasuredResult(
        runner_error=limited_a5_validation_error(limited_target_soc),
        rollback_kind="target_capability",
    )


def _evaluate_leased_workspace(
    workspace: Path,
    *,
    leases: Mapping[str, Mapping[str, Any]],
    manifest_path: Path,
):
    """Gate the leased lanes, run the frozen evaluator, and adapt its report."""
    precision_device = int(leases["precision"]["device"])
    performance_device = int(leases["performance"]["device"])
    health_error = _lane_health_gate_error([precision_device, performance_device])
    if health_error is not None:
        raise NpubenchLeaseError(health_error)
    npubench_runner = importlib.import_module("npubench.npubench_runner")
    npubench_target = importlib.import_module("npubench.npubench_target")
    from reference_source import load_durable_state, resolve_reference_binding

    # Evaluation outputs live under ``workspace`` too.  Freeze the worker
    # candidate before either lane starts so adapter/evidence writes cannot
    # alter the provenance digest between precision and performance.
    candidate_snapshot = npubench_runner.materialize_candidate_snapshot(workspace)
    reference = resolve_reference_binding(load_durable_state(workspace))
    if reference.get("source") != "npubench":
        raise NpubenchLeaseError("durable reference is not npubench")
    # Do not run an old-format benchmark task in the controller process
    # merely because a remote A5 needs staging.  The transport has one
    # explicit local-target mode and otherwise creates a fresh tokenized
    # A5 workspace; it never relies on the shared legacy current_task
    # synchronization directory.
    evaluation = npubench_target.evaluate_npubench_on_target(
        workspace=workspace,
        reference=reference,
        candidate_snapshot=candidate_snapshot,
        precision_device=precision_device,
        performance_device=performance_device,
        lease_manifest=manifest_path,
    )
    if isinstance(evaluation, Mapping):
        # This is supplemental engine evidence.  The binding itself keeps
        # the content digest; the stable path makes the immutable snapshot
        # inspectable without trusting a worker-supplied candidate path.
        evaluation = dict(evaluation)
        evaluation["candidate_snapshot"] = str(candidate_snapshot)
    return _measured_result_from_evaluation(
        evaluation,
        manifest_path=manifest_path,
        leases=leases,
    )


def npubench_verify_runner(
    workspace: Path,
    op: str,
    lane: int = 0,
    *,
    extra_lanes: Iterable[int] = (),
):
    """Run an O5 evaluation and adapt its evidence to ``MeasuredResult``.

    The caller owns scheduling.  ``lane`` is always the precision lane.  The
    first distinct assigned extra lane is used for performance when it can be
    leased; otherwise both stages use ``lane`` and evidence records
    ``degraded_single_lane``.  A failure to reserve the mandatory primary lane
    is an infrastructure failure, never a silent device fallback.
    """
    from phase_o5 import MeasuredResult

    workspace = Path(workspace)
    try:
        early_result = _controlled_build_precheck(workspace, lane)
        if early_result is not None:
            return early_result
        leases, manifest_path = _acquire_leases(
            workspace, op, primary_device=lane, extra_lanes=extra_lanes
        )
    except Exception as exc:
        return MeasuredResult(
            runner_error=f"npubench lane lease failed: {type(exc).__name__}: {exc}",
            rollback_kind="infra",
        )

    result = None
    try:
        result = _evaluate_leased_workspace(
            workspace, leases=leases, manifest_path=manifest_path
        )
    except Exception as exc:
        result = MeasuredResult(
            runner_error=f"npubench evaluation raised {type(exc).__name__}: {exc}",
            rollback_kind="infra",
        )
    finally:
        try:
            _release_leases(workspace, leases, manifest_path)
        except Exception as exc:
            # A successful evaluation is not safe to publish while its lease
            # cleanup is unknown: the next run could observe a stale live lease.
            # Mutate the already-built result so the historical MeasuredResult
            # contract and any evaluation error/evidence remain intact.
            result = _mark_cleanup_failure(
                result,
                cleanup_error=exc,
                manifest_path=manifest_path,
                leases=leases,
                measured_result_type=MeasuredResult,
            )
    return result


def _mark_cleanup_failure(
    result: object,
    *,
    cleanup_error: BaseException,
    manifest_path: Path,
    leases: Mapping[str, Mapping[str, Any]],
    measured_result_type: type,
):
    """Turn uncertain lease cleanup into an auditable infrastructure failure.

    Cleanup runs in ``finally`` and must not replace the evaluator's result
    object: callers rely on its pass/performance/provider fields even when the
    runner failed.  The result is nevertheless never publishable as PASS once
    cleanup has failed, so append the cleanup error and force ``infra``.
    """
    cleanup_message = (
        "npubench lease cleanup failed: "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )
    if result is None:
        # Defensive only: every ordinary Exception path above creates a
        # MeasuredResult before finally runs.
        result = measured_result_type(
            runner_error="npubench evaluation produced no result",
            rollback_kind="infra",
        )

    existing_error = getattr(result, "runner_error", None)
    result.runner_error = (
        f"{existing_error}; {cleanup_message}"
        if existing_error
        else cleanup_message
    )
    result.rollback_kind = "infra"

    cleanup_devices: set[int] = set()
    for label in ("precision", "performance"):
        lease = leases.get(label)
        device = lease.get("device") if isinstance(lease, Mapping) else None
        if isinstance(device, int):
            cleanup_devices.add(int(device))
    audit_record: dict[str, Any] = {
        "status": "ERROR",
        "error": cleanup_message,
        "exception_type": type(cleanup_error).__name__,
        "recorded_at": _iso(_utcnow()),
        "manifest_path": str(manifest_path),
        "devices": sorted(cleanup_devices),
    }
    evidence = getattr(result, "provider_evidence", None)
    if not isinstance(evidence, dict):
        evidence = {}
        result.provider_evidence = evidence
    evidence["lease_cleanup"] = dict(audit_record)

    # Best effort: the MeasuredResult is the authoritative fail-closed signal;
    # if the manifest itself is still writable, leave a durable cleanup marker
    # for operators diagnosing the stale-lease condition.
    try:
        manifest = _read_json_object(manifest_path) or {}
        manifest["state"] = "cleanup_failed"
        manifest["cleanup_failure"] = dict(audit_record)
        _write_json_atomic(manifest_path, manifest)
    except Exception as audit_error:
        audit_record["audit_write_error"] = (
            f"{type(audit_error).__name__}: {audit_error}"
        )
        evidence["lease_cleanup"] = dict(audit_record)

    return result


# Host H2D copy-path faults reproduce without the candidate kernel ever
# running, so they are infrastructure faults rather than authoring faults.
_HOST_COPY_FAULT_MARKERS = (
    "copy_between_host_and_device",
    "CopyKernelOpApi",
    "copy_with_slice",
    "timed out",
    "TimeoutExpired",
)


def _host_copy_fault(reason: str, precision_reason: str) -> bool:
    """Detect the host-side copy fault signature in either failure reason."""
    for marker in _HOST_COPY_FAULT_MARKERS:
        if marker in reason or marker in precision_reason:
            return True
    return False


def _candidate_execution_fault(reason: str, precision_reason: str) -> bool:
    """Detect a candidate-kernel execution fault (a worker-fixable failure)."""
    if "RuntimeError" in reason or "RuntimeError" in precision_reason:
        return True
    return "device error" in reason


def _error_result_from_evaluation(evaluation: Mapping[str, Any]):
    """Route an evaluator ERROR back to the worker or to an infra retry.

    Candidate-execution faults (RuntimeError / NPU device error from the
    candidate's own kernels) are worker-fixable authoring failures: route them
    back to await_worker like any other candidate failure instead of the
    infra-terminal stop.

    BUT the host H2D copy path faults (torch_npu CopyKernelOpApi /
    copy_between_host_and_device, e.g. a TBE Slice kernel raising 507035
    "vector core execution is abnormal") are NOT candidate faults: they
    reproduce without the candidate kernel ever running, on multiple cards,
    during host-level bad-state windows that also hang npu-smi (2026-08-22 A5
    campaign: CoT/SDPA/BAM precision+perf, cards 3/5).  Routing those to
    await_worker burns worker iterations on an innocent kernel; they belong to
    infra so O5 retries in place instead.
    """
    from phase_o5 import MeasuredResult

    reason = str(evaluation.get("reason") or "npubench evaluator reported ERROR")
    try:
        precision_report = _report_from_evaluation(evaluation, "precision")
        prec_reason = (
            str(precision_report.get("reason") or "")
            if isinstance(precision_report, Mapping)
            else ""
        )
    except Exception:
        # Best effort only: an unreadable precision sidecar must never change
        # the aggregate ERROR routing, so fall back to the empty reason.
        prec_reason = ""
    rollback_kind = "infra"
    host_copy_fault = _host_copy_fault(reason, prec_reason)
    candidate_fault = _candidate_execution_fault(reason, prec_reason)
    if not host_copy_fault and candidate_fault:
        rollback_kind = "candidate"
    return MeasuredResult(
        runner_error=reason,
        rollback_kind=rollback_kind,
    )


def _measured_result_from_evaluation(
    evaluation: object,
    *,
    manifest_path: Path,
    leases: Mapping[str, Mapping[str, Any]],
):
    """Translate structured runner evidence without inventing a PASS state."""
    from phase_o5 import MeasuredResult

    if not isinstance(evaluation, Mapping):
        return MeasuredResult(
            runner_error="npubench evaluator returned a non-object report",
            rollback_kind="infra",
        )
    status = evaluation.get("status")
    if status == "ERROR":
        return _error_result_from_evaluation(evaluation)
    if evaluation.get("target_receipt_path") != _TARGET_RECEIPT_PATH:
        return MeasuredResult(
            runner_error="npubench target evaluation omitted the canonical target receipt path",
            rollback_kind="infra",
        )
    if not _is_sha256(evaluation.get("target_receipt_sha256")):
        return MeasuredResult(
            runner_error="npubench target evaluation omitted a valid target receipt digest",
            rollback_kind="infra",
        )
    precision = _report_from_evaluation(evaluation, "precision")
    performance = _report_from_evaluation(evaluation, "performance")
    if precision is None or performance is None:
        return MeasuredResult(
            runner_error="npubench evaluator omitted precision or performance evidence",
            rollback_kind="infra",
        )
    pass_a = precision.get("pass_a")
    if not isinstance(pass_a, Mapping):
        pass_a = {}
    provider_evidence = {
        "precision": dict(precision),
        "performance": dict(performance),
        "evaluate": dict(evaluation),
        "lease_manifest": str(manifest_path),
        "leases": {name: dict(value) for name, value in leases.items()},
    }
    for field in ("target_receipt_path", "target_receipt_sha256", "transport"):
        value = evaluation.get(field)
        if value is not None:
            provider_evidence[field] = value
    return MeasuredResult(
        pass_a=dict(pass_a),
        perf=dict(performance),
        provider_evidence=provider_evidence,
    )


def _report_from_evaluation(evaluation: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    """Read a report from either the aggregate result or its fixed sidecar."""
    direct = evaluation.get(name)
    if isinstance(direct, Mapping):
        return direct
    direct = evaluation.get(f"{name}_report")
    if isinstance(direct, Mapping):
        return direct
    candidate = Path(str(evaluation.get(f"{name}_report_path") or ""))
    if candidate.is_file():
        try:
            value = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None
    return None


def _lease_root(workspace: Path) -> Path:
    """Keep a shared local lease registry beside all operation workspaces."""
    return workspace.resolve().parent / ".npubench_leases"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish lease state atomically and durably enough for a local scheduler."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _owner_start_time() -> int:
    """Current process start time (jiffies, /proc/self/stat field 22).  Sandbox
    pid namespaces make a bare pid unreliable (the recorded pid may be reused
    by an unrelated container process), so the start time disambiguates the
    true owner.
    """
    try:
        stat_line = Path("/proc/self/stat").read_text()
        after_paren = stat_line.rsplit(")", 1)[-1].split()
        return int(after_paren[19])
    except (OSError, ValueError, IndexError):
        return 0


def _owner_pid_is_dead(record: Mapping[str, Any]) -> bool:
    """Detect a lease whose owning process is gone (e.g. the orchestrator was
    killed without running its release path).  When the record carries an
    owner_starttime, a pid that is missing OR now belongs to a different
    process (start time mismatch) is dead.  Missing/legacy records without an
    owner_pid stay protected — only expiry can reclaim them.
    """
    pid = record.get("owner_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    start_time = record.get("owner_starttime")
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True  # owner process no longer exists
    if isinstance(start_time, int) and start_time > 0:
        try:
            after_paren = stat_line.rsplit(")", 1)[-1].split()
            current_start = int(after_paren[19])
        except (ValueError, IndexError):
            current_start = -1
        if current_start != start_time:
            return True  # pid was reused by a different process
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _lease_is_reclaimable(record: Mapping[str, Any], now: datetime) -> bool:
    """Reclaim a lease when its owner process is dead OR it has expired.
    A live/unknown owner remains protected.
    """
    if _owner_pid_is_dead(record):
        return True
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed <= now


def _requested_devices(primary_device: int, extra_lanes: Iterable[int]) -> list[int]:
    if not isinstance(primary_device, int) or primary_device < 0:
        raise NpubenchLeaseError(f"invalid primary NPU device {primary_device!r}")
    devices = [primary_device]
    for device in extra_lanes:
        if not isinstance(device, int) or device < 0:
            raise NpubenchLeaseError(f"invalid extra NPU device {device!r}")
        if device != primary_device:
            devices.append(device)
    return devices


def _reserve_device_locked(
    root: Path,
    device: int,
    now: datetime,
    template: Mapping[str, Any],
    primary_device: int,
) -> list[dict[str, Any]]:
    """Reserve one device while the caller already holds the registry lock.

    Returns ``[record]`` when the lease was written and ``[]`` when an optional
    extra lane is already held by a live lease.  A busy primary lane raises.
    """
    state_path = root / f"device_{device}.json"
    existing = _read_json_object(state_path)
    if existing is not None and not _lease_is_reclaimable(existing, now):
        if device == primary_device:
            raise NpubenchLeaseError(
                f"assigned precision device {device} is leased by another evaluation"
            )
        return []
    record = dict(template)
    record["device"] = device
    record["token"] = secrets.token_hex(16)
    _write_json_atomic(state_path, record)
    return [record]


def _rollback_reservations(root: Path, acquired: Iterable[Mapping[str, Any]]) -> None:
    """Drop only the lease records this attempt wrote; never a foreign record."""
    for record in acquired:
        state_path = root / f"device_{record['device']}.json"
        existing = _read_json_object(state_path)
        if existing is not None and existing.get("token") == record["token"]:
            state_path.unlink(missing_ok=True)


def _publish_lease_manifest(
    workspace: Path,
    acquired: list[dict[str, Any]],
    *,
    primary_device: int,
    run_id: str,
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Shape the acquired records into lanes and publish the durable manifest."""
    primary = next((item for item in acquired if item["device"] == primary_device), None)
    if primary is None:  # Defensive: primary either acquired or raised above.
        raise NpubenchLeaseError("mandatory precision lease was not acquired")
    performance = next((item for item in acquired if item["device"] != primary_device), primary)
    parallel = performance["device"] != primary["device"]
    leases = {
        "precision": dict(primary),
        "performance": dict(performance),
        "parallelism": {
            "mode": "parallel" if parallel else "degraded_single_lane",
            "reason": None if parallel else "no second assigned safe lease",
        },
    }
    evidence_dir = workspace / "npubench_evidence"
    manifest_path = evidence_dir / "leases" / f"{run_id}.json"
    _write_json_atomic(
        manifest_path,
        {
            "schema": LEASE_SCHEMA,
            "run_id": run_id,
            "state": "active",
            "precision": leases["precision"],
            "performance": leases["performance"],
            "parallelism": leases["parallelism"],
        },
    )
    return leases, manifest_path


def _acquire_leases(
    workspace: Path,
    op: str,
    *,
    primary_device: int,
    extra_lanes: Iterable[int],
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Acquire the mandatory lane and at most one optional perf lane atomically."""
    root = _lease_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".lock"
    requested = _requested_devices(primary_device, extra_lanes)
    now = _utcnow()
    run_id = f"{os.getpid()}-{secrets.token_hex(8)}"
    template = {
        "schema": LEASE_SCHEMA,
        "op": op,
        "workspace": str(workspace.resolve()),
        "owner_pid": os.getpid(),
        "owner_starttime": _owner_start_time(),
        "run_id": run_id,
        "issued_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=LEASE_TTL_SECONDS)),
    }
    acquired: list[dict[str, Any]] = []
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            for device in requested:
                acquired.extend(
                    _reserve_device_locked(root, device, now, template, primary_device)
                )
        except Exception:
            # Transactional acquisition: if a later requested lane is busy or
            # a write fails, release every lease written by this attempt before
            # exposing the failure.  A leaked primary lease would otherwise
            # make a retry look like a competing evaluator.
            _rollback_reservations(root, acquired)
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return _publish_lease_manifest(
        workspace, acquired, primary_device=primary_device, run_id=run_id
    )


def _release_owned_leases(
    root: Path, leases: Mapping[str, Mapping[str, Any]]
) -> list[int]:
    """Delete the registry records this run owns; returns the released devices."""
    released: list[int] = []
    seen: set[int] = set()
    for label in ("precision", "performance"):
        lease = leases.get(label)
        if not isinstance(lease, Mapping):
            continue
        device = lease.get("device")
        token = lease.get("token")
        if not isinstance(device, int) or device in seen or not isinstance(token, str):
            continue
        seen.add(device)
        state_path = root / f"device_{device}.json"
        existing = _read_json_object(state_path)
        if existing is not None and existing.get("token") == token:
            state_path.unlink(missing_ok=True)
            released.append(device)
    return released


def _release_leases(
    workspace: Path,
    leases: Mapping[str, Mapping[str, Any]],
    manifest_path: Path,
) -> None:
    """Release only records carrying this exact token, then stamp evidence."""
    root = _lease_root(workspace)
    lock_path = root / ".lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            released = _release_owned_leases(root, leases)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    manifest = _read_json_object(manifest_path) or {}
    manifest["state"] = "released"
    manifest["released_at"] = _iso(_utcnow())
    manifest["released_devices"] = released
    _write_json_atomic(manifest_path, manifest)

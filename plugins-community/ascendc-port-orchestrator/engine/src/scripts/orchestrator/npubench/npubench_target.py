# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Small A5 transport for frozen, old-format NPUKernelBench evidence.

It stages the original task/sidecar closure, the read-only candidate snapshot,
and the repository quick profiler into a tokenized target workspace.  The
controller imports only fixed evidence paths and writes the final receipt.

This module is the public face of the transport and keeps the controlled
candidate-build protocol plus the staging/execution/import pipeline.  The
supporting layers live in sibling modules and are re-exported here verbatim so
every historical ``npubench_target.<name>`` import keeps working:

* ``npubench_target_base``      -- exceptions and filesystem/JSON/cleanup primitives
* ``npubench_candidate_contract`` -- authored-candidate delivery contract and digests
* ``npubench_build_receipt``    -- build contract, identity binding, receipt HMAC
* ``npubench_target_toolchain`` -- target Python/CANN/npu-smi resolution and the build process
"""
from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

from npubench import npubench_runner
from a5_target_capability import (
    a5_soc_version,
    is_known_a5_soc,
    limited_a5_validation_error,
)
from a5_target_transport import (
    _Target,
    _maybe_sudo,
    _resolve_target,
    _scp_command,
    _ssh_command,
)
from a5_target_transport import _run as _a5_run
from npubench.npubench_inputs import NPUBENCH_SOURCE, verify_npubench_stage

# Re-exported so `npubench_target.<name>` keeps resolving for every importer.
from npubench.npubench_build_receipt import (  # noqa: F401
    TILELANG2ASCENDC_BUILD_IDENTITY_SCHEMA,
    TILELANG2ASCENDC_BUILD_RECEIPT_FILENAME,
    TILELANG2ASCENDC_BUILD_RECEIPT_PATH,
    TILELANG2ASCENDC_BUILD_RECEIPT_SCHEMA,
    _BUILD_IDENTITY_CONFIG_KEYS,
    _BUILD_IDENTITY_PROCESS_KEYS,
    _RECEIPT_KEY_DIR,
    _build_contract,
    _build_receipt_reusable,
    _candidate_build_error_payload,
    _candidate_build_identity,
    _classify_controlled_compile_failure,
    _local_runtime_container,
    _local_runtime_observation,
    _path_identity,
    _read_json_if_present,
    _read_runtime_marker,
    _receipt_auth_bytes,
    _receipt_auth_hmac,
    _receipt_auth_key,
    _receipt_auth_key_path,
    _receipt_payload_sha256,
    _receipt_payload_valid,
    _target_identity,
    _write_candidate_build_receipt,
)
from npubench.npubench_candidate_contract import (  # noqa: F401
    TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA,
    TILELANG2ASCENDC_CANDIDATE_KIND,
    TILELANG2ASCENDC_SOURCE_KIND,
    TILELANG2ASCENDC_STABLE_AUTHORED_FILES,
    _HOST_IS_AARCH64,
    _TILELANG2ASCENDC_MODEL_FRAMEWORK_PATTERNS,
    _ast_assignment_names,
    _ast_attribute_chain,
    _ast_contains_name,
    _authored_cmake_sha256,
    _candidate_delivery_files,
    _candidate_source_digest,
    _cxx_code_without_comments,
    _cxx_code_without_comments_or_literals,
    _normalised_authored_text,
    _python_code_without_comments,
    _python_code_without_comments_or_literals,
    _read_candidate_text,
    _validate_candidate_for_controlled_build,
    _validate_tilelang2ascendc_candidate_for_build,
    _validate_tilelang2ascendc_kernel_boundary,
)
from npubench.npubench_target_base import (  # noqa: F401
    CandidateContractError,
    CleanupFailure,
    TargetTransportError,
    _DirectBuildTimeout,
    _atomic_json,
    _audit_cleanup_errors,
    _collect_cleanup_error,
    _contract_value,
    _device,
    _file,
    _json_tail,
    _raise_cleanup_failures,
    _raise_preserving,
    _read_json,
    _real_directory,
    _remove_tree,
    _sha,
    _sha_ok,
    _unlink,
)
from npubench.npubench_target_toolchain import (  # noqa: F401
    _direct_build_preflight,
    _npu_smi_command,
    _positive_timeout_from_env,
    _probe_target_soc,
    _process_output_text,
    _resolve_cann_set_env,
    _run_direct_build_process,
    _target_python,
    _terminate_direct_build_process_group,
)


TARGET_RECEIPT_SCHEMA = "cannbot.npubench_target_evaluation/v1"
TARGET_RECEIPT_FILENAME = "target_receipt.json"
PREFLIGHT_RECEIPT_FILENAME = "preflight_target_receipt.json"
TARGET_RECEIPT_PATH = f"{npubench_runner.EVIDENCE_DIRNAME}/{TARGET_RECEIPT_FILENAME}"
PREFLIGHT_RECEIPT_PATH = f"{npubench_runner.EVIDENCE_DIRNAME}/{PREFLIGHT_RECEIPT_FILENAME}"
_REPORT = {
    "preflight": npubench_runner.PREFLIGHT_REPORT_FILENAME,
    "precision": npubench_runner.PRECISION_REPORT_FILENAME,
    "performance": npubench_runner.PERFORMANCE_REPORT_FILENAME,
    "evaluate": npubench_runner.EVALUATE_REPORT_FILENAME,
}
_EVAL = ("precision", "performance", "evaluate")
_REFERENCE_INPUT_KEYS = (
    "bundle_sha256",
    "task_sha256",
    "sidecar_sha256",
    "bundle_manifest_sha256",
)


class _ControlledBuildAbort(Exception):
    """Internal signal: stop the controlled build and persist an ERROR receipt.

    Every stage of the controlled build fails by raising this with exactly the
    receipt fields it has observed so far, so the one writer at the top of
    ``_build_controlled_candidate_on_target`` stays the only place that
    persists a failure receipt.
    """

    def __init__(self, reason: str, **fields: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fields = fields


@dataclass
class _ControlledBuild:
    """Evidence accumulated while one controlled candidate build progresses."""

    workspace: Path
    contract: dict[str, str]
    source_kind: str | None
    attempt: str
    state: Mapping[str, Any]
    target: _Target
    target_identity: dict[str, Any] = field(default_factory=dict)
    source_stage_digest: str | None = None
    soc: str | None = None
    candidate_digest: str | None = None
    authored_cmake_sha256: str | None = None
    observed_soc: str | None = None
    build_identity: dict[str, Any] | None = None
    build_mode: str = "controlled_authored_cmake"


def _abort_controlled_build(reason: str, **fields: Any) -> NoReturn:
    raise _ControlledBuildAbort(reason, **fields)


def _digest_fields(build: _ControlledBuild) -> dict[str, Any]:
    return {"source_stage_digest": build.source_stage_digest}


def _soc_fields(build: _ControlledBuild) -> dict[str, Any]:
    return {"source_stage_digest": build.source_stage_digest, "soc": build.soc}


def _candidate_fields(build: _ControlledBuild) -> dict[str, Any]:
    return {
        "source_stage_digest": build.source_stage_digest,
        "candidate_digest": build.candidate_digest,
        "authored_cmake_sha256": build.authored_cmake_sha256,
        "soc": build.soc,
    }


def _identity_fields(build: _ControlledBuild) -> dict[str, Any]:
    return {
        **_candidate_fields(build),
        "observed_soc": build.observed_soc,
        "build_identity": build.build_identity,
    }


def _controlled_build_error(build: _ControlledBuild, reason: str, **fields: Any) -> dict[str, Any]:
    """Persist the single authenticated ERROR receipt for this build attempt."""
    return _write_candidate_build_receipt(
        build.workspace,
        build.contract,
        _candidate_build_error_payload(
            build.contract,
            build.source_kind,
            reason,
            target=build.target_identity,
            **fields,
        ),
    )


def _controlled_build_attempt_id(build_attempt_id: str | None) -> str:
    """Return the caller's attempt token, or mint a fresh one.

    The attempt token is not a worker input.  It binds the in-process build
    return to the receipt written by this exact O5 invocation, preventing a
    previously signed ERROR receipt from being replayed as the current
    candidate result.
    """
    if isinstance(build_attempt_id, str) and re.fullmatch(r"[0-9a-f]{32}", build_attempt_id):
        return build_attempt_id
    return uuid.uuid4().hex


def build_tilelang2ascendc_candidate_on_target(
    workspace: Path, lane: int, *, build_attempt_id: str | None = None
) -> dict[str, Any]:
    """Build the authored TileLang2AscendC project with its own contract."""
    return _build_controlled_candidate_on_target(
        workspace, lane, source_kind=TILELANG2ASCENDC_SOURCE_KIND, build_attempt_id=build_attempt_id
    )


def build_generic_kernel_project_on_target(
    workspace: Path, lane: int, *, build_attempt_id: str | None = None
) -> dict[str, Any]:
    """Build a generic authored kernel project (``model_new_ascendc.py`` + ``kernel/``).

    Used for routes whose durable state carries no ``port_source.kind``
    binding (e.g. opgen_mode=port_a3_to_a5).  The build protocol, receipt
    authentication, and build script are shared with the source-bound routes;
    the payload's ``source_kind`` stays ``None`` to record the route taken.
    """
    return _build_controlled_candidate_on_target(
        workspace, lane, source_kind=None, build_attempt_id=build_attempt_id
    )


def _build_controlled_candidate_on_target(
    workspace: Path,
    lane: int,
    *,
    source_kind: str | None,
    build_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Run the one controlled CANN build protocol shared by supported sources."""
    workspace = Path(workspace)
    attempt = _controlled_build_attempt_id(build_attempt_id)
    contract = {**_build_contract(source_kind), "build_attempt_id": attempt}
    state = _read_json(workspace / ".opgen_state.json", "durable state")
    source = state.get("port_source")
    if source_kind is not None and (
        not isinstance(source, Mapping) or source.get("kind") != source_kind
    ):
        return {
            "schema": _contract_value(contract, "schema"),
            "status": "SKIPPED",
            "reason": f"durable state does not select {source_kind}",
        }
    target = _target(workspace, lane)
    build = _ControlledBuild(
        workspace=workspace,
        contract=contract,
        source_kind=source_kind,
        attempt=attempt,
        state=state,
        target=target,
        target_identity=_target_identity(target),
    )
    try:
        return _run_controlled_candidate_build(build)
    except _ControlledBuildAbort as abort:
        return _controlled_build_error(build, abort.reason, **abort.fields)


def _run_controlled_candidate_build(build: _ControlledBuild) -> dict[str, Any]:
    """Drive the controlled build stages in their fixed order."""
    _resolve_controlled_source_digest(build)
    _require_local_controlled_target(build)
    _require_known_controlled_soc(build)
    script, missing_script_name = _controlled_build_script(build)
    prebuild_identity = _controlled_prebuild_identity(build, script)
    source_manifest = _controlled_source_manifest(build)
    _record_best_effort_candidate_digests(build)
    proof = _validate_controlled_candidate(build, source_manifest, prebuild_identity)
    _require_controlled_build_script(build, script, missing_script_name)
    set_env = _resolve_cann_set_env(Path(build.target.cann_path).expanduser())
    _probe_controlled_build_soc(build, set_env)
    python = _target_python(build.target.env)
    reusable = _reusable_controlled_receipt(build, python, script, set_env)
    if reusable is not None:
        return reusable
    _run_controlled_build_preflight(build, python)
    shell_command, env = _controlled_build_command(build, python, script, set_env)
    completed = _run_controlled_build_command(build, shell_command, env, script)
    _require_stable_candidate_after_build(build, int(completed.returncode))
    payload = _controlled_build_success_payload(build, completed, script, proof)
    return _write_candidate_build_receipt(build.workspace, build.contract, payload)


def _resolve_controlled_source_digest(build: _ControlledBuild) -> None:
    try:
        build.source_stage_digest = _source_stage_digest(
            build.workspace, build.state, build.source_kind
        )
    except TargetTransportError as exc:
        _abort_controlled_build(str(exc))


def _require_local_controlled_target(build: _ControlledBuild) -> None:
    if build.target.container.lower() == "local":
        return
    _abort_controlled_build(
        "target build requires a local target workspace; remote target build receipt protocol is not enabled",
        **_digest_fields(build),
    )


def _require_known_controlled_soc(build: _ControlledBuild) -> None:
    build.soc = a5_soc_version(build.target.env)
    if not is_known_a5_soc(build.soc):
        _abort_controlled_build(limited_a5_validation_error(build.soc), **_soc_fields(build))


def _controlled_build_script(build: _ControlledBuild) -> tuple[Path, str]:
    """Return the controlled build script plus its human-facing filename."""
    if build.source_kind in (TILELANG2ASCENDC_SOURCE_KIND, None):
        script = Path(__file__).resolve().parents[2] / "patches" / "build_tilelang2ascendc.py"
        return script, "build_tilelang2ascendc.py"
    raise TargetTransportError(
        f"unsupported controlled candidate source kind: {build.source_kind!r}"
    )


def _controlled_prebuild_identity(build: _ControlledBuild, script: Path) -> dict[str, Any]:
    # Candidate validation happens before CANN is invoked, but its repair
    # receipt still needs a current toolchain identity.  Bind the configured
    # target/runtime inputs now; ``observed_soc`` remains None because no
    # target probe is needed to classify an authored candidate defect.
    return _candidate_build_identity(
        build.target,
        build.soc,
        None,
        _target_python(build.target.env),
        script,
        _resolve_cann_set_env(Path(build.target.cann_path).expanduser()),
        identity_schema=_contract_value(build.contract, "identity_schema"),
    )


def _controlled_source_manifest(build: _ControlledBuild) -> Mapping[str, Any] | None:
    try:
        if build.source_kind != TILELANG2ASCENDC_SOURCE_KIND:
            return None
        return _verified_tilelang2ascendc_source_manifest(build.workspace, build.state)
    except (TargetTransportError, OSError) as exc:
        _abort_controlled_build(f"candidate rejected before build: {exc}", **_soc_fields(build))


def _record_best_effort_candidate_digests(build: _ControlledBuild) -> None:
    """Capture whatever candidate identity exists before the validator runs.

    In particular, a missing CMake file must still carry the model/kernel
    digest so O5 can distinguish this current repairable defect from a
    replayed receipt.
    """
    try:
        build.candidate_digest = _candidate_source_digest(build.workspace)
    except (TargetTransportError, OSError):
        # Best effort only: an unreadable candidate is reported by the
        # contract validator below with a precise diagnostic.
        pass
    try:
        build.authored_cmake_sha256 = _authored_cmake_sha256(build.workspace)
    except (TargetTransportError, OSError):
        # Same: the validator owns the actionable message for a missing CMake.
        pass


def _validate_controlled_candidate(
    build: _ControlledBuild,
    source_manifest: Mapping[str, Any] | None,
    prebuild_identity: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    try:
        proof = _validate_candidate_for_controlled_build(
            build.workspace, build.source_kind, source_manifest
        )
        # A successful validation must have both identity fields.  Recompute
        # strictly here so a future validator cannot accidentally authorize a
        # build without the bound candidate/CMake digests.
        build.authored_cmake_sha256 = _authored_cmake_sha256(build.workspace)
        build.candidate_digest = _candidate_source_digest(build.workspace)
        return proof
    except CandidateContractError as exc:
        reason = str(exc)
        if not reason.startswith("candidate rejected before build:"):
            reason = f"candidate rejected before build: {reason}"
        _abort_controlled_build(
            reason,
            failure_kind="candidate_contract",
            build_identity=prebuild_identity,
            **_candidate_fields(build),
        )
    except (TargetTransportError, OSError) as exc:
        _abort_controlled_build(str(exc), **_soc_fields(build))


def _require_controlled_build_script(
    build: _ControlledBuild, script: Path, missing_script_name: str
) -> None:
    if script.is_symlink() or not script.is_file():
        _abort_controlled_build(
            f"{missing_script_name} is unavailable: {script}", **_candidate_fields(build)
        )


def _probe_controlled_build_soc(build: _ControlledBuild, set_env: Path | None) -> None:
    try:
        build.observed_soc = _probe_target_soc(build.target, set_env)
    except TargetTransportError as exc:
        _abort_controlled_build(str(exc), **_candidate_fields(build))


def _reusable_controlled_receipt(
    build: _ControlledBuild, python: Path, script: Path, set_env: Path | None
) -> dict[str, Any] | None:
    """Return a still-valid PASS receipt for identical inputs, else None."""
    build.build_identity = _candidate_build_identity(
        build.target,
        build.soc,
        build.observed_soc,
        python,
        script,
        set_env,
        identity_schema=_contract_value(build.contract, "identity_schema"),
    )
    receipt_path = build.workspace / _contract_value(build.contract, "receipt_path")
    existing = _read_json_if_present(receipt_path)
    if _build_receipt_reusable(
        existing,
        contract=build.contract,
        source_kind=build.source_kind,
        source_stage_digest=build.source_stage_digest,
        candidate_digest=build.candidate_digest,
        authored_cmake_sha256=build.authored_cmake_sha256,
        target=build.target_identity,
        build_identity=build.build_identity,
        workspace=build.workspace,
        build_mode=build.build_mode,
        build_attempt_id=build.attempt,
    ):
        return dict(existing)
    return None


def _run_controlled_build_preflight(build: _ControlledBuild, python: Path) -> None:
    preflight_error = _direct_build_preflight(python, build.target)
    if preflight_error is not None:
        _abort_controlled_build(
            preflight_error, **_identity_fields(build), python=str(python)
        )


def _controlled_build_command(
    build: _ControlledBuild, python: Path, script: Path, set_env: Path | None
) -> tuple[str, dict[str, str]]:
    """Compose the controlled build shell command and its process environment."""
    command = [str(python), str(script), str(build.workspace), "-v", str(build.soc), "--clean"]
    env = os.environ.copy()
    cann_path = Path(build.target.cann_path).expanduser()
    if cann_path.is_dir():
        env["ASCEND_INSTALL_PATH"] = str(cann_path.resolve())
        env["ASCEND_HOME_PATH"] = str(cann_path.resolve())
    shell_command = " ".join(shlex.quote(item) for item in command)
    if set_env is not None and set_env.is_file():
        shell_command = f"source {shlex.quote(str(set_env))} >/dev/null 2>&1 && {shell_command}"
    return shell_command, env


def _run_controlled_build_command(
    build: _ControlledBuild, shell_command: str, env: Mapping[str, str], script: Path
) -> subprocess.CompletedProcess[str]:
    timeout_sec = _positive_timeout_from_env("CANNBOT_DIRECT_BUILD_TIMEOUT_SEC", 1200)
    try:
        return _run_direct_build_process(
            shell_command,
            cwd=build.workspace,
            env=env,
            timeout_sec=timeout_sec,
        )
    except _DirectBuildTimeout as exc:
        _abort_controlled_build(
            str(exc),
            **_identity_fields(build),
            build_script=str(script),
            build_dir="kernel/build",
            build_mode=build.build_mode,
            timed_out=True,
            timeout_sec=exc.timeout_sec,
            stdout_tail=exc.stdout[-4096:],
            stderr_tail=exc.stderr[-4096:],
        )
    except (TargetTransportError, OSError, subprocess.SubprocessError) as exc:
        _abort_controlled_build(
            f"candidate build failed to start: {type(exc).__name__}: {exc}",
            **_identity_fields(build),
        )


def _require_stable_candidate_after_build(build: _ControlledBuild, returncode: int) -> None:
    """Reject a build that mutated the authored candidate it was given."""
    try:
        authored_cmake_after = _authored_cmake_sha256(build.workspace)
        candidate_after = _candidate_source_digest(build.workspace)
    except TargetTransportError as exc:
        _abort_controlled_build(
            str(exc),
            **_identity_fields(build),
            build_mode=build.build_mode,
            returncode=returncode,
        )
    if candidate_after != build.candidate_digest:
        _abort_controlled_build(
            "authored candidate changed during controlled candidate build",
            **_identity_fields(build),
            candidate_source_sha256_after=candidate_after,
            build_mode=build.build_mode,
            returncode=returncode,
        )
    if authored_cmake_after != build.authored_cmake_sha256:
        _abort_controlled_build(
            "authored CMake changed during controlled candidate build",
            **_identity_fields(build),
            build_mode=build.build_mode,
            returncode=returncode,
            authored_cmake_sha256_after=authored_cmake_after,
        )


def _controlled_independence_proof(
    build: _ControlledBuild, proof: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return the independence gate fields recorded on a completed build."""
    if build.source_kind != TILELANG2ASCENDC_SOURCE_KIND:
        return {
            "candidate_independence_gate": None,
            "candidate_independence_schema": None,
            "candidate_independence_proof": dict(proof) if proof is not None else None,
        }
    return {
        "candidate_independence_gate": "PASS",
        "candidate_independence_schema": TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA,
        "candidate_independence_proof": dict(proof) if proof is not None else None,
    }


def _controlled_build_success_payload(
    build: _ControlledBuild,
    completed: subprocess.CompletedProcess[str],
    script: Path,
    proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the receipt payload for a controlled build that actually ran."""
    payload: dict[str, Any] = {
        "schema": _contract_value(build.contract, "schema"),
        "status": "PASS" if completed.returncode == 0 else "ERROR",
        "build_attempt_id": build.attempt,
        "source_kind": build.source_kind,
        "source_stage_digest": build.source_stage_digest,
        "build_mode": build.build_mode,
        "returncode": int(completed.returncode),
        "target": build.target_identity,
        "soc": build.soc,
        "observed_soc": build.observed_soc,
        "authored_cmake_sha256": build.authored_cmake_sha256,
        "candidate_source_sha256": build.candidate_digest,
        **_controlled_independence_proof(build, proof),
        "build_identity": build.build_identity,
        "build_script": str(script),
        "build_dir": "kernel/build",
        # 16 KiB tails: external-project builds print the first compile
        # error well before the gmake cascade, and the O5 classifier needs
        # the diagnostic inside the window to route candidate compile
        # failures to await_worker (2026-08-22 SDPA: error missed in a
        # 4 KiB window → misclassified infra → terminal).
        "stdout_tail": str(completed.stdout or "")[-16384:],
        "stderr_tail": str(completed.stderr or "")[-16384:],
    }
    if completed.returncode != 0:
        payload["failure_kind"] = _classify_controlled_compile_failure(
            str(completed.stdout or ""), str(completed.stderr or "")
        )
        payload["reason"] = f"{script.name} exited non-zero"
    return payload


def _source_stage_digest(
    workspace: Path,
    state: Mapping[str, Any],
    source_kind: str | None,
) -> str:
    """Return the immutable source-stage digest bound to this build route."""
    if source_kind is None:
        digest = state.get("source_stage_digest")
        if not _sha_ok(digest):
            raise TargetTransportError(
                "generic durable state has no usable source-stage digest"
            )
        return str(digest)
    source = state.get("port_source")
    if not isinstance(source, Mapping) or source.get("kind") != source_kind:
        raise TargetTransportError(f"durable state does not select {source_kind}")
    if source_kind == TILELANG2ASCENDC_SOURCE_KIND:
        _verified_tilelang2ascendc_source_manifest(workspace, state)

    values = [
        source.get("tree_sha256"),
        source.get("digest"),
        source.get("source_stage_digest"),
        state.get("source_stage_digest"),
    ]
    supplied = [value for value in values if value is not None]
    if not supplied:
        raise TargetTransportError(f"{source_kind} durable state has no source-stage digest")
    if not all(_sha_ok(value) for value in supplied):
        raise TargetTransportError(f"{source_kind} durable state has a malformed source-stage digest")
    if len(set(supplied)) != 1:
        raise TargetTransportError(f"{source_kind} durable state has conflicting source-stage digests")
    return str(supplied[0])


def _verified_tilelang2ascendc_source_manifest(
    workspace: Path,
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate the complete TileLang2AscendC source-stage binding."""
    root = Path(workspace).resolve() / ".tilelang2ascendc_source"
    manifest_path = root / ".tilelang2ascendc_source_manifest.json"
    unsafe_root = root.is_symlink() or not root.is_dir()
    unsafe_manifest = manifest_path.is_symlink() or not manifest_path.is_file()
    if unsafe_root or unsafe_manifest:
        raise TargetTransportError("TileLang2AscendC source stage or manifest is missing/unsafe")
    try:
        from tilelang2ascendc_source import verify_tilelang2ascendc_source_stage

        valid, reason, manifest = verify_tilelang2ascendc_source_stage(workspace, state)
    except Exception as exc:
        raise TargetTransportError(f"TileLang2AscendC source verifier failed: {exc}") from exc
    if not valid or not isinstance(manifest, Mapping):
        raise TargetTransportError(f"TileLang2AscendC source stage rejected: {reason}")
    if (
        manifest.get("tree_sha256") != state.get("source_stage_digest")
        or manifest.get("file_count") != state.get("source_stage_file_count")
    ):
        raise TargetTransportError("TileLang2AscendC durable source digest/file count differs from manifest")
    return manifest


def preflight_npubench_on_target(workspace: Path, reference: Mapping[str, Any], lane: int) -> dict[str, Any]:
    """Run the old task's API preflight in the configured A5 environment."""
    return _guard(
        "preflight",
        lambda: _execute(
            workspace, reference, "preflight", lane, candidate=None, lease_manifest=None
        ),
    )


def evaluate_npubench_on_target(
    workspace: Path,
    reference: Mapping[str, Any],
    candidate_snapshot: Path,
    precision_device: int,
    performance_device: int,
    *legacy_lease_manifest: Path | None,
    lease_manifest: Path | None = None,
) -> dict[str, Any]:
    """Run precision and fixed quick W3/R5 profiling on one explicit lane.

    ``lease_manifest`` is keyword-first; the variadic tail keeps the historical
    six-positional call form working for existing callers without adding a
    sixth named positional parameter.
    """
    if legacy_lease_manifest:
        if len(legacy_lease_manifest) != 1 or lease_manifest is not None:
            raise TargetTransportError("target evaluation got a duplicate lease manifest")
        lease_manifest = legacy_lease_manifest[0]
    manifest = lease_manifest

    def run() -> dict[str, Any]:
        _device(precision_device, "precision_device")
        _device(performance_device, "performance_device")
        if precision_device != performance_device:
            raise TargetTransportError("NPUKernelBench target route supports one precision/performance lane")
        return _execute(
            workspace,
            reference,
            "evaluate",
            precision_device,
            candidate=candidate_snapshot,
            lease_manifest=manifest,
        )

    return _guard("evaluate", run)


def validate_target_evidence_receipt(
    workspace: Path,
    reference: Mapping[str, Any],
    evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    """Read-only finalizer gate for the controller-generated evaluate receipt."""
    try:
        workspace = _reference(workspace, reference)
        receipt = _target_receipt_for_evidence(workspace, evidence, reference)
        candidate = _receipt_candidate_snapshot(workspace, receipt)
        _verify_reports(workspace, "evaluate", reports, reports.get("evaluate"), candidate)
        _check_receipt_binding(workspace, receipt, candidate, reports)
        _check_receipt_report_digests(workspace, receipt, reports)
        _check_receipt_profile(workspace, receipt, reports)
    except (TargetTransportError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, "NPUKernelBench target receipt verified"


def _target_receipt_for_evidence(
    workspace: Path, evidence: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the canonical target receipt named by the finalizer evidence."""
    if evidence.get("target_execution_receipt") != TARGET_RECEIPT_PATH:
        raise TargetTransportError("target receipt path is not canonical")
    pointer = evidence.get("target_execution_receipt_sha256")
    receipt_path = workspace / TARGET_RECEIPT_PATH
    _file(receipt_path, "target receipt")
    if not _sha_ok(pointer) or _sha(receipt_path) != pointer:
        raise TargetTransportError("target receipt digest differs from evidence pointer")
    receipt = _read_json(receipt_path, "target receipt")
    _receipt_basics(receipt, reference)
    return receipt


def _receipt_candidate_snapshot(workspace: Path, receipt: Mapping[str, Any]) -> Path:
    candidate_sha = receipt.get("candidate_tree_sha256")
    if not _sha_ok(candidate_sha):
        raise TargetTransportError("target receipt candidate digest is invalid")
    return _snapshot(workspace, workspace / npubench_runner.SNAPSHOT_DIRNAME / candidate_sha)


def _check_receipt_binding(
    workspace: Path,
    receipt: Mapping[str, Any],
    candidate: Path,
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_binding = npubench_runner.build_evaluation_binding(workspace, candidate)
    if (
        receipt.get("binding_sha256") != expected_binding["binding_sha256"]
        or receipt.get("evaluation_binding") != expected_binding
    ):
        raise TargetTransportError("target receipt binding differs from frozen inputs")
    if receipt.get("status") != reports["evaluate"].get("status"):
        raise TargetTransportError("target receipt status differs from evaluate report")


def _check_receipt_report_digests(
    workspace: Path, receipt: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]
) -> None:
    digests = receipt.get("reports")
    if not isinstance(digests, Mapping):
        raise TargetTransportError("target receipt report digests are missing")
    for name in _EVAL:
        path = workspace / npubench_runner.EVIDENCE_DIRNAME / _REPORT[name]
        _file(path, f"canonical {name} report")
        if not _sha_ok(digests.get(name)) or _sha(path) != digests.get(name):
            raise TargetTransportError(f"target receipt {name} report digest differs")
        if _read_json(path, f"canonical {name} report") != reports.get(name):
            raise TargetTransportError(f"canonical {name} report differs from finalizer evidence")


def _require_empty_receipt_profile(profile: Any) -> None:
    if (
        not isinstance(profile, Mapping)
        or profile.get("path") is not None
        or profile.get("tree_sha256") is not None
    ):
        raise TargetTransportError(
            "target receipt profile must be empty for deferred performance"
        )


def _check_receipt_profile(
    workspace: Path, receipt: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]
) -> None:
    profile = receipt.get("profile")
    performance = reports["performance"]
    if performance.get("perf_deferred") is True:
        _require_empty_receipt_profile(profile)
        return
    if (
        not isinstance(profile, Mapping)
        or profile.get("path") != performance.get("profile_archive")
        or profile.get("tree_sha256") != performance.get("profile_tree_sha256")
    ):
        raise TargetTransportError("target receipt profile differs from performance report")
    _profile(workspace, performance)


def _execute(
    workspace: Path,
    reference: Mapping[str, Any],
    verb: str,
    lane: int,
    *,
    candidate: Path | None,
    lease_manifest: Path | None,
) -> dict[str, Any]:
    workspace = _reference(workspace, reference)
    _device(lane, "lane")
    if verb == "evaluate":
        if candidate is None:
            raise TargetTransportError("target evaluation has no candidate")
        candidate = _snapshot(workspace, candidate)
    target = _target(workspace, lane)
    if target.container.lower() == "local":
        returned = (
            npubench_runner.preflight_workspace(workspace)
            if verb == "preflight"
            else npubench_runner.evaluate_workspace(
                workspace, candidate, precision_device=lane, performance_device=lane,
                lease_manifest=lease_manifest,
            )
        )
        reports = _reports(workspace, verb)
        _verify_reports(workspace, verb, reports, returned, candidate)
        return _publish(workspace, reference, target, verb, reports, "local_target", lane)
    reports = _remote(workspace, reference, target, verb, lane, candidate, lease_manifest)
    return _publish(workspace, reference, target, verb, reports, "ssh_target", lane)


def _reference(workspace: Path, reference: Mapping[str, Any]) -> Path:
    workspace = Path(workspace)
    if workspace.is_symlink() or not workspace.is_dir():
        raise TargetTransportError("workspace must be a real directory")
    workspace = workspace.resolve()
    if not isinstance(reference, Mapping) or reference.get("source") != NPUBENCH_SOURCE:
        raise TargetTransportError("durable reference does not select npubench")
    state_path = workspace / ".opgen_state.json"
    _file(state_path, "durable state")
    if _read_json(state_path, "durable state").get("reference") != dict(reference):
        raise TargetTransportError("provided reference differs from durable state")
    valid, reason, _ = verify_npubench_stage(workspace, reference)
    if not valid:
        raise TargetTransportError("staged NPUKernelBench bundle rejected: " + reason)
    return workspace


def _snapshot(workspace: Path, path: Path) -> Path:
    try:
        root = (workspace / npubench_runner.SNAPSHOT_DIRNAME).resolve(strict=True)
        path = Path(path).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TargetTransportError("candidate is outside the frozen snapshot root") from exc
    if not _sha_ok(path.name):
        raise TargetTransportError("candidate snapshot name is not a content digest")
    try:
        # Bound by name (not attribute access) inside the guard so a UT patch of
        # the runner module is still observed on every call.
        from npubench.npubench_runner import _candidate_entry, _require_snapshot

        _require_snapshot(path, path.name)
        _candidate_entry(path)
    except Exception as exc:
        raise TargetTransportError(f"candidate snapshot is invalid: {exc}") from exc
    return path


def _target(workspace: Path, lane: int) -> _Target:
    try:
        return _resolve_target(workspace, lane)
    except Exception as exc:
        raise TargetTransportError(f"cannot resolve A5 target: {exc}") from exc


def _cleanup_remote_run(
    target: _Target,
    token: str,
    remote: Mapping[str, str],
    local: Mapping[str, Path | None],
) -> list[BaseException]:
    """Drop the local staging artifacts and the remote workspace, auditing failures."""
    cleanup_errors: list[BaseException] = []
    for label, path in local.items():
        _collect_cleanup_error(
            cleanup_errors,
            f"unlink local {label} {path}",
            lambda bound=path: _unlink(bound, required=True, cleanup_errors=cleanup_errors),
        )
    _collect_cleanup_error(
        cleanup_errors,
        "remote target cleanup",
        lambda: _cleanup(target, remote["root"], remote["stage"], remote["result"], token),
    )
    return cleanup_errors


def _remote(
    workspace: Path,
    reference: Mapping[str, Any],
    target: _Target,
    verb: str,
    lane: int,
    *candidate_and_lease: Path | None,
) -> dict[str, dict[str, Any]]:
    """Stage, run and import one evaluation on a remote target.

    ``candidate_and_lease`` groups the two related per-run inputs --
    ``(candidate, lease_manifest)`` -- into one variadic parameter instead of
    two more positional ones, while keeping every existing call form working.
    """
    if len(candidate_and_lease) != 2:
        raise TargetTransportError("target remote run needs (candidate, lease_manifest)")
    candidate, lease_manifest = candidate_and_lease
    token = uuid.uuid4().hex
    remote_root = f"{target.benchmark_root.rstrip('/')}/npubench_target/{token}"
    remote_stage = f"/tmp/cannbot_npubench_stage_{token}.tar"
    remote_result = f"/tmp/cannbot_npubench_result_{token}.tar"
    stage: Path | None = None
    result: Path | None = None
    returned_reports: dict[str, dict[str, Any]] | None = None
    primary_error: BaseException | None = None
    try:
        state = _read_json(workspace / ".opgen_state.json", "durable state")
        stage = _stage(workspace, reference, candidate, lease_manifest, state=state)
        _must(
            _run(
                _scp(str(stage), f"{target.user}@{target.host}:{remote_stage}", target),
                "target stage upload",
            ),
            "target stage upload",
        )
        _prepare(target, remote_root, remote_stage, token)
        returned = _invoke(
            target, remote_root, verb, lane, candidate=candidate, lease=lease_manifest
        )
        _pack(target, remote_root, remote_result, token, verb)
        result = _download(remote_result, target)
        returned_reports = _import(workspace, result, verb, returned, candidate)
    except BaseException as exc:
        primary_error = exc

    cleanup_errors = _cleanup_remote_run(
        target,
        token,
        {"root": remote_root, "stage": remote_stage, "result": remote_result},
        {"stage": stage, "result": result},
    )
    if primary_error is not None:
        _raise_preserving(primary_error, cleanup_errors)
    if cleanup_errors:
        _raise_cleanup_failures(cleanup_errors)
    if returned_reports is None:
        raise TargetTransportError("target remote run produced no imported reports")
    return returned_reports


def _stage(
    workspace: Path,
    reference: Mapping[str, Any],
    candidate: Path | None,
    lease: Path | None,
    *,
    state: Mapping[str, Any] | None = None,
) -> Path:
    bundle_sha = reference.get("bundle_sha256")
    if not _sha_ok(bundle_sha):
        raise TargetTransportError("bundle digest is invalid")
    bundle = workspace / "reference_inputs" / NPUBENCH_SOURCE / bundle_sha
    if bundle.is_symlink() or not bundle.is_dir():
        raise TargetTransportError("staged NPUKernelBench bundle is missing")
    if candidate is not None:
        _snapshot(workspace, candidate)
    if lease is not None:
        _file(lease, "lease manifest")
    state_path = workspace / ".opgen_state.json"
    _file(state_path, "durable state")
    runner = Path(npubench_runner.__file__).resolve()
    # The runner is split across sibling modules and imports them at module
    # top, so shipping npubench_runner.py alone would make the remote
    # `python npubench_runner.py <verb>` die with ModuleNotFoundError.  Ship
    # exactly the set the runner declares.
    runner_modules = tuple(
        (runner.with_name(name), name) for name in npubench_runner.RUNNER_MODULE_FILENAMES
    )
    # Bound by name inside the function so a UT patch of the runner module is
    # still observed here (and no protected attribute is accessed).
    from npubench.npubench_runner import _default_profiler_summary

    summary = _default_profiler_summary()
    required_files = (
        (workspace / ".opgen_state.json", "durable state"),
        *((path, f"runner module {name}") for path, name in runner_modules),
        (summary, "quick profiler summary"),
    )
    for path, label in required_files:
        _file(path, label)
    with tempfile.NamedTemporaryFile(prefix="cannbot_npubench_stage_", suffix=".tar", delete=False) as file:
        archive = Path(file.name)
    try:
        with tarfile.open(archive, "w") as tar:
            _add(tar, workspace / ".opgen_state.json", ".opgen_state.json")
            _add(tar, bundle, f"reference_inputs/{NPUBENCH_SOURCE}/{bundle_sha}")
            for module_path, module_name in runner_modules:
                _add(tar, module_path, module_name)
            _add(tar, summary, "ops/ops-profiling/scripts/msprof_perf_summary.py")
            if candidate is not None:
                _add(tar, candidate, f"{npubench_runner.SNAPSHOT_DIRNAME}/{candidate.name}")
            if lease is not None:
                _add(tar, lease, "leases/lease_manifest.json")
        return archive
    except BaseException as exc:
        cleanup_error = _unlink(archive)
        if cleanup_error is not None:
            _audit_cleanup_errors(exc, [cleanup_error])
        raise


def _add(tar: tarfile.TarFile, source: Path, name: str) -> None:
    def keep(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        path = PurePosixPath(member.name)
        if "__pycache__" in path.parts or path.name.endswith(".pyc"):
            return None
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise TargetTransportError("target stage contains a non-regular entry")
        return member

    tar.add(str(source), arcname=name, filter=keep)


def _add_json(tar: tarfile.TarFile, value: Mapping[str, Any], name: str) -> None:
    """Add controller-authored JSON without materializing a transient source file."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    encoded = (payload + "\n").encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(encoded)
    info.mode = 0o400
    info.mtime = 0
    tar.addfile(info, io.BytesIO(encoded))


def _prepare(target: _Target, root: str, stage: str, token: str) -> None:
    q = shlex.quote
    if target.host_mode:
        command = f"rm -rf {q(root)} && mkdir -p {q(root)} && tar -xf {q(stage)} -C {q(root)} && rm -f {q(stage)}"
    else:
        inside = f"/tmp/cannbot_npubench_stage_{token}.tar"
        extract = f"rm -rf {q(root)} && mkdir -p {q(root)} && tar -xf {q(inside)} -C {q(root)} && rm -f {q(inside)}"
        command = _sudo(
            f"docker cp {q(stage)} {q(f'{target.container}:{inside}')} && "
            f"docker exec {q(target.container)} bash -c {q(extract)} && rm -f {q(stage)}",
            target,
        )
    _must(_run(_ssh(command, target), "target stage extract"), "target stage extract")


def _invoke(
    target: _Target,
    root: str,
    verb: str,
    lane: int,
    *,
    candidate: Path | None,
    lease: Path | None,
) -> dict[str, Any]:
    args = [verb, "--workspace", "."]
    if verb == "evaluate":
        if candidate is None:
            raise TargetTransportError("target evaluation has no candidate")
        args += [
            "--candidate-dir",
            f"{npubench_runner.SNAPSHOT_DIRNAME}/{candidate.name}",
            "--precision-device",
            str(lane),
            "--performance-device",
            str(lane),
        ]
        if lease is not None:
            args += ["--lease-manifest", "leases/lease_manifest.json"]
    result = _run(
        _target_command(
            target,
            root,
            args,
        ),
        f"target {verb}",
    )
    report = _json_tail(result.stdout)
    if not isinstance(report, dict) or report.get("schema") != f"cannbot.npubench.{verb}/v1":
        raise TargetTransportError(f"target {verb} produced no valid JSON report")
    return report


def _target_command(
    target: _Target,
    root: str,
    args: Sequence[str],
) -> list[str]:
    from phase_o5_helpers import _resolve_extra_ld, _resolve_extra_pythonpath, _resolve_npu_python_bin

    q, env = shlex.quote, target.env
    argv = " ".join(q(str(arg)) for arg in args)
    set_env = q(str(Path(target.cann_path) / "set_env.sh"))
    npu_python = _resolve_npu_python_bin(env, target.name)
    extra_ld = _resolve_extra_ld(env, target.name)
    extra_pythonpath = _resolve_extra_pythonpath(env, target.name)
    if not target.host_mode:
        from phase_o5_verify import _container_npu_python_setup

        python, setup = _container_npu_python_setup(target.cann_path, npu_python, extra_ld, extra_pythonpath)
        inside = (
            f"cd {q(root)} && export ASCEND_RT_VISIBLE_DEVICES={target.visible_device} && "
            f"if [ -f {set_env} ]; then source {set_env}; fi; {setup}{python} "
            f"npubench_runner.py {argv}"
        )
        return _ssh(_sudo(f"docker exec {q(target.container)} bash -c {q(inside)}", target), target)
    python = str(env.get("A5_HOST_PYTHON") or "python3")
    home = os.path.dirname(os.path.dirname(python))
    libraries = (
        f"{extra_ld + ':' if extra_ld else ''}{home}/lib/python3.11/site-packages/torch/lib:"
        f"{home}/lib/python3.11/site-packages/torch_npu/lib:{target.cann_path}/lib64:"
        "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
    )
    runner = f"{q(python)} npubench_runner.py {argv}"
    host_setup = (
        f"set +eu; set +o pipefail; if [ -f {set_env} ]; then source {set_env} || true; fi; "
        f"export LD_LIBRARY_PATH={q(libraries)}:${{LD_LIBRARY_PATH:-}}; "
        f"export PYTHONPATH={q(extra_pythonpath)}:${{PYTHONPATH:-}}; "
        f"export PATH={q(os.path.dirname(python))}:$PATH; "
        f"export ASCEND_RT_VISIBLE_DEVICES={target.visible_device}; cd {q(root)} && "
    )
    return _ssh(host_setup + runner, target)


def _pack(target: _Target, root: str, result: str, token: str, verb: str) -> None:
    q = shlex.quote
    names = (
        [f"{npubench_runner.EVIDENCE_DIRNAME}/{_REPORT['preflight']}"]
        if verb == "preflight"
        else [f"{npubench_runner.EVIDENCE_DIRNAME}/{_REPORT[name]}" for name in _EVAL]
    )
    fixed, profiles = " ".join(q(name) for name in names), q(f"{npubench_runner.EVIDENCE_DIRNAME}/profiles")
    create = (
        f"cd {q(root)} && if [ -d {profiles} ]; then tar -cf {q(result)} -- {fixed} "
        f"{profiles}; else tar -cf {q(result)} -- {fixed}; fi"
    )
    if not target.host_mode:
        inside = f"/tmp/cannbot_npubench_result_{token}.tar"
        create = create.replace(q(result), q(inside))
        create = _sudo(
            f"docker exec {q(target.container)} bash -c {q(create)} && "
            f"docker cp {q(f'{target.container}:{inside}')} {q(result)} && "
            f"docker exec {q(target.container)} rm -f {q(inside)}",
            target,
        )
    _must(_run(_ssh(create, target), "target result package"), "target result package")


def _download(remote: str, target: _Target) -> Path:
    with tempfile.NamedTemporaryFile(prefix="cannbot_npubench_result_", suffix=".tar", delete=False) as file:
        local = Path(file.name)
    try:
        _must(
            _run(
                _scp(f"{target.user}@{target.host}:{remote}", str(local), target),
                "target result fetch",
            ),
            "target result fetch",
        )
    except BaseException as exc:
        cleanup_errors: list[BaseException] = []
        _collect_cleanup_error(
            cleanup_errors,
            f"unlink downloaded result {local}",
            lambda: _unlink(local, required=True, cleanup_errors=cleanup_errors),
        )
        # ``_raise_preserving`` is annotated NoReturn: control never reaches
        # the single ``return`` below on this path.
        _raise_preserving(exc, cleanup_errors)
    return local


def _import(
    workspace: Path,
    archive: Path,
    verb: str,
    returned: Mapping[str, Any],
    candidate: Path | None,
) -> dict[str, dict[str, Any]]:
    staged = Path(tempfile.mkdtemp(prefix=".npubench_target_", dir=workspace))
    imported_reports: dict[str, dict[str, Any]] | None = None
    primary_error: BaseException | None = None
    try:
        _extract(archive, staged, verb)
        reports = _reports(staged, verb)
        _verify_reports(workspace, verb, reports, returned, candidate, profile_root=staged)
        _commit(workspace, staged, reports, verb)
        imported_reports = reports
    except BaseException as exc:
        primary_error = exc

    cleanup_errors: list[BaseException] = []
    _collect_cleanup_error(
        cleanup_errors,
        f"remove import staging {staged}",
        lambda: _remove_tree(staged, required=True, cleanup_errors=cleanup_errors),
    )
    if primary_error is not None:
        _raise_preserving(primary_error, cleanup_errors)
    if cleanup_errors:
        _raise_cleanup_failures(cleanup_errors)
    if imported_reports is None:
        raise TargetTransportError("target result import produced no reports")
    return imported_reports


def _is_result_parent_dir(name: str, required: set[str]) -> bool:
    """Report whether an archive directory legitimately holds a fixed report."""
    if name in {npubench_runner.EVIDENCE_DIRNAME, f"{npubench_runner.EVIDENCE_DIRNAME}/profiles"}:
        return True
    return any(item.startswith(name + "/") for item in required)


def _checked_result_member_name(
    member: tarfile.TarInfo, seen: set[str], required: set[str]
) -> str:
    """Validate one archive entry and return its normalized member name."""
    path = PurePosixPath(member.name)
    if not member.name or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise TargetTransportError("target result archive has an unsafe path")
    name = path.as_posix()
    if name in seen:
        raise TargetTransportError("target result archive has duplicate entries")
    allowed = name in required or name.startswith(f"{npubench_runner.EVIDENCE_DIRNAME}/profiles/")
    parent = _is_result_parent_dir(name, required)
    if not allowed and not (member.isdir() and parent):
        raise TargetTransportError("target result archive has an unexpected entry")
    if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
        raise TargetTransportError("target result archive has a link or special entry")
    return name


def _extract_result_member(tar: tarfile.TarFile, member: tarfile.TarInfo, output: Path) -> None:
    """Materialize one validated archive entry below the staging directory."""
    if member.isdir():
        output.mkdir(parents=True, exist_ok=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        raise TargetTransportError("cannot read target result entry")
    with source, output.open("xb") as file:
        shutil.copyfileobj(source, file)


def _extract(archive: Path, destination: Path, verb: str) -> None:
    required = (
        {f"{npubench_runner.EVIDENCE_DIRNAME}/{_REPORT['preflight']}"}
        if verb == "preflight"
        else {f"{npubench_runner.EVIDENCE_DIRNAME}/{_REPORT[name]}" for name in _EVAL}
    )
    seen: set[str] = set()
    try:
        with tarfile.open(archive, "r") as tar:
            for member in tar.getmembers():
                name = _checked_result_member_name(member, seen, required)
                seen.add(name)
                _extract_result_member(tar, member, destination / PurePosixPath(name))
    except tarfile.TarError as exc:
        raise TargetTransportError(f"target result archive is invalid: {exc}") from exc
    if not required.issubset(seen):
        raise TargetTransportError("target result archive omitted a fixed report")


def _reports(root: Path, verb: str) -> dict[str, dict[str, Any]]:
    names = ("preflight",) if verb == "preflight" else _EVAL
    reports: dict[str, dict[str, Any]] = {}
    for name in names:
        path = Path(root) / npubench_runner.EVIDENCE_DIRNAME / _REPORT[name]
        _file(path, f"{name} report")
        reports[name] = _read_json(path, f"{name} report")
    return reports


def _verify_reports(
    workspace: Path,
    verb: str,
    reports: Mapping[str, Mapping[str, Any]],
    returned: Mapping[str, Any] | None,
    candidate: Path | None,
    *,
    profile_root: Path | None = None,
) -> None:
    if verb == "preflight":
        report = reports.get("preflight")
        if not isinstance(report, Mapping) or report != returned:
            raise TargetTransportError("target preflight report differs from returned evidence")
        ok, reason = npubench_runner.verify_evidence_report(workspace, report, expected_verb="preflight")
        if not ok:
            raise TargetTransportError("target preflight evidence rejected: " + reason)
        return
    if candidate is None or any(not isinstance(reports.get(name), Mapping) for name in _EVAL):
        raise TargetTransportError("target evaluation omitted a frozen candidate or fixed report")
    precision, performance, evaluate = reports["precision"], reports["performance"], reports["evaluate"]
    if evaluate != returned:
        raise TargetTransportError("target evaluate report differs from returned evidence")
    for name, report in (("precision", precision), ("performance", performance), ("evaluate", evaluate)):
        ok, reason = npubench_runner.verify_evidence_report(
            workspace,
            report,
            expected_verb=name,
            candidate_dir=candidate,
        )
        if not ok:
            raise TargetTransportError(f"target {name} evidence rejected: {reason}")
    if evaluate.get("precision") != precision or evaluate.get("performance") != performance:
        raise TargetTransportError("target evaluate report does not bind lane reports")
    perf_deferred = (
        performance.get("status") == "DEFERRED"
        and performance.get("perf_deferred") is True
    )
    if perf_deferred:
        # Precision-first mode: evaluate status follows precision alone
        # (2026-08-23 SDPA/BAM: the old hard-coded PASS+FAIL contract
        # rejected the deferred aggregate).
        expected_status = "PASS" if precision.get("status") == "PASS" else "FAIL"
    else:
        expected_status = (
            "PASS"
            if precision.get("status") == performance.get("status") == "PASS"
            else "FAIL"
        )
    if evaluate.get("status") != expected_status:
        raise TargetTransportError("target evaluate status is inconsistent")
    if not perf_deferred:
        _profile(profile_root or workspace, performance)


def _profile(root: Path, performance: Mapping[str, Any]) -> None:
    value, digest = performance.get("profile_archive"), performance.get("profile_tree_sha256")
    if value is None:
        if digest is not None:
            raise TargetTransportError("profile digest has no archive")
        return
    path = PurePosixPath(str(value))
    unsafe_shape = path.is_absolute() or len(path.parts) != 3
    wrong_root = path.parts[:2] != (npubench_runner.EVIDENCE_DIRNAME, "profiles")
    if unsafe_shape or wrong_root or not _sha_ok(digest):
        raise TargetTransportError("profile evidence is invalid")
    try:
        if npubench_runner.profile_tree_sha256(Path(root) / path) != digest:
            raise TargetTransportError("profile tree digest differs from report")
    except TargetTransportError:
        raise
    except Exception as exc:
        raise TargetTransportError(f"profile tree is invalid: {exc}") from exc


def _commit(workspace: Path, staged: Path, reports: Mapping[str, Mapping[str, Any]], verb: str) -> None:
    evidence = _real_directory(
        workspace / npubench_runner.EVIDENCE_DIRNAME,
        "evidence directory",
    )
    if verb == "evaluate" and reports["performance"].get("profile_archive") is not None:
        relative = PurePosixPath(str(reports["performance"]["profile_archive"]))
        source = staged / relative
        profiles = _real_directory(evidence / "profiles", "profile evidence directory")
        destination = profiles / relative.name
        if destination.exists() or destination.is_symlink():
            raise TargetTransportError("profile destination already exists")
        os.replace(source, destination)
    for name in (("preflight",) if verb == "preflight" else _EVAL):
        destination = evidence / _REPORT[name]
        if destination.is_symlink():
            raise TargetTransportError("canonical report path is a symlink")
        os.replace(staged / npubench_runner.EVIDENCE_DIRNAME / _REPORT[name], destination)


def _target_receipt_target_block(target: _Target, lane: int) -> dict[str, Any]:
    """Describe the endpoint that produced the evidence, without credentials."""
    block: dict[str, Any] = {
        "name": target.name,
        "host": target.host,
        "container": "host" if target.host_mode else target.container,
        "lane": lane,
        "visible_device": target.visible_device,
    }
    runtime_container = _local_runtime_container(target)
    if runtime_container is not None:
        block["runtime_container"] = runtime_container
        block["runtime_observation"] = _local_runtime_observation()
    return block


def _target_receipt_binding_block(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the evaluation binding and its adapter summary into the receipt."""
    adapter = binding.get("input_adapter")
    case_count = adapter.get("case_count") if isinstance(adapter, Mapping) else None
    return {
        "evaluation_binding": dict(binding),
        "input_adapter": dict(binding.get("input_adapter", {})),
        "case_count": case_count,
        "candidate_tree_sha256": binding.get("candidate_tree_sha256"),
    }


def _publish(
    workspace: Path,
    reference: Mapping[str, Any],
    target: _Target,
    verb: str,
    *receipt_inputs: Any,
) -> dict[str, Any]:
    """Write the controller-side target receipt and return the primary report.

    ``receipt_inputs`` groups the three related receipt inputs -- ``(reports,
    transport, lane)`` -- into one variadic parameter instead of three more
    positional ones, while keeping every existing call form working.
    """
    if len(receipt_inputs) != 3:
        raise TargetTransportError("target receipt publish needs (reports, transport, lane)")
    reports, transport, lane = receipt_inputs
    primary = reports["preflight" if verb == "preflight" else "evaluate"]
    binding = primary.get("evaluation_binding")
    if not isinstance(binding, Mapping):
        raise TargetTransportError("target report has no evaluation binding")
    evidence = workspace / npubench_runner.EVIDENCE_DIRNAME
    receipt = {
        "schema": TARGET_RECEIPT_SCHEMA,
        "verb": verb,
        "execution_site": "a5_target",
        "transport": transport,
        "target": _target_receipt_target_block(target, lane),
        "runner_contract_version": npubench_runner.RUNNER_CONTRACT_VERSION,
        "input": {key: reference.get(key) for key in _REFERENCE_INPUT_KEYS},
        "binding_sha256": primary.get("binding_sha256"),
        **_target_receipt_binding_block(binding),
        "reports": {name: _sha(evidence / _REPORT[name]) for name in reports},
        "profile": {
            "path": reports.get("performance", {}).get("profile_archive"),
            "tree_sha256": reports.get("performance", {}).get("profile_tree_sha256"),
        },
        "status": primary.get("status"),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    filename = PREFLIGHT_RECEIPT_FILENAME if verb == "preflight" else TARGET_RECEIPT_FILENAME
    path = evidence / filename
    _atomic_json(path, receipt)
    result = dict(primary)
    result.update(
        {
            "transport": transport,
            "target_receipt_path": f"{npubench_runner.EVIDENCE_DIRNAME}/{filename}",
            "target_receipt_sha256": _sha(path),
        }
    )
    return result


def _receipt_basics(receipt: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema") != TARGET_RECEIPT_SCHEMA
        or receipt.get("verb") != "evaluate"
        or receipt.get("execution_site") != "a5_target"
    ):
        raise TargetTransportError("target receipt schema is invalid")
    if (
        receipt.get("transport") not in {"local_target", "ssh_target"}
        or receipt.get("runner_contract_version")
        != npubench_runner.RUNNER_CONTRACT_VERSION
    ):
        raise TargetTransportError("target receipt transport is invalid")
    target = receipt.get("target")
    if not isinstance(target, Mapping):
        raise TargetTransportError("target receipt identity is invalid")
    ssh_without_host = receipt.get("transport") == "ssh_target" and not target.get("host")
    if str(target.get("name", "")).upper() != "A5" or ssh_without_host:
        raise TargetTransportError("target receipt identity is invalid")
    _device(target.get("lane"), "target lane")
    _device(target.get("visible_device"), "target visible device")
    inputs = receipt.get("input")
    if not isinstance(inputs, Mapping):
        raise TargetTransportError("target receipt input binding differs")
    if any(inputs.get(key) != reference.get(key) for key in _REFERENCE_INPUT_KEYS):
        raise TargetTransportError("target receipt input binding differs")
    if receipt.get("status") not in {"PASS", "FAIL"}:
        raise TargetTransportError("target receipt status is invalid")
    binding = receipt.get("evaluation_binding")
    adapter = binding.get("input_adapter") if isinstance(binding, Mapping) else None
    if adapter is not None:
        if receipt.get("input_adapter") != adapter:
            raise TargetTransportError("target receipt input adapter differs from evaluation binding")
        if receipt.get("case_count") != adapter.get("case_count"):
            raise TargetTransportError("target receipt case_count differs from input adapter")


def _cleanup(target: _Target, root: str, stage: str, result: str, token: str) -> None:
    if os.environ.get("CANNBOT_KEEP_NPUBENCH_TARGET_WORKSPACE") == "1":
        return
    q = shlex.quote
    if target.host_mode:
        command = f"rm -rf {q(root)} && rm -f {q(stage)} {q(result)}"
    else:
        command = _sudo(
            f"docker exec {q(target.container)} rm -rf {q(root)} "
            f"/tmp/cannbot_npubench_stage_{token}.tar "
            f"/tmp/cannbot_npubench_result_{token}.tar; rm -f {q(stage)} {q(result)}",
            target,
        )
    _must(_run(_ssh(command, target), "target cleanup", 60), "target cleanup")


def _guard(verb: str, action) -> dict[str, Any]:
    try:
        return action()
    except (TargetTransportError, OSError, ValueError) as exc:
        return _error(verb, str(exc))
    except Exception as exc:  # pragma: no cover - defensive transport boundary
        return _error(verb, f"{type(exc).__name__}: {exc}")


def _error(verb: str, reason: str) -> dict[str, Any]:
    return {
        "schema": f"cannbot.npubench.{verb}/v1",
        "status": "ERROR",
        "runner_contract_version": npubench_runner.RUNNER_CONTRACT_VERSION,
        "reason": str(reason)[:1024],
        "transport": "target_unavailable",
    }


def _must(result: subprocess.CompletedProcess[str], what: str) -> None:
    if result.returncode != 0:
        raise TargetTransportError(f"{what} failed (rc={result.returncode}; stderr_tail={result.stderr[-300:]!r})")


def _timeout() -> int:
    try:
        return max(60, int(os.environ.get("CANNBOT_NPUBENCH_TARGET_TIMEOUT", "1800")))
    except ValueError:
        return 1800


def _ssh(command: str, target: _Target) -> list[str]:
    return _ssh_command(command, target)


def _scp(source: str, destination: str, target: _Target) -> list[str]:
    return _scp_command(source, destination, target)


def _sudo(command: str, target: _Target) -> str:
    return _maybe_sudo(command, target)


def _run(command: list[str], what: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return _a5_run(command, timeout=timeout or _timeout(), what=what)
    except Exception as exc:
        raise TargetTransportError(str(exc)) from exc

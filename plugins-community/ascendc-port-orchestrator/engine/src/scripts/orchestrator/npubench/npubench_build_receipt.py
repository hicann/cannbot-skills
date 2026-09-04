# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Controlled-build receipt: contract, identity binding and HMAC authentication.

This module owns the persisted evidence of one controlled CANN candidate build:
the per-source build contract, the target/toolchain/runtime identity recorded
with it, the failure payload shape, and the payload digest plus private-key
HMAC that let O5 trust a receipt it did not itself produce.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
from pathlib import Path
from typing import Any, Mapping

from npubench import npubench_runner
from a5_target_capability import a5_soc_version
from a5_target_transport import _Target
from npubench.npubench_candidate_contract import (
    TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA,
    TILELANG2ASCENDC_SOURCE_KIND,
)
from npubench.npubench_target_base import (
    TargetTransportError,
    _atomic_json,
    _contract_value,
    _real_directory,
    _sha,
    _sha_ok,
)


TILELANG2ASCENDC_BUILD_RECEIPT_FILENAME = "tilelang2ascendc_build_receipt.json"


TILELANG2ASCENDC_BUILD_RECEIPT_PATH = f"{npubench_runner.EVIDENCE_DIRNAME}/{TILELANG2ASCENDC_BUILD_RECEIPT_FILENAME}"


TILELANG2ASCENDC_BUILD_RECEIPT_SCHEMA = "cannbot.tilelang2ascendc_candidate_build/v1"


TILELANG2ASCENDC_BUILD_IDENTITY_SCHEMA = "cannbot.tilelang2ascendc_candidate_build_identity/v1"


_RECEIPT_KEY_DIR = ".cannbot_candidate_receipt_keys"


_BUILD_IDENTITY_CONFIG_KEYS = (
    "A5_SOC_VERSION",
    "A5_PYTHON",
    "A5_HOST_PYTHON",
    "A5_NPU_PYTHON_BIN",
    "NPU_PYTHON_BIN",
    "A5_CANN_PATH",
    "CANN_PATH",
    "A5_CONTAINER",
    "A5_LOCAL_RUNTIME_CONTAINER",
    "A5_HOST_MODE",
    "A5_EXTRA_LD",
    "A5_EXTRA_PYTHONPATH",
    "A5_NPU_SMI_BIN",
    "NPU_SMI_BIN",
)


_BUILD_IDENTITY_PROCESS_KEYS = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CC",
    "CXX",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "ASCEND_INSTALL_PATH",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
)


def _build_contract(source_kind: str | None) -> dict[str, str]:
    # The receipt protocol (schema, identity binding, HMAC authentication) is
    # source-agnostic; the generic authored-kernel route (source_kind=None)
    # shares it, with the payload's ``source_kind`` field recording the route.
    if source_kind in (TILELANG2ASCENDC_SOURCE_KIND, None):
        return {
            "schema": TILELANG2ASCENDC_BUILD_RECEIPT_SCHEMA,
            "identity_schema": TILELANG2ASCENDC_BUILD_IDENTITY_SCHEMA,
            "receipt_filename": TILELANG2ASCENDC_BUILD_RECEIPT_FILENAME,
            "receipt_path": TILELANG2ASCENDC_BUILD_RECEIPT_PATH,
        }
    raise TargetTransportError(f"unsupported controlled candidate source kind: {source_kind!r}")


def _candidate_build_error_payload(
    contract: Mapping[str, str],
    source_kind: str | None,
    reason: str,
    *,
    failure_kind: str = "target_build",
    target: Mapping[str, Any],
    source_stage_digest: str | None = None,
    candidate_digest: str | None = None,
    authored_cmake_sha256: str | None = None,
    soc: str | None = None,
    observed_soc: str | None = None,
    build_identity: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Keep failed receipts bound to every input observed before failure."""
    payload: dict[str, Any] = {
        "schema": _contract_value(contract, "schema"),
        "status": "ERROR",
        # Keep the routing category in the authenticated receipt.  It is
        # supplied by the typed validation boundary above, never inferred from
        # mutable human-facing diagnostics.
        "failure_kind": failure_kind,
        "build_attempt_id": contract.get("build_attempt_id"),
        "source_kind": source_kind,
        "source_stage_digest": source_stage_digest,
        "candidate_source_sha256": candidate_digest,
        "authored_cmake_sha256": authored_cmake_sha256,
        "target": dict(target),
        "soc": soc,
        "observed_soc": observed_soc,
        "build_identity": dict(build_identity) if build_identity is not None else None,
        "reason": reason,
    }
    payload.update(extra)
    if source_kind == TILELANG2ASCENDC_SOURCE_KIND:
        payload.setdefault(
            "candidate_independence_gate",
            "PASS" if candidate_digest is not None else "NOT_REACHED",
        )
        payload.setdefault(
            "candidate_independence_schema",
            TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA,
        )
        payload.setdefault("build_mode", "controlled_authored_cmake")
    return payload


def _classify_controlled_compile_failure(stdout: str, stderr: str) -> str:
    """Classify compiler diagnostics that clearly point at authored code.

    A controlled build can fail because the target/toolchain is unavailable or
    because the delivered candidate does not compile.  Keep the latter on the
    worker repair path, but require a compiler diagnostic naming a file below
    the candidate ``kernel/`` tree so missing SDK/tool errors remain
    target-build failures.

    A candidate can also break an SDK header transitively: an include chain
    rooted below ``kernel/`` (for example a tiling header that pulls
    ``tiling_api.h`` in the wrong compile pass) makes the compiler emit its
    errors inside SDK headers, so no diagnostic names a candidate file even
    though only the candidate can reorder the include.  Treat such cascades as
    candidate failures too, but keep ``file not found`` cascades fail-closed:
    a missing SDK header is a toolchain/target problem, not authored code.
    """
    diagnostics = f"{stdout}\n{stderr}"
    if re.search(
        r"(?m)\bkernel/[^:\n]+:\d+:\d+:\s*(?:fatal\s+)?error:",
        diagnostics,
    ):
        return "candidate_contract"
    candidate_rooted_cascade = re.search(
        # clang prints one "In file included from <path>:N:" per chain entry;
        # gcc folds the tail into indented "from <path>:N," continuation
        # lines.  Both name the candidate file that rooted the chain.
        r"(?m)(?:^In file included from |^\s+from )[^:\n]*/kernel/[^:\n]+:\d+",
        diagnostics,
    )
    if candidate_rooted_cascade:
        for match in re.finditer(
            r"(?m)^[^:\n]+:\d+:\d+:\s*(?:fatal\s+)?error:\s*(?P<message>[^\n]*)",
            diagnostics,
        ):
            if "file not found" not in match.group("message"):
                return "candidate_contract"
    return "target_build"


def _build_receipt_reusable(
    receipt: Mapping[str, Any] | None,
    *,
    contract: Mapping[str, str],
    source_kind: str | None,
    source_stage_digest: str,
    candidate_digest: str,
    authored_cmake_sha256: str,
    target: Mapping[str, Any],
    build_identity: Mapping[str, Any],
    workspace: Path,
    build_attempt_id: str,
    build_mode: str = "controlled_authored_cmake",
) -> bool:
    """Accept only a PASS receipt for identical inputs and this invocation."""
    return (
        isinstance(receipt, Mapping)
        and _receipt_payload_valid(receipt, workspace)
        and receipt.get("schema") == _contract_value(contract, "schema")
        and receipt.get("status") == "PASS"
        # A fresh O5 invocation receives a fresh attempt token.  A PASS from a
        # prior invocation must therefore be rebuilt, rather than silently
        # reused as if it were produced by the current controlled build.
        and receipt.get("build_attempt_id") == build_attempt_id
        and receipt.get("source_kind") == source_kind
        and receipt.get("source_stage_digest") == source_stage_digest
        and receipt.get("build_mode") == build_mode
        and receipt.get("returncode") == 0
        and receipt.get("candidate_source_sha256") == candidate_digest
        and _sha_ok(receipt.get("authored_cmake_sha256"))
        and receipt.get("authored_cmake_sha256") == authored_cmake_sha256
        and receipt.get("target") == dict(target)
        and receipt.get("build_identity") == dict(build_identity)
        and (
            source_kind != TILELANG2ASCENDC_SOURCE_KIND
            or (
                receipt.get("candidate_independence_gate") == "PASS"
                and receipt.get("candidate_independence_schema")
                == TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA
                and isinstance(receipt.get("candidate_independence_proof"), Mapping)
            )
        )
        and (Path(workspace) / "kernel" / "build").is_dir()
    )


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_runtime_marker(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "unavailable"
    # Keep receipts bounded while retaining the cgroup path needed for an
    # independent container-vs-host audit.
    return value.strip()[:4096]


def _local_runtime_observation() -> dict[str, Any]:
    """Capture non-secret process/container identity for a local receipt."""
    hostname = socket.gethostname()
    etc_hostname = _read_runtime_marker(Path("/etc/hostname"))
    cgroup = _read_runtime_marker(Path("/proc/1/cgroup"))
    return {
        "probe": "python-process",
        "hostname": hostname,
        "etc_hostname": etc_hostname,
        "proc_1_cgroup": cgroup,
        "proc_1_cgroup_sha256": hashlib.sha256(cgroup.encode("utf-8")).hexdigest(),
    }


def _local_runtime_container(target: _Target) -> str | None:
    """Return the outer Docker container for an explicit local target.

    ``A5_CONTAINER=local`` describes the transport mode, not the shell's
    actual namespace.  When the orchestrator is launched through
    ``docker exec`` (for example in ``cjm_cann2``), persist that name so a
    build/preflight receipt can distinguish a real NPU container from a Mac or
    bare host controller.  The field is optional for native local execution.
    """
    if target.container.strip().lower() != "local":
        return None
    value = str(
        target.env.get("A5_LOCAL_RUNTIME_CONTAINER")
        or os.environ.get("CANNBOT_LOCAL_RUNTIME_CONTAINER", "")
    ).strip()
    if not value:
        return None
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise TargetTransportError(
            "A5_LOCAL_RUNTIME_CONTAINER must be a Docker container name"
        )
    return value


def _target_identity(target: _Target) -> dict[str, Any]:
    identity = {
        "name": target.name,
        "host": target.host,
        "container": target.container,
        "host_mode": target.host_mode,
        "cann_path": target.cann_path,
        "benchmark_root": target.benchmark_root,
        "visible_device": target.visible_device,
        "configured_soc": a5_soc_version(target.env),
    }
    runtime_container = _local_runtime_container(target)
    if runtime_container is not None:
        identity["runtime_container"] = runtime_container
        identity["runtime_observation"] = _local_runtime_observation()
    return identity


def _path_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "resolved_path": None,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "executable": False,
            "sha256": None,
        }
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
        exists = candidate.exists()
        is_file = candidate.is_file()
        is_dir = candidate.is_dir()
        executable = os.access(candidate, os.X_OK)
        digest = _sha(candidate) if is_file else None
    except OSError:
        resolved = candidate.resolve(strict=False)
        exists = False
        is_file = False
        is_dir = False
        executable = False
        digest = None
    return {
        "path": str(candidate),
        "resolved_path": str(resolved),
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "executable": executable,
        "sha256": digest,
    }


def _candidate_build_identity(
    target: _Target,
    soc: str,
    observed_soc: str | None,
    python: Path,
    *toolchain: Path | None,
    identity_schema: str,
) -> dict[str, Any]:
    """Describe every target/toolchain/runtime input that can affect a build.

    ``toolchain`` groups the two related toolchain paths -- ``(build_script,
    cann_set_env)`` -- into one variadic parameter instead of two more
    positional ones, while keeping the historical positional call form working
    for every existing caller.

    This deliberately records allowlisted environment keys only.  The target
    environment can contain credentials, so persisting the complete mapping
    would turn a build receipt into a secret-bearing artifact.
    """
    if len(toolchain) != 2:
        raise TargetTransportError(
            "candidate build identity needs exactly (build_script, cann_set_env)"
        )
    script, set_env = toolchain
    return {
        "schema": identity_schema,
        "target": _target_identity(target),
        "soc": soc,
        "hardware": {
            "visible_device": target.visible_device,
            "observed_soc": observed_soc,
        },
        "toolchain": {
            "build_script": _path_identity(script),
            "cann_path": _path_identity(Path(target.cann_path).expanduser()),
            "set_env": _path_identity(set_env),
        },
        "runtime": {
            "python": _path_identity(python),
            "configured": {
                key: target.env.get(key) for key in _BUILD_IDENTITY_CONFIG_KEYS
            },
            "process": {
                key: os.environ.get(key) for key in _BUILD_IDENTITY_PROCESS_KEYS
            },
        },
    }


def _write_candidate_build_receipt(
    workspace: Path,
    contract: Mapping[str, str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist a controlled-build receipt with a private HMAC."""
    evidence = _real_directory(Path(workspace) / npubench_runner.EVIDENCE_DIRNAME, "evidence directory")
    path = evidence / _contract_value(contract, "receipt_filename")
    persisted = dict(payload)
    persisted.pop("payload_sha256", None)
    persisted.pop("receipt_auth_hmac", None)
    persisted["payload_sha256"] = _receipt_payload_sha256(persisted)
    persisted["receipt_auth_hmac"] = _receipt_auth_hmac(workspace, persisted)
    _atomic_json(path, persisted)
    result = dict(persisted)
    result["receipt_path"] = _contract_value(contract, "receipt_path")
    result["receipt_sha256"] = _sha(path)
    return result


def _receipt_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical receipt content excluding the self-referential field."""
    normalized = dict(payload)
    normalized.pop("payload_sha256", None)
    normalized.pop("receipt_auth_hmac", None)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TargetTransportError(f"build receipt is not JSON canonicalizable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _receipt_auth_key_path(workspace: Path) -> Path:
    """Return a private key path outside the worker-visible workspace."""
    resolved_workspace = Path(workspace).resolve()
    key_dir = resolved_workspace.parent / _RECEIPT_KEY_DIR
    if key_dir.exists() or key_dir.is_symlink():
        metadata = key_dir.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise TargetTransportError("candidate receipt key directory is unsafe")
    else:
        key_dir.mkdir(mode=0o700, exist_ok=False)
    key_name = hashlib.sha256(str(resolved_workspace).encode("utf-8")).hexdigest() + ".key"
    return key_dir / key_name


def _unsafe_receipt_key_metadata(metadata: os.stat_result) -> bool:
    """Reject a key that is a link, not a lone regular file, or group/world readable."""
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return True
    return metadata.st_nlink != 1 or bool(metadata.st_mode & 0o077)


def _receipt_auth_key(workspace: Path) -> bytes:
    """Load or create the target-run secret used to authenticate receipts."""
    path = _receipt_auth_key_path(workspace)
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if _unsafe_receipt_key_metadata(metadata):
            raise TargetTransportError("candidate receipt key is unsafe")
        key = path.read_bytes()
        if len(key) != 32:
            raise TargetTransportError("candidate receipt key has invalid length")
        return key
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _receipt_auth_key(workspace)
    try:
        key = secrets.token_bytes(32)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())
        return key
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _receipt_auth_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = dict(payload)
    normalized.pop("receipt_auth_hmac", None)
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TargetTransportError(f"build receipt is not JSON canonicalizable: {exc}") from exc


def _receipt_auth_hmac(workspace: Path, payload: Mapping[str, Any]) -> str:
    return hmac.new(
        _receipt_auth_key(workspace), _receipt_auth_bytes(payload), hashlib.sha256
    ).hexdigest()


def _receipt_payload_valid(
    payload: Mapping[str, Any], workspace: Path | None = None
) -> bool:
    actual = payload.get("payload_sha256")
    if not _sha_ok(actual):
        return False
    try:
        if actual != _receipt_payload_sha256(payload):
            return False
        if workspace is None:
            return True
        auth = payload.get("receipt_auth_hmac")
        return (
            isinstance(auth, str)
            and _sha_ok(auth)
            and hmac.compare_digest(auth, _receipt_auth_hmac(workspace, payload))
        )
    except (TargetTransportError, OSError):
        # A malformed or non-canonical persisted receipt is not reusable.  It
        # must trigger a fresh controlled build rather than aborting O5 before
        # the build boundary can repair the evidence.
        return False

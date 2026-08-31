# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Immutable staging for original NPUKernelBench task bundles.

This module deliberately understands only the old, file-pair NPUKernelBench
contract: one task ``.py`` plus exactly one same-stem ``.json`` or ``.jsonl``
sidecar.  It does *not* translate the task into ``model.py``/``test.py`` and it
does not import task code.  The runner is responsible for executing the frozen
bundle later; this boundary only validates, copies, hashes, and verifies bytes.
"""
from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping


REFERENCE_INPUTS_DIR = "reference_inputs"
NPUBENCH_DIR = "npubench"
INCOMING_DIR = ".incoming"
BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
BUNDLE_SCHEMA = "cannbot.npubench_inputs/v1"
REFERENCE_SCHEMA_VERSION = 3
NPUBENCH_SOURCE = "npubench"
SEMANTIC_BINDING = "npubench_old_format_task_bundle"
RUNNER_CONTRACT_VERSION = "npubench/v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class NpubenchInputError(ValueError):
    """Raised when an NPUKernelBench input bundle is unsafe or ambiguous."""


@dataclass(frozen=True)
class NormalizedNpubenchArgs:
    """Validated task/root arguments before workspace mutation."""

    task_path: Path
    root_path: Path
    task_relative_path: Path
    sidecar_path: Path
    sidecar_relative_path: Path
    sidecar_encoding: str


@dataclass(frozen=True)
class NpubenchStage:
    """Published, content-addressed original NPUKernelBench bundle."""

    root: Path
    task_path: Path
    sidecar_path: Path
    manifest_path: Path
    bundle_sha256: str
    task_sha256: str
    sidecar_sha256: str
    manifest_sha256: str
    task_relative_path: str
    sidecar_relative_path: str
    sidecar_encoding: str

    def state_block(self) -> dict[str, Any]:
        """Return the complete immutable durable reference binding."""
        # The state is portable with its workspace.  Do not persist an
        # absolute controller path merely because the published stage happens
        # to have one in this process; consumers derive the canonical bundle
        # location from ``bundle_sha256`` and verify it again before use.
        manifest_relative = (
            Path(REFERENCE_INPUTS_DIR)
            / NPUBENCH_DIR
            / self.bundle_sha256
            / BUNDLE_MANIFEST_FILENAME
        )
        return {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "source": NPUBENCH_SOURCE,
            "semantic_binding": SEMANTIC_BINDING,
            "runner_contract_version": RUNNER_CONTRACT_VERSION,
            "bundle_manifest_path": manifest_relative.as_posix(),
            "bundle_manifest_sha256": self.manifest_sha256,
            "bundle_sha256": self.bundle_sha256,
            "task_relative_path": self.task_relative_path,
            "task_sha256": self.task_sha256,
            "sidecar_relative_path": self.sidecar_relative_path,
            "sidecar_sha256": self.sidecar_sha256,
            "sidecar_encoding": self.sidecar_encoding,
        }


_IMMUTABLE_STATE_FIELDS = (
    "schema_version",
    "source",
    "semantic_binding",
    "runner_contract_version",
    "bundle_manifest_path",
    "bundle_manifest_sha256",
    "bundle_sha256",
    "task_relative_path",
    "task_sha256",
    "sidecar_relative_path",
    "sidecar_sha256",
    "sidecar_encoding",
)


@dataclass(frozen=True)
class _StagedBundleDigests:
    """The four digests that identify one published NPUKernelBench bundle."""

    bundle_sha256: str
    task_sha256: str
    sidecar_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class _StagePaths:
    """Controller-owned directories used to publish one staged bundle."""

    workspace_root: Path
    provider_root: Path
    incoming_root: Path
    temp_root: Path


def _discover_task_sidecar(task_path: Path) -> Path:
    """Resolve the task's single same-stem ``.json``/``.jsonl`` sidecar."""
    sidecars = []
    for suffix in (".json", ".jsonl"):
        candidate = task_path.with_suffix(suffix)
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NpubenchInputError(
                f"cannot inspect NPUKernelBench sidecar {candidate}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise NpubenchInputError(
                f"NPUKernelBench sidecar must be a non-symlink regular file: {candidate}"
            )
        sidecars.append(candidate)
    if not sidecars:
        raise NpubenchInputError(
            "--npubench-task requires exactly one same-stem .json or .jsonl sidecar"
        )
    if len(sidecars) != 1:
        raise NpubenchInputError(
            "NPUKernelBench task has both .json and .jsonl sidecars; select an unambiguous task"
        )
    return _validate_regular_file(sidecars[0], "NPUKernelBench sidecar")


def validate_cli_npubench_args(
    npubench_task: Path | None, npubench_root: Path | None = None
) -> NormalizedNpubenchArgs | None:
    """Validate the original NPUKernelBench task/root matrix without writing.

    ``None`` means neither provider flag was supplied.  A root without a task,
    a non-Python task, a symlink, an outside-root task, ambiguous sidecars, and
    non-JSON/JSONL sidecar contents are all hard errors.  The byte parser only
    recognizes content; it never materializes values or synthesizes inputs.
    """
    if npubench_task is None:
        if npubench_root is not None:
            raise NpubenchInputError("--npubench-root requires --npubench-task")
        return None

    task_path = _validate_regular_file(Path(npubench_task), "--npubench-task")
    if task_path.suffix != ".py":
        raise NpubenchInputError(
            f"--npubench-task must name a .py file: {task_path}"
        )
    if npubench_root is None:
        root_path = _validate_regular_directory(task_path.parent, "--npubench-root")
    else:
        root_path = _validate_regular_directory(Path(npubench_root), "--npubench-root")
    try:
        task_relative_path = task_path.relative_to(root_path)
    except ValueError as exc:
        raise NpubenchInputError(
            "--npubench-task must be contained by --npubench-root"
        ) from exc

    sidecar_path = _discover_task_sidecar(task_path)
    try:
        sidecar_relative_path = sidecar_path.relative_to(root_path)
    except ValueError as exc:  # Defensive after sibling discovery.
        raise NpubenchInputError("NPUKernelBench sidecar escapes --npubench-root") from exc
    sidecar_encoding = _recognize_sidecar_encoding(sidecar_path)
    return NormalizedNpubenchArgs(
        task_path=task_path,
        root_path=root_path,
        task_relative_path=task_relative_path,
        sidecar_path=sidecar_path,
        sidecar_relative_path=sidecar_relative_path,
        sidecar_encoding=sidecar_encoding,
    )


def _npubench_source_inventory(args: NormalizedNpubenchArgs) -> list[tuple[Path, Path]]:
    """Collect the source closure and reject a root that cannot be staged."""
    inventory = _collect_source_inventory(args.root_path)
    inventory_by_rel = {rel: path for rel, path in inventory}
    if Path(BUNDLE_MANIFEST_FILENAME) in inventory_by_rel:
        raise NpubenchInputError(
            "NPUKernelBench source root contains reserved top-level "
            f"{BUNDLE_MANIFEST_FILENAME}; choose a narrower root"
        )
    if args.task_relative_path not in inventory_by_rel:
        raise NpubenchInputError("NPUKernelBench task disappeared during source inventory")
    if args.sidecar_relative_path not in inventory_by_rel:
        raise NpubenchInputError("NPUKernelBench sidecar disappeared during source inventory")
    return inventory


def _npubench_stage_roots(workspace: Path) -> tuple[Path, Path, Path]:
    """Create the controller-owned staging hierarchy one direct child at a time.

    A recursive ``mkdir(parents=True)`` would silently traverse a pre-existing
    ``reference_inputs`` symlink and place staged user bytes outside the
    workspace.  This is a practical accidental-write guard, not a claim of
    hostile same-UID sandboxing.
    """
    workspace_root = _workspace_root(workspace)
    reference_inputs_root = _ensure_real_child_directory(
        workspace_root, REFERENCE_INPUTS_DIR, mode=0o700, private=False
    )
    provider_root = _ensure_real_child_directory(
        reference_inputs_root, NPUBENCH_DIR, mode=0o700, private=True
    )
    incoming_root = _ensure_real_child_directory(
        provider_root, INCOMING_DIR, mode=0o700, private=True
    )
    return workspace_root, provider_root, incoming_root


def _stage_inventory_copy(
    temp_root: Path, inventory: list[tuple[Path, Path]]
) -> tuple[list[dict[str, Any]], dict[Path, tuple[str, int]]]:
    """Copy every source file into the temporary bundle, hashing as it goes."""
    entries: list[dict[str, Any]] = []
    digests: dict[Path, tuple[str, int]] = {}
    for relative_path, source_path in inventory:
        destination = temp_root / relative_path
        _create_parent_directories(temp_root, destination.parent)
        digest, size = _copy_regular_file_with_digest(source_path, destination)
        digests[relative_path] = (digest, size)
        entries.append(
            {"path": relative_path.as_posix(), "sha256": digest, "size": size}
        )
    return entries, digests


def _write_bundle_manifest(
    temp_root: Path, args: NormalizedNpubenchArgs, entries: list[dict[str, Any]]
) -> tuple[str, str]:
    """Write the fixed bundle manifest and return its bundle/manifest digests."""
    bundle_sha256 = _bundle_digest(
        entries,
        task_relative_path=args.task_relative_path.as_posix(),
        sidecar_relative_path=args.sidecar_relative_path.as_posix(),
        sidecar_encoding=args.sidecar_encoding,
    )
    manifest_payload = {
        "schema": BUNDLE_SCHEMA,
        "bundle_sha256": bundle_sha256,
        "task_relative_path": args.task_relative_path.as_posix(),
        "sidecar_relative_path": args.sidecar_relative_path.as_posix(),
        "sidecar_encoding": args.sidecar_encoding,
        "files": entries,
    }
    manifest_path = temp_root / BUNDLE_MANIFEST_FILENAME
    _write_json_exclusive(manifest_path, manifest_payload)
    return bundle_sha256, _file_sha256(manifest_path)


def _existing_stage_for_args(
    workspace_root: Path,
    args: NormalizedNpubenchArgs,
    digests: _StagedBundleDigests,
) -> NpubenchStage:
    """Rebind an identical bundle that another writer published first."""
    return _existing_stage(
        workspace_root,
        bundle_sha256=digests.bundle_sha256,
        task_relative_path=args.task_relative_path.as_posix(),
        sidecar_relative_path=args.sidecar_relative_path.as_posix(),
        sidecar_encoding=args.sidecar_encoding,
        task_sha256=digests.task_sha256,
        sidecar_sha256=digests.sidecar_sha256,
        manifest_sha256=digests.manifest_sha256,
    )


def _replace_temp_with_final(paths: _StagePaths, final_root: Path) -> bool:
    """Move the temporary bundle into place; ``False`` if it already exists."""
    _fsync_tree(paths.temp_root)
    _fsync_directory(paths.incoming_root)
    try:
        os.replace(paths.temp_root, final_root)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            return False
        raise NpubenchInputError(
            f"could not atomically publish NPUKernelBench bundle: {exc}"
        ) from exc
    return True


def _sealed_stage(
    paths: _StagePaths, args: NormalizedNpubenchArgs, digests: _StagedBundleDigests
) -> NpubenchStage:
    """Seal the freshly published bundle and return its verified binding."""
    final_root = paths.provider_root / digests.bundle_sha256
    # macOS requires write permission on the moved directory itself for
    # rename(2).  The final bundle is still unreachable to a runner
    # until the state binding below is written, so seal it before any
    # caller can bind it as reference truth.
    _make_stage_read_only(final_root)
    _fsync_directory(paths.provider_root)
    stage = NpubenchStage(
        root=final_root,
        task_path=final_root / args.task_relative_path,
        sidecar_path=final_root / args.sidecar_relative_path,
        manifest_path=final_root / BUNDLE_MANIFEST_FILENAME,
        bundle_sha256=digests.bundle_sha256,
        task_sha256=digests.task_sha256,
        sidecar_sha256=digests.sidecar_sha256,
        manifest_sha256=digests.manifest_sha256,
        task_relative_path=args.task_relative_path.as_posix(),
        sidecar_relative_path=args.sidecar_relative_path.as_posix(),
        sidecar_encoding=args.sidecar_encoding,
    )
    valid, reason, _manifest = verify_npubench_stage(paths.workspace_root, stage.state_block())
    if not valid:
        raise NpubenchInputError(
            f"published NPUKernelBench bundle is invalid: {reason}"
        )
    return stage


def _publish_or_rebind_stage(
    paths: _StagePaths, args: NormalizedNpubenchArgs, digests: _StagedBundleDigests
) -> tuple[NpubenchStage, bool]:
    """Publish the staged bundle, or rebind the identical one a racer published."""
    final_root = paths.provider_root / digests.bundle_sha256
    with _bundle_publish_lock(paths.provider_root):
        if final_root.exists() or final_root.is_symlink():
            return _existing_stage_for_args(paths.workspace_root, args, digests), False
        if not _replace_temp_with_final(paths, final_root):
            return _existing_stage_for_args(paths.workspace_root, args, digests), False
        return _sealed_stage(paths, args, digests), True


def stage_npubench_inputs(
    workspace: Path,
    *,
    npubench_task: Path,
    npubench_root: Path | None = None,
) -> NpubenchStage:
    """Atomically freeze an old-format NPUKernelBench source-root closure.

    All ordinary files below the declared root are copied under their original
    relative names.  This is intentionally broader than a task/sidecar pair:
    old tasks can use sibling helpers and package relative imports.  Symlinks
    and special files are rejected rather than copied or dereferenced.
    """
    args = validate_cli_npubench_args(npubench_task, npubench_root)
    if args is None:  # Defensive: the keyword-only task is required.
        raise NpubenchInputError("NPUKernelBench task is required")
    inventory = _npubench_source_inventory(args)
    workspace_root, provider_root, incoming_root = _npubench_stage_roots(workspace)
    temp_root = incoming_root / uuid.uuid4().hex
    _mkdir_exclusive(temp_root, mode=0o700)
    paths = _StagePaths(
        workspace_root=workspace_root,
        provider_root=provider_root,
        incoming_root=incoming_root,
        temp_root=temp_root,
    )
    published = False
    try:
        entries, file_digests = _stage_inventory_copy(temp_root, inventory)
        task_digest, _ = file_digests[args.task_relative_path]
        sidecar_digest, _ = file_digests[args.sidecar_relative_path]
        bundle_sha256, manifest_sha256 = _write_bundle_manifest(temp_root, args, entries)
        digests = _StagedBundleDigests(
            bundle_sha256=bundle_sha256,
            task_sha256=task_digest,
            sidecar_sha256=sidecar_digest,
            manifest_sha256=manifest_sha256,
        )
        stage, published = _publish_or_rebind_stage(paths, args, digests)
        return stage
    finally:
        if not published and temp_root.exists():
            _remove_owned_incoming_tree(temp_root, incoming_root)


def bind_npubench_state(
    state: MutableMapping[str, Any], stage: NpubenchStage
) -> dict[str, Any]:
    """Bind a staged NPUKernelBench bundle without allowing reference drift."""
    if not isinstance(state, MutableMapping):
        raise NpubenchInputError("durable state must be a mutable JSON object")
    expected = stage.state_block()
    existing = state.get("reference")
    if existing is None:
        state["reference"] = copy.deepcopy(expected)
        return state  # type: ignore[return-value]
    if not isinstance(existing, Mapping):
        raise NpubenchInputError("durable state reference block is not an object")
    for field in _IMMUTABLE_STATE_FIELDS:
        if existing.get(field) != expected[field]:
            raise NpubenchInputError(
                "durable state already binds a different NPUKernelBench bundle: "
                f"field {field!r} differs"
            )
    state["reference"] = copy.deepcopy(dict(existing))
    return state  # type: ignore[return-value]


@dataclass(frozen=True)
class _ReferenceBinding:
    """The parsed, already-validated fields of a durable reference block."""

    bundle_sha256: str
    task_relative: Path
    sidecar_relative: Path
    sidecar_encoding: str
    task_sha256: str
    sidecar_sha256: str


def _parse_reference_binding(
    reference_block: Mapping[str, Any]
) -> tuple[_ReferenceBinding | None, str]:
    """Validate a durable reference block and return its parsed binding."""
    if reference_block.get("source") != NPUBENCH_SOURCE:
        return None, "reference.source is not npubench"
    if reference_block.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        return None, "unsupported NPUKernelBench state schema"
    if reference_block.get("semantic_binding") != SEMANTIC_BINDING:
        return None, "NPUKernelBench semantic binding mismatch"
    if reference_block.get("runner_contract_version") != RUNNER_CONTRACT_VERSION:
        return None, "unsupported NPUKernelBench runner contract"
    bundle_sha256 = reference_block.get("bundle_sha256")
    if not _is_sha256(bundle_sha256):
        return None, "NPUKernelBench bundle_sha256 is invalid"
    task_relative = _safe_relative_path(reference_block.get("task_relative_path"))
    sidecar_relative = _safe_relative_path(reference_block.get("sidecar_relative_path"))
    if task_relative is None or task_relative.suffix != ".py":
        return None, "NPUKernelBench task_relative_path is invalid"
    if sidecar_relative is None or sidecar_relative.suffix not in {".json", ".jsonl"}:
        return None, "NPUKernelBench sidecar_relative_path is invalid"
    if task_relative.with_suffix(sidecar_relative.suffix) != sidecar_relative:
        return None, "NPUKernelBench sidecar is not the task's same-stem companion"
    sidecar_encoding = reference_block.get("sidecar_encoding")
    if sidecar_encoding not in {"json", "jsonl"}:
        return None, "NPUKernelBench sidecar_encoding is invalid"
    for field in ("task_sha256", "sidecar_sha256", "bundle_manifest_sha256"):
        if not _is_sha256(reference_block.get(field)):
            return None, f"NPUKernelBench {field} is invalid"
    binding = _ReferenceBinding(
        bundle_sha256=str(bundle_sha256),
        task_relative=task_relative,
        sidecar_relative=sidecar_relative,
        sidecar_encoding=str(sidecar_encoding),
        task_sha256=str(reference_block.get("task_sha256")),
        sidecar_sha256=str(reference_block.get("sidecar_sha256")),
    )
    return binding, ""


def _verified_stage_root(workspace: Path, bundle_sha256: str) -> Path:
    """Resolve the read-only staged bundle directory for a bound digest."""
    workspace_root = _existing_workspace_root(workspace)
    reference_inputs_root = workspace_root / REFERENCE_INPUTS_DIR
    _require_directory_no_symlink(
        reference_inputs_root, "NPUKernelBench reference input root"
    )
    provider_root = reference_inputs_root / NPUBENCH_DIR
    _require_directory_no_symlink(provider_root, "NPUKernelBench input root")
    stage_root = provider_root / bundle_sha256
    _require_directory_no_symlink(stage_root, "NPUKernelBench bundle")
    for directory in _collect_tree_directories(stage_root):
        _require_read_only(directory, "NPUKernelBench staged directory")
    return stage_root


def _staged_manifest_violation(
    stage_root: Path, reference_block: Mapping[str, Any], binding: _ReferenceBinding
) -> tuple[str, dict[str, Any]]:
    """Load the staged manifest and bind it to durable state; empty when valid."""
    manifest_path = stage_root / BUNDLE_MANIFEST_FILENAME
    if not _has_rebasable_manifest_suffix(
        reference_block.get("bundle_manifest_path"), binding.bundle_sha256
    ):
        return "state bundle_manifest_path does not name the fixed staged manifest", {}
    _require_regular_file_no_symlink(manifest_path, "NPUKernelBench bundle manifest")
    _require_read_only(manifest_path, "NPUKernelBench bundle manifest")
    if _file_sha256(manifest_path) != reference_block.get("bundle_manifest_sha256"):
        return "NPUKernelBench bundle manifest digest differs from durable state", {}
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        return "unsupported NPUKernelBench bundle manifest schema", manifest
    if manifest.get("bundle_sha256") != binding.bundle_sha256:
        return "NPUKernelBench manifest bundle digest mismatch", manifest
    if manifest.get("task_relative_path") != binding.task_relative.as_posix():
        return "NPUKernelBench manifest task path mismatch", manifest
    if manifest.get("sidecar_relative_path") != binding.sidecar_relative.as_posix():
        return "NPUKernelBench manifest sidecar path mismatch", manifest
    if manifest.get("sidecar_encoding") != binding.sidecar_encoding:
        return "NPUKernelBench manifest sidecar encoding mismatch", manifest
    return "", manifest


def _staged_bytes_violation(
    stage_root: Path, binding: _ReferenceBinding, manifest: dict[str, Any]
) -> str:
    """Check staged bytes and inventory against the manifest and durable state."""
    files = _validate_manifest_files(manifest.get("files"))
    expected_bundle = _bundle_digest(
        files,
        task_relative_path=binding.task_relative.as_posix(),
        sidecar_relative_path=binding.sidecar_relative.as_posix(),
        sidecar_encoding=binding.sidecar_encoding,
    )
    if expected_bundle != binding.bundle_sha256:
        return "NPUKernelBench bundle digest does not bind manifest inventory"
    disk_inventory = _collect_staged_inventory(stage_root)
    if disk_inventory != files:
        return "NPUKernelBench staged inventory/hash mismatch"
    file_map = {Path(entry["path"]): entry for entry in files}
    task_entry = file_map.get(binding.task_relative)
    sidecar_entry = file_map.get(binding.sidecar_relative)
    if task_entry is None or sidecar_entry is None:
        return "NPUKernelBench task or sidecar missing from bundle manifest"
    if task_entry["sha256"] != binding.task_sha256:
        return "NPUKernelBench task digest differs from durable state"
    if sidecar_entry["sha256"] != binding.sidecar_sha256:
        return "NPUKernelBench sidecar digest differs from durable state"
    detected_encoding = _recognize_sidecar_encoding(stage_root / binding.sidecar_relative)
    if detected_encoding != binding.sidecar_encoding:
        return "NPUKernelBench sidecar content encoding changed"
    return ""


def verify_npubench_stage(
    workspace: Path, reference: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Verify a published NPUKernelBench bundle against its durable binding.

    The signature is intentionally stable for the future ``npubench_runner``:
    callers receive only checked manifest data and need not re-open an external
    user path.  A malformed, partial, modified, or symlinked stage fails closed.
    """
    reference_block = _reference_block(reference)
    if reference_block is None:
        return False, "NPUKernelBench reference block is missing or malformed", {}
    binding, binding_error = _parse_reference_binding(reference_block)
    if binding is None:
        return False, binding_error, {}
    manifest: dict[str, Any] = {}
    try:
        stage_root = _verified_stage_root(workspace, binding.bundle_sha256)
        manifest_error, manifest = _staged_manifest_violation(
            stage_root, reference_block, binding
        )
        if manifest_error:
            return False, manifest_error, manifest
        bytes_error = _staged_bytes_violation(stage_root, binding, manifest)
        if bytes_error:
            return False, bytes_error, manifest
    except NpubenchInputError as exc:
        return False, str(exc), {}
    # json.JSONDecodeError is a ValueError subclass, so listing it here would
    # capture the same exception class twice.
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"NPUKernelBench stage validation failed: {type(exc).__name__}: {exc}", {}
    return True, f"NPUKernelBench bundle verified ({binding.bundle_sha256[:12]})", manifest


def atomic_write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    """Atomically publish a complete durable state after provider staging.

    It is intentionally public so the command handler can pair a published
    bundle with one all-or-nothing state write.  It never creates or trusts an
    alternate state location.
    """
    workspace_root = _existing_workspace_root(workspace)
    state_path = workspace_root / ".opgen_state.json"
    temporary = workspace_root / f".{state_path.name}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(temporary, flags, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, state_path)
        _fsync_directory(workspace_root)
    except OSError as exc:
        raise NpubenchInputError(f"cannot atomically publish durable state: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _reference_block(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidate: Any = value.get("reference") if "reference" in value else value
    return candidate if isinstance(candidate, Mapping) else None


def _validate_regular_file(path_value: Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    if _has_control_characters(path):
        raise NpubenchInputError(f"{label} path contains a forbidden control character")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NpubenchInputError(f"{label} cannot be inspected: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NpubenchInputError(f"{label} must be a non-symlink regular file: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpubenchInputError(f"{label} cannot be resolved: {path}: {exc}") from exc


def _validate_regular_directory(path_value: Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    if _has_control_characters(path):
        raise NpubenchInputError(f"{label} path contains a forbidden control character")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NpubenchInputError(f"{label} cannot be inspected: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise NpubenchInputError(f"{label} must be a non-symlink directory: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpubenchInputError(f"{label} cannot be resolved: {path}: {exc}") from exc


def _collect_source_inventory(root: Path) -> list[tuple[Path, Path]]:
    """Walk a source closure without following or silently ignoring unsafe nodes."""
    pending = [root]
    files: list[tuple[Path, Path]] = []
    while pending:
        directory = pending.pop()
        _require_directory_no_symlink(directory, "NPUKernelBench source directory")
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise NpubenchInputError(
                f"cannot enumerate NPUKernelBench source directory {directory}: {exc}"
            ) from exc
        for path in children:
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise NpubenchInputError(f"cannot inspect source path {path}: {exc}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise NpubenchInputError(
                    f"NPUKernelBench source closure cannot contain symlink: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append((path.relative_to(root), path))
            else:
                raise NpubenchInputError(
                    f"NPUKernelBench source closure cannot contain non-regular file: {path}"
                )
    return sorted(files, key=lambda item: item[0].as_posix())


def _collect_staged_inventory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative_path, path in _collect_source_inventory(root):
        if relative_path == Path(BUNDLE_MANIFEST_FILENAME):
            continue
        _require_read_only(path, f"NPUKernelBench staged file {relative_path}")
        entries.append(
            {
                "path": relative_path.as_posix(),
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
        )
    return entries


def _collect_tree_directories(root: Path) -> list[Path]:
    """Enumerate every staged directory without following a symlink."""
    pending = [root]
    directories: list[Path] = []
    while pending:
        directory = pending.pop()
        _require_directory_no_symlink(directory, "NPUKernelBench staged directory")
        directories.append(directory)
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise NpubenchInputError(
                f"cannot enumerate NPUKernelBench staged directory {directory}: {exc}"
            ) from exc
        for child in children:
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise NpubenchInputError(
                    f"cannot inspect NPUKernelBench staged path {child}: {exc}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise NpubenchInputError(
                    f"NPUKernelBench staged bundle cannot contain symlink: {child}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
    return directories


def _recognize_sidecar_encoding(path: Path) -> str:
    """Recognize JSON versus JSONL strictly from bytes, preserving those bytes."""
    try:
        text = _read_regular_bytes(path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NpubenchInputError(
            f"NPUKernelBench sidecar must be UTF-8 JSON/JSONL: {path}"
        ) from exc
    if not text.strip():
        raise NpubenchInputError(f"NPUKernelBench sidecar is empty: {path}")
    if path.suffix == ".json":
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            pass
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        raise NpubenchInputError(f"NPUKernelBench sidecar is empty: {path}")
    try:
        for line in nonempty_lines:
            json.loads(line)
    except json.JSONDecodeError as exc:
        raise NpubenchInputError(
            f"NPUKernelBench sidecar is neither strict JSON nor JSONL: {path}: {exc}"
        ) from exc
    return "jsonl"


def _workspace_root(workspace: Path) -> Path:
    path = Path(workspace).expanduser()
    if _has_control_characters(path):
        raise NpubenchInputError("workspace path contains a forbidden control character")
    try:
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpubenchInputError(f"workspace cannot be prepared: {path}: {exc}") from exc
    _require_directory_no_symlink(resolved, "workspace")
    return resolved


def _existing_workspace_root(workspace: Path) -> Path:
    path = Path(workspace).expanduser()
    if _has_control_characters(path):
        raise NpubenchInputError("workspace path contains a forbidden control character")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpubenchInputError(f"workspace is missing or unreadable: {path}: {exc}") from exc
    _require_directory_no_symlink(resolved, "workspace")
    return resolved


def _ensure_real_child_directory(
    parent: Path,
    name: str,
    *,
    mode: int,
    private: bool,
) -> Path:
    """Create/validate one real child without traversing an existing link.

    ``parent`` has already crossed the workspace boundary.  Keeping this as a
    one-level operation makes the stage destination auditable and avoids the
    surprising behaviour of ``mkdir(parents=True)`` through a user-created
    intermediate symlink.  Only the provider's private directories have their
    mode normalized; an existing shared ``reference_inputs`` directory is not
    chmodded by this feature.
    """
    _require_directory_no_symlink(parent, str(parent))
    path = parent / name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=mode)
        except OSError as exc:
            raise NpubenchInputError(f"cannot create {path}: {exc}") from exc
    except OSError as exc:
        raise NpubenchInputError(f"cannot inspect {path}: {exc}") from exc
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise NpubenchInputError(
                f"{path} must be a non-symlink directory"
            )
    _require_directory_no_symlink(path, str(path))
    if not private:
        return path
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise NpubenchInputError(f"cannot set safe permissions on {path}: {exc}") from exc
    return path


def _mkdir_exclusive(path: Path, *, mode: int) -> None:
    try:
        path.mkdir(mode=mode)
    except OSError as exc:
        raise NpubenchInputError(f"cannot create staging directory {path}: {exc}") from exc
    _require_directory_no_symlink(path, "staging directory")


def _create_parent_directories(root: Path, destination_parent: Path) -> None:
    try:
        relative = destination_parent.relative_to(root)
    except ValueError as exc:
        raise NpubenchInputError("staging destination escapes its temporary root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists():
            _require_directory_no_symlink(current, "NPUKernelBench staging directory")
            continue
        try:
            current.mkdir(mode=0o700)
        except OSError as exc:
            raise NpubenchInputError(f"cannot create staging directory {current}: {exc}") from exc


def _require_directory_no_symlink(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NpubenchInputError(f"{label} is missing or unreadable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise NpubenchInputError(f"{label} must be a non-symlink directory: {path}")


def _require_regular_file_no_symlink(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NpubenchInputError(f"{label} is missing or unreadable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NpubenchInputError(f"{label} must be a non-symlink regular file: {path}")


def _require_read_only(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NpubenchInputError(f"{label} is missing or unreadable: {path}: {exc}") from exc
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise NpubenchInputError(f"{label} must be read-only: {path}")


def _open_staged_destination(destination: Path) -> int:
    """Create the staged copy exclusively, owner-readable only."""
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        write_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        write_flags |= os.O_CLOEXEC
    try:
        return os.open(destination, write_flags, 0o600)
    except OSError as exc:
        raise NpubenchInputError(f"cannot create staged NPUKernelBench input {destination}: {exc}") from exc


def _stream_copy_with_digest(source_fd: int, destination_fd: int) -> tuple[str, int]:
    """Copy one open descriptor into another, returning the digest and size."""
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
        _write_all(destination_fd, chunk)
    os.fsync(destination_fd)
    return digest.hexdigest(), size


def _copy_regular_file_with_digest(source: Path, destination: Path) -> tuple[str, int]:
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        read_flags |= os.O_CLOEXEC
    try:
        source_fd = os.open(source, read_flags)
    except OSError as exc:
        raise NpubenchInputError(f"cannot securely open NPUKernelBench input {source}: {exc}") from exc
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise NpubenchInputError(f"NPUKernelBench input is not a regular file: {source}")
        destination_fd = _open_staged_destination(destination)
        try:
            return _stream_copy_with_digest(source_fd, destination_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def _read_regular_bytes(path: Path) -> bytes:
    _require_regular_file_no_symlink(path, str(path))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise NpubenchInputError(f"cannot securely read {path}: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise NpubenchInputError(f"cannot read non-regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise NpubenchInputError("short write while staging NPUKernelBench input")
        view = view[written:]


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise NpubenchInputError(f"cannot create NPUKernelBench bundle manifest {path}: {exc}") from exc
    try:
        _write_all(fd, serialized)
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _bundle_digest(
    entries: list[dict[str, Any]],
    *,
    task_relative_path: str,
    sidecar_relative_path: str,
    sidecar_encoding: str,
) -> str:
    canonical = {
        "schema": BUNDLE_SCHEMA,
        "task_relative_path": task_relative_path,
        "sidecar_relative_path": sidecar_relative_path,
        "sidecar_encoding": sidecar_encoding,
        "files": entries,
    }
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_manifest_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise NpubenchInputError("NPUKernelBench manifest files inventory is missing or empty")
    entries: list[dict[str, Any]] = []
    previous = ""
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping):
            raise NpubenchInputError("NPUKernelBench manifest file entry is not an object")
        relative = _safe_relative_path(entry.get("path"))
        digest = entry.get("sha256")
        size = entry.get("size")
        has_valid_size = isinstance(size, int) and size >= 0
        if relative is None or not _is_sha256(digest) or not has_valid_size:
            raise NpubenchInputError("NPUKernelBench manifest file entry is invalid")
        normalized = relative.as_posix()
        if normalized in seen or normalized <= previous:
            raise NpubenchInputError("NPUKernelBench manifest files are not unique and sorted")
        previous = normalized
        seen.add(normalized)
        entries.append({"path": normalized, "sha256": digest, "size": size})
    return entries


def _safe_relative_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, NpubenchInputError) as exc:
        raise NpubenchInputError(f"NPUKernelBench bundle manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NpubenchInputError("NPUKernelBench bundle manifest must be a JSON object")
    return payload


def _existing_stage(
    workspace_root: Path,
    *,
    bundle_sha256: str,
    task_relative_path: str,
    sidecar_relative_path: str,
    sidecar_encoding: str,
    task_sha256: str,
    sidecar_sha256: str,
    manifest_sha256: str,
) -> NpubenchStage:
    root = workspace_root / REFERENCE_INPUTS_DIR / NPUBENCH_DIR / bundle_sha256
    stage = NpubenchStage(
        root=root,
        task_path=root / task_relative_path,
        sidecar_path=root / sidecar_relative_path,
        manifest_path=root / BUNDLE_MANIFEST_FILENAME,
        bundle_sha256=bundle_sha256,
        task_sha256=task_sha256,
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=manifest_sha256,
        task_relative_path=task_relative_path,
        sidecar_relative_path=sidecar_relative_path,
        sidecar_encoding=sidecar_encoding,
    )
    valid, reason, _manifest = verify_npubench_stage(workspace_root, stage.state_block())
    if not valid:
        raise NpubenchInputError(
            "existing NPUKernelBench bundle conflicts with staged content: " + reason
        )
    return stage


def _make_stage_read_only(root: Path) -> None:
    # Child files before directories: a directory read-only transition must not
    # prevent a later child chmod on stricter filesystems.
    files = _collect_source_inventory(root)
    for _relative, path in files:
        try:
            os.chmod(path, 0o400)
        except OSError as exc:
            raise NpubenchInputError(f"cannot mark staged input read-only: {path}: {exc}") from exc
    manifest_path = root / BUNDLE_MANIFEST_FILENAME
    try:
        os.chmod(manifest_path, 0o400)
    except OSError as exc:
        raise NpubenchInputError(f"cannot mark staged manifest read-only: {manifest_path}: {exc}") from exc
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir() and not path.is_symlink())
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        try:
            os.chmod(path, 0o500)
        except OSError as exc:
            raise NpubenchInputError(f"cannot mark staged directory read-only: {path}: {exc}") from exc


@contextmanager
def _bundle_publish_lock(provider_root: Path) -> Iterator[None]:
    lock_path = provider_root / ".publish.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise NpubenchInputError(f"cannot open NPUKernelBench publish lock: {exc}") from exc
    locked = False
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise NpubenchInputError("NPUKernelBench publish lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    except OSError as exc:
        raise NpubenchInputError(f"cannot lock NPUKernelBench publisher: {exc}") from exc
    finally:
        try:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fsync_tree(root: Path) -> None:
    for _relative, path in _collect_source_inventory(root):
        # Files were fsync'ed at creation; reopening here makes the publication
        # ordering explicit and detects a post-copy mutation before rename.
        _file_sha256(path)
    _fsync_directory(root)
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise NpubenchInputError(f"cannot open directory for durable staging: {path}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise NpubenchInputError(f"filesystem cannot durably stage NPUKernelBench input at {path}: {exc}") from exc
    finally:
        os.close(fd)


def _remove_owned_incoming_tree(path: Path, incoming_root: Path) -> None:
    try:
        if path.parent != incoming_root or path.is_symlink():
            return
        shutil.rmtree(path)
    except OSError:
        pass


def _has_rebasable_manifest_suffix(value: Any, bundle_sha256: str) -> bool:
    if not isinstance(value, str) or _has_control_characters(value):
        return False
    parts = [part for part in value.replace("\\", "/").split("/") if part not in ("", ".")]
    return parts[-4:] == [REFERENCE_INPUTS_DIR, NPUBENCH_DIR, bundle_sha256, BUNDLE_MANIFEST_FILENAME]


def _has_control_characters(path: Path | str) -> bool:
    return any(char in str(path) for char in ("\x00", "\n", "\r"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _SHA256_HEX for char in value)

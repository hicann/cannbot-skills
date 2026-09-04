# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Immutable staging for the TileLang2AscendC project output format.

TileLang2AscendC emits an AscendC custom-op project rather than an ops-nn
ACLNN project and rather than a direct PyBind kernel.  The public port mode
keeps ``aclnn`` in its name for compatibility with the surrounding workflow,
but this adapter intentionally owns a separate schema, stage directory, and
candidate contract.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tokenize
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import fcntl

from source_op import is_safe_op_name, require_safe_op_name


TILELANG2ASCENDC_SOURCE_KIND = "port-aclnn-tilelang2ascendc"
TILELANG2ASCENDC_SOURCE_KIND_ALIAS = "port_aclnn_tilelang2ascendc"
TILELANG2ASCENDC_SOURCE_SCHEMA = "cannbot.tilelang2ascendc_source/v1"
TILELANG2ASCENDC_STAGE_DIR = ".tilelang2ascendc_source"
TILELANG2ASCENDC_MANIFEST = ".tilelang2ascendc_source_manifest.json"
TILELANG2ASCENDC_ROOT_LAYOUT = "model_and_kernel"

_SOURCE_DIRS = frozenset({".git", ".cache", "build", "dist", "__pycache__"})
_SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".asc",
    ".bak", ".cmake", ".csv", ".json", ".jsonl", ".log", ".md", ".py", ".sh", ".txt", ".yaml", ".yml",
})
_BINARY_SUFFIXES = frozenset({
    ".a", ".bin", ".dll", ".dylib", ".elf", ".exe", ".o", ".obj", ".pyc",
    ".so", ".whl", ".zip", ".tar", ".gz",
})
_SAFE_SUFFIXLESS_NAMES = frozenset({"CMakeLists.txt", "Makefile", "LICENSE", "NOTICE"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GENERATED_SUFFIX = re.compile(r"-[0-9a-f]{8,}\Z", re.IGNORECASE)
_MAX_FILES = 4096
_MAX_DEPTH = 32
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_CXX_TEXT_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".asc")
_MANIFEST_BINDINGS = (
    ("schema", TILELANG2ASCENDC_SOURCE_SCHEMA),
    ("kind", TILELANG2ASCENDC_SOURCE_KIND),
    ("root_layout", TILELANG2ASCENDC_ROOT_LAYOUT),
    ("source_arch", "arch35"),
    ("target_arch", "arch35"),
)


class Tilelang2AscendcSourceError(ValueError):
    """Raised when a TileLang2AscendC project is unsafe or incomplete."""


@dataclass(frozen=True)
class Tilelang2AscendcSourceDetection:
    arch: str | None
    supported: bool
    method: str
    confidence: str
    evidence: tuple[str, ...]
    analyzed_paths: tuple[str, ...]

    def state_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        payload["analyzed_paths"] = list(self.analyzed_paths)
        return payload


@dataclass(frozen=True)
class Tilelang2AscendcSourceStage:
    root: Path
    manifest: Path
    digest: str
    file_count: int
    op: str
    detection: Tilelang2AscendcSourceDetection
    files: tuple[dict[str, Any], ...]


def _error(message: str) -> str:
    return f"TILELANG2ASCENDC_SOURCE_INVALID: {message}"


def _real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Tilelang2AscendcSourceError(_error(f"{label} cannot be inspected: {exc}")) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Tilelang2AscendcSourceError(_error(f"{label} must be a real non-symlink directory: {path}"))


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Tilelang2AscendcSourceError(_error(f"{label} cannot be inspected: {exc}")) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise Tilelang2AscendcSourceError(_error(f"{label} must not be a symlink: {path}"))
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Tilelang2AscendcSourceError(_error(f"{label} must be one regular non-hardlinked file: {path}"))
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_text(path: Path, relative: str) -> str:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise Tilelang2AscendcSourceError(_error(f"source file is not UTF-8: {relative}")) from exc
    if b"\0" in raw:
        raise Tilelang2AscendcSourceError(_error(f"binary content is forbidden: {relative}"))
    return text


def _cxx_without_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _python_without_comments(text: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return tokenize.untokenize(
            (token.type, "" if token.type == tokenize.COMMENT else token.string)
            for token in tokens
        )
    except (SyntaxError, tokenize.TokenError):
        # IndentationError is a SyntaxError subclass, so the parent already covers it.
        return re.sub(r"#[^\n]*", "", text)


def _logical_op(root: Path) -> str:
    name = root.name
    if name in {"kernel", "full_task", "simple_task", "output"}:
        name = root.parent.name
    name = _GENERATED_SUFFIX.sub("", name)
    try:
        return require_safe_op_name(name)
    except ValueError as exc:
        raise Tilelang2AscendcSourceError(_error(str(exc))) from exc


def _normalise_root(source_root: Path) -> Path:
    supplied = Path(source_root).expanduser()
    _real_directory(supplied, "source root")
    root = supplied.resolve(strict=True)
    # Accept the generator's full output directory and its kernel subdirectory
    # as a convenience, but always stage the project root when model_new is
    # present.  This also handles full_task/kernel-style caller paths.
    if root.name == "kernel" and (root.parent / "model_new_ascendc.py").is_file():
        root = root.parent
    if (root / "kernel").is_dir() and (root / "model_new_ascendc.py").is_file():
        return root
    if (
        (root / "op_host").is_dir()
        and (root / "op_kernel").is_dir()
        and (root.parent / "model_new_ascendc.py").is_file()
    ):
        return root.parent
    return root


def logical_op_name(source_root: Path) -> str:
    """Return the logical operator name for a full project or ``kernel/`` path."""
    return _logical_op(_normalise_root(source_root))


def _required_layout(root: Path) -> tuple[Path, ...]:
    required = (
        root / "model_new_ascendc.py",
        root / "kernel" / "CMakeLists.txt",
        root / "kernel" / "register.cpp",
        root / "kernel" / "op_host",
        root / "kernel" / "op_kernel",
    )
    for path in required:
        label = path.relative_to(root).as_posix()
        if path.is_dir():
            _real_directory(path, label)
        else:
            _regular(path, label)
    return required


def _inventory_guard_path(relative: Path, *, manifest_scoped: bool) -> None:
    """Reject reserved or sidecar names before a tree entry is inspected."""
    if not manifest_scoped and relative == Path(TILELANG2ASCENDC_MANIFEST):
        raise Tilelang2AscendcSourceError(_error("source tree uses the reserved manifest name"))
    if any(part.startswith("._") for part in relative.parts):
        raise Tilelang2AscendcSourceError(
            _error(f"AppleDouble sidecar is forbidden: {relative.as_posix()}")
        )


def _inventory_queue_dir(item: Path, depth: int, pending: list[tuple[Path, int]]) -> None:
    """Queue a subdirectory for traversal unless it is an ignored build directory."""
    if item.name in _SOURCE_DIRS or item.name.endswith(".egg-info"):
        return
    if depth + 1 > _MAX_DEPTH:
        raise Tilelang2AscendcSourceError(_error("source tree exceeds directory depth limit"))
    pending.append((item, depth + 1))


def _inventory_entry(item: Path, relative: Path, metadata: os.stat_result) -> dict[str, Any]:
    """Validate one regular file and return its inventory entry."""
    name = relative.as_posix()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Tilelang2AscendcSourceError(_error(f"source tree contains non-regular file: {name}"))
    suffix = item.suffix.lower()
    if suffix in _BINARY_SUFFIXES:
        raise Tilelang2AscendcSourceError(_error(f"binary artifact is forbidden: {name}"))
    if suffix not in _SOURCE_SUFFIXES and item.name not in _SAFE_SUFFIXLESS_NAMES:
        raise Tilelang2AscendcSourceError(_error(f"unsupported source file suffix: {name}"))
    if metadata.st_size > _MAX_FILE_BYTES:
        raise Tilelang2AscendcSourceError(_error(f"source file exceeds size limit: {name}"))
    raw = item.read_bytes()
    if len(raw) != metadata.st_size:
        raise Tilelang2AscendcSourceError(_error(f"source file changed while inspecting: {name}"))
    _read_text(item, name)
    return {"path": name, "size": metadata.st_size, "sha256": hashlib.sha256(raw).hexdigest()}


def _inventory(root: Path, *, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total = 0
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        current, depth = pending.pop()
        for item in sorted(current.iterdir(), key=lambda path: path.name):
            relative = item.relative_to(root)
            if manifest_path is not None and item == manifest_path:
                continue
            _inventory_guard_path(relative, manifest_scoped=manifest_path is not None)
            metadata = item.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise Tilelang2AscendcSourceError(
                    _error(f"source tree contains symlink: {relative.as_posix()}")
                )
            if stat.S_ISDIR(metadata.st_mode):
                _inventory_queue_dir(item, depth, pending)
                continue
            file_entry = _inventory_entry(item, relative, metadata)
            total += metadata.st_size
            if total > _MAX_TOTAL_BYTES:
                raise Tilelang2AscendcSourceError(_error("source tree exceeds total size limit"))
            entries.append(file_entry)
            if len(entries) > _MAX_FILES:
                raise Tilelang2AscendcSourceError(_error("source tree exceeds file-count limit"))
    return sorted(entries, key=lambda entry: entry["path"])


def _collect_inventory_violations(root: Path) -> list[str]:
    """Collect ALL source-inventory violations instead of the first one.

    The strict inventory is a deliberate safety net (frozen source stages must
    be portable/reproducible — binary artifacts and unreadable files are not).
    Failing on the FIRST violation makes operators fix debris one file at a
    time (2026-08-23 FlashBwd/Nystrom cold-starts: bin debris, x86_64 .so,
    .simplified suffix — each surfaced only after the previous was deleted).
    The detector keeps failing closed; this helper only makes the rejection
    message list every offending path at once.
    """
    violations: list[str] = []
    pending = [(root, 0)]
    while pending and len(violations) < 30:
        current, depth = pending.pop()
        if depth > _MAX_DEPTH:
            continue
        try:
            items = sorted(current.iterdir())
        except OSError as exc:
            violations.append(f"unreadable directory: {exc}")
            continue
        for item in items:
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                violations.append(f"symlink is forbidden: {relative}")
                continue
            if item.is_dir():
                pending.append((item, depth + 1))
                continue
            try:
                metadata = item.stat()
            except OSError as exc:
                violations.append(f"unstatable file: {relative}: {exc}")
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                violations.append(f"non-regular file: {relative}")
                continue
            suffix = item.suffix.lower()
            if suffix in _BINARY_SUFFIXES:
                violations.append(f"binary artifact is forbidden: {relative}")
                continue
            if suffix not in _SOURCE_SUFFIXES and item.name not in _SAFE_SUFFIXLESS_NAMES:
                violations.append(f"unsupported source file suffix: {relative}")
                continue
            if metadata.st_size > _MAX_FILE_BYTES:
                violations.append(f"file exceeds size limit: {relative}")
                continue
            try:
                _read_text(item, relative)
            except Exception as exc:
                violations.append(f"unreadable source file: {relative}: {exc}")
    return violations


def _registered_ops(root: Path) -> tuple[str, ...]:
    register = _cxx_without_comments(_read_text(root / "kernel" / "register.cpp", "kernel/register.cpp"))
    if not re.search(
        r"\bTORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*npu\s*,", register
    ) or not re.search(
        r"\bTORCH_LIBRARY_IMPL\s*\(\s*npu\s*,", register
    ):
        raise Tilelang2AscendcSourceError(
            _error("kernel/register.cpp lacks TORCH_LIBRARY registration for namespace npu")
        )
    names = tuple(sorted(set(re.findall(r'\bm\.def\s*\(\s*["\']([A-Za-z0-9_]+)', register))))
    if not names:
        names = tuple(sorted(set(re.findall(r'\bm\.impl\s*\(\s*["\']([A-Za-z0-9_]+)', register))))
    if not names:
        raise Tilelang2AscendcSourceError(_error("kernel/register.cpp has no registered operator"))
    model = _python_without_comments(_read_text(root / "model_new_ascendc.py", "model_new_ascendc.py"))
    calls = tuple(sorted(set(re.findall(r"torch\s*\.\s*ops\s*\.\s*npu\s*\.\s*([A-Za-z0-9_]+)", model))))
    if not calls:
        raise Tilelang2AscendcSourceError(_error("model_new_ascendc.py does not call torch.ops.npu.<op>"))
    if not set(calls).intersection(names):
        raise Tilelang2AscendcSourceError(_error("model_new_ascendc.py calls no operator registered by register.cpp"))
    return names


def _inspect(source_root: Path, *, expected_op: str | None = None, manifest_path: Path | None = None):
    root = _normalise_root(source_root)
    _required_layout(root)
    op = require_safe_op_name(expected_op) if expected_op is not None else _logical_op(root)
    entries = _inventory(root, manifest_path=manifest_path)
    registered = _registered_ops(root)
    host_files = [entry["path"] for entry in entries if entry["path"].startswith("kernel/op_host/")]
    kernel_files = [entry["path"] for entry in entries if entry["path"].startswith("kernel/op_kernel/")]
    if not host_files or not kernel_files:
        raise Tilelang2AscendcSourceError(_error("kernel/op_host and kernel/op_kernel must contain source files"))
    cxx_paths = [entry["path"] for entry in entries if entry["path"].lower().endswith(_CXX_TEXT_SUFFIXES)]
    cxx = "\n".join(_read_text(root / path, path) for path in cxx_paths)
    evidence = [f"op:{op}", "layout:model_new_ascendc+kernel", *registered, *host_files, *kernel_files]
    if "__global__" in cxx and "__aicore__" in cxx and "AscendC::" in cxx:
        evidence.append("device:global-aicore-ascendc")
    if "EXEC_KERNEL_CMD" in cxx or "aclrtlaunch_" in cxx or "ACLRT_LAUNCH_KERNEL" in cxx:
        evidence.append("host:kernel-launch")
    detection = Tilelang2AscendcSourceDetection(
        arch="arch35",
        supported=True,
        method="tilelang2ascendc_project_layout",
        confidence="high",
        evidence=tuple(sorted(set(evidence))),
        analyzed_paths=tuple(entry["path"] for entry in entries),
    )
    return detection, entries, op


def detect_tilelang2ascendc_source(source_root: Path) -> Tilelang2AscendcSourceDetection:
    try:
        detection, _entries, _op = _inspect(source_root)
        return detection
    except Exception as exc:
        detail = str(exc)
        evidence: list[str] = [detail]
        # Report every inventory violation at once so the operator can clean
        # all debris in one pass (instead of serial one-at-a-time discovery).
        try:
            violations = _collect_inventory_violations(source_root)
        except Exception:
            violations = []
        if violations:
            evidence.append("all violations: " + "; ".join(violations))
        return Tilelang2AscendcSourceDetection(
            arch=None,
            supported=False,
            method="invalid_tilelang2ascendc_project",
            confidence="high",
            evidence=tuple(evidence),
            analyzed_paths=(),
        )


@contextmanager
def _publish_lock(workspace: Path) -> Iterator[None]:
    lock = workspace / ".tilelang2ascendc_source.publish.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as exc:
        raise Tilelang2AscendcSourceError(_error(f"cannot open publisher lock: {exc}")) from exc
    try:
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise Tilelang2AscendcSourceError(_error("publish lock is unsafe"))
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise Tilelang2AscendcSourceError(_error(f"cannot lock publisher: {exc}")) from exc
        # Keep exceptions raised by the protected publish operation outside the
        # lock-acquisition error handler; ENOSPC/EACCES at os.replace is a
        # publish failure, not evidence that flock failed.
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _stat_identical(before: os.stat_result, after: os.stat_result) -> bool:
    """Report whether both stat results describe the same unchanged file."""
    return (
        before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _stream_copy(source: Path, output: Any) -> tuple[int, str]:
    """Copy *source* into the already-open *output*, returning (size, sha256)."""
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    output.flush()
    os.fsync(output.fileno())
    return size, digest.hexdigest()


def _copy_regular(source: Path, destination: Path, relative: str) -> dict[str, Any]:
    before = _regular(source, "source file")
    before_digest = _file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            size, copied = _stream_copy(source, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = _regular(source, "source file")
    # Re-hashing the source stays the last term so it only runs when the cheap
    # stat/size/digest comparisons already agree (same short-circuit as before).
    changed = not _stat_identical(before, after) or size != before.st_size or before_digest != copied
    if changed or _file_sha256(source) != copied:
        raise Tilelang2AscendcSourceError(_error(f"source changed while copying: {relative}"))
    return {"path": relative, "size": size, "sha256": copied}


def _cleanup(path: Path) -> None:
    try:
        if path.name.startswith(f"..{TILELANG2ASCENDC_STAGE_DIR}.incoming-") and not path.is_symlink():
            shutil.rmtree(path)
    except OSError:
        pass


def _manifest_document(
    op: str,
    digest: str,
    copied: list[dict[str, Any]],
    detection: Tilelang2AscendcSourceDetection,
) -> dict[str, Any]:
    """Build the manifest document that binds a staged source tree."""
    return {
        "schema": TILELANG2ASCENDC_SOURCE_SCHEMA,
        "kind": TILELANG2ASCENDC_SOURCE_KIND,
        "root_layout": TILELANG2ASCENDC_ROOT_LAYOUT,
        "op": op,
        "source_arch": "arch35",
        "target_arch": "arch35",
        "tree_sha256": digest,
        "file_count": len(copied),
        "files": copied,
        "source_detection": detection.state_payload(),
    }


def _write_manifest(incoming: Path, payload: dict[str, Any]) -> None:
    """Write the manifest into the incoming stage and flush it to disk."""
    with (incoming / TILELANG2ASCENDC_MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _copy_tree(source_root: Path, incoming: Path, inspected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy every inspected file into the incoming stage, re-verifying each one."""
    copied: list[dict[str, Any]] = []
    for entry in inspected:
        item_path = entry["path"]
        copied.append(_copy_regular(source_root / item_path, incoming / item_path, item_path))
    return copied


def stage_tilelang2ascendc_source_tree(source_root: Path, workspace: Path) -> Tilelang2AscendcSourceStage:
    source_root = _normalise_root(source_root)
    detection, inspected, op = _inspect(source_root)
    workspace = Path(workspace).expanduser()
    if workspace.exists() or workspace.is_symlink():
        _real_directory(workspace, "workspace")
    else:
        workspace.mkdir(parents=True, exist_ok=False)
    workspace = workspace.resolve()
    incoming = workspace / f"..{TILELANG2ASCENDC_STAGE_DIR}.incoming-{uuid.uuid4().hex}"
    incoming.mkdir(mode=0o700, exist_ok=False)
    try:
        copied = _copy_tree(source_root, incoming, inspected)
        if copied != inspected:
            raise Tilelang2AscendcSourceError(_error("source changed while staging"))
        staged_detection, staged_entries, staged_op = _inspect(incoming, expected_op=op)
        if staged_detection != detection or staged_entries != copied or staged_op != op:
            raise Tilelang2AscendcSourceError(_error("incoming source stage does not match inspected source"))
        digest = _tree_digest(copied)
        _write_manifest(incoming, _manifest_document(op, digest, copied, detection))
        final_root = workspace / TILELANG2ASCENDC_STAGE_DIR
        with _publish_lock(workspace):
            if final_root.exists() or final_root.is_symlink():
                raise Tilelang2AscendcSourceError(
                    "TILELANG2ASCENDC_SOURCE_STAGE_EXISTS: existing stage must not be replaced"
                )
            os.replace(incoming, final_root)
        return Tilelang2AscendcSourceStage(
            final_root,
            final_root / TILELANG2ASCENDC_MANIFEST,
            digest,
            len(copied),
            op,
            detection,
            tuple(copied),
        )
    except Exception:
        _cleanup(incoming)
        raise


def _valid_manifest_size(value: object) -> bool:
    """Report whether *value* is a plain int within the per-file limit.

    ``bool`` is rejected on purpose: ``isinstance(True, int)`` is true, and a
    manifest size of ``True`` must not be accepted as the integer 1.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return 0 <= value <= _MAX_FILE_BYTES


def _normalise_inventory(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > _MAX_FILES:
        raise Tilelang2AscendcSourceError(_error("manifest inventory is malformed"))
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise Tilelang2AscendcSourceError(_error("manifest inventory is malformed"))
        rel = Path(item["path"])
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise Tilelang2AscendcSourceError(_error("manifest path is unsafe"))
        if not _valid_manifest_size(item.get("size")):
            raise Tilelang2AscendcSourceError(_error("manifest file size is malformed"))
        if not isinstance(item.get("sha256"), str) or not _HASH.fullmatch(item["sha256"]):
            raise Tilelang2AscendcSourceError(_error("manifest file digest is malformed"))
        result.append({"path": rel.as_posix(), "size": item["size"], "sha256": item["sha256"]})
    ordered = sorted(result, key=lambda entry: entry["path"])
    unique_paths = {entry["path"] for entry in result}
    if result != ordered or len(unique_paths) != len(result):
        raise Tilelang2AscendcSourceError(_error("manifest inventory is not canonical"))
    return result


def _manifest_binding_ok(payload: object) -> bool:
    """Report whether the manifest header carries every expected binding."""
    if not isinstance(payload, dict):
        return False
    for field, expected in _MANIFEST_BINDINGS:
        if payload.get(field) != expected:
            return False
    if not is_safe_op_name(payload.get("op")):
        return False
    tree_sha = payload.get("tree_sha256")
    return isinstance(tree_sha, str) and _HASH.fullmatch(tree_sha) is not None


def _load_manifest(path: Path) -> dict[str, Any]:
    _regular(path, "TileLang2AscendC source manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Tilelang2AscendcSourceError(_error(f"manifest is unreadable: {type(exc).__name__}")) from exc
    if not _manifest_binding_ok(payload):
        raise Tilelang2AscendcSourceError(_error("manifest binding is malformed"))
    payload["files"] = _normalise_inventory(payload.get("files"))
    if payload.get("file_count") != len(payload["files"]) or not isinstance(payload.get("source_detection"), Mapping):
        raise Tilelang2AscendcSourceError(_error("manifest file count or detection is malformed"))
    return payload


def tilelang2ascendc_state_block(stage: Tilelang2AscendcSourceStage) -> dict[str, Any]:
    return {
        "schema": TILELANG2ASCENDC_SOURCE_SCHEMA,
        "kind": TILELANG2ASCENDC_SOURCE_KIND,
        "stage_root": str(stage.root),
        "manifest": str(stage.manifest),
        "digest": stage.digest,
        "tree_sha256": stage.digest,
        "file_count": stage.file_count,
        "source_arch": "arch35",
        "target_arch": "arch35",
        "root_layout": TILELANG2ASCENDC_ROOT_LAYOUT,
        "op": stage.op,
        "source_detection": stage.detection.state_payload(),
        "files": [dict(entry) for entry in stage.files],
    }


def _stage_inventory_matches(
    payload: Mapping[str, Any],
    entries: list[dict[str, Any]],
    detected: list[dict[str, Any]],
    digest: str,
) -> bool:
    """Report whether manifest, re-inspection and tree digest all agree."""
    return entries == detected and payload["files"] == entries and payload["tree_sha256"] == digest


def _durable_block_matches(block: Mapping[str, Any], entries: list[dict[str, Any]], digest: str) -> bool:
    """Report whether the durable state block still mirrors the verified stage."""
    if block.get("schema") != TILELANG2ASCENDC_SOURCE_SCHEMA or block.get("files") != entries:
        return False
    if block.get("tree_sha256") != digest or block.get("digest") != digest:
        return False
    return block.get("file_count") == len(entries)


def verify_tilelang2ascendc_source_stage(workspace: Path, state: Mapping[str, Any] | None = None):
    try:
        workspace = Path(workspace).expanduser().resolve(strict=True)
        _real_directory(workspace, "workspace")
        if state is None:
            state = json.loads((workspace / ".opgen_state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping) or state.get("source_kind") != TILELANG2ASCENDC_SOURCE_KIND:
            return False, "TileLang2AscendC top-level source_kind binding is missing or mismatched", {}
        block = state.get("port_source")
        if not isinstance(block, Mapping) or block.get("kind") != TILELANG2ASCENDC_SOURCE_KIND:
            return False, "workspace is not a TileLang2AscendC source run", {}
        root = workspace / TILELANG2ASCENDC_STAGE_DIR
        manifest_path = root / TILELANG2ASCENDC_MANIFEST
        _real_directory(root, "TileLang2AscendC source stage")
        payload = _load_manifest(manifest_path)
        entries = _inventory(root, manifest_path=manifest_path)
        digest = _tree_digest(entries)
        detection, detected_entries, op = _inspect(root, expected_op=payload["op"], manifest_path=manifest_path)
        if not _stage_inventory_matches(payload, entries, detected_entries, digest):
            raise Tilelang2AscendcSourceError(_error("source stage inventory or digest mismatch"))
        if not _durable_block_matches(block, entries, digest):
            raise Tilelang2AscendcSourceError(_error("durable source inventory or digest mismatch"))
        expected = {
            "port_a3_source": str(root),
            "source_stage_manifest": str(manifest_path),
            "graybox_source_dir": str(root),
            "source_stage_digest": digest,
            "source_stage_file_count": len(entries),
            "source_arch": "arch35",
            "target_arch": "arch35",
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise Tilelang2AscendcSourceError(_error(f"durable {field} binding mismatch"))
        detected_payload = detection.state_payload()
        op_bound = block.get("op") == op and state.get("op") in (None, op)
        detection_bound = (
            detected_payload == payload.get("source_detection")
            and detected_payload == block.get("source_detection")
        )
        if not op_bound or not detection_bound:
            raise Tilelang2AscendcSourceError(_error("source op or detection binding mismatch"))
        return True, f"TileLang2AscendC source stage verified ({len(entries)} files, sha256={digest[:12]})", payload
    except Exception as exc:
        return False, str(exc), locals().get("payload", {})

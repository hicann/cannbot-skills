# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Deterministic source-architecture detection for cross-generation migration."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from source_op import (
    SOURCE_STAGE_DIR,
    SOURCE_STAGE_MANIFEST,
    SOURCE_STAGE_SCHEMA,
    is_safe_op_name,
    require_safe_op_name,
)


_SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cxx"}
_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "arch35",
    "ascend950",
    "build",
    "dist",
    "install",
    "output",
}
_TARGET_NAME_MARKERS = ("arch35", "ascend950", "dav_c310", "v351")
_BINARY_SUFFIXES = {".a", ".bin", ".o", ".pyc", ".so"}
_BUILD_SOURCE_REGISTRY_DIR = ".port_a3_build_sources"
_BUILD_SOURCE_REGISTRY_SCHEMA = "cannbot.port_a3_build_source/v1"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_SOURCE_MARKERS = (
    re.compile(r"\barch22\b", re.IGNORECASE),
    re.compile(r"\bdav_c220\b", re.IGNORECASE),
    re.compile(r"__CCE_AICORE__\s*==\s*220\b"),
    re.compile(r"__NPU_ARCH__\s*==\s*2201\b"),
)
_ALGORITHM_PRIMITIVE = re.compile(
    r"\b(?:DataCopy|TQue|TBuf|LocalTensor|GlobalTensor|Matmul|Mmad|"
    r"PipeBarrier|CrossCoreSetFlag|CrossCoreWaitFlag)\s*[<(]"
)
_CLASS_PROCESS_BODY = re.compile(
    r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*[^{]*\{[\s\S]*?\bProcess\s*\([^)]*\)\s*\{"
)
_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)
_SUBSTANTIVE_CODE = re.compile(
    r"\b(?:class|struct|template|namespace|using|extern|constexpr|return|"
    r"if|for|while|void|bool|char|float|double|(?:u?int(?:8|16|32|64)?_t)|"
    r"__aicore__|__global__)\b"
)


@dataclass(frozen=True)
class SourceArchDetection:
    arch: str | None
    supported: bool
    method: str
    confidence: str
    evidence: tuple[str, ...]
    analyzed_paths: tuple[str, ...]

    def state_payload(self) -> dict:
        """JSON-ready, deterministic payload for durable state and reports."""
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        payload["analyzed_paths"] = list(self.analyzed_paths)
        return payload


@dataclass(frozen=True)
class SourceStage:
    """Immutable description of a source-only arch22 staging snapshot."""

    root: Path
    manifest: Path
    digest: str
    file_count: int
    detection: SourceArchDetection


def _is_forbidden_name(name: str) -> bool:
    """Reject target/generated names without touching the corresponding inode."""
    lower = name.lower()
    return lower in _IGNORED_DIRS or any(marker in lower for marker in _TARGET_NAME_MARKERS)


def _strip_comments_and_literals(text: str) -> str:
    """Remove C/C++ comments and string/character literals while preserving lines.

    Architecture markers and primitive names inside prose, comments, or strings are
    not executable evidence.  Replacing bytes with spaces keeps token boundaries and
    line structure stable for the remaining regex checks.
    """
    out: list[str] = []
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out.extend((" ", " "))
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                out.extend((" ", " "))
                i += 2
                state = "block_comment"
                continue
            if ch in {'"', "'"}:
                quote = ch
                out.append(" ")
                i += 1
                state = "literal"
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line_comment":
            if ch == "\n":
                out.append("\n")
                state = "code"
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                out.extend((" ", " "))
                i += 2
                state = "code"
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        # string/character literal
        if ch == "\\" and nxt:
            out.extend((" ", "\n" if nxt == "\n" else " "))
            i += 2
        elif ch == quote:
            out.append(" ")
            i += 1
            state = "code"
        else:
            out.append("\n" if ch == "\n" else " ")
            i += 1
    return "".join(out)


def _has_substantive_code(path: Path, text: str) -> bool:
    if path.name == "CMakeLists.txt":
        return False
    return bool(_SUBSTANTIVE_CODE.search(_strip_comments_and_literals(text)))


def _is_source_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix.lower() in _SOURCE_SUFFIXES


def _code_files(root: Path) -> list[Path]:
    """Enumerate source-side candidates without entering target directories.

    ``Path.rglob`` is intentionally not used: even filtering its results would
    first enumerate the target implementation tree.  This walker rejects
    source-side symlink directories and prunes target/build/output names before
    calling any filesystem predicate on their contents.
    """
    if not root.is_dir():
        return []
    files: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        for path in sorted(current.iterdir()):
            if _is_forbidden_name(path.name):
                continue
            if path.is_symlink():
                # Source-file links are validated by _collect_source_texts.
                # Directory links are also returned so they fail closed rather
                # than becoming an unbounded alternate traversal root.
                if _is_source_file(path) or path.suffix == "":
                    files.append(path)
                continue
            if path.is_dir():
                pending.append(path)
            elif _is_source_file(path) and path.is_file():
                files.append(path)
    return sorted(files)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _collect_source_texts(
    op_dir: Path, kernel_dir: Path
) -> tuple[dict[Path, str], tuple[str, ...], tuple[str, ...]]:
    """Read candidate sources once and report unsafe paths/read failures.

    A source symlink that resolves outside the operator directory is not valid
    architecture evidence.  Likewise, an unreadable candidate cannot be
    silently treated as empty: doing so could let weaker evidence select a
    supported architecture while contradictory code remains unexamined.
    """
    texts: dict[Path, str] = {}
    escapes: list[str] = []
    unreadable: list[str] = []
    for path in _code_files(kernel_dir):
        rel = str(path.relative_to(op_dir))
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            unreadable.append(f"{rel}: resolve failed ({type(exc).__name__})")
            continue
        if not _is_within(resolved, op_dir):
            escapes.append(f"{rel}: resolves outside operator directory")
            continue
        try:
            texts[path] = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            unreadable.append(f"{rel}: read failed ({type(exc).__name__})")
    return texts, tuple(sorted(escapes)), tuple(sorted(unreadable))


def _substantive_under(texts: dict[Path, str], root: Path) -> list[Path]:
    """Return files that contain executable/declarative C++ rather than prose."""
    return sorted(
        path
        for path, text in texts.items()
        if _is_within(path, root) and _has_substantive_code(path, text)
    )


def _relative(op_dir: Path, paths: list[Path]) -> tuple[str, ...]:
    return tuple(sorted(str(path.relative_to(op_dir)) for path in paths))


def detect_source_arch(op_dir: Path) -> SourceArchDetection:
    """Detect arch22 evidence without opening a co-located target tree.

    The current customer route supports only arch22 to arch35.  Explicit arch22
    directories win over the target directory; otherwise the established ops-nn
    top-level source layout is used.  The target directory is pruned before
    traversal and never contributes evidence.  Ambiguous or source-less layouts
    fail closed.
    """
    try:
        op_dir = op_dir.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="invalid_source_root",
            confidence="high",
            evidence=(f"source root cannot be resolved ({type(exc).__name__})",),
            analyzed_paths=(),
        )
    kernel_dir = op_dir / "op_kernel"
    if not kernel_dir.is_dir():
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="missing_op_kernel",
            confidence="high",
            evidence=("op_kernel directory is missing",),
            analyzed_paths=(),
        )

    try:
        source_texts, path_escapes, unreadable = _collect_source_texts(
            op_dir, kernel_dir
        )
    except (OSError, RuntimeError) as exc:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="unreadable_source",
            confidence="high",
            evidence=(f"source traversal failed ({type(exc).__name__})",),
            analyzed_paths=(),
        )
    analyzed = _relative(op_dir, list(source_texts))
    if path_escapes:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="source_path_escape",
            confidence="high",
            evidence=path_escapes,
            analyzed_paths=analyzed,
        )
    if unreadable:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="unreadable_source",
            confidence="high",
            evidence=unreadable,
            analyzed_paths=analyzed,
        )

    arch22_files = _substantive_under(source_texts, kernel_dir / "arch22")

    sibling_arch_sources: list[Path] = []
    for child in sorted(kernel_dir.iterdir()):
        # Name-prune before every predicate: a hostile/unreadable target inode
        # must never be stat'ed merely to decide that it is out of scope.
        if _is_forbidden_name(child.name) or child.name.lower() == "arch22":
            continue
        if not child.is_dir():
            continue
        if (
            re.fullmatch(r"arch[0-9]+", child.name, re.IGNORECASE)
            and _substantive_under(source_texts, child)
        ):
            sibling_arch_sources.append(child)

    if arch22_files and sibling_arch_sources:
        siblings = ", ".join(path.name for path in sibling_arch_sources)
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="conflicting_source_dirs",
            confidence="high",
            evidence=(f"op_kernel/arch22 and source sibling(s) {siblings} both contain code",),
            analyzed_paths=analyzed,
        )

    if arch22_files:
        return SourceArchDetection(
            arch="arch22",
            supported=True,
            method="explicit_source_dir",
            confidence="high",
            evidence=_relative(op_dir, arch22_files),
            analyzed_paths=analyzed,
        )

    marker_evidence: list[str] = []
    for path in _substantive_under(source_texts, kernel_dir):
        relative_parts = path.relative_to(kernel_dir).parts
        if any(part.startswith(".") or part.lower() in _IGNORED_DIRS for part in relative_parts[:-1]):
            continue
        path_has_marker = any("dav_c220" in part.lower() for part in relative_parts)
        code = _strip_comments_and_literals(source_texts[path])
        if path_has_marker or any(pattern.search(code) for pattern in _SOURCE_MARKERS):
            marker_evidence.append(str(path.relative_to(op_dir)))

    top_level_files = sorted(
        path for path in source_texts if path.parent == kernel_dir and _is_source_file(path)
    )
    algorithm_evidence: list[Path] = []
    pure_target_shells: list[Path] = []
    for path in top_level_files:
        text = source_texts[path]
        code = _strip_comments_and_literals(text)
        includes = _LOCAL_INCLUDE.findall(text)
        includes_target_only = bool(includes) and all(
            include.replace("\\", "/").startswith("arch35/") for include in includes
        )
        has_real_algorithm = bool(
            _ALGORITHM_PRIMITIVE.search(code) or _CLASS_PROCESS_BODY.search(code)
        )
        if includes_target_only and not _ALGORITHM_PRIMITIVE.search(code):
            pure_target_shells.append(path)
        elif has_real_algorithm:
            algorithm_evidence.append(path)

    if sibling_arch_sources and (marker_evidence or algorithm_evidence):
        siblings = ", ".join(path.name for path in sibling_arch_sources)
        mixed = tuple(sorted(marker_evidence)) + _relative(op_dir, algorithm_evidence)
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="conflicting_source_dirs",
            confidence="high",
            evidence=(
                f"source sibling(s) {siblings} coexist with arch22 marker/algorithm evidence",
                *mixed,
            ),
            analyzed_paths=analyzed,
        )

    if marker_evidence:
        return SourceArchDetection(
            arch="arch22",
            supported=True,
            method="source_marker",
            confidence="medium",
            evidence=tuple(sorted(marker_evidence)),
            analyzed_paths=analyzed,
        )

    if algorithm_evidence:
        return SourceArchDetection(
            arch="arch22",
            supported=True,
            method="ops_nn_top_level_algorithm",
            confidence="medium",
            evidence=_relative(op_dir, algorithm_evidence),
            analyzed_paths=analyzed,
        )

    if pure_target_shells:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="target_dispatch_only",
            confidence="high",
            evidence=_relative(op_dir, pure_target_shells),
            analyzed_paths=analyzed,
        )

    if sibling_arch_sources:
        return SourceArchDetection(
            arch=None,
            supported=False,
            method="unsupported_source_dir",
            confidence="high",
            evidence=tuple(f"op_kernel/{path.name}" for path in sibling_arch_sources),
            analyzed_paths=analyzed,
        )

    return SourceArchDetection(
        arch=None,
        supported=False,
        method="unknown",
        confidence="low",
        evidence=("no recognizable source entry was found under op_kernel",),
        analyzed_paths=analyzed,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _snapshot_entries(root: Path) -> list[dict]:
    entries: list[dict] = []
    pending = [root]
    while pending:
        current = pending.pop()
        for item in sorted(current.iterdir()):
            if item.name == SOURCE_STAGE_MANIFEST:
                continue
            if _is_forbidden_name(item.name):
                raise ValueError(f"source stage contains forbidden target/generated name: {item.name}")
            if item.is_symlink():
                raise ValueError(f"source stage contains symlink: {item.relative_to(root)}")
            if item.is_dir():
                pending.append(item)
                continue
            if not item.is_file():
                raise ValueError(f"source stage contains non-regular file: {item.relative_to(root)}")
            rel = item.relative_to(root).as_posix()
            entries.append({"path": rel, "size": item.stat().st_size, "sha256": _file_sha256(item)})
    return sorted(entries, key=lambda entry: entry["path"])


def _source_only_entries(root: Path) -> list[dict]:
    """Hash an original checkout with the same filters used by staging."""
    entries: list[dict] = []
    pending = [root]
    while pending:
        current = pending.pop()
        for item in sorted(current.iterdir()):
            if item.name == SOURCE_STAGE_MANIFEST:
                continue
            if _is_forbidden_name(item.name):
                continue
            if item.is_symlink():
                raise ValueError(
                    f"source input contains symlink: {item.relative_to(root)}"
                )
            if item.is_dir():
                pending.append(item)
                continue
            if not item.is_file():
                raise ValueError(
                    f"source input contains non-regular file: {item.relative_to(root)}"
                )
            if item.suffix.lower() in _BINARY_SUFFIXES:
                continue
            entries.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "size": item.stat().st_size,
                    "sha256": _file_sha256(item),
                }
            )
    return sorted(entries, key=lambda entry: entry["path"])


def _validate_build_source_digest(source_root: Path, expected_digest: str) -> None:
    try:
        current_digest = _tree_digest(_source_only_entries(source_root))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"build source cannot be hashed safely: {exc}") from exc
    if current_digest != expected_digest:
        raise ValueError(
            "build source content no longer matches the immutable source stage"
        )


def stage_source_tree(source_root: Path, workspace: Path) -> SourceStage:
    """Create a deterministic source-only snapshot below ``workspace``.

    Target/generated names are rejected by name before filesystem predicates;
    symlinks and compiled artifacts fail closed.  The durable state must point
    only at the returned snapshot, never at the original customer checkout.
    """
    source_root = source_root.expanduser().resolve(strict=True)
    op_name = require_safe_op_name(source_root.name)
    workspace = workspace.expanduser().resolve()
    detection = detect_source_arch(source_root)
    if not detection.supported or detection.arch != "arch22":
        raise ValueError(
            f"source-only staging requires arch22 input, got {detection.arch!r} "
            f"via {detection.method}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    stage_root = workspace / SOURCE_STAGE_DIR
    tmp_root = workspace / f"{SOURCE_STAGE_DIR}.tmp-{uuid.uuid4().hex}"
    tmp_root.mkdir(mode=0o700)
    try:
        pending: list[tuple[Path, Path]] = [(source_root, tmp_root)]
        while pending:
            src_dir, dst_dir = pending.pop()
            for item in sorted(src_dir.iterdir()):
                if _is_forbidden_name(item.name):
                    continue
                if item.is_symlink():
                    raise ValueError(
                        f"source input contains symlink: {item.relative_to(source_root)}"
                    )
                destination = dst_dir / item.name
                if item.is_dir():
                    destination.mkdir()
                    pending.append((item, destination))
                elif item.is_file():
                    if item.suffix.lower() in _BINARY_SUFFIXES:
                        continue
                    shutil.copy2(item, destination)
                else:
                    raise ValueError(
                        f"source input contains non-regular file: {item.relative_to(source_root)}"
                    )
        staged_detection = detect_source_arch(tmp_root)
        if not staged_detection.supported or staged_detection.arch != "arch22":
            raise ValueError(
                "source-only snapshot lost the required arch22 evidence: "
                f"{staged_detection.method}"
            )
        entries = _snapshot_entries(tmp_root)
        tree_sha256 = _tree_digest(entries)
        manifest_payload = {
            "schema": SOURCE_STAGE_SCHEMA,
            "op": op_name,
            "source_arch": "arch22",
            "target_arch": "arch35",
            "tree_sha256": tree_sha256,
            "file_count": len(entries),
            "files": entries,
            "source_arch_detection": staged_detection.state_payload(),
        }
        manifest = tmp_root / SOURCE_STAGE_MANIFEST
        manifest.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")
        if stage_root.exists():
            if stage_root.is_symlink() or not stage_root.is_dir():
                raise ValueError(f"refusing to replace unsafe source stage path: {stage_root}")
            shutil.rmtree(stage_root)
        os.replace(tmp_root, stage_root)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
    return SourceStage(
        root=stage_root,
        manifest=stage_root / SOURCE_STAGE_MANIFEST,
        digest=tree_sha256,
        file_count=len(entries),
        detection=staged_detection,
    )


def _build_source_record_path(workspace: Path) -> Path:
    """Return the harness-private record path for one migration workspace.

    The registry is a sibling of per-op workspaces.  Graybox workers receive a
    bind mount for the individual workspace only, so they cannot read this
    original-checkout pointer even though it survives a separate resume
    process.
    """
    workspace = workspace.expanduser().resolve(strict=True)
    op_name = require_safe_op_name(workspace.name)
    return workspace.parent / _BUILD_SOURCE_REGISTRY_DIR / f"{op_name}.json"


def _validate_private_registry(registry: Path) -> None:
    try:
        metadata = registry.lstat()
    except OSError as exc:
        raise ValueError("build-source registry is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("build-source registry is not a regular directory")
    if metadata.st_uid != os.geteuid():
        raise ValueError("build-source registry is not owned by the current user")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("build-source registry permissions must be 0700")


def _validate_private_record(record: Path) -> None:
    try:
        metadata = record.lstat()
    except OSError as exc:
        raise ValueError("private build-source record is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("private build-source record is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise ValueError("private build-source record is not owned by the current user")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("private build-source record permissions must be 0600")


def record_port_a3_build_source(
    workspace: Path,
    source_root: Path,
    *,
    source_stage_digest: str,
) -> Path:
    """Persist the trusted build checkout outside the worker-visible workspace.

    Durable migration state remains bound only to ``.source_arch22``.  This
    private harness record exists solely because an unshipped source operator
    may need the repository-level ``build.sh`` after a process restart.
    """
    workspace = workspace.expanduser().resolve(strict=True)
    source_root = source_root.expanduser().resolve(strict=True)
    op_name = require_safe_op_name(workspace.name)
    if source_root.name != op_name:
        raise ValueError(
            f"build source op mismatch: expected {op_name!r}, got {source_root.name!r}"
        )
    if _is_within(source_root, workspace):
        raise ValueError("build source must not resolve inside the migration workspace")
    if not (source_root / "op_host").is_dir() or not (source_root / "op_kernel").is_dir():
        raise ValueError("build source is missing op_host/ or op_kernel/")
    if not isinstance(source_stage_digest, str) or not _SHA256_HEX.fullmatch(
        source_stage_digest
    ):
        raise ValueError("source-stage digest is not a lowercase SHA-256 value")
    detection = detect_source_arch(source_root)
    if not detection.supported or detection.arch != "arch22":
        raise ValueError(
            "build source no longer has supported arch22 evidence: "
            f"{detection.method}"
        )
    _validate_build_source_digest(source_root, source_stage_digest)

    record = _build_source_record_path(workspace)
    registry = record.parent
    registry.mkdir(mode=0o700, exist_ok=True)
    _validate_private_registry(registry)
    if record.exists() or record.is_symlink():
        _validate_private_record(record)
    payload = {
        "schema": _BUILD_SOURCE_REGISTRY_SCHEMA,
        "op": op_name,
        "source_root": str(source_root),
        "source_stage_digest": source_stage_digest,
    }
    temporary = registry / f".{op_name}.tmp-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, record)
        _validate_private_record(record)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return record


def load_port_a3_build_source(
    workspace: Path,
    *,
    source_stage_digest: str,
) -> Path:
    """Load and validate the private source checkout binding for resume."""
    workspace = workspace.expanduser().resolve(strict=True)
    op_name = require_safe_op_name(workspace.name)
    record = _build_source_record_path(workspace)
    registry = record.parent
    _validate_private_registry(registry)
    _validate_private_record(record)
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"private build-source record is unreadable: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("private build-source record is not a JSON object")
    if payload.get("schema") != _BUILD_SOURCE_REGISTRY_SCHEMA:
        raise ValueError("private build-source record schema mismatch")
    if payload.get("op") != op_name:
        raise ValueError("private build-source record op mismatch")
    if payload.get("source_stage_digest") != source_stage_digest:
        raise ValueError("private build-source record stage digest mismatch")
    raw_source = payload.get("source_root")
    if not isinstance(raw_source, str) or not raw_source or "\n" in raw_source:
        raise ValueError("private build-source record has an invalid source path")
    candidate = Path(raw_source)
    if not candidate.is_absolute():
        raise ValueError("private build-source path must be absolute")
    try:
        source_root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"private build-source path cannot be resolved: {type(exc).__name__}"
        ) from exc
    if source_root.name != op_name or _is_within(source_root, workspace):
        raise ValueError("private build-source path is not bound to this operator")
    if not (source_root / "op_host").is_dir() or not (source_root / "op_kernel").is_dir():
        raise ValueError("private build source is missing op_host/ or op_kernel/")
    detection = detect_source_arch(source_root)
    if not detection.supported or detection.arch != "arch22":
        raise ValueError(
            "private build source no longer has supported arch22 evidence: "
            f"{detection.method}"
        )
    _validate_build_source_digest(source_root, source_stage_digest)
    return source_root


def verify_source_stage(
    workspace: Path, state: dict | None = None
) -> tuple[bool, str, dict]:
    """Validate the fixed source snapshot and its durable-state binding."""
    workspace = workspace.expanduser().resolve()
    state_path = workspace / ".opgen_state.json"
    if state is None:
        try:
            loaded = json.loads(state_path.read_text())
            state = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            return False, f"migration state unreadable: {type(exc).__name__}", {}
    expected_root = workspace / SOURCE_STAGE_DIR
    expected_manifest = expected_root / SOURCE_STAGE_MANIFEST
    if state.get("opgen_mode") != "port_a3_to_a5":
        return False, "workspace is not a migration run", {}
    if state.get("port_a3_source") != str(expected_root):
        return False, "port_a3_source is not the fixed source-only stage", {}
    if state.get("source_stage_manifest") != str(expected_manifest):
        return False, "source_stage_manifest does not name the fixed manifest", {}
    if state.get("graybox_sandbox") is not True:
        return False, "migration sandbox is not mandatory in durable state", {}
    if state.get("graybox_arch22_dir") != str(expected_root):
        return False, "graybox source bind is not the fixed source-only stage", {}
    try:
        if expected_root.is_symlink() or not expected_root.is_dir():
            return False, "source-only stage is missing or is a symlink", {}
        if expected_manifest.is_symlink() or not expected_manifest.is_file():
            return False, "source-stage manifest is missing, non-regular, or a symlink", {}
        payload = json.loads(expected_manifest.read_text())
        entries = _snapshot_entries(expected_root)
        digest = _tree_digest(entries)
    except Exception as exc:
        return False, f"source-only stage validation failed: {exc}", {}
    if payload.get("schema") != SOURCE_STAGE_SCHEMA:
        return False, "unsupported source-stage manifest schema", payload
    manifest_op = payload.get("op")
    if not is_safe_op_name(manifest_op):
        return False, "source-stage manifest has an invalid op name", payload
    if state.get("op") is not None and state.get("op") != manifest_op:
        return False, "source-stage op binding mismatch", payload
    if payload.get("files") != entries or payload.get("file_count") != len(entries):
        return False, "source-stage file inventory/hash mismatch", payload
    if payload.get("tree_sha256") != digest:
        return False, "source-stage tree digest mismatch", payload
    if state.get("source_stage_digest") != digest:
        return False, "durable state source_stage_digest mismatch", payload
    detection = detect_source_arch(expected_root)
    if not detection.supported or detection.arch != "arch22":
        return False, f"staged source is no longer arch22 ({detection.method})", payload
    return True, f"source-only stage verified ({len(entries)} files, sha256={digest[:12]})", payload

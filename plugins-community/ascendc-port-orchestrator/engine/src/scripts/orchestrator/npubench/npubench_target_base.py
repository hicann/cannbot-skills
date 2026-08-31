# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Primitive building blocks shared by the NPUKernelBench target transport.

This module owns the transport's exception hierarchy plus the small, dependency
free filesystem/JSON/cleanup primitives that every other ``npubench_target*``
module builds on.  It deliberately imports nothing from the orchestrator so the
higher layers (candidate contract, build receipt, toolchain, transport) can form
an acyclic import graph.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


class TargetTransportError(RuntimeError):
    """Target staging, execution, or evidence import failed."""


class CandidateContractError(TargetTransportError):
    """The authored candidate violates the provider's delivery contract."""


class CleanupFailure(TargetTransportError):
    """One or more cleanup operations failed."""

    def __init__(self, failures: Sequence[str | BaseException], *, required: bool) -> None:
        self.failures = tuple(str(failure) for failure in failures)
        kind = "required" if required else "optional"
        super().__init__(f"{kind} cleanup failed: " + "; ".join(self.failures))


class _DirectBuildTimeout(TargetTransportError):
    """A direct build exceeded its deadline after its process group was stopped."""

    def __init__(self, timeout_sec: int, stdout: str, stderr: str) -> None:
        super().__init__(f"direct candidate build timed out after {timeout_sec}s; process group terminated")
        self.timeout_sec = timeout_sec
        self.stdout = stdout
        self.stderr = stderr


def _file(path: Path, label: str) -> None:
    if Path(path).is_symlink() or not Path(path).is_file():
        raise TargetTransportError(f"{label} must be a regular file")


def _real_directory(path: Path, label: str) -> Path:
    """Create one controller-owned directory without following a link."""
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise TargetTransportError(f"{label} must be a real directory")
    else:
        path.mkdir()
    if path.is_symlink() or not path.is_dir():
        raise TargetTransportError(f"{label} must be a real directory")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TargetTransportError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TargetTransportError(f"{label} is not a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish JSON atomically, never leaking the mkstemp descriptor or file."""
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        # os.fdopen has not taken ownership of the descriptor, so the audited
        # cleanup block below never runs for this path: close the descriptor and
        # delete the temporary file here instead of leaking both.
        os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            # Best-effort removal: the descriptor is already closed and the
            # original failure re-raised below is what the caller must see.
            pass
        raise
    primary_error: BaseException | None = None
    try:
        with stream as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException as exc:
        primary_error = exc

    cleanup_errors: list[BaseException] = []
    _collect_cleanup_error(
        cleanup_errors,
        f"unlink atomic JSON temporary {temporary}",
        lambda: _unlink(temporary, required=True, cleanup_errors=cleanup_errors),
    )
    if primary_error is not None:
        _raise_preserving(primary_error, cleanup_errors)
    if cleanup_errors:
        _raise_cleanup_failures(cleanup_errors)


def _contract_value(contract: Mapping[str, str], key: str) -> str:
    """Read a required build-contract field without an implicit KeyError."""
    value = contract.get(key)
    if not isinstance(value, str) or not value:
        raise TargetTransportError(f"build contract is missing {key}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_ok(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _device(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TargetTransportError(f"{label} must be a non-negative integer")


def _json_tail(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _unlink(
    path: Path | None,
    *,
    required: bool = False,
    cleanup_errors: list[BaseException] | None = None,
) -> CleanupFailure | None:
    if path is None:
        return None
    try:
        Path(path).unlink()
    except FileNotFoundError:
        # The desired post-condition already holds, including after os.replace.
        return None
    except OSError as exc:
        failure = CleanupFailure(
            (f"unlink {Path(path)}: {type(exc).__name__}: {exc}",),
            required=required,
        )
        if cleanup_errors is not None:
            cleanup_errors.append(failure)
            return None
        if required:
            raise failure from exc
        return failure
    return None


def _remove_tree(
    path: Path | None,
    *,
    required: bool = False,
    cleanup_errors: list[BaseException] | None = None,
) -> CleanupFailure | None:
    if path is None:
        return None
    try:
        shutil.rmtree(Path(path))
    except FileNotFoundError:
        return None
    except OSError as exc:
        failure = CleanupFailure(
            (f"remove tree {Path(path)}: {type(exc).__name__}: {exc}",),
            required=required,
        )
        if cleanup_errors is not None:
            cleanup_errors.append(failure)
            return None
        if required:
            raise failure from exc
        return failure
    return None


def _collect_cleanup_error(errors: list[BaseException], label: str, action) -> None:
    """Collect cleanup failures so a cleanup error cannot mask a primary error."""
    try:
        action()
    except BaseException as exc:
        if isinstance(exc, CleanupFailure):
            errors.append(exc)
        else:
            errors.append(
                CleanupFailure(
                    (f"{label}: {type(exc).__name__}: {exc}",),
                    required=True,
                )
            )


def _audit_cleanup_errors(primary: BaseException, errors: Sequence[BaseException]) -> None:
    for error in errors:
        note = f"cleanup failure: {error}"
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:  # Python 3.10 has no BaseException.add_note().
            # Python 3.10 has no public add_note(), but BaseException instances
            # still accept an attribute.  Preserve the same auditable notes
            # contract for the local controller runtime before falling back to
            # args for unusual exception implementations that reject attrs.
            try:
                notes = list(getattr(primary, "__notes__", ()))
                notes.append(note)
                setattr(primary, "__notes__", notes)
            except (AttributeError, TypeError):
                primary.args = (*primary.args, note)


def _raise_preserving(primary: BaseException, cleanup_errors: Sequence[BaseException]) -> NoReturn:
    _audit_cleanup_errors(primary, cleanup_errors)
    raise primary.with_traceback(primary.__traceback__)


def _raise_cleanup_failures(errors: Sequence[BaseException]) -> NoReturn:
    if len(errors) == 1 and isinstance(errors[0], CleanupFailure):
        raise errors[0]
    raise CleanupFailure(errors, required=True)

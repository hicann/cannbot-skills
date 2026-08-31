# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Profiler evidence and device leases for the NPUKernelBench runner.

This module owns what the parent does with a finished measurement: locating
the packaged msprof summary engine, independently revalidating the quick
profiler's per-case report before any of it is believed, resolving and
archiving only the raw PROF directories that report legitimately names, and
validating the device leases a lane was assigned.

It imports only ``npubench_core``.  ``npubench_runner`` re-exports its public
surface, so importers keep using the runner module path.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from npubench_core import (
    EVIDENCE_DIRNAME,
    NpuBenchRunnerError,
    REPEATS,
    WARM_UP,
    _atomic_json,
    _create_real_child_directory,
    _ensure_real_child_directory,
    _make_tree_read_only,
    _positive_case_count,
    _require_regular,
    _safe_prof_tag,
    _workspace_runtime_directory,
    profile_tree_sha256,
    tree_sha256,
)


def _default_profiler_summary() -> Path:
    """Locate the packaged quick-profiler engine.

    The plugin owns NPUKernelBench's native fixture adapter.  It deliberately
    reuses the existing summary engine for the actual msprof W3/R5 collection
    and parsing, without changing the shared ``ops/`` implementation.  A
    Claude marketplace install contains only this plugin directory, so the
    byte-identical runtime copy in the ``vendor/`` directory beside this
    runner package is authoritative.  The
    checkout lookup is retained solely to support older package layouts while
    preserving the same invocation contract.
    """
    here = Path(__file__).resolve()
    # The runner modules live in the orchestrator/npubench/ subpackage in the
    # checkout (vendor/ is its sibling under orchestrator/), but are copied
    # out as a FLAT staged directory for isolated execution.  Probe both
    # layouts.
    for base in (here.parent.parent, here.parent):
        bundled = base / "vendor" / "msprof_perf_summary.py"
        if bundled.is_file() and not bundled.is_symlink():
            return bundled
    for parent in here.parents:
        script = parent / "ops" / "ops-profiling" / "scripts" / "msprof_perf_summary.py"
        if script.is_file() and not script.is_symlink():
            return script
    raise NpuBenchRunnerError(
        "cannot locate packaged msprof_perf_summary.py; reinstall the "
        "ascendc-port-orchestrator plugin"
    )


def _load_quick_profiler_summary(
    adapter: Path,
    *,
    expected_case_count: int,
    expected_valid_case_indices: Sequence[int],
    run_id: str,
    profile_root: Path,
    native: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Read and independently validate complete quick-profile coverage.

    ``performance.json`` was written by code that imported the benchmark
    candidate, so it is an input to the parent publisher rather than evidence
    by itself.  A successful process with one convenient case out of N must
    not be promoted to a gate PASS.  Validate every case, exact tagged raw
    artifact location, and all timing values before copying anything out of
    the evaluation child.
    """
    summary = _read_quick_profiler_report(Path(adapter))
    expected_valid = _validated_expected_valid_cases(expected_case_count, expected_valid_case_indices)
    native_case_records = _native_case_records(native)
    by_index = _indexed_quick_profiler_rows(summary, expected_case_count, len(expected_valid))
    sanitized_rows: list[dict[str, Any]] = []
    for index in range(expected_case_count):
        sanitized_rows.append(
            _sanitized_quick_profiler_row(
                by_index[index],
                index,
                native_case=native_case_records.get(index) if native is not None else None,
                is_valid=index in expected_valid,
                run_id=run_id,
                profile_root=profile_root,
            )
        )
    # Return only fields whose semantics the parent recomputed or validated;
    # never publish arbitrary candidate-controlled report keys verbatim.
    speedups = [row["speedup"] for row in sanitized_rows if "speedup" in row]
    sanitized: dict[str, Any] = {
        "profiling_mode": "quick",
        "gate_eligible": True,
        "warmup": WARM_UP,
        "repeats": REPEATS,
        "profiler_artifacts_retained": True,
        "n_cases_total": expected_case_count,
        "n_cases_valid": len(expected_valid),
        "geomean_speedup": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "mean_speedup": sum(speedups) / len(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "per_case": sanitized_rows,
    }
    if native is not None:
        sanitized.update(
            {
                "native_input_manifest_sha256": native["manifest_sha256"],
                "native_fixture_sha256": native["fixture_sha256"],
            }
        )
    return sanitized


def _read_quick_profiler_report(adapter: Path) -> Mapping[str, Any]:
    """Read ``performance.json`` and check the quick-mode measurement protocol."""
    summary_path = adapter / "performance.json"
    _require_regular(summary_path, "quick profiler performance.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"quick profiler report is unreadable: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise NpuBenchRunnerError("quick profiler report must be an object")
    if summary.get("profiling_mode") != "quick":
        raise NpuBenchRunnerError("profiler did not produce quick evidence")
    if summary.get("warmup") != WARM_UP or summary.get("repeats") != REPEATS:
        raise NpuBenchRunnerError("profiler report does not prove quick warmup=3/repeats=5")
    return summary


def _validated_expected_valid_cases(
    expected_case_count: int, expected_valid_case_indices: Sequence[int]
) -> set[int]:
    """Return the frozen fixture's non-empty case indices after range checks."""
    if not _positive_case_count(expected_case_count):
        raise NpuBenchRunnerError("native fixture has no performance cases")
    expected_valid = {int(index) for index in expected_valid_case_indices}
    if not expected_valid:
        raise NpuBenchRunnerError("native fixture has no non-empty performance cases")
    if expected_valid - set(range(expected_case_count)):
        raise NpuBenchRunnerError("native fixture valid case indices are invalid")
    return expected_valid


def _native_case_records(native: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    """Index the frozen fixture's per-case metadata by case number."""
    records_by_index: dict[int, Mapping[str, Any]] = {}
    if native is None:
        return records_by_index
    manifest_sha = native.get("manifest_sha256")
    fixture_sha = native.get("fixture_sha256")
    records = native.get("case_fixtures")
    digests_present = isinstance(manifest_sha, str) and isinstance(fixture_sha, str)
    if not digests_present or not isinstance(records, list):
        raise NpuBenchRunnerError("native fixture metadata is incomplete")
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("case"), int):
            raise NpuBenchRunnerError("native fixture case metadata is invalid")
        records_by_index[int(record["case"])] = record
    return records_by_index


def _indexed_quick_profiler_rows(
    summary: Mapping[str, Any], expected_case_count: int, expected_valid_count: int
) -> dict[int, Mapping[str, Any]]:
    """Index the reported per-case rows, rejecting gaps and duplicates."""
    if summary.get("n_cases_total") != expected_case_count:
        raise NpuBenchRunnerError("quick profiler case total does not match frozen native fixture")
    if summary.get("n_cases_valid") != expected_valid_count:
        raise NpuBenchRunnerError("quick profiler valid case coverage is incomplete")
    rows = summary.get("per_case")
    if not isinstance(rows, list) or len(rows) != expected_case_count:
        raise NpuBenchRunnerError("quick profiler per_case coverage is incomplete")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise NpuBenchRunnerError("quick profiler has an invalid per_case row")
        index = row.get("case")
        numeric_index = not isinstance(index, bool) and isinstance(index, int)
        if not numeric_index or index not in range(expected_case_count):
            raise NpuBenchRunnerError("quick profiler has an invalid case index")
        if index in by_index:
            raise NpuBenchRunnerError("quick profiler emitted a duplicate case")
        by_index[index] = row
    if set(by_index) != set(range(expected_case_count)):
        raise NpuBenchRunnerError("quick profiler omitted one or more cases")
    return by_index


def _skipped_quick_profiler_row(
    row: Mapping[str, Any], index: int, native_case: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Check that a frozen empty case was explicitly skipped, and normalize it."""
    if row.get("skipped") != "empty_tensor":
        raise NpuBenchRunnerError("quick profiler did not explicitly skip a frozen empty case")
    if any(
        row.get(field) is not None
        for field in ("ref_us", "asc_us", "speedup", "ref_prof_dir", "asc_prof_dir")
    ):
        raise NpuBenchRunnerError("quick profiler emitted timing for a frozen empty case")
    skipped_row: dict[str, Any] = {"case": index, "skipped": "empty_tensor"}
    if native_case is not None:
        skipped_row.update(
            {
                "native_case_fixture_sha256": native_case["sha256"],
                "native_case_path": native_case["path"],
            }
        )
    return skipped_row


def _quick_profiler_timings(row: Mapping[str, Any], index: int) -> dict[str, float]:
    """Extract the three positive finite timings and re-derive the speedup."""
    values: dict[str, float] = {}
    for field in ("ref_us", "asc_us", "speedup"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NpuBenchRunnerError(f"quick profiler {field} is not numeric for case {index}")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise NpuBenchRunnerError(f"quick profiler {field} is not positive finite for case {index}")
        values[field] = number
    derived = values["ref_us"] / values["asc_us"]
    if not math.isclose(values["speedup"], derived, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise NpuBenchRunnerError(f"quick profiler speedup does not match timings for case {index}")
    return values


def _sanitized_quick_profiler_row(
    row: Mapping[str, Any],
    index: int,
    *,
    native_case: Mapping[str, Any] | None,
    is_valid: bool,
    run_id: str,
    profile_root: Path,
) -> dict[str, Any]:
    """Validate one reported case row and return only recomputed fields."""
    if not is_valid:
        return _skipped_quick_profiler_row(row, index, native_case)
    if row.get("skipped") not in (None, ""):
        raise NpuBenchRunnerError("quick profiler skipped a required case")
    values = _quick_profiler_timings(row, index)
    for field in ("ref_error", "asc_error"):
        if row.get(field) not in (None, ""):
            raise NpuBenchRunnerError(f"quick profiler reported {field} for required case {index}")
    ref_path = _validate_reported_profile_dir(
        row.get("ref_prof_dir"), profile_root, run_id, index, "ref"
    )
    asc_path = _validate_reported_profile_dir(
        row.get("asc_prof_dir"), profile_root, run_id, index, "asc"
    )
    sanitized_row: dict[str, Any] = {
            "case": index,
            "ref_us": values["ref_us"],
            "asc_us": values["asc_us"],
            "speedup": values["speedup"],
            "ref_prof_dir": str(ref_path),
            "asc_prof_dir": str(asc_path),
        }
    if native_case is not None:
        sanitized_row.update(
            {
                "native_case_fixture_sha256": native_case["sha256"],
                "native_case_path": native_case["path"],
            }
        )
    return sanitized_row


def _archive_retained_profiles(
    workspace: Path,
    adapter: Path,
    run_id: str,
    summary: Mapping[str, Any],
    *,
    expected_valid_case_indices: Sequence[int] | None = None,
    profile_root: Path | None = None,
    source_profile_root: Path | None = None,
    raw_path_mapper: Callable[[Path], Path] | None = None,
) -> tuple[str, str]:
    """Copy only profiler-reported raw directories into immutable evidence.

    The profiler runs under a temporary output root.  Persisting its raw paths
    is not durable evidence, and trusting arbitrary paths from a JSON file
    would be an exfiltration primitive.  Only real, non-symlink ``/tmp/msprof``
    directories (or an adapter-local directory) are accepted and copied.
    """
    raw_paths = _raw_profile_paths(
        summary,
        adapter=adapter,
        run_id=run_id,
        expected_valid_case_indices=expected_valid_case_indices,
        profile_root=profile_root,
        raw_path_mapper=raw_path_mapper,
    )
    if not raw_paths:
        raise NpuBenchRunnerError("quick profiler retained no raw PROF directories")
    destination, incoming = _profile_archive_slots(workspace, run_id)
    try:
        inventory = _copy_raw_profiles(
            raw_paths,
            incoming,
            adapter,
            source_profile_root if source_profile_root is not None else profile_root,
        )
        _atomic_json(
            incoming / "archive_manifest.json",
            {
                "schema": "cannbot.npubench.profile_archive/v1",
                "run_id": run_id,
                "profiles": inventory,
            },
        )
        os.replace(incoming, destination)
        _make_tree_read_only(destination)
        digest = profile_tree_sha256(destination)
        relative = destination.relative_to(Path(workspace)).as_posix()
        return relative, digest
    finally:
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True)


def _identity_path(path: Path) -> Path:
    """Default raw-path mapper: the profiler-reported path is used as-is."""
    return path


def _profile_archive_slots(workspace: Path, run_id: str) -> tuple[Path, Path]:
    """Reserve the evidence publication and staging directories for one run."""
    evidence_parent = _workspace_runtime_directory(
        workspace, EVIDENCE_DIRNAME, "NPUKernelBench evidence root"
    )
    evidence_root = _ensure_real_child_directory(
        evidence_parent, "profiles", "NPUKernelBench profile evidence root"
    )
    destination = evidence_root / run_id
    if destination.exists():
        raise NpuBenchRunnerError("profile archive destination already exists")
    incoming_parent = _ensure_real_child_directory(
        evidence_root, ".incoming", "NPUKernelBench profile incoming root"
    )
    incoming = _create_real_child_directory(
        incoming_parent, run_id, "NPUKernelBench profile incoming"
    )
    return destination, incoming


def _copy_raw_profiles(
    raw_paths: Sequence[Path], incoming: Path, adapter: Path, profile_root: Path | None
) -> list[dict[str, str]]:
    """Copy each allowed raw profile directory and re-verify its digest."""
    inventory: list[dict[str, str]] = []
    for index, source in enumerate(raw_paths):
        _require_allowed_profile_dir(source, adapter, profile_root=profile_root)
        source_digest = tree_sha256(source)
        target = incoming / f"profile_{index:03d}"
        shutil.copytree(source, target, symlinks=False)
        if tree_sha256(target) != source_digest:
            raise NpuBenchRunnerError("raw profile changed while being archived")
        inventory.append({"name": target.name, "tree_sha256": source_digest})
    return inventory


def _valid_profile_case_indices(
    cases: Sequence[Any], expected_valid_case_indices: Sequence[int] | None
) -> set[int]:
    """Return the case indices whose raw profile directories must exist."""
    if expected_valid_case_indices is not None:
        return {int(index) for index in expected_valid_case_indices}
    return {
        int(case.get("case"))
        for case in cases
        if isinstance(case, Mapping) and case.get("skipped") in (None, "")
    }


def _indexed_profile_cases(cases: Sequence[Any]) -> dict[int, Mapping[str, Any]]:
    """Index reported per_case rows by case number, rejecting duplicates."""
    by_index: dict[int, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise NpuBenchRunnerError("quick profiler report has invalid per_case row")
        index = case.get("case")
        numeric_index = not isinstance(index, bool) and isinstance(index, int)
        if not numeric_index or index in by_index:
            raise NpuBenchRunnerError("quick profiler report has invalid per_case case index")
        by_index[index] = case
    return by_index


def _case_raw_profile_paths(
    case: Mapping[str, Any],
    index: int,
    *,
    root: Path,
    run_id: str,
    mapper: Callable[[Path], Path],
) -> list[Path]:
    """Resolve one case's two reported raw profile directories."""
    paths: list[Path] = []
    for field in ("ref_prof_dir", "asc_prof_dir"):
        value = case.get(field)
        if not isinstance(value, str) or not value:
            raise NpuBenchRunnerError(f"quick profiler {field} is invalid")
        role = "ref" if field == "ref_prof_dir" else "asc"
        reported = _validate_reported_profile_dir(value, root, run_id, index, role)
        paths.append(Path(mapper(reported)).resolve())
    return paths


def _raw_profile_paths(
    summary: Mapping[str, Any],
    *,
    adapter: Path,
    run_id: str,
    expected_valid_case_indices: Sequence[int] | None,
    profile_root: Path | None,
    raw_path_mapper: Callable[[Path], Path] | None,
) -> list[Path]:
    """Resolve every raw profile directory the report claims for a valid case."""
    cases = summary.get("per_case")
    if not isinstance(cases, list):
        raise NpuBenchRunnerError("quick profiler report has no per_case list")
    valid = _valid_profile_case_indices(cases, expected_valid_case_indices)
    if not valid:
        return []
    root = Path(profile_root) if profile_root is not None else Path("/tmp")
    mapper = raw_path_mapper if raw_path_mapper is not None else _identity_path
    by_index = _indexed_profile_cases(cases)
    if set(by_index) != valid | (set(by_index) - valid):
        raise NpuBenchRunnerError("quick profiler report has invalid per_case coverage")
    raw_paths: list[Path] = []
    for index in sorted(valid):
        case = by_index.get(index)
        if case is None:
            raise NpuBenchRunnerError("quick profiler report omitted a required raw profile")
        raw_paths.extend(
            _case_raw_profile_paths(case, index, root=root, run_id=run_id, mapper=mapper)
        )
    if len({str(path) for path in raw_paths}) != len(raw_paths):
        raise NpuBenchRunnerError("quick profiler reused a raw profile directory")
    return raw_paths


def _expected_profile_dir(profile_root: Path, run_id: str, case_index: int, role: str) -> Path:
    """Return the deterministic parent used by the shared quick profiler."""
    if role not in {"ref", "asc"} or isinstance(case_index, bool) or case_index < 0:
        raise NpuBenchRunnerError("invalid deterministic profiler artifact identity")
    if not _safe_prof_tag(run_id):
        raise NpuBenchRunnerError("invalid deterministic profiler run tag")
    return Path(profile_root) / f"msprof_quick_{role}_{run_id}_c{case_index}_{run_id}"


def _validate_reported_profile_dir(
    value: Any,
    profile_root: Path,
    run_id: str,
    case_index: int,
    role: str,
) -> Path:
    """Accept only the direct ``PROF_*`` child returned by the summary engine."""
    if not isinstance(value, str) or not value:
        raise NpuBenchRunnerError("quick profiler profile directory is invalid")
    reported = Path(value)
    parent = _expected_profile_dir(profile_root, run_id, case_index, role)
    if reported.parent != parent or not reported.name.startswith("PROF_"):
        raise NpuBenchRunnerError("quick profiler raw artifact tag/path mismatch")
    return reported


def _require_allowed_profile_dir(
    path: Path,
    adapter: Path,
    *,
    profile_root: Path | None,
) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise NpuBenchRunnerError(f"retained profiler path is not a real directory: {path}")
    adapter_root = Path(adapter).resolve()
    tmp_root = Path(profile_root if profile_root is not None else "/tmp").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(adapter_root)
        return
    except ValueError:
        pass
    try:
        resolved.relative_to(tmp_root)
    except ValueError as exc:
        raise NpuBenchRunnerError("retained profiler path is outside adapter and /tmp") from exc
    relative = resolved.relative_to(tmp_root)
    if (
        len(relative.parts) != 2
        or not relative.parts[0].startswith("msprof_quick_")
        or not relative.parts[1].startswith("PROF_")
    ):
        raise NpuBenchRunnerError("retained /tmp profiler path has an unsafe name")


def _validate_requested_lease(lease_manifest: Path | None, *, role: str, device: int) -> None:
    if lease_manifest is None:
        return
    leases = _read_lease_manifest(lease_manifest)
    lease = leases.get(role)
    if not isinstance(lease, Mapping):
        raise NpuBenchRunnerError(f"lease manifest has no {role} lease")
    if lease.get("device") != device:
        raise NpuBenchRunnerError(f"{role} lease device does not match requested device")
    token = lease.get("token")
    if not isinstance(token, str) or not token:
        raise NpuBenchRunnerError(f"{role} lease has no token")


def _validate_parallel_leases(lease_manifest: Path | None, precision_device: int, performance_device: int) -> bool:
    if lease_manifest is None:
        return False
    leases = _read_lease_manifest(lease_manifest)
    for role, device in (("precision", precision_device), ("performance", performance_device)):
        lease = leases.get(role)
        if not isinstance(lease, Mapping):
            return False
        if lease.get("device") != device:
            return False
        if not isinstance(lease.get("token"), str) or not lease["token"]:
            return False
    return True


def _read_lease_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"lease manifest is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise NpuBenchRunnerError("lease manifest must be an object")
    leases = payload.get("leases", payload)
    if not isinstance(leases, Mapping):
        raise NpuBenchRunnerError("lease manifest leases must be an object")
    return leases


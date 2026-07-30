# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 2 of /aog-prior-art-verify — stage prior-art files for build/verify.

Given a `.prior_art_scan.json` (Phase 1 output) that detected prior art,
this module copies the relevant kernel + host + config files into
`workspace/<op>/.prior_art_candidate/` so Phase 3 (build) has an isolated
build root. The staged copy is read-only intent; downstream phases must
not mutate the candidate dir.

Layout mirrored to .prior_art_candidate/:
  op_kernel/arch35/<op_files>.{h,cpp}    # from Mode A or Mode B
  op_kernel/<op>_apt.cpp                  # if upstream apt.cpp present
  op_host/config/ascend950/*.{json,ini}   # if upstream A5 config present
  manifest.json                           # source-of-truth: file list + sha + provenance

Usage:
  python3 stage_candidate.py --op <name> --port-source <path> --workspace <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StageReport:
    op: str
    candidate_dir: Path
    sources_staged: list[str] = field(default_factory=list)  # type names from scan
    files_staged: list[dict] = field(default_factory=list)
    repo_name: Optional[str] = None
    op_repo_rel_path: Optional[str] = None
    candidate_digest: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


_STAGEABLE_SOURCE_TYPES = {
    "upstream_arch35",
    "upstream_shared_common",
    "upstream_apt",
    "upstream_ascend950_config",
    "upstream_op_def",
}
_TARGET_SOURCE_TYPES = {
    "upstream_arch35",
    "upstream_shared_common",
    "upstream_apt",
    "upstream_ascend950_config",
    "upstream_op_def",
}
_IMPLEMENTATION_SOURCE_TYPES = {
    "upstream_arch35",
    "upstream_shared_common",
    "upstream_apt",
}


@dataclass(frozen=True)
class _StageItem:
    source: Path
    candidate_rel_path: Path
    build_rel_path: Path
    sha256: str
    source_type: str


def _repo_root_for(port_source: Path) -> Path:
    """Find the containing CANN operator repository."""
    resolved = port_source.resolve()
    for path in (resolved, *resolved.parents):
        if path.name.startswith("ops-"):
            return path
    return resolved.parent


def _safe_relative(path: Path, root: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root}: {path}") from exc
    if relative == Path(".") or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"unsafe {label}: {relative}")
    return relative


def _candidate_relative(source: Path, port_source: Path, source_type: str) -> Path:
    if source_type == "upstream_shared_common":
        return _safe_relative(source, port_source.parent, "shared candidate path")
    return _safe_relative(source, port_source, "candidate path")


def _collect_stage_plan(op: str, port_source: Path, scan_result: dict,
                        rep: StageReport) -> list[_StageItem]:
    """Validate authorization and all source digests before copying anything."""
    if scan_result.get("op") != op:
        rep.errors.append(
            f"scan op mismatch: expected {op!r}, got {scan_result.get('op')!r}"
        )
        return []
    recorded_port_source = scan_result.get("port_source")
    if not recorded_port_source:
        rep.errors.append("scan result has no port_source binding")
        return []
    resolved_source = port_source.resolve()
    if Path(recorded_port_source).resolve() != resolved_source:
        rep.errors.append(
            f"scan port_source mismatch: {recorded_port_source!r} != {str(port_source)!r}"
        )
        return []

    repo_root = _repo_root_for(resolved_source)
    rep.repo_name = repo_root.name
    try:
        rep.op_repo_rel_path = str(_safe_relative(
            resolved_source, repo_root, "operator repository path"
        ))
    except ValueError as exc:
        rep.errors.append(str(exc))
        return []

    consulted_target = scan_result.get("consulted_a5_sources") is True
    plan: list[_StageItem] = []
    destinations: dict[Path, str] = {}
    source_records = scan_result.get("sources")
    if not isinstance(source_records, list):
        rep.errors.append("scan sources must be a list")
        return []
    for source_record in source_records:
        if not isinstance(source_record, dict):
            rep.errors.append("scan contains a malformed source record")
            continue
        source_type = source_record.get("type")
        if source_type not in _STAGEABLE_SOURCE_TYPES:
            continue
        if source_type in _TARGET_SOURCE_TYPES and not consulted_target:
            rep.errors.append(
                f"scan did not authorize target-source staging: {source_type}"
            )
            continue
        records = source_record.get("file_records")
        if not isinstance(records, list) or not records:
            rep.errors.append(
                f"scan source {source_type} lacks digest-bound file_records"
            )
            continue
        for record in records:
            source_value = record.get("path") if isinstance(record, dict) else None
            expected_sha = record.get("sha256") if isinstance(record, dict) else None
            if not isinstance(source_value, str) or not isinstance(expected_sha, str):
                rep.errors.append(f"malformed file record in {source_type}")
                continue
            if len(expected_sha) != 64:
                rep.errors.append(
                    f"scan digest for {source_value} is not a full SHA-256"
                )
                continue
            source = Path(source_value).resolve()
            if not source.is_file():
                rep.errors.append(f"authorized source disappeared after scan: {source}")
                continue
            actual_sha = _sha(source)
            if actual_sha != expected_sha:
                rep.errors.append(
                    f"source changed after scan: {source} "
                    f"(expected {expected_sha}, got {actual_sha})"
                )
                continue
            try:
                build_rel = _safe_relative(source, repo_root, "build overlay path")
                candidate_rel = _candidate_relative(source, resolved_source, source_type)
            except ValueError as exc:
                rep.errors.append(str(exc))
                continue
            prior_sha = destinations.get(candidate_rel)
            if prior_sha is not None and prior_sha != actual_sha:
                rep.errors.append(
                    f"conflicting sources map to candidate path {candidate_rel}"
                )
                continue
            if prior_sha is not None:
                continue
            destinations[candidate_rel] = actual_sha
            plan.append(_StageItem(
                source=source,
                candidate_rel_path=candidate_rel,
                build_rel_path=build_rel,
                sha256=actual_sha,
                source_type=source_type,
            ))

    if plan and not any(
            item.source_type in _IMPLEMENTATION_SOURCE_TYPES for item in plan):
        rep.errors.append("scan authorized only companion files, not a target implementation")
        return []
    if not plan and not rep.errors:
        rep.errors.append(
            "scan contains no digest-bound candidate source authorized for staging"
        )
    return sorted(plan, key=lambda item: str(item.candidate_rel_path))


def compute_candidate_digest(op: str, repo_name: str,
                             files: list[dict]) -> str:
    """Hash the logical candidate without local absolute source paths."""
    normalized = {
        "op": op,
        "repo_name": repo_name,
        "files": sorted(
            [
                {
                    "rel_path": item["rel_path"],
                    "build_rel_path": item["build_rel_path"],
                    "sha": item["sha"],
                    "source_type": item["source_type"],
                }
                for item in files
            ],
            key=lambda item: (item["build_rel_path"], item["rel_path"]),
        ),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage(op: str, port_source: Path, workspace: Path,
          scan_result: Optional[dict] = None) -> StageReport:
    """Phase 2 entry. If `scan_result` is None, reads workspace/.prior_art_scan.json.

    Returns StageReport; caller writes manifest + handles errors.
    """
    candidate_dir = workspace / ".prior_art_candidate"
    rep = StageReport(op=op, candidate_dir=candidate_dir)

    # Idempotent: clear stale candidate dir if present (manifest will be rewritten)
    if candidate_dir.is_dir():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)

    # Load scan result for has_prior_art check
    if scan_result is None:
        scan_path = workspace / ".prior_art_scan.json"
        if not scan_path.is_file():
            rep.errors.append(
                ".prior_art_scan.json missing — run scan_prior_art first"
            )
            return rep
        try:
            scan_result = json.loads(scan_path.read_text())
        except Exception as e:
            rep.errors.append(f"failed to parse .prior_art_scan.json: {e}")
            return rep

    if not scan_result.get("has_prior_art"):
        rep.errors.append("scan reported has_prior_art=False; nothing to stage")
        return rep

    plan = _collect_stage_plan(op, port_source, scan_result, rep)
    if rep.errors:
        return rep

    for item in plan:
        destination = candidate_dir / item.candidate_rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
        if _sha(destination) != item.sha256:
            rep.errors.append(
                f"staged copy digest mismatch for {item.candidate_rel_path}"
            )
            continue
        rep.files_staged.append({
            "rel_path": str(item.candidate_rel_path),
            "build_rel_path": str(item.build_rel_path),
            "sha": item.sha256,
            "source_type": item.source_type,
            "origin": str(item.source),
        })
        if item.source_type not in rep.sources_staged:
            rep.sources_staged.append(item.source_type)

    if rep.errors:
        return rep
    if not rep.files_staged:
        rep.errors.append("no candidate files staged")
        return rep
    rep.candidate_digest = compute_candidate_digest(
        op, rep.repo_name or "", rep.files_staged
    )

    return rep


def write_manifest(rep: StageReport) -> Path:
    """Write manifest.json into candidate_dir with full provenance."""
    manifest = {
        "schema_version": 2,
        "op": rep.op,
        "candidate_dir": str(rep.candidate_dir),
        "repo_name": rep.repo_name,
        "op_repo_rel_path": rep.op_repo_rel_path,
        "candidate_digest": rep.candidate_digest,
        "sources_staged": rep.sources_staged,
        "file_count": len(rep.files_staged),
        "files": rep.files_staged,
        "warnings": rep.warnings,
        "errors": rep.errors,
    }
    out = rep.candidate_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True)
    p.add_argument("--port-source", required=True, type=Path)
    p.add_argument("--workspace", required=True, type=Path)
    args = p.parse_args(argv)
    rep = stage(args.op, args.port_source, args.workspace)
    manifest_path = write_manifest(rep)
    print(f"staged {len(rep.files_staged)} files → {manifest_path}")
    if rep.errors:
        print("ERRORS:", file=sys.stderr)
        for e in rep.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    if rep.warnings:
        print("WARNINGS:")
        for w in rep.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

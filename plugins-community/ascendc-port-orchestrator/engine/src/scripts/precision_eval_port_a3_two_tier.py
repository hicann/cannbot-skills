#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Grade arch35 outputs against fresh arch22 NPU captures.

The compatibility filename and tier-shaped result fields are retained for existing
callers. The authoritative verdict is single-source: ``a5_capture.pt`` is compared
with the harness-owned live arch22 NPU tensors in ``edge_dataset.pt``. A valid
``a3_capture/v2`` manifest binds those tensors to the current source-only snapshot
and hashes every capture input.

CPU fp64 truth and CPU native-dtype output are optional diagnostics. They cannot
authorize a pass, replace a missing arch22 tensor, or change the primary verdict.
Synthetic fallbacks, partial captures, stale manifests, and case-count mismatches
fail closed before grading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

# Reuse the canonical per-case judge — single source of truth for tier math.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision_eval_two_tier import classify_output  # noqa: E402


# A tagged native CPU capture may be used only for supplemental CPU diagnostics.
# It is never passed to the source-NPU primary comparison.
from cannbench_grader import NATIVE_CARVEOUT_WHITELIST  # noqa: E402


# ---------------------------------------------------------------------------
# Explicit precision-standard selection (controllable-harness #4, 2026-07-04)
# ---------------------------------------------------------------------------
# THE INVARIANT (docs/design/CONTROLLABLE_HARNESS_DESIGN.md): the precision
# standard is an EXPLICIT, selectable choice that MUST be (1) provenance-recorded
# in verification.json, (2) fail-loud when ambiguous/invalid (a *documented*
# default may default, but it is RECORDED, not silent), (3) never a silent
# coverage-narrowing knob.
#
# These legacy choices are retained for report compatibility. They select only
# the optional CPU diagnostic route; the arch22-NPU primary verdict is invariant.
VALID_PRECISION_STANDARDS = ("ecosystem", "commercial")
# Map the compatibility option to the supplemental CPU-diagnostic route.
_PRECISION_STANDARD_TO_ROUTE = {"ecosystem": "ecosystem", "commercial": "commercial"}
# DOCUMENTED default (owner-directed 2026-06-30). Allowed to apply silently ONLY
# because it is documented here AND recorded as precision_standard_source="default".
DEFAULT_PRECISION_STANDARD = "ecosystem"


def _precision_standard_from_env_file(env_path: Path) -> tuple[bool, Optional[str]]:
    """Return whether an env file declares the standard and its raw value."""
    try:
        if not env_path.is_file():
            return False, None
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("PRECISION_STANDARD="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                return True, value or None
    except OSError:
        return False, None
    return False, None


def _read_ascendc_env_precision_standard(workspace: Path) -> "tuple[Optional[str], Optional[Path]]":
    """Read `PRECISION_STANDARD=` from the canonical `.ascendc_env`.

    The canonical file for a port_a3 workspace (`_PROJECT_ROOT/workspace/<op>/`) is
    `workspace.parent/.ascendc_env` (== briefs._common.DEFAULT_ASCENDC_ENV). We read
    that; if absent, we also check `workspace/.ascendc_env` (belt for hermetic test
    fixtures / vendored layouts). Returns (value_or_None, path_read_or_None). The
    value is NOT validated here — the resolver validates + fails loud so the error
    names the source.
    """
    for env_path in (workspace.parent / ".ascendc_env", workspace / ".ascendc_env"):
        found, value = _precision_standard_from_env_file(env_path)
        if found:
            return value, env_path
    return None, None


def resolve_precision_standard(
    workspace: Path,
    cli_value: "Optional[str]" = None,
    *,
    env: "Optional[dict]" = None,
) -> "tuple[str, str]":
    """Resolve the effective precision standard + its provenance (THE INVARIANT #4).

    Precedence (first authoritative source wins):
      1. `cli_value`            (the `--precision-standard` flag) — source="cli"
      2. `PRECISION_STANDARD`   env var                          — source="env"
      3. `.ascendc_env` file    `PRECISION_STANDARD=`            — source="ascendc_env"
      4. DEFAULT_PRECISION_STANDARD (documented)                 — source="default"

    FAIL-LOUD (raises ValueError, never a silent wrong-pick):
      - any EXPLICIT value (cli/env/file) that is non-empty but not in
        VALID_PRECISION_STANDARDS → the caller mistyped / picked a non-existent
        standard; refuse rather than silently fall through to the default.
      - env var AND `.ascendc_env` BOTH set to CONFLICTING valid values → two
        authorities disagree; the choice is ambiguous → refuse (do not guess).

    Returns (standard, source). `standard` ∈ VALID_PRECISION_STANDARDS.
    """
    environ = env if env is not None else os.environ

    def _check(val: "Optional[str]", where: str) -> "Optional[str]":
        if val is None:
            return None
        v = val.strip()
        if not v:
            return None
        if v not in VALID_PRECISION_STANDARDS:
            raise ValueError(
                f"precision_standard invalid: {v!r} (from {where}). "
                f"Valid choices: {list(VALID_PRECISION_STANDARDS)} "
                f"(生态=ecosystem, 商用=commercial). Fix the {where} value — "
                f"refusing to silently fall back to the default (would grade with "
                f"the WRONG standard)."
            )
        return v

    cli = _check(cli_value, "--precision-standard")
    if cli is not None:
        return cli, "cli"

    env_val = _check(environ.get("PRECISION_STANDARD"), "PRECISION_STANDARD env var")
    file_raw, file_path = _read_ascendc_env_precision_standard(workspace)
    file_val = _check(file_raw, f".ascendc_env ({file_path})" if file_path else ".ascendc_env")

    # Ambiguity: env AND file both set, and they DISAGREE → fail loud, don't guess.
    if env_val is not None and file_val is not None and env_val != file_val:
        raise ValueError(
            f"precision_standard AMBIGUOUS: PRECISION_STANDARD env var = {env_val!r} "
            f"but {file_path} says {file_val!r}. Two authorities disagree — refusing "
            f"to silently pick one (would grade with a possibly-wrong standard). "
            f"Set them to the same value, or drop one."
        )

    if env_val is not None:
        return env_val, "env"
    if file_val is not None:
        return file_val, "ascendc_env"
    return DEFAULT_PRECISION_STANDARD, "default"


# ---------------------------------------------------------------------------
# Synth-conflation / A3-availability detection
# ---------------------------------------------------------------------------
def is_synth_conflation(workspace: Path) -> bool:
    """True iff edge_dataset.pt's a3_outputs is the CPU-canonical synth fallback
    (NOT a genuine A3-NPU capture).

    Detected via the `.truth_source_override` marker that
    `phase_o25_a3_ref.synthesize_cpu_truth_dataset` drops, whose `truth_source=`
    field is `cpu_canonical_via_synthetic_edge_dataset`. When this marker is
    present, `a3_outputs` == CPU truth (the SAME tensor), so it cannot serve as a
    genuine T2 competitor: tier2 is A3_UNAVAILABLE for the whole op.
    """
    marker = workspace / ".truth_source_override"
    if not marker.is_file():
        return False
    try:
        body = marker.read_text(errors="ignore")
    except OSError:
        return False
    return "cpu_canonical" in body


# ---------------------------------------------------------------------------
# Per-case source-NPU classifier
# ---------------------------------------------------------------------------
def _source_npu_unavailable_result(error: str) -> dict[str, Any]:
    """Build the fail-closed result used when source-NPU comparison is impossible."""
    return {
        "verdict": "EVAL_ERR",
        "error": error,
        "tier1_pass": False,
        "tier2_status": "A3_UNAVAILABLE",
        "primary_truth": "source_npu_arch22",
    }


def _classify_source_npu_output(
    ours: torch.Tensor,
    a3_cann: torch.Tensor,
    cand_orig_dtype: "torch.dtype | None",
) -> tuple[dict[str, Any], Any]:
    """Classify against source-NPU truth and return the chosen candidate dtype."""
    ref_dtype = a3_cann.dtype
    cand_dtype = (
        cand_orig_dtype
        if cand_orig_dtype is not None
        else getattr(ours, "dtype", ref_dtype)
    )
    try:
        ours_cmp = (
            ours.detach().cpu().to(ref_dtype)
            if ours.dtype != ref_dtype
            else ours.detach().cpu()
        )
        a3_cmp = a3_cann.detach().cpu()
    except (RuntimeError, AttributeError) as exc:
        return _source_npu_unavailable_result(
            f"source-NPU comparison cast failed: {exc!r}"
        ), cand_dtype

    # Reuse the canonical tolerance engine with the source-NPU tensor on both
    # reference axes. Thus no CPU value can influence the primary verdict.
    result = classify_output(
        ours_cmp,
        a3_cmp,
        a3_cmp,
        cand_orig_dtype=cand_dtype,
        native_output=None,
        route="ecosystem",
    )
    source_pass = result.get("verdict") in ("PASS_T1", "PASS_T1_SMALLVAL")
    if result.get("verdict") != "EVAL_ERR":
        result["verdict"] = "PASS_T1" if source_pass else "FAIL"
    result["tier1_pass"] = source_pass
    result["tier2_status"] = "N/A_SOURCE_NPU_PRIMARY"
    result["primary_truth"] = "source_npu_arch22"
    result["truth_source"] = "fresh_live_arch22_npu_capture"
    result["cpu_truth_role"] = "diagnostic_only"
    return result, cand_dtype


def _add_cpu_diagnostic(
    result: dict[str, Any],
    ours: torch.Tensor,
    cpu_truth: Optional[torch.Tensor],
    native: Optional[torch.Tensor],
    cand_dtype: Any,
    route: str,
) -> None:
    """Attach the optional CPU-only diagnostic without changing the source verdict."""
    if cpu_truth is None:
        return
    try:
        cpu_cmp = cpu_truth.detach().cpu()
        ours_cpu = ours.detach().cpu().to(cpu_cmp.dtype)
        native_cmp = native.detach().cpu() if native is not None else None
        diagnostic = classify_output(
            ours_cpu,
            cpu_cmp,
            cpu_cmp,
            cand_orig_dtype=cand_dtype,
            native_output=native_cmp,
            route=route,
        )
        result["cpu_diagnostic"] = {
            key: diagnostic.get(key)
            for key in ("verdict", "ours_mere", "ours_mare", "threshold")
        }
    except (RuntimeError, AttributeError) as exc:
        result["cpu_diagnostic"] = {"error": repr(exc)}


def classify_port_a3_case(
    ours: torch.Tensor,
    a3_cann: Optional[torch.Tensor],
    cpu_truth: Optional[torch.Tensor],
    *,
    native: Optional[torch.Tensor] = None,
    a3_available: bool = True,
    cand_orig_dtype: "torch.dtype | None" = None,
    route: str = "ecosystem",
) -> dict[str, Any]:
    """Grade one target output against the mandatory source-NPU capture.

    CPU/native references are optional diagnostics only.  Neither can authorize
    a migration PASS or substitute for a missing source-device tensor.
    """
    if not a3_available or a3_cann is None:
        return _source_npu_unavailable_result(
            "fresh source-NPU arch22 truth is unavailable"
        )
    result, cand_dtype = _classify_source_npu_output(
        ours, a3_cann, cand_orig_dtype
    )
    if result.get("tier2_status") == "A3_UNAVAILABLE":
        return result
    _add_cpu_diagnostic(result, ours, cpu_truth, native, cand_dtype, route)
    return result


# ---------------------------------------------------------------------------
# Op-level summary
# ---------------------------------------------------------------------------
def summarize(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate source-NPU verdicts into the legacy tier-shaped report."""
    n_total = len(per_case)
    n_t1 = sum(1 for c in per_case if c.get("tier1_pass") is True)
    n_t2 = sum(1 for c in per_case if c.get("tier2_status") == "PASS")
    n_fail = sum(1 for c in per_case if c.get("verdict") == "FAIL")
    n_err = sum(1 for c in per_case if c.get("verdict") == "EVAL_ERR")
    n_a3_unavail = sum(1 for c in per_case if c.get("tier2_status") == "A3_UNAVAILABLE")
    # Compatibility diagnostic for a missing mandatory source tensor.
    n_unresolved_a3 = 0
    for case in per_case:
        if (
            case.get("tier2_status") == "A3_UNAVAILABLE"
            and case.get("verdict") == "FAIL"
        ):
            n_unresolved_a3 += 1

    if n_total == 0:
        op_tier2_status = "N/A_NO_CASES"
    elif n_fail > 0 or n_err > 0:
        op_tier2_status = "FAIL"
    elif n_t1 == n_total:
        op_tier2_status = "N/A_ALL_T1"
    elif n_a3_unavail > 0 and n_t2 == 0:
        # Some cases failed T1 but no genuine A3 competitor existed for any of them.
        op_tier2_status = "A3_UNAVAILABLE"
    else:
        op_tier2_status = "PASS"

    status = "PASS" if (n_fail == 0 and n_err == 0 and n_total > 0) else "FAIL"

    return {
        "tier1_pass": n_t1,
        "tier2_pass": n_t2,
        "tier1_pass_inclusive": n_t1 + n_t2,
        "total": n_total,
        "tier2_status": op_tier2_status,
        "n_fail": n_fail,
        "n_err": n_err,
        "n_a3_unavailable": n_a3_unavail,
        "n_unresolved_a3_unavailable": n_unresolved_a3,
        "status": status,
        "method": "precision_eval_port_a3_two_tier (A5 vs fresh arch22 NPU truth)",
        "primary_truth": "source_npu_arch22",
        "truth_source": "fresh_live_arch22_npu_capture",
        "cpu_truth_role": "diagnostic_only",
        "results": per_case,
    }


# ---------------------------------------------------------------------------
# Loaders — robust to the two edge_dataset.pt provenances + worker capture shapes
# ---------------------------------------------------------------------------
def _coerce_case_list(obj):
    """Coerce an INT-KEYED edge_dataset dict {0:case, 1:case, ...} → ordered list.

    aog-a3-author-authored run_a3_reference.py scripts sometimes emit the per-case
    dataset as an int-keyed dict (e.g. flash_attention_score: {0:{name,a3_outputs,
    meta}, ..., 42}) instead of the documented list-of-case-dict. That shape has no
    top-level a3_outputs/outputs/cases key, so _to_tensor_list below would fall to
    `return [blob]` and wrap ALL N cases as a SINGLE element → the grader silently
    grades 1 case, not N (a hollow PASS far more dangerous than a hard fail).

    Coerces a dict whose keys are EXACTLY the contiguous ints 0..N-1 (or their str
    forms) into [obj[0], ..., obj[N-1]]. Leaves lists + the documented dict schemas
    untouched. Pure-stdlib. Robustness fix (fix-harness-for-next-customer), NOT a
    per-archive data patch.

    NOTE: keep the coercion LOGIC in sync with phase_o25_a3_ref._coerce_case_list
    (mirrored, not shared-imported, to avoid the in-container sibling-module import
    fragility that bit DEBT-184; docstrings differ per file).
    """
    if not isinstance(obj, dict) or not obj:
        return obj
    # A documented schema (aligned-lists / cases-keyed) — never touch.
    if any(k in obj for k in ("a3_outputs", "inputs", "outputs", "cases")):
        return obj
    try:
        indexed_keys = sorted((int(key), key) for key in obj)
    except (TypeError, ValueError):
        return obj  # non-int keys → not an int-keyed dataset
    if any(index != position for position, (index, _) in enumerate(indexed_keys)):
        return obj  # not contiguous 0..N-1 → don't guess
    return [obj[key] for _, key in indexed_keys]


def _to_tensor_list(blob: Any) -> list:
    """Normalize a loaded *.pt blob to a per-case list. Accepts:
      - list  → as-is
      - int-keyed dict {0:case,...,N} → ordered list (via _coerce_case_list)
      - dict with 'a3_outputs' / 'outputs' / 'cases' → that list
      - else  → wrap single item in a 1-list
    """
    blob = _coerce_case_list(blob)
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in ("a3_outputs", "outputs", "cases"):
            if k in blob:
                return blob[k]
        return [blob]
    return [blob]


def _first_tensor(item: Any) -> Optional[torch.Tensor]:
    """Extract the comparison tensor from a per-case capture record. Handles a bare
    tensor, a list/tuple (first tensor), a dict (per-case record with a3_outputs /
    output / out), or None (A3-error slot)."""
    if item is None:
        return None
    if isinstance(item, torch.Tensor):
        return item
    if isinstance(item, dict):
        if item.get("a3_error"):
            return None
        for k in ("a3_outputs", "output", "out", "result"):
            if k in item:
                return _first_tensor(item[k])
        # dict of tensors with no known key → first tensor value
        for v in item.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(item, (list, tuple)):
        for v in item:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_stage_state_error(
    state: dict, stage_root: Path, stage_manifest: Path
) -> Optional[str]:
    """Return the durable-state binding error, if the source snapshot is unbound."""
    if state.get("port_a3_source") != str(stage_root):
        return "migration state does not use the fixed source-only snapshot"
    if state.get("graybox_arch22_dir") != str(stage_root):
        return "sandbox source bind does not use the fixed source-only snapshot"
    if state.get("source_stage_manifest") != str(stage_manifest):
        return "migration state points at an unexpected source manifest"
    if state.get("graybox_sandbox") is not True:
        return "migration sandbox is not mandatory in durable state"
    return None


def _load_source_stage_manifest(
    stage_root: Path, stage_manifest: Path
) -> tuple[Optional[dict], Optional[str]]:
    """Load a concrete source-stage manifest without following an unsafe root."""
    try:
        if stage_root.is_symlink() or not stage_root.is_dir():
            raise ValueError("fixed source snapshot is missing or symlinked")
        return json.loads(stage_manifest.read_text()), None
    except Exception as exc:
        return None, f"source-only snapshot unreadable: {exc}"


def _source_stage_metadata_error(stage_payload: dict) -> Optional[str]:
    """Check the source-only manifest fields needed before its file walk."""
    if stage_payload.get("schema") != "source_stage/v1":
        return "unsupported source-only snapshot schema"
    if (
        stage_payload.get("source_arch") != "arch22"
        or stage_payload.get("target_arch") != "arch35"
    ):
        return "source-only snapshot architecture binding is invalid"
    detection = stage_payload.get("source_arch_detection")
    if not isinstance(detection, dict) or detection.get("arch") != "arch22":
        return "source-only snapshot has no arch22 detection evidence"
    if detection.get("supported") is not True:
        return "source-only snapshot architecture is unsupported"
    return None


def _source_stage_item(
    item: Path,
    stage_root: Path,
    stage_manifest: Path,
    forbidden_dirs: set[str],
    forbidden_markers: tuple[str, ...],
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Validate one staged item and return either its directory or file inventory."""
    if item == stage_manifest:
        return None, None
    lower = item.name.lower()
    if lower in forbidden_dirs or any(marker in lower for marker in forbidden_markers):
        raise ValueError(f"forbidden target/generated name: {item.relative_to(stage_root)}")
    if item.is_symlink():
        raise ValueError(
            f"source-only snapshot contains symlink: {item.relative_to(stage_root)}"
        )
    if item.is_dir():
        return item, None
    if item.is_file():
        return None, {
            "path": item.relative_to(stage_root).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256(item),
        }
    raise ValueError(
        f"source-only snapshot contains a non-file: {item.relative_to(stage_root)}"
    )


def _walk_source_stage_entries(
    stage_root: Path, stage_manifest: Path
) -> list[dict[str, Any]]:
    """Walk and inventory the source-only snapshot in deterministic order."""
    forbidden_dirs = {
        ".git",
        "__pycache__",
        "arch35",
        "ascend950",
        "build",
        "dist",
        "install",
        "output",
    }
    forbidden_markers = ("arch35", "ascend950", "dav_c310", "v351")
    entries: list[dict[str, Any]] = []
    pending = [stage_root]
    while pending:
        current = pending.pop()
        for item in sorted(current.iterdir()):
            directory, entry = _source_stage_item(
                item, stage_root, stage_manifest, forbidden_dirs, forbidden_markers
            )
            if directory is not None:
                pending.append(directory)
            elif entry is not None:
                entries.append(entry)
    return entries


def _collect_source_stage_entries(
    stage_root: Path, stage_manifest: Path
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Return a deterministic source-stage file inventory or its validation error."""
    try:
        entries = _walk_source_stage_entries(stage_root, stage_manifest)
    except Exception as exc:
        return None, f"source-only snapshot validation failed: {exc}"
    return entries, None


def _validate_source_stage_binding(workspace: Path, state: dict) -> tuple[bool, str]:
    """Independently verify the fixed source-only snapshot on the verify host."""
    stage_root = workspace / ".source_arch22"
    stage_manifest = stage_root / ".source_stage_manifest.json"
    state_error = _source_stage_state_error(state, stage_root, stage_manifest)
    if state_error:
        return False, state_error
    stage_payload, read_error = _load_source_stage_manifest(stage_root, stage_manifest)
    if read_error:
        return False, read_error
    metadata_error = _source_stage_metadata_error(stage_payload)
    if metadata_error:
        return False, metadata_error
    entries, entries_error = _collect_source_stage_entries(stage_root, stage_manifest)
    if entries_error:
        return False, entries_error
    entries.sort(key=lambda entry: entry["path"])
    digest = _source_tree_digest(entries)
    if stage_payload.get("files") != entries:
        return False, "source-only snapshot file inventory/hash mismatch"
    if stage_payload.get("file_count") != len(entries):
        return False, "source-only snapshot file count mismatch"
    if stage_payload.get("tree_sha256") != digest:
        return False, "source-only snapshot manifest digest mismatch"
    if state.get("source_stage_digest") != digest:
        return False, "migration state source digest mismatch"
    return True, f"source-only arch22 snapshot verified ({len(entries)} files)"


def _load_live_capture_files(workspace: Path) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Load the capture manifest and migration state needed by the live-capture gate."""
    try:
        manifest = json.loads((workspace / "a3_capture_manifest.json").read_text())
        state = json.loads((workspace / ".opgen_state.json").read_text())
    except Exception as exc:
        return None, None, f"fresh source-NPU provenance missing/unreadable: {exc}"
    return manifest, state, None


def _capture_binding_error(manifest: dict, state: dict) -> Optional[str]:
    """Validate capture identity fields after the source stage itself is verified."""
    if manifest.get("source_stage_digest") != state.get("source_stage_digest"):
        return "capture is not bound to the current source snapshot"
    if manifest.get("source_arch") != "arch22":
        return "capture source architecture is not arch22"
    device = manifest.get("device")
    if not isinstance(device, dict) or device.get("kind") != "NPU":
        return "capture has no source-NPU device provenance"
    if not isinstance(manifest.get("capture_id"), str) or not manifest["capture_id"]:
        return "capture_id missing"
    return None


def _capture_timestamp_error(manifest: dict) -> Optional[str]:
    """Validate the ISO timestamps and their chronological ordering."""
    try:
        started = datetime.fromisoformat(
            str(manifest["capture_started_ts"]).replace("Z", "+00:00")
        )
        completed = datetime.fromisoformat(
            str(manifest["capture_completed_ts"]).replace("Z", "+00:00")
        )
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise ValueError("invalid timestamp ordering/timezone")
    except Exception as exc:
        return f"capture timestamps invalid: {exc}"
    return None


def _capture_evidence_error(workspace: Path, manifest: dict) -> Optional[str]:
    """Verify every capture file against its manifest path, size, and digest."""
    expected = {
        "edge_inputs": "edge_inputs.pt",
        "edge_dataset": "edge_dataset.pt",
        "a3_baseline_perf": "a3_baseline_perf.json",
        "runner": "run_a3_reference.py",
    }
    files = manifest.get("files")
    if not isinstance(files, dict):
        return "capture file inventory missing"
    for label, filename in expected.items():
        path = workspace / filename
        entry = files.get(label)
        try:
            if not isinstance(entry, dict) or entry.get("path") != filename:
                raise ValueError("manifest entry missing")
            if path.is_symlink() or not path.is_file():
                raise ValueError("file missing/symlinked")
            if entry.get("size") != path.stat().st_size:
                raise ValueError("size mismatch")
            if entry.get("sha256") != _sha256(path):
                raise ValueError("SHA256 mismatch")
        except Exception as exc:
            return f"capture evidence {label} invalid: {exc}"
    return None


def _validate_live_a3_manifest(workspace: Path) -> tuple[bool, str, dict]:
    """Self-contained verifier for the harness-owned live-capture manifest.

    This evaluator is copied to the verification host as a standalone script,
    so it validates the capture binding without importing orchestrator modules.
    """
    workspace = workspace.resolve()
    manifest, state, read_error = _load_live_capture_files(workspace)
    if read_error:
        return False, read_error, {}
    if manifest.get("schema") != "a3_capture/v2" or manifest.get("live_exec") is not True:
        return False, "source truth is not a fresh harness-owned live capture", manifest
    if state.get("opgen_mode") != "port_a3_to_a5":
        return False, "workspace is not an authorized migration run", manifest
    stage_ok, stage_reason = _validate_source_stage_binding(workspace, state)
    if not stage_ok:
        return False, stage_reason, manifest
    binding_error = _capture_binding_error(manifest, state)
    if binding_error:
        return False, binding_error, manifest
    timestamp_error = _capture_timestamp_error(manifest)
    if timestamp_error:
        return False, timestamp_error, manifest
    evidence_error = _capture_evidence_error(workspace, manifest)
    if evidence_error:
        return False, evidence_error, manifest
    n_total = manifest.get("n_total")
    n_captured = manifest.get("n_captured")
    if not isinstance(n_total, int) or n_total <= 0 or n_captured != n_total:
        return False, "source-NPU capture is empty or partial", manifest
    return True, f"fresh live source-NPU capture verified ({n_total} cases)", manifest


def _resolve_load_precision(
    workspace: Path,
    precision_standard: "Optional[str]",
    precision_standard_source: "Optional[str]",
) -> tuple[str, str, str]:
    """Resolve and validate the diagnostic precision route for a load operation."""
    if precision_standard is None:
        precision_standard, precision_standard_source = resolve_precision_standard(
            workspace, cli_value=precision_standard
        )
    elif precision_standard not in VALID_PRECISION_STANDARDS:
        raise ValueError(
            f"precision_standard invalid: {precision_standard!r} "
            f"(valid: {list(VALID_PRECISION_STANDARDS)})"
        )
    if precision_standard_source is None:
        precision_standard_source = "explicit"
    return (
        precision_standard,
        precision_standard_source,
        _PRECISION_STANDARD_TO_ROUTE[precision_standard],
    )


def _failed_load_summary(
    error: str,
    *,
    total: int = 0,
    include_primary_truth: bool = False,
) -> dict[str, Any]:
    """Build the existing fail-closed summary shape for pre-grading failures."""
    summary: dict[str, Any] = {
        "tier1_pass": 0,
        "tier2_pass": 0,
        "tier1_pass_inclusive": 0,
        "total": total,
        "tier2_status": "FAIL",
        "status": "FAIL",
        "error": error,
        "results": [],
    }
    if include_primary_truth:
        summary["primary_truth"] = "source_npu_arch22"
    return summary


def _load_case_lists(
    a5_capture_path: Path, edge_dataset_path: Path, cpu_truth_path: Path
) -> tuple[list, list, list]:
    """Load target, source-NPU, and optional CPU diagnostic case sequences."""
    ours_list = _to_tensor_list(torch.load(a5_capture_path, weights_only=False))
    edge_blob = torch.load(edge_dataset_path, weights_only=False)
    a3_list = _to_tensor_list(edge_blob)
    if cpu_truth_path.is_file():
        cpu_list = _to_tensor_list(torch.load(cpu_truth_path, weights_only=False))
    else:
        cpu_list = []
    return ours_list, a3_list, cpu_list


def _load_native_diagnostic(
    native_path: Path,
) -> tuple[bool, Any, bool, Optional[list]]:
    """Load only the whitelisted native capture used for CPU diagnostics."""
    native_blob = torch.load(native_path, weights_only=False) if native_path.is_file() else None
    native_file_present = native_blob is not None
    native_kind = (
        native_blob.get("native_kind")
        if isinstance(native_blob, dict)
        else ("untagged_bare_list" if native_file_present else None)
    )
    native_usable = (
        isinstance(native_blob, dict) and native_kind in NATIVE_CARVEOUT_WHITELIST
    )
    native_list = _to_tensor_list(native_blob) if native_usable else None
    return native_file_present, native_kind, native_usable, native_list


def _case_capture_tensors(
    index: int,
    ours_item: Any,
    a3_list: list,
    cpu_list: list,
    native_list: Optional[list],
    native_usable: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract the target, source, CPU, and optional native tensors for one case."""
    native_tensor = (
        _first_tensor(native_list[index])
        if native_usable and native_list and index < len(native_list)
        else None
    )
    return (
        _first_tensor(ours_item),
        _first_tensor(a3_list[index]) if index < len(a3_list) else None,
        _first_tensor(cpu_list[index]) if index < len(cpu_list) else None,
        native_tensor,
    )


def _grade_capture_cases(
    ours_list: list,
    a3_list: list,
    cpu_list: list,
    native_list: Optional[list],
    native_usable: bool,
    route: str,
    verbose: bool,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    """Grade aligned captures while retaining the legacy native-diagnostic fields."""
    per_case: list[dict[str, Any]] = []
    n_native_used = 0
    missing_native_case_ids: list[int] = []
    for index, ours_item in enumerate(ours_list):
        ours_tensor, a3_tensor, cpu_tensor, native_tensor = _case_capture_tensors(
            index, ours_item, a3_list, cpu_list, native_list, native_usable
        )
        if ours_tensor is None or a3_tensor is None:
            per_case.append(
                {
                    "case": index,
                    "verdict": "EVAL_ERR",
                    "error": "ours or source-NPU truth tensor missing",
                    "tier2_status": "A3_UNAVAILABLE",
                }
            )
            continue
        if native_tensor is not None:
            n_native_used += 1
        else:
            missing_native_case_ids.append(index)
        result = classify_port_a3_case(
            ours_tensor,
            a3_tensor,
            cpu_tensor,
            native=native_tensor,
            a3_available=True,
            cand_orig_dtype=getattr(ours_tensor, "dtype", None),
            route=route,
        )
        result["case"] = index
        per_case.append(result)
        if verbose:
            print(
                f"case[{index}]: {result.get('verdict')} "
                f"(tier1_pass={result.get('tier1_pass')}, "
                f"tier2={result.get('tier2_status')})"
            )
    return per_case, n_native_used, missing_native_case_ids


def _annotate_capture_summary(
    summary: dict[str, Any],
    *,
    precision_standard: str,
    precision_standard_source: str,
    capture_manifest: dict,
    provenance_reason: str,
    native_file_present: bool,
    native_kind: Any,
    native_usable: bool,
    n_native_used: int,
    missing_native_case_ids: list[int],
) -> dict[str, Any]:
    """Attach non-authoritative diagnostics and capture provenance to a summary."""
    summary["synth_conflation"] = False
    summary["a3_available"] = True
    summary["precision_standard"] = precision_standard
    summary["precision_standard_source"] = precision_standard_source
    summary["precision_standard_role"] = "cpu_diagnostic_only"
    summary["grader"] = "fresh arch22 NPU truth (harness-side, author≠measurer)"
    summary["capture_id"] = capture_manifest.get("capture_id")
    summary["capture_source_stage_digest"] = capture_manifest.get("source_stage_digest")
    summary["capture_provenance"] = provenance_reason
    summary["native_capture_present"] = native_file_present
    summary["native_kind"] = native_kind
    summary["native_usable"] = native_usable
    summary["n_native_used"] = n_native_used
    summary["n_native_missing"] = len(missing_native_case_ids)
    summary["missing_native_case_ids"] = missing_native_case_ids
    summary["native_provision_ok"] = bool(
        native_usable and summary["n_native_missing"] == 0
    )
    return summary


def _print_capture_summary(summary: dict[str, Any]) -> None:
    """Print the established human-readable precision summary."""
    print("\n--- arch22-to-arch35 precision summary ---")
    print(
        f"  total={summary['total']} tier1_pass={summary['tier1_pass']} "
        f"tier2_pass={summary['tier2_pass']} "
        f"tier1_pass_inclusive={summary['tier1_pass_inclusive']}"
    )
    print(f"  op tier2_status={summary['tier2_status']} status={summary['status']}")


def _capture_file_paths(workspace: Path) -> tuple[Path, Path, Path]:
    """Return the required target/source paths and optional CPU diagnostic path."""
    return (
        workspace / "a5_capture.pt",
        workspace / "edge_dataset.pt",
        workspace / "cpu_truth_outputs.pt",
    )


def _missing_required_capture_files(*paths: Path) -> list[str]:
    """List missing required capture filenames in the established input order."""
    return [path.name for path in paths if not path.is_file()]


def _preflight_capture_data(
    workspace: Path,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Validate, load, and align all artifacts needed before per-case grading."""
    a5_capture_path, edge_dataset_path, cpu_truth_path = _capture_file_paths(workspace)
    provenance_ok, provenance_reason, capture_manifest = _validate_live_a3_manifest(
        workspace
    )
    if not provenance_ok:
        return None, _failed_load_summary(
            provenance_reason, include_primary_truth=True
        )
    if is_synth_conflation(workspace):
        return None, _failed_load_summary(
            "synthetic/CPU truth cannot substitute for source-NPU capture",
            include_primary_truth=True,
        )
    missing = _missing_required_capture_files(a5_capture_path, edge_dataset_path)
    if missing:
        return None, _failed_load_summary(
            f"missing required capture file(s): {missing}"
        )
    ours_list, a3_list, cpu_list = _load_case_lists(
        a5_capture_path, edge_dataset_path, cpu_truth_path
    )
    native_file_present, native_kind, native_usable, native_list = (
        _load_native_diagnostic(workspace / "native_capture.pt")
    )
    expected_cases = capture_manifest.get("n_total")
    if len(ours_list) != len(a3_list) or len(a3_list) != expected_cases:
        return None, _failed_load_summary(
            "case-count mismatch: "
            f"a5={len(ours_list)} source_npu={len(a3_list)} "
            f"manifest={expected_cases}",
            total=len(ours_list),
            include_primary_truth=True,
        )
    if len(a3_list) == 0:
        return None, _failed_load_summary("no comparable cases (empty capture lists)")
    return {
        "ours_list": ours_list,
        "a3_list": a3_list,
        "cpu_list": cpu_list,
        "native_file_present": native_file_present,
        "native_kind": native_kind,
        "native_usable": native_usable,
        "native_list": native_list,
        "capture_manifest": capture_manifest,
        "provenance_reason": provenance_reason,
    }, None


def load_and_classify(
    workspace: Path,
    verbose: bool = True,
    *,
    precision_standard: "Optional[str]" = None,
    precision_standard_source: "Optional[str]" = None,
) -> dict[str, Any]:
    """Grade target captures against source-NPU truth, failing closed on bad inputs."""
    workspace = workspace.resolve()
    precision_standard, precision_standard_source, route = _resolve_load_precision(
        workspace, precision_standard, precision_standard_source
    )
    capture_data, failed_summary = _preflight_capture_data(workspace)
    if failed_summary:
        return failed_summary

    per_case, n_native_used, missing_native_case_ids = _grade_capture_cases(
        capture_data["ours_list"],
        capture_data["a3_list"],
        capture_data["cpu_list"],
        capture_data["native_list"],
        capture_data["native_usable"],
        route,
        verbose,
    )
    summary = _annotate_capture_summary(
        summarize(per_case),
        precision_standard=precision_standard,
        precision_standard_source=precision_standard_source,
        capture_manifest=capture_data["capture_manifest"],
        provenance_reason=capture_data["provenance_reason"],
        native_file_present=capture_data["native_file_present"],
        native_kind=capture_data["native_kind"],
        native_usable=capture_data["native_usable"],
        n_native_used=n_native_used,
        missing_native_case_ids=missing_native_case_ids,
    )
    if verbose:
        _print_capture_summary(summary)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", help="workspace/<op> dir with a5_capture.pt + "
                                     "edge_dataset.pt + cpu_truth_outputs.pt")
    p.add_argument("--json")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--precision-standard", "-p", default=None,
        choices=list(VALID_PRECISION_STANDARDS),
        help="EXPLICIT grading standard: ecosystem (生态, default) | commercial (商用). "
             "When omitted, resolved from PRECISION_STANDARD env var, then .ascendc_env, "
             "then the documented default; an invalid/ambiguous choice FAILS LOUD. The "
             "resolved standard + source are recorded in the summary JSON.")
    args = p.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    # Resolve HERE so a bad --precision-standard / env / .ascendc_env fails loud with a
    # clear message before any capture I/O (THE INVARIANT #2).
    try:
        std, src = resolve_precision_standard(workspace, cli_value=args.precision_standard)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"precision_standard={std} (source={src})")
    s = load_and_classify(workspace, verbose=not args.quiet,
                          precision_standard=std, precision_standard_source=src)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, default=str))
        print(f"wrote json: {args.json}")
    return 0 if s.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

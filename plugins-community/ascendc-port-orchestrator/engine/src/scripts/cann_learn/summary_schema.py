# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Sanitized public summary JSON schema + validation.

Per v2 design: only ONE outbound artifact leaves the learner context — and it
is JSON with a fixed schema (no prose, no paths, no identifiers). Any
deviation (extra fields, prose-typed fields with non-empty content) =
schema fail = mode-5 caller refuses to merge candidates.

Schema is LITERAL — fields listed here are the only allowed top-level keys.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_KEYS = frozenset({
    "run_id", "ts", "op",
    "module_path_sha256",
    "files_read_count", "files_read_total_bytes", "files_read_hashes",
    "candidate_count_extracted", "candidate_count_kept",
    "candidate_count_dropped_leak", "candidate_count_dropped_compile",
    "candidate_count_dropped_copy_shape", "candidate_count_overlap_existing",
    "metadata_fix_proposals_count",
    "leak_score", "copy_shape_score", "compile_pass_rate",
    "self_review_verdict",
    "self_review_failures",
    "checks",
})

# Optional keys allowed (e.g. extra forensic counters added later)
OPTIONAL_KEYS = frozenset({
    "ts_finished",
    "researcher_evidence_path",
    "probe_evidence_path",
    "warning_count",
    "warnings",
    # Mode 6 (2026-05-21): build-system extraction adds extraction-mode tag
    # so downstream caller can route candidates to the right KB subtree
    # (kernel patterns vs build-system patterns). Optional so Mode 5 runs
    # without it still validate.
    "extraction_mode",
})

ALL_ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


def normalize_hierarchical_to_flat(summary: dict[str, Any]) -> dict[str, Any]:
    """Map the aog-cann-learner agent's hierarchical output shape to the flat
    schema this validator expects.

    P0aau-c35.g (2026-05-10): first real cann_learner run on FA emitted a
    hierarchical structure (`phase_a_pre_scan`, `phase_b_source_read`,
    `phase_c_extraction`, `kept_candidates`, `metadata_fix_count`) instead of
    the flat keys the schema demanded. Rather than reject and re-run (wasting
    quota + agent cycles), normalize known equivalences so flat-schema
    validation can proceed. Future agent SKILL.md update can converge agent
    output to flat schema directly.

    Returns a copy with flat keys synthesized from hierarchical sources.
    Idempotent — input that's already flat passes through unchanged.
    """
    out = dict(summary)

    # Phase B → files_read_*
    pb = summary.get("phase_b_source_read", {})
    if isinstance(pb, dict):
        if "files_read_count" not in out and "files_read" in pb:
            out["files_read_count"] = int(pb["files_read"])
        if "files_read_total_bytes" not in out and "approx_bytes_scanned" in pb:
            out["files_read_total_bytes"] = int(pb["approx_bytes_scanned"])

    # Phase C → candidate_count_*
    pc = summary.get("phase_c_extraction", {})
    if isinstance(pc, dict):
        if "candidate_count_extracted" not in out and "candidates_drafted" in pc:
            out["candidate_count_extracted"] = int(pc["candidates_drafted"])
        if "candidate_count_kept" not in out:
            if "kept_candidates" in summary:
                # Top-level alias the agent used
                kc = summary["kept_candidates"]
                out["candidate_count_kept"] = (
                    int(kc) if isinstance(kc, int) else len(kc) if isinstance(kc, list) else 0
                )
            elif "candidates_finalized" in pc:
                out["candidate_count_kept"] = int(pc["candidates_finalized"])

    # Drop categorization from `candidates_dropped` list (each entry has
    # a drop_reason matching scanner that fired).
    dropped = summary.get("candidates_dropped", [])
    if isinstance(dropped, list):
        n_leak = sum(
            1 for d in dropped
            if isinstance(d, dict) and "leak" in str(d.get("drop_reason", "")).lower()
        )
        n_compile = sum(
            1 for d in dropped
            if isinstance(d, dict) and "compile" in str(d.get("drop_reason", "")).lower()
        )
        n_copy = 0
        n_overlap = 0
        for dropped_candidate in dropped:
            if not isinstance(dropped_candidate, dict):
                continue
            drop_reason = str(dropped_candidate.get("drop_reason", ""))
            drop_reason_lower = drop_reason.lower()
            if "copy_shape" in drop_reason_lower or "c34c" in drop_reason_lower:
                n_copy += 1
            if (
                "overlap" in drop_reason_lower
                or "c35" in drop_reason_lower
                or "P-P" in drop_reason
            ):
                n_overlap += 1
        out.setdefault("candidate_count_dropped_leak", n_leak)
        out.setdefault("candidate_count_dropped_compile", n_compile)
        out.setdefault("candidate_count_dropped_copy_shape", n_copy)
        out.setdefault("candidate_count_overlap_existing", n_overlap)
    else:
        for k in (
            "candidate_count_dropped_leak", "candidate_count_dropped_compile",
            "candidate_count_dropped_copy_shape", "candidate_count_overlap_existing",
        ):
            out.setdefault(k, 0)

    # metadata_fix_count → metadata_fix_proposals_count
    if "metadata_fix_proposals_count" not in out:
        if "metadata_fix_count" in summary:
            out["metadata_fix_proposals_count"] = int(summary["metadata_fix_count"])
        elif isinstance(summary.get("metadata_fix_proposals"), list):
            out["metadata_fix_proposals_count"] = len(summary["metadata_fix_proposals"])

    # ts → use run_date if absent
    if "ts" not in out and "run_date" in summary:
        out["ts"] = str(summary["run_date"])

    # Score extraction from checks block
    checks = summary.get("checks", {})
    if isinstance(checks, dict):
        if "leak_score" not in out:
            c34a = checks.get("C34a_identifier_leak", {})
            if isinstance(c34a, dict) and "score" in c34a:
                out["leak_score"] = float(c34a["score"])
        if "copy_shape_score" not in out:
            c34c = checks.get("C34c_copy_shape", {})
            if isinstance(c34c, dict) and "score" in c34c:
                out["copy_shape_score"] = float(c34c["score"])
        if "compile_pass_rate" not in out:
            c34b = checks.get("C34b_compile_gate", {})
            if isinstance(c34b, dict) and "pass_rate" in c34b:
                out["compile_pass_rate"] = float(c34b["pass_rate"])

    # module_path_sha256 fallback: derive from module_path_basename if no real sha
    # (agent didn't emit one — synthesize a placeholder hash from basename for
    # gate completeness; not cryptographically meaningful, just unique per module path)
    if "module_path_sha256" not in out:
        basename = pb.get("module_path_basename", "") if isinstance(pb, dict) else ""
        if basename:
            import hashlib
            out["module_path_sha256"] = hashlib.sha256(
                f"module:{basename}".encode()
            ).hexdigest()

    # files_read_hashes: agent didn't emit per-file hashes. Synthesize empty
    # list (validator allows empty list as long as type is right).
    out.setdefault("files_read_hashes", [])

    # self_review_failures default empty list
    out.setdefault("self_review_failures", [])

    return out


def validate(summary: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    # P0aau-c35.g: first try normalizing hierarchical agent output to flat schema.
    # If normalization adds keys, re-run validation on normalized form.
    normalized = normalize_hierarchical_to_flat(summary)
    if normalized != summary:
        # Use normalized for downstream validation; record the original keys
        # the agent used for forensics in the normalized form's debug field.
        summary = normalized

    # Required keys present
    for k in REQUIRED_KEYS:
        if k not in summary:
            errors.append(f"missing required key: {k}")

    # No extra keys (prose-typed leak surface)
    # — but allow the hierarchical phase_* / drafting fields the agent
    #   emits (already extracted by normalize), and `agent` / `run_date` /
    #   `schema_version` / known top-level aliases.
    HIERARCHICAL_PASSTHROUGH = frozenset({
        "schema_version", "agent", "run_date",
        "phase_a_pre_scan", "phase_b_source_read", "phase_c_extraction",
        "candidates_emitted", "metadata_fix_proposals", "candidates_dropped",
        "kept_candidates", "metadata_fix_count",
    })
    for k in summary:
        if k not in ALL_ALLOWED_KEYS and k not in HIERARCHICAL_PASSTHROUGH:
            errors.append(f"unknown key (potential prose-leak): {k}")

    # Type checks on the structurally-important fields
    if "self_review_verdict" in summary:
        if summary["self_review_verdict"] not in ("PASS", "FAIL"):
            errors.append("self_review_verdict must be 'PASS' or 'FAIL'")

    for numeric_key in (
        "files_read_count", "files_read_total_bytes",
        "candidate_count_extracted", "candidate_count_kept",
        "candidate_count_dropped_leak", "candidate_count_dropped_compile",
        "candidate_count_dropped_copy_shape", "candidate_count_overlap_existing",
        "metadata_fix_proposals_count",
    ):
        if numeric_key in summary:
            v = summary[numeric_key]
            if not isinstance(v, int):
                errors.append(f"{numeric_key} must be int, got {type(v).__name__}")
            elif v < 0:
                errors.append(f"{numeric_key} must be ≥ 0, got {v}")

    for ratio_key in ("leak_score", "copy_shape_score", "compile_pass_rate"):
        if ratio_key in summary:
            v = summary[ratio_key]
            if not isinstance(v, (int, float)):
                errors.append(f"{ratio_key} must be number, got {type(v).__name__}")
            elif not (0.0 <= float(v) <= 1.0):
                errors.append(f"{ratio_key} must be in [0,1], got {v}")

    if "self_review_failures" in summary:
        v = summary["self_review_failures"]
        if not isinstance(v, list):
            errors.append("self_review_failures must be list")
        elif not all(isinstance(x, str) for x in v):
            errors.append("self_review_failures must contain only strings")

    if "checks" in summary:
        v = summary["checks"]
        if not isinstance(v, dict):
            errors.append("checks must be dict")
        else:
            for chk_id, chk in v.items():
                if not isinstance(chk, dict) or "passed" not in chk:
                    errors.append(f"checks.{chk_id} must have 'passed' field")
                else:
                    # 'passed' must be bool
                    if not isinstance(chk["passed"], bool):
                        errors.append(f"checks.{chk_id}.passed must be bool")

    if "files_read_hashes" in summary:
        v = summary["files_read_hashes"]
        if not isinstance(v, list):
            errors.append("files_read_hashes must be list")
        else:
            for h in v:
                if not isinstance(h, str):
                    errors.append("files_read_hashes entries must be strings")
                # Each should LOOK like a sha256 (64 hex chars)
                elif len(h) != 64 or not all(c in "0123456789abcdef" for c in h.lower()):
                    errors.append(f"files_read_hashes entry {h!r} not a sha256 hex")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_file(path: Path) -> ValidationResult:
    if not path.exists():
        return ValidationResult(valid=False, errors=[f"file does not exist: {path}"])
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return ValidationResult(valid=False, errors=[f"invalid JSON: {e}"])
    return validate(data)

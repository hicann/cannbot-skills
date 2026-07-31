#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""finalize_checks_provenance — binary & verify-path provenance / KB / plugin-dispatch finalize gate CHECK functions.

Behavior-neutral extraction from finalize_checks.py (DEBT-201 god-file
sub-split, 2026-07-06). Byte-identical function bodies; only relocated.
finalize_checks re-imports these (bottom import) so call sites + import
paths (`from finalize_checks import ...`) are unaffected."""
from __future__ import annotations
import logging
import ast as _ast
import hashlib
import json
import re
import shutil
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path
from typing import Optional

import perf_irm_provenance

from finalize_pipeline import _get_active_plugin


def _check_binary_provenance(workspace: Path, vj: dict) -> Optional[str]:
    """DEBT-091 binary-provenance gate — plugin-dispatched (DEBT-094 phase 1).

    Plugin dispatch: looks up the plugin for `workspace` and delegates to
    `plugin.check_binary_provenance(workspace, vj)`. Each plugin owns its
    own proof model. The migration plugin proves generated-source to
    built-artifact lineage; the backward plugin owns its generated-reference
    provenance.

    Behavior-preserving: callers continue calling _check_binary_provenance;
    they don't need to know about plugins.
    """
    try:
        from plugins import detect_plugin
    except ImportError:
        # Pre-import; plugin layer not yet on sys.path. Fallback to no-op.
        return None
    plugin = detect_plugin(workspace)
    if plugin is None:
        return None
    return plugin.check_binary_provenance(workspace, vj)


def _check_a5_verify_path_provenance(workspace: Path, vj: dict) -> Optional[str]:
    """Plugin-dispatched verify-path provenance gate (P80 phase 2 of DEBT-094).

    Behavior-preserving wrapper. The actual rule lives in
    plugins.port_a3.PortA3Plugin.check_verify_path_provenance. The backward
    mode currently returns None (no equivalent gate).
    """
    try:
        from plugins import detect_plugin
    except ImportError:
        return None
    plugin = detect_plugin(workspace)
    if plugin is None:
        return None
    return plugin.check_verify_path_provenance(workspace, vj)


_GE_OPHOST_RECIPE_REFERENCE = (
    "Fix: GENERATE the GE op_host by following "
    "`kb/target/ascendc/patterns/domains/fa_class/"
    "templates/op_host/GE_HOST_TRANSFORM_RECIPE.md` — CARRY def/infershape "
    "from the A3 (arch22) input, REPLACE-HOOK tiling.cpp onto the KB shared "
    "layer (`#include \"wp_fa_host_tiling.h\"` + call `wfh::Calc*` /\n"
    "`wp_fa_host::Calc*`). Prior target implementations may be consulted "
    "for research, but every retained upstream file must be declared with "
    "its SHA256 in `.upstream_prestaged.json`."
)


def _is_port_a3_fa_workspace(workspace: Path) -> bool:
    """Return whether the raw-copy gate applies to this workspace."""
    from plugins import detect_plugin as _detect_plugin
    from plugins.base import is_attention_named as _is_fa_named
    from plugins.base import is_fa_class as _is_fa_tag

    active_plugin = _detect_plugin(workspace)
    if active_plugin is None or active_plugin.name != "port_a3_to_a5":
        return False
    op_classification = workspace / "op_classification.json"
    op_class = ""
    if op_classification.is_file():
        try:
            tags = json.loads(op_classification.read_text()).get("op_class_tags") or []
            op_class = " ".join(tags) if isinstance(tags, list) else str(tags)
        except Exception:
            op_class = ""
    return _is_fa_named(workspace.name) or _is_fa_tag(op_class)


def _find_ge_ophost_sources(workspace: Path) -> tuple[list[Path], list[Path]]:
    """Return GE op-host sources and its tiling implementations in scan order."""
    op_host_dir = workspace / "op_host"
    if not op_host_dir.is_dir():
        return [], []
    ge_files = [
        source
        for source in list(op_host_dir.rglob("*.cpp")) + list(op_host_dir.rglob("*.h"))
        if source.is_file()
    ]
    tiling_files = [
        source
        for source in ge_files
        if source.name.endswith("_tiling.cpp") or source.name == "tiling.cpp"
    ]
    return ge_files, tiling_files


def _structural_ge_ophost_offenders(
    workspace: Path, tiling_files: list[Path]
) -> list[str]:
    """Return tiling sources that do not use the shared FA host layer."""
    offenders: list[str] = []
    for tiling_file in tiling_files:
        try:
            text = tiling_file.read_text(encoding="utf-8", errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            continue
        has_include = bool(re.search(r'#include\s+"[^"]*wp_fa_host_tiling\.h"', text))
        has_call = bool(re.search(r'\b(?:wfh|wp_fa_host)::', text))
        if has_include and has_call:
            continue
        relative_path = (
            tiling_file.relative_to(workspace)
            if workspace in tiling_file.parents
            else tiling_file.name
        )
        missing: list[str] = []
        if not has_include:
            missing.append('#include "wp_fa_host_tiling.h"')
        if not has_call:
            missing.append("wfh::/wp_fa_host:: call")
        offenders.append(f"  {relative_path}: missing {', '.join(missing)}")
    return offenders


def _md5(path: Path) -> Optional[str]:
    """Return a file MD5 digest, or None when the source cannot be read."""
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _target_source_hashes(target_root: Path) -> dict[str, set[str]]:
    """Index target GE source MD5s by basename in the historical scan order."""
    scan_root = target_root / "ops-transformer" / "attention"
    if not scan_root.is_dir():
        scan_root = target_root
    target_hashes: dict[str, set[str]] = {}
    for candidate in scan_root.rglob("*"):
        if not candidate.is_file() or candidate.suffix not in (".cpp", ".h"):
            continue
        digest = _md5(candidate)
        if digest is not None:
            target_hashes.setdefault(candidate.name, set()).add(digest)
    return target_hashes


def _verified_prestaged_sources(workspace: Path) -> set[str]:
    """Return manifest entries whose SHA256 still matches their source file."""
    manifest_path = workspace / ".upstream_prestaged.json"
    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
        staged_files = manifest.get("staged_files", {})
        if not isinstance(staged_files, dict):
            return set()
        root = workspace.resolve()
        verified: set[str] = set()
        for relative_path, recorded_sha in staged_files.items():
            if not isinstance(relative_path, str) or not isinstance(recorded_sha, str):
                continue
            staged_path = workspace / relative_path
            staged = staged_path.resolve()
            if root not in staged.parents or staged_path.is_symlink() or not staged.is_file():
                continue
            actual_sha = hashlib.sha256(staged.read_bytes()).hexdigest()
            if actual_sha == recorded_sha.lower():
                verified.add(relative_path)
        return verified
    except (OSError, ValueError):
        return set()


def _byte_identical_ge_sources(
    workspace: Path,
    ge_files: list[Path],
    target_hashes: dict[str, set[str]],
    verified_prestage: set[str],
) -> list[str]:
    """Return non-prestaged GE sources that match a target source byte-for-byte."""
    offenders: list[str] = []
    for generated in ge_files:
        relative_path = str(generated.relative_to(workspace))
        if relative_path in verified_prestage:
            continue
        digest = _md5(generated)
        if digest is not None and digest in target_hashes.get(generated.name, set()):
            offenders.append(
                f"  {relative_path}: md5 {digest} == target source {generated.name}"
            )
    return offenders


def _check_ge_ophost_raw_cann_copy(workspace: Path) -> Optional[str]:
    """Reject untracked raw GE-host copies while honoring prestage provenance."""
    import os as _os

    if not _is_port_a3_fa_workspace(workspace):
        return None
    ge_files, tiling_files = _find_ge_ophost_sources(workspace)
    if not tiling_files:
        return None
    structural_offenders = _structural_ge_ophost_offenders(workspace, tiling_files)
    if structural_offenders:
        return "\n".join(
            [
                "GE_OPHOST_RAW_CANN_COPY: port_a3 FA GE op_host tiling.cpp does "
                "not consume the KB shared arch35 tiling layer.",
                "",
                "Offending file(s):",
                *structural_offenders,
                "",
                _GE_OPHOST_RECIPE_REFERENCE,
            ]
        )
    target_root = Path(
        _os.environ.get("CANN_SOURCE_ROOT", str(Path.home() / "workspace" / "cann"))
    )
    if not target_root.is_dir():
        return None
    byte_offenders = _byte_identical_ge_sources(
        workspace,
        ge_files,
        _target_source_hashes(target_root),
        _verified_prestaged_sources(workspace),
    )
    if not byte_offenders:
        return None
    return "\n".join(
        [
            "GE_OPHOST_RAW_CANN_COPY: untracked GE op_host file(s) are "
            "BYTE-IDENTICAL to target source. Record sanctioned prestage "
            "files with their SHA256 or generate the deliverable via the recipe.",
            "",
            f"Target source root: {target_root}",
            "Byte-identical file(s):",
            *byte_offenders,
            "",
            _GE_OPHOST_RECIPE_REFERENCE,
        ]
    )


def _check_kb_writeup(workspace: Path, vj: dict) -> Optional[str]:
    """KB-writeup gate (P0aax 2026-05-07). knowledge_update.md must exist
    (workspace root OR .harness/), be ≥ 100 bytes, and carry a `## Findings`
    section. Extracted verbatim from check_finalize_eligibility PASS branch."""
    prec = vj.get("precision", {}) or {}
    status = prec.get("status")
    ku = workspace / "knowledge_update.md"
    if not ku.exists():
        ku = workspace / ".harness" / "knowledge_update.md"
    if not ku.exists():
        return (
            f"precision.status={status} but knowledge_update.md missing "
            "(checked workspace root AND .harness/ subdir) — "
            "every PASS handoff must include a KB writeup. "
            "Respawn worker with directive: write knowledge_update.md only "
            "(no kernel edits). See kw_brief.py Phase E for required structure."
        )
    body = ku.read_text(encoding="utf-8", errors="replace")
    if len(body) < 100:
        return (
            f"precision.status={status} but knowledge_update.md is only "
            f"{len(body)} bytes (< 100) — KB writeup must be "
            "non-trivial. Respawn worker with directive: expand "
            "knowledge_update.md per Phase E structure (Findings, "
            "KB-promotable patterns, Cited KB, Anti-patterns)."
        )
    if "## Findings" not in body and "## findings" not in body.lower():
        return (
            f"precision.status={status} but knowledge_update.md lacks "
            "`## Findings` section — file present but does "
            "not follow Phase E required structure. Respawn worker with "
            "directive: rewrite knowledge_update.md per kw_brief.py "
            "Phase E template."
        )
    return None


def _check_verifier_uses_modelnew(workspace: Path, vj: dict) -> Optional[str]:
    """P96 follow-up (2026-05-15) — plugin-dispatched VERIFIER_USES_MODELNEW.
    Extracted verbatim: only fires when the active plugin defines the check."""
    plugin = _get_active_plugin(workspace)
    if plugin is None:
        return None
    try:
        return plugin.check_verifier_uses_modelnew(workspace, vj)
    except Exception as exc:
        return (
            "verifier_uses_modelnew gate failed to inspect the verifier: "
            f"{exc!r}"
        )


def _delegation_source_directory_names(workspace: Path) -> list[str]:
    """Return the producer-defined C++ source directories, with its fallback."""
    try:
        from handoff_audit import delegation_cpp_dir_names as _cpp_dirs

        return _cpp_dirs(workspace)
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
        return ["kernel", "op_host", "op_kernel"]


def _newest_delegation_source_mtime(workspace: Path) -> float:
    """Return the newest mtime used by the delegation-marker producer."""
    newest_mtime = 0.0
    source_suffixes = (".h", ".hpp", ".cpp", ".cc", ".cxx", ".py")
    for directory_name in _delegation_source_directory_names(workspace):
        kernel_dir = workspace / directory_name
        if not kernel_dir.is_dir():
            continue
        for source_file in kernel_dir.rglob("*"):
            if source_file.is_file() and source_file.suffix in source_suffixes:
                newest_mtime = max(newest_mtime, source_file.stat().st_mtime)
    for workspace_file in (
        workspace / "model_new_ascendc.py",
        workspace / "pybind11.cpp",
    ):
        if workspace_file.exists():
            newest_mtime = max(newest_mtime, workspace_file.stat().st_mtime)
    return newest_mtime


def _check_delegation_scan_marker(workspace: Path, vj: dict) -> Optional[str]:
    """Require a present, fresh delegation scan marker before finalization."""
    status = (vj.get("precision", {}) or {}).get("status")
    deleg_marker = workspace / ".delegation_scan_passed"
    if not deleg_marker.exists():
        return (
            f"precision.status={status} but .delegation_scan_passed marker "
            "absent — orchestrator must run "
            "scan_delegation_cheating.py and produce the marker before "
            "finalize. verification.json self-claim alone is NOT accepted "
            "(codex review 2026-05-07: that bypasses the scanner)."
        )
    try:
        marker_mtime = deleg_marker.stat().st_mtime
        newest_kernel_mtime = _newest_delegation_source_mtime(workspace)
        if newest_kernel_mtime > marker_mtime + 1.0:
            return (
                f"precision.status={status} but .delegation_scan_passed "
                f"marker is STALE (mtime {marker_mtime:.0f} < newest kernel "
                f"file mtime {newest_kernel_mtime:.0f}). Re-run "
                "scan_delegation_cheating.py against current kernel state."
            )
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    return None


def _post_worker_audit_content(
    workspace: Path, status: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Return an audit document or its pre-existing rejection diagnostic."""
    audit_doc = workspace / "audit_self_critic_post_worker.md"
    if not audit_doc.exists():
        return (
            f"precision.status={status} but audit_self_critic_post_worker.md "
            "missing — every PASS handoff must include a "
            "post-worker self-critic audit. Orchestrator should invoke "
            "/aog-self-critic with workspace=<workspace> AFTER kw return "
            "and BEFORE finalize_op. See aog-self-critic SKILL.md C13/C18/"
            "C25/C26 catalog (post-worker watchpoints).",
            None,
        )
    audit_body = audit_doc.read_text(encoding="utf-8", errors="replace")
    if len(audit_body.strip()) < 50:
        return (
            f"precision.status={status} but audit_self_critic_post_worker.md "
            f"is empty/trivial ({len(audit_body)} bytes) — "
            "self-critic must produce a substantive verdict + findings.",
            None,
        )
    return None, audit_body


def _post_worker_audit_verdict(audit_body: str) -> Optional[str]:
    """Find the first explicit self-critic verdict in its historical order."""
    audit_lower = audit_body.lower()
    verdict_line_status = None
    lines = audit_lower.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if "verdict" not in line:
            continue
        if re.search(r"verdict[*\s:_>-]*[^:]{0,30}?\b(fail|block|reject)\b", line):
            verdict_line_status = "FAIL"
            break
        if re.search(r"verdict[*\s:_>-]*[^:]{0,30}?❌\s*block", line):
            verdict_line_status = "FAIL"
            break
        if re.search(r"verdict[*\s:_>-]*[^:]{0,30}?\bpass\b", line):
            verdict_line_status = "PASS"
            break
        if re.search(r"verdict[*\s:_>-]*[^:]{0,30}?\bpartial\b", line):
            verdict_line_status = "PARTIAL"
            break
        heading_only_strip = re.sub(r"[#*:>\s]+", "", line)
        if heading_only_strip == "verdict":
            for following_index in range(index + 1, min(index + 4, len(lines))):
                next_line = lines[following_index].strip()
                if not next_line:
                    continue
                if re.search(r"\b(fail|block|reject)\b", next_line) or "❌" in next_line:
                    verdict_line_status = "FAIL"
                    break
                if re.search(r"\bpartial\b", next_line):
                    verdict_line_status = "PARTIAL"
                    break
                if re.search(r"\bpass\b", next_line):
                    verdict_line_status = "PASS"
                    break
                break
            if verdict_line_status is not None:
                break
    return verdict_line_status


def _post_worker_audit_allows_finalize(audit_body: str) -> bool:
    """Return whether the audit has a PASS or an explicitly waived PARTIAL."""
    verdict = _post_worker_audit_verdict(audit_body)
    has_pass = verdict == "PASS"
    has_partial_with_waiver = verdict == "PARTIAL" and bool(
        re.search(r"(?<!no )(?<!without )\bwaiver\b", audit_body.lower())
    )
    return verdict != "FAIL" and (has_pass or has_partial_with_waiver)


def _check_post_worker_pass_b(precision: dict, status: Optional[str]) -> Optional[str]:
    """Validate the existing independent post-verify status contract."""
    pass_b = precision.get("pass_b", {}) or {}
    pass_b_status = pass_b.get("status")
    pass_b_reason = pass_b.get("reason", "")
    if pass_b_status not in ("PASS", "PASS_WITHIN_TOLERANCE", "N/A", "SKIPPED"):
        return (
            f"precision.status={status} but precision.pass_b.status is "
            f"{pass_b_status!r} — explicit value required: PASS / "
            "PASS_WITHIN_TOLERANCE (independent post-verify ran), N/A "
            "(with reason — typically OL-68 Case A), or SKIPPED (with "
            "reason). Silent-skip rejected."
        )
    if pass_b_status in ("N/A", "SKIPPED") and not pass_b_reason:
        return (
            f"precision.pass_b.status={pass_b_status} but no reason given — "
            "explicit reason required when post-verify is "
            "N/A or SKIPPED. Typical reasons: 'OL-68 Case A — torch_npu "
            "reference unrunnable on Ascend950PR' (Path A) or specific "
            "harness-level explanation."
        )
    return None


def _check_post_worker_audit(workspace: Path, vj: dict) -> Optional[str]:
    """Require a self-critic audit, delegation proof, and post-verify contract."""
    precision = vj.get("precision", {}) or {}
    status = precision.get("status")
    audit_error, audit_body = _post_worker_audit_content(workspace, status)
    if audit_error:
        return audit_error
    if audit_body is None or not _post_worker_audit_allows_finalize(audit_body):
        return (
            f"precision.status={status} but audit_self_critic_post_worker.md "
            "verdict is not PASS (and not PARTIAL+waiver) — "
            "self-critic surfaced unwaived issues on kw output. Address "
            "findings before finalize."
        )
    marker_reason = _check_delegation_scan_marker(workspace, vj)
    if marker_reason:
        return marker_reason
    pass_b_error = _check_post_worker_pass_b(precision, status)
    if pass_b_error:
        return pass_b_error
    return _check_perf_irm_provenance(vj, status)


def _requires_independent_perf_measure(performance: dict) -> bool:
    """Return whether this performance result requires the IRM proof."""
    status = performance.get("status")
    if status not in ("PASS", "PASS_WITHIN_TOLERANCE", "BELOW_THRESHOLD"):
        return False
    ratio_baseline = performance.get("ratio_baseline", "") or ""
    normalized_baseline = ratio_baseline.lower()
    return "cpu-truth" not in normalized_baseline and "path a" not in normalized_baseline


def _validate_independent_perf_measure(
    independent_measure: object, precision_status: Optional[str], performance_status: object
) -> Optional[str]:
    """Validate the existing independent-measure shape and truthfulness fields."""
    if not isinstance(independent_measure, dict) or not independent_measure:
        return (
            f"precision.status={precision_status} + performance.status="
            f"{performance_status} but performance.independent_re_measure "
            "missing or empty — CLAUDE.md hard rule "
            "'NEVER trust skill-reported performance numbers'. "
            "Set independent_re_measure to {ran: true, ratio: "
            "<num>, delta_vs_kw_self_report: <num>} OR {status: "
            "'N/A', reason: '<Path A / explicit reason>'}."
        )
    independent_status = independent_measure.get("status")
    independently_ran = independent_measure.get("ran")
    if independent_status not in ("N/A", "SKIPPED") and independently_ran is not True:
        return (
            "performance.independent_re_measure present but "
            "ran is not true and status is not N/A/SKIPPED — "
            "independent re-measure must be RAN "
            "(with ratio number) OR explicit N/A/SKIPPED with reason."
        )
    if independent_status in ("N/A", "SKIPPED") and not independent_measure.get("reason"):
        return (
            f"performance.independent_re_measure.status="
            f"{independent_status} but no reason — "
            "explicit reason required when N/A/SKIPPED."
        )
    if independently_ran is True and "ratio" not in independent_measure:
        return (
            "performance.independent_re_measure.ran=true but "
            "no ratio field — ran=true requires "
            "the actual measured ratio to be recorded."
        )
    return _validate_independent_perf_source(independent_measure, independently_ran)


def _validate_independent_perf_source(
    independent_measure: dict, independently_ran: object
) -> Optional[str]:
    """Require an orchestrator source for a positive IRM claim."""
    if independently_ran is not True or perf_irm_provenance.is_orchestrator_measured(
        independent_measure
    ):
        return None
    return (
        "performance.independent_re_measure.ran=true but its "
        f"`source` ({independent_measure.get('source')!r}) does not "
        "name an orchestrator-side measurer — `ran: true` asserts "
        "author!=measurer, so it requires a re-measure the "
        "ORCHESTRATOR performed. A worker-authored block cannot "
        "satisfy it: the kernel's author is not an independent "
        "measurer of its own perf. Either let "
        "fsm_phase_finalize._run_perf_capture stamp this field "
        "(it runs automatically at finalize; force a real capture "
        "with AOG_PERF_CAPTURE_OVERRIDE_WORKER=1), or record the "
        "honest unmeasured state: {ran: false, status: 'N/A', "
        "reason: '<why the orchestrator could not re-measure>'}."
    )


def _check_perf_irm_provenance(vj: dict, status: Optional[str]) -> Optional[str]:
    """Require an orchestrator-stamped independent perf measure when needed."""
    performance = vj.get("performance", {}) or {}
    if not _requires_independent_perf_measure(performance):
        return None
    return _validate_independent_perf_measure(
        performance.get("independent_re_measure"),
        status,
        performance.get("status"),
    )

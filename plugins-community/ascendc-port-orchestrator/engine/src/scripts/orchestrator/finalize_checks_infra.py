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

"""finalize_checks_infra — infrastructure blame / retry / paper-over finalize gate CHECK functions.

Behavior-neutral extraction from finalize_checks.py (DEBT-201 god-file
sub-split, 2026-07-06). Byte-identical function bodies; only relocated.
finalize_checks re-imports these (bottom import) so call sites + import
paths (`from finalize_checks import ...`) are unaffected."""
from __future__ import annotations
import ast as _ast
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path
from typing import Optional

from finalize_shared import (  # DEBT-201: shared pure leaves (breaks the finalize_pipeline cycle)
    _is_kernel_caused_context_window, _is_negative_assertion_window, _INFRA_PAPER_OVER_PHRASES,
    _NPU_BASELINE_ERROR_CODES)


_logger = logging.getLogger(__name__)
_platform_blame_doc_names = (
    "PROGRESS.md",
    "analysis.md",
    "knowledge_update.md",
    "probe_report.md",
    "self_critic_report.md",
)
_infra_paper_over_doc_names = (
    "PROGRESS.md",
    "orchestrator_events.jsonl",
    "analysis.md",
    "knowledge_update.md",
    "self_critic_report.md",
)
_infra_retry_doc_names = ("PROGRESS.md", "orchestrator_events.jsonl")
_gate_own_markers = (
    "c-infra-baseline-paper-over",
    "infra_baseline_paper_over",
    "p135.sl",
)
_anti_pattern_heading_re = re.compile(
    r"^#{1,6}\s+(anti-patterns?|anti pattern|antipattern)s?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _workspace_documents(workspace: Path, names: tuple[str, ...]) -> list[Path]:
    """Return the existing workspace documents for a fixed gate catalog."""
    return [workspace / name for name in names if (workspace / name).is_file()]


def _read_optional_text(document: Path) -> Optional[str]:
    """Read a gate input, logging recoverable filesystem failures."""
    text: Optional[str] = None
    try:
        text = document.read_text(errors="ignore")
    except Exception as error:
        _logger.debug("Recoverable operation failed.", exc_info=error)
    return text


def _find_platform_blame_hits(documents: list[Path], phrases: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return at most one platform-attribution phrase hit per document."""
    hits: list[tuple[str, str]] = []
    for document in documents:
        text = _read_optional_text(document)
        if text is None:
            continue
        for phrase in phrases:
            if phrase in text.lower():
                hits.append((document.name, phrase))
                break
    return hits


def _platform_blame_evidence(workspace: Path, documents: list[Path]) -> tuple[bool, bool, bool, bool]:
    """Collect the evidence flags that permit a platform-attribution claim."""
    has_probe = (workspace / "probes").is_dir() and any(
        path.suffix == ".py" for path in (workspace / "probes").iterdir()
    )
    has_msprof = any(workspace.glob("*msprof*.json")) or any(workspace.glob("*.msprof.json"))
    has_hw_citation = False
    has_pb_citation = False
    for document in documents:
        text = _read_optional_text(document)
        if text is None:
            continue
        has_hw_citation = has_hw_citation or ("hardware/" in text and ".md" in text)
        has_pb_citation = has_pb_citation or (
            "PLATFORM_BUGS.md" in text or bool(re.search(r"\bPB-\d+\b", text))
        )
    return has_probe, has_msprof, has_hw_citation, has_pb_citation


def _platform_blame_failure(hits: list[tuple[str, str]], evidence: tuple[bool, bool, bool, bool]) -> str:
    """Render the existing diagnostic for an unbacked platform claim."""
    has_probe, has_msprof, has_hw_citation, has_pb_citation = evidence
    return (
        f"P94 PLATFORM-BLAME: workspace docs contain platform-attribution "
        f"phrases {hits[:3]} without forensic backing. Required evidence "
        f"(at least ONE of): "
        f"(a) workspace/probes/*.py empirical probe script, "
        f"(b) workspace/*msprof*.json hardware-counter trace, "
        f"(c) doc-level citation of references/hardware/<chip>.md, "
        f"(d) doc-level citation of PLATFORM_BUGS.md / PB-<N>. "
        f"Found: probe={has_probe}, msprof={has_msprof}, "
        f"hw_citation={has_hw_citation}, pb_citation={has_pb_citation}. "
        f"Platform-blame WITHOUT evidence is reward-hacking via "
        f"attribution offload (DS audit 2026-05-15 found 9 production "
        f"ops with this pattern). Fix: produce probe OR remove the "
        f"platform-blame language and diagnose root cause."
    )


def _check_platform_blame_backed(workspace: Path) -> Optional[str]:
    """P94 attack-id PLATFORM-BLAME (2026-05-15, DS audit msg 1504774693):
    detect platform-attribution claims in worker docs that lack
    evidentiary backing.

    DS found 9 production ops with phrases like "V220 limitation",
    "HBM-blocked", "c10 ABI" in PROGRESS.md / analysis.md WITHOUT
    corresponding probe artifact, msprof trace, or KB citation.
    These are reward-hacking via platform attribution — the agent
    offloads responsibility to "platform" without forensic proof.

    Gate scans workspace docs (PROGRESS.md, analysis.md,
    knowledge_update.md, probe_report.md) for platform-blame phrases.
    If found, requires evidence in same workspace:
    - workspace/probes/*.py (empirical probe script), OR
    - workspace/*.msprof.json or workspace/msprof_*.json, OR
    - explicit citation of references/hardware/<chip>.md or
      references/target/ascendc/PLATFORM_BUGS.md (PB-N) in the doc.

    Returns None on pass, error string on fail.
    """
    # P94 PLATFORM-BLAME phrase catalog (synced with aog-self-critic C5
    # expansion). Match case-insensitive, word-aware via simple substr.
    blame_phrases = (
        "v220 limitation", "v300 limitation", "v351 limitation", "hbm-blocked", "hbm bandwidth limit",
        "c10 abi", "fp16 not supported", "no scalar half", "no half arithmetic",
        "cast intrinsic unavailable", "aicpu fallback expected",
        "pytorch dispatcher quirk", "torch_npu deprecated",
        # Original C5 phrases (kept for symmetry)
        "platform bug", "hardware limitation", "expected behavior",
        "known limitation",
    )

    documents = _workspace_documents(workspace, _platform_blame_doc_names)
    hits = _find_platform_blame_hits(documents, blame_phrases)
    if not hits:
        return None  # no platform-blame language → gate inactive

    # Platform-blame phrase(s) found. Require ONE of three evidence types.
    evidence = _platform_blame_evidence(workspace, documents)
    if any(evidence):
        return None  # evidence present, gate inactive
    return _platform_blame_failure(hits, evidence)


def _strip_gate_self_references(name: str, text: str) -> str:
    """Exclude the gate's own JSONL audit trail and Markdown anti-patterns."""
    if name == "orchestrator_events.jsonl":
        return "\n".join(
            line for line in text.splitlines()
            if not any(marker in line.lower() for marker in _gate_own_markers)
        )

    output: list[str] = []
    skip_depth: Optional[int] = None
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading and _anti_pattern_heading_re.match(line):
            skip_depth = len(heading.group(1))
            continue
        if heading and skip_depth is not None and len(heading.group(1)) <= skip_depth:
            skip_depth = None
        if skip_depth is None:
            output.append(line)
    return "\n".join(output)


def _is_active_paper_over_phrase(text: str, phrase: str) -> bool:
    """Check one configured phrase while honoring negative assertions."""
    if ".*" in phrase:
        match = re.search(phrase, text)
        if match is None:
            return False
        return not _is_negative_assertion_window(text, match.start(), match.end())
    index = text.find(phrase)
    return index >= 0 and not _is_negative_assertion_window(text, index, index + len(phrase))


def _find_paper_over_hits(documents: list[Path]) -> list[tuple[str, str]]:
    """Return at most one actionable paper-over signal per document."""
    hits: list[tuple[str, str]] = []
    for document in documents:
        text = _read_optional_text(document)
        if text is None:
            continue
        filtered = _strip_gate_self_references(document.name, text).lower()
        for phrase in _INFRA_PAPER_OVER_PHRASES:
            if _is_active_paper_over_phrase(filtered, phrase):
                hits.append((document.name, phrase))
                break
    return hits


def _contains_uncontextualized_baseline_error(text: str) -> bool:
    """Return whether a baseline error code lacks kernel-caused context."""
    for code in _NPU_BASELINE_ERROR_CODES:
        search_from = 0
        while True:
            index = text.find(code, search_from)
            if index < 0:
                break
            if not _is_kernel_caused_context_window(text, index, index + len(code)):
                return True
            search_from = index + len(code)
    return False


def _baseline_error_status(documents: list[Path]) -> tuple[bool, bool]:
    """Return uncontextualized-error and documented-handoff flags."""
    has_uncontextualized_error = False
    has_baseline_violated_handoff = False
    for document in documents:
        text = _read_optional_text(document)
        if text is None:
            continue
        has_baseline_violated_handoff = has_baseline_violated_handoff or (
            "INFRA_BASELINE_VIOLATED" in text or "INFRA_TRANSIENT_RETRY_EXHAUSTED" in text
        )
        has_uncontextualized_error = has_uncontextualized_error or (
            _contains_uncontextualized_baseline_error(text.lower())
        )
    return has_uncontextualized_error, has_baseline_violated_handoff


def _infra_paper_over_failure(paper_over_hits: list[tuple[str, str]], npu_unescalated: bool) -> str:
    """Render the existing P96 paper-over diagnostic."""
    message_parts: list[str] = []
    if paper_over_hits:
        message_parts.append(f"workspace docs contain paper-over keywords {paper_over_hits[:3]}")
    if npu_unescalated:
        message_parts.append(
            f"NPU baseline error codes "
            f"({', '.join(_NPU_BASELINE_ERROR_CODES)}) cited in docs without "
            f"corresponding INFRA_BASELINE_VIOLATED / INFRA_TRANSIENT_RETRY_EXHAUSTED handoff"
        )
    return (
        f"C-INFRA-BASELINE-PAPER-OVER (P96): worker performed structural "
        f"env workaround instead of escalating to preflight. "
        f"Violations: {'; '.join(message_parts)}. "
        f"Required: worker MUST emit "
        f"`→ orchestrator: await_user_decision — INFRA_BASELINE_VIOLATED <symptom>` "
        f"with forensic artifacts (error transcript + lib md5/size + path) "
        f"and STOP. Replacing install libs / bypass-ing build pipeline / "
        f"hand-editing binary_info_config produces non-reproducible verify "
        f"artifacts. See ANTI_PRESSURE_PROTOCOLS.md §P9 + "
        f"docs/baseline/environment_baseline.yaml."
    )


def _check_infra_paper_over(workspace: Path) -> Optional[str]:
    """Reject un-escalated environment paper-over workarounds in workspace artifacts."""
    documents = _workspace_documents(workspace, _infra_paper_over_doc_names)
    paper_over_hits = _find_paper_over_hits(documents)
    has_uncontextualized_error, has_baseline_violated_handoff = _baseline_error_status(documents)
    npu_unescalated = has_uncontextualized_error and not has_baseline_violated_handoff
    if not paper_over_hits and not npu_unescalated:
        return None
    return _infra_paper_over_failure(paper_over_hits, npu_unescalated)


def _high_retry_count(documents: list[Path]) -> int:
    """Count retry/attempt records at or over the allowed retry cap."""
    count = 0
    for document in documents:
        text = _read_optional_text(document)
        if text is None:
            continue
        for match in re.finditer(r"\b(?:retry|attempt #?)\s*(\d{1,2})\b", text, re.IGNORECASE):
            if int(match.group(1)) >= 4:
                count += 1
    return count


def _has_tracked_retry_counter(state_file: Path) -> bool:
    """Return whether the state file records at least four tracked retries."""
    state: object = None
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text())
        except Exception as error:
            _logger.debug("Recoverable operation failed.", exc_info=error)
    has_tracked_counter = False
    if isinstance(state, dict):
        try:
            has_tracked_counter = state.get("transient_retry_count", 0) >= 4
        except Exception as error:
            _logger.debug("Recoverable operation failed.", exc_info=error)
    return has_tracked_counter


def _has_retry_exhausted_handoff(documents: list[Path]) -> bool:
    """Return whether any gate input records the retry-exhausted handoff."""
    for document in documents:
        text = _read_optional_text(document)
        if text is not None and "INFRA_TRANSIENT_RETRY_EXHAUSTED" in text:
            return True
    return False


def _infra_retry_failure(high_retry_count: int) -> str:
    """Render the existing P96 retry-budget diagnostic."""
    return (
        f"C-INFRA-RETRY-WITHOUT-CAP (P96): worker performed >= 4 retries "
        f"on env errors ({high_retry_count} retry-keyword hits with N >= 4) "
        f"without orchestrator-tracked counter "
        f"(.opgen_state.json.transient_retry_count) or proper "
        f"INFRA_TRANSIENT_RETRY_EXHAUSTED handoff. "
        f"Required: bound retries to <= 3 with exponential backoff; on 4th "
        f"attempt or after 60s wallclock on same error, emit handoff. "
        f"See ANTI_PRESSURE_PROTOCOLS.md §P9 + "
        f"docs/baseline/environment_baseline.yaml.transient_retry_budgets."
    )


def _check_infra_retry_budget(workspace: Path) -> Optional[str]:
    """Reject high retry counts without state tracking and an exhaustion handoff."""
    documents = _workspace_documents(workspace, _infra_retry_doc_names)
    high_retry_count = _high_retry_count(documents)
    if high_retry_count == 0:
        return None
    has_tracked_counter = _has_tracked_retry_counter(workspace / ".opgen_state.json")
    if has_tracked_counter and _has_retry_exhausted_handoff(documents):
        return None
    return _infra_retry_failure(high_retry_count)

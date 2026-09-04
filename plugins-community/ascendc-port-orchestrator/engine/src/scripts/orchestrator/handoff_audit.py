# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Handoff extraction + audit artifacts + applied-user-decision consumption.

Mechanically extracted from orchestrator.py (god-file decomposition 2026-06-30, per
ORCHESTRATOR_REFACTOR_AND_UT_SPEC §1). Behavior unchanged — function bodies are VERBATIM.
Re-imported into orchestrator's namespace so existing call-sites + `orchestrator.<name>` external
access are preserved (e.g. run_single_op still reads _CANONICAL_HANDOFF_PREFIXES via the re-import).

DAG: imports stdlib + `logging_config` + `resolution` siblings. resolution does not import
orchestrator or handoff_audit → acyclic; this module imports standalone without pulling orchestrator.

Constants moved here (single source of truth, re-imported back into orchestrator):
_CANONICAL_HANDOFF_PREFIXES / _ARROW_TO_AT_FORM / _VALID_ARROW_KEYWORDS (consumed by
extract_canonical_handoff; _CANONICAL_HANDOFF_PREFIXES is also read by run_single_op in core via the
re-import) and SELF_CRITIC_POST_WORKER_TIMEOUT_SEC (sole consumer is _ensure_audit_artifacts).
`__file__`-derived paths in _ensure_audit_artifacts are identical here (same orchestrator/ dir).

MONKEYPATCH NOTE (durable — OL-160-class latent-coupling guard): the functions and module-level
constants/logger here are re-imported into orchestrator's namespace, which preserves
`orchestrator.<name>` attribute LOOKUP only — it does NOT rebind THIS module's own globals. A test
that overrides a symbol one of these functions reads must `monkeypatch.setattr(<this_module>,
'<name>', ...)` on THIS module, NOT on `orchestrator` (patching orchestrator silently misses the
binding used here). No current test patches these on orchestrator; this note prevents a future one
from a silent no-op.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import types as _types
from pathlib import Path
from typing import Optional

from logging_config import get_logger
# Harness-decoupling: the claude self-critic invocation goes through the Backend, not hardcoded.
from backends import get_backend

_backend = get_backend()
from resolution import op_name_from_workspace

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical handoff extraction (Day 2 finding 2026-05-04)
# ---------------------------------------------------------------------------
# state_machine.py uses handoff.startswith(arg) for handoff_match conditions.
# Workers under the Python orchestrator emit a full markdown stdout where the
# canonical handoff line is at the END, so the raw stdout never satisfies
# startswith() and every transition fell through to the abort catch-all.
# Solution: scan stdout from the bottom for the LAST line beginning with one
# of the canonical handoff prefixes, return that single line as the handoff
# the state machine sees.
_CANONICAL_HANDOFF_PREFIXES = (
    "→ orchestrator:",
    "@aog-precision-probe",
    "@aog-kernel-optimizer",
    "@aog-fused-optimizer",
    "@aog-determinism-analyzer",
    "@aog-researcher",
    "@orchestrator:",
    # DEBT-103 (2026-05-20): an E2E run found a worker emitting
    # arrow form `→ aog-X` instead of `@aog-X` for inter-agent handoff.
    # Accept both forms; normalize arrow → @ in extract_canonical_handoff
    # so downstream state_machine handoff_match (which keys on @aog-X) hits.
    "→ aog-precision-probe",
    "→ aog-kernel-optimizer",
    "→ aog-fused-optimizer",
    "→ aog-determinism-analyzer",
    "→ aog-researcher",
)

# DEBT-103: map arrow-form prefix → @-form (state_machine uses @-form)
_ARROW_TO_AT_FORM = {
    "→ aog-precision-probe": "@aog-precision-probe",
    "→ aog-kernel-optimizer": "@aog-kernel-optimizer",
    "→ aog-fused-optimizer": "@aog-fused-optimizer",
    "→ aog-determinism-analyzer": "@aog-determinism-analyzer",
    "→ aog-researcher": "@aog-researcher",
}

# Valid forms WHEN line starts with `→ orchestrator:` per worker brief.
# Anything else (e.g. "→ orchestrator: handoff to @aog-X") is improvised
# wrapper — strip the wrapper and recover the inner @aog-X form (P0s).
_VALID_ARROW_KEYWORDS = (
    "done",
    # NPUBench workers report candidate readiness only; Phase O5 owns the
    # target build/evaluation after this provider-bound handoff.
    "build-ready",
    "PARTIAL_PERSIST",
    "await_user_decision",
    "research_done",
    "research_partial",
    "research_blocked",
    "PARTIAL_PERF_STRUCTURAL_CEILING",
    "cann_learn_blocked",
    "cann_learn_done",
    "cann_learn_empty",
    "pipeline_done",
    "build stuck",
    "infra unreachable",
    "BLOCKED",
    # 2026-05-20 — structural-rewrite escalation sentinel.
    # S4 (PR #31) added kw_brief guidance for workers to emit
    # `→ orchestrator: structural_rewrite_needed — <reason>` when scope spans
    # ≥2 design axes + objective signal fires. Without adding the keyword here,
    # extract_canonical_handoff rejects the line as malformed → orchestrator
    # falls to abort with "worker exited without recognized handoff — contract
    # violation" (caught empirically during S5 cold-start on 3_FusionAttention:
    # kw-1 correctly emitted the sentinel but the receiver-side recognizer was
    # the missing piece). S3c.2's await_worker route-in references handoff_match
    # on this canonical form, so recognition has to land here too.
    "structural_rewrite_needed",
    # FOLLOWUPS v3.1 A.2 (2026-08-30, 2_FFN_evo lesson): a worker whose mandate
    # requires NO code change (e.g. waiting on pending evidence the operator
    # must supply) previously had no honest handoff — improvisations were
    # rejected as malformed or misrouted. `hold` is the canonical no-change
    # handoff: it matches no specific YAML rule and lands on the P0abk
    # `→ orchestrator:` catch-all → await_user_decision, unchanged.
    "hold",
)

_AOG_HANDOFF_PREFIXES = (
    "@aog-precision-probe",
    "@aog-kernel-optimizer",
    "@aog-fused-optimizer",
    "@aog-determinism-analyzer",
    "@aog-researcher",
)


# Post-worker self-critic skill call timeout (env-overridable, default 1800s).
SELF_CRITIC_POST_WORKER_TIMEOUT_SEC = int(os.environ.get("AOG_SELF_CRITIC_POST_WORKER_TIMEOUT_SEC", "1800"))


def _embedded_aog_handoff(tail: str) -> Optional[str]:
    """Return a nested @aog handoff from an improvised wrapper, if present.
    """
    for inner_prefix in _AOG_HANDOFF_PREFIXES:
        inner_idx = tail.find(inner_prefix)
        if inner_idx != -1:
            return tail[inner_idx:].rstrip("`* \t")
    return None


def _keyword_prefix_matches(text: str, keyword: str) -> bool:
    """Match a handoff keyword without accepting an alphanumeric suffix."""
    if not text.startswith(keyword):
        return False
    suffix = text[len(keyword):]
    return not suffix or bool(re.match(r"^[^\w-]", suffix))


def _normalize_canonical_handoff(prefix: str, tail: str, line: str, prefix_idx: int) -> str:
    """Normalize one matched canonical handoff while retaining malformed forms.
    """
    if prefix in _ARROW_TO_AT_FORM:
        at_form_prefix = _ARROW_TO_AT_FORM[prefix]
        return at_form_prefix + tail[len(prefix):]

    if prefix == "@orchestrator:":
        if prefix_idx >= 2 and line[prefix_idx - 2:prefix_idx] == "→ ":
            return tail
        rest = tail[len(prefix):].lstrip()
        if any(_keyword_prefix_matches(rest, keyword) for keyword in _VALID_ARROW_KEYWORDS):
            return "→ orchestrator: " + rest
        return tail

    if prefix != "→ orchestrator:":
        return tail

    rest = tail[len(prefix):].lstrip()
    if any(_keyword_prefix_matches(rest, keyword) for keyword in _VALID_ARROW_KEYWORDS):
        return "→ orchestrator: " + rest
    return _embedded_aog_handoff(tail) or tail


def extract_canonical_handoff(output_text: str) -> str:
    """Return the canonical handoff line from agent stdout (LAST match wins).

    Strategy:
      1. Reverse-scan stdout lines (last-match-wins so worker's final decision
         beats any earlier mention).
      2. For each line, look for any canonical prefix as a substring.
      3. If found, return the prefix-onwards content with surrounding
         markdown noise (backticks, asterisks, list bullets, trailing
         spaces/tabs) stripped.

    P0h (Day 4 op#10 finding): workers sometimes wrap the handoff in
    markdown like ``**Exit handoff**: `→ orchestrator: await_user_decision` ``.
    Pure startswith() missed those, falling through to abort. Substring
    search handles them; the reverse-scan + last-match-wins behavior
    preserves the original semantics.

    P0s (2026-05-05 op#10 kw-2 finding): workers sometimes mash forms,
    e.g. `→ orchestrator: handoff to @aog-kernel-optimizer per V3.8.4`.
    The `→ orchestrator:` prefix matches but the remainder is not a valid
    arrow-form keyword (done / PARTIAL_PERSIST / await_user_decision /
    research_*). When the line is wrapped this way AND contains an inner
    `@aog-X` agent reference, strip the wrapper and return the inner
    agent-handoff form (which IS canonical for await_worker → await_X).

    Returns full stripped text if no canonical prefix appears anywhere
    (so downstream state machine's startswith() will fail and route to
    the contract-violation abort, preserving forensic intent).
    """
    if not output_text:
        return ""
    for line in reversed(output_text.splitlines()):
        s = line.strip()
        for prefix in _CANONICAL_HANDOFF_PREFIXES:
            idx = s.find(prefix)
            if idx == -1:
                continue
            tail = s[idx:].rstrip("`* \t")
            return _normalize_canonical_handoff(prefix, tail, s, idx)
    return output_text.strip()


def delegation_cpp_dir_names(workspace: Path) -> list[str]:
    """C++ directories the delegation scanner covers for `workspace` — the
    active plugin's `kernel_cpp_dirs()` (e.g. port_a3 → op_host/op_kernel) plus
    the historical `kernel/`. Kept in lock-step with
    scan_delegation_cheating.scan_op_workspace so the marker-freshness check
    (producer + gate) never misses a source dir the scanner reads.

    DEBT-211 directory-level: before op_host/op_kernel were included here, an
    edit to op_kernel/ did NOT invalidate a stale `.delegation_scan_passed`
    marker, so a rebuilt port_a3 kernel could re-ship its old clean marker.

    Best-effort: on any plugin-resolution failure, fall back to the union of
    every known AscendC C++ dir name (never UNDER-covers a real source dir).
    """
    try:
        import sys as _sys
        _orch = Path(__file__).resolve().parent
        if str(_orch) not in _sys.path:
            _sys.path.insert(0, str(_orch))
        from plugins import detect_plugin
        plug = detect_plugin(workspace)
        if plug is not None:
            workspace_aware = getattr(plug, "kernel_cpp_dirs_for_workspace", None)
            declared = (
                workspace_aware(workspace)
                if callable(workspace_aware)
                else plug.kernel_cpp_dirs()
            )
            return list(dict.fromkeys([*declared, "kernel"]))
    except Exception:
        # This freshness helper is best-effort and conservatively scans the
        # union of every supported C++ directory when ownership cannot be
        # resolved.  The authoritative scanner calls detect_plugin directly
        # and remains fail-loud; do not catch process-control BaseExceptions.
        pass
    return ["kernel", "op_host", "op_kernel"]


def _newest_matching_file_mtime(
        directory: Path, suffixes: tuple[str, ...], *, recursive: bool) -> float:
    """Return the newest matching file mtime below or directly in `directory`.
    """
    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    newest = 0.0
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix in suffixes:
            newest = max(newest, candidate.stat().st_mtime)
    return newest


def _newest_workspace_file_mtime(workspace: Path, filenames: tuple[str, ...]) -> float:
    """Return the newest mtime among the named direct workspace artifacts.
    """
    newest = 0.0
    for filename in filenames:
        ws_file = workspace / filename
        if ws_file.exists():
            newest = max(newest, ws_file.stat().st_mtime)
    return newest


def _newest_delegation_input_mtime(workspace: Path) -> float:
    """Return the newest delegation-scanned input mtime for `workspace`.
    """
    newest = 0.0
    cpp_suffixes = (".h", ".hpp", ".cpp", ".cc", ".cxx", ".py")
    for directory_name in delegation_cpp_dir_names(workspace):
        kernel_dir = workspace / directory_name
        if kernel_dir.is_dir():
            newest = max(
                newest,
                _newest_matching_file_mtime(kernel_dir, cpp_suffixes, recursive=True),
            )
    workspace_files = ("model_new_ascendc.py", "pybind11.cpp")
    return max(newest, _newest_workspace_file_mtime(workspace, workspace_files))


def deleg_marker_needs_refresh(workspace: Path) -> bool:
    """DEBT-155 (2026-06-13): does the .delegation_scan_passed marker need a (re)scan?

    True if the marker is ABSENT *or STALE* (older than the newest kernel-source
    file). Mirrors the finalize gate's freshness check (finalize_pipeline.py ~2825)
    so the producer (`_ensure_audit_artifacts`) and the consumer (the gate) agree.

    Before this fix, `_ensure_audit_artifacts` re-ran the scan only when the marker
    was ABSENT. After a worker REBUILT the kernel (e.g. a P149 fix), the stale marker
    was never refreshed → the gate kept rejecting "stale marker" → rollback →
    worker no-op (kernel already correct) → finalize again skipped the re-scan →
    infinite rollback → LOOP-BREAK → await_user_decision (observed twice on mul_grad).

    DEBT-211 directory-level: the source set now includes the mode's declared
    C++ dirs (op_host/op_kernel for port_a3) via `delegation_cpp_dir_names`, so
    an edit to op_kernel/ invalidates a stale marker just like a kernel/ edit.
    """
    marker = workspace / ".delegation_scan_passed"
    if not marker.exists():
        return True
    try:
        marker_mtime = marker.stat().st_mtime
        newest = _newest_delegation_input_mtime(workspace)
        return newest > marker_mtime + 1.0  # 1s slack — same as the gate
    except Exception:
        return False  # best-effort: if we can't tell, don't force a re-scan


def _newest_audit_input_mtime(workspace: Path) -> float:
    """Return the newest post-worker self-critic input mtime for `workspace`.
    """
    newest = 0.0
    kernel_dir = workspace / "kernel"
    if kernel_dir.exists():
        kernel_suffixes = (".h", ".cpp", ".py")
        newest = _newest_matching_file_mtime(kernel_dir, kernel_suffixes, recursive=False)

    workspace_files = (
        "model_new_ascendc.py",
        "pybind11.cpp",
        "verification.json",
        "edge_dataset.pt",
        "pass_a_runner.py",
        "knowledge_update.md",
        "PROGRESS.md",
    )
    return max(newest, _newest_workspace_file_mtime(workspace, workspace_files))


def audit_doc_needs_refresh(workspace: Path) -> bool:
    """DEBT-168 (2026-06-24): does audit_self_critic_post_worker.md need (re)gen?

    True if the audit doc is ABSENT *or STALE* (older than the newest kw-output
    artifact the audit evaluates). Sibling to deleg_marker_needs_refresh
    (DEBT-155), but with a BROADER trigger set: the post-worker audit evaluates
    not just kernel *code* but also the worker's CLAIMED RESULTS and docs — a
    worker respawn that fixes a coverage gap (e.g. kw-4 adding bf16 + Q-path
    cases) rewrites verification.json / edge_dataset.pt / pass_a_runner.py
    WITHOUT touching kernel code. A kernel-only freshness check (like
    deleg_marker_needs_refresh) would MISS that, leaving the stale audit in
    place. Hence this function tracks code + claimed-results + progress docs.

    Before this fix, `_ensure_audit_artifacts` regenerated the audit only when
    ABSENT. After a worker respawn closed the audit's own findings, the stale
    PARTIAL doc was never refreshed → post_worker_audit gate re-read the stale
    verdict → rollback → respawn → LOOP-BREAK → await_user_decision (observed
    on top_k_top_p_sample 2026-06-24). Mirrors the DEBT-155 infinite-rollback
    class, one layer up (audit doc instead of delegation marker).
    """
    audit_doc = workspace / "audit_self_critic_post_worker.md"
    if not audit_doc.exists():
        return True
    try:
        audit_mtime = audit_doc.stat().st_mtime
        newest = _newest_audit_input_mtime(workspace)
        return newest > audit_mtime + 1.0  # 1s slack — same as the gate
    except Exception:
        return False  # best-effort: if we can't tell, don't force a re-gen


def _record_delegation_scan_result(workspace: Path, result: subprocess.CompletedProcess) -> None:
    """Write the delegation scanner's pass marker or violation evidence.
    """
    deleg_marker = workspace / ".delegation_scan_passed"
    if result.returncode == 0:
        deleg_marker.write_text(
            f"scanner=scan_delegation_cheating.py violations=0 "
            f"ts={_dt.datetime.now(_dt.timezone.utc).isoformat()}\n"
        )
        log.info("audit: delegation scan PASS (0 violations)")
        return

    violations_path = workspace / ".delegation_scan_violations.json"
    violations_path.write_text(result.stdout or "{}")
    log.info(
        "audit: delegation scan FOUND violations "
        "(see .delegation_scan_violations.json) — finalize gate will block"
    )


def _refresh_delegation_scan_artifact(workspace: Path) -> None:
    """Refresh delegation scan evidence when the source marker is stale.
    """
    if not deleg_marker_needs_refresh(workspace):
        return

    scan_path = Path(__file__).resolve().parent.parent / "scan_delegation_cheating.py"
    if not scan_path.exists():
        return

    try:
        result = subprocess.run(
            ["python3", str(scan_path), "--workspace", str(workspace), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        _record_delegation_scan_result(workspace, result)
    except Exception as error:
        log.info(f"audit: delegation scan ERROR ({error})")


def _archive_stale_audit_doc(audit_doc: Path) -> None:
    """Archive the stale audit document without allowing archive failures to escape.
    """
    import time as _time

    stale = audit_doc.with_name(f"audit_self_critic_post_worker.STALE_{int(_time.time())}.md")
    try:
        audit_doc.rename(stale)
        log.info(f"audit: archived stale post-worker audit → {stale.name}")
    except Exception as error:
        log.info(f"audit: could not archive stale audit ({error})")


def _post_worker_audit_prompt(workspace: Path, op_name: str) -> str:
    """Build the self-critic request without changing the gate contract.
    """
    return (
        f"Run /aog-self-critic skill in post-worker mode for op {op_name}. "
        f"Workspace: {workspace}. The kw has returned, verification.json "
        "is in place. Scan the post-worker C-catalog watchpoints (C13 "
        "hallucinated metrics, C18 delegation cheating, C25/C26 anti-overfit, "
        "and any other catalog item that fires on kw output). Write "
        "audit_self_critic_post_worker.md in the workspace with verdict + "
        "findings. Verdict must be PASS, PARTIAL+waiver, or BLOCK."
    )


def _dispatch_post_worker_self_critic(op_name: str, prompt: str) -> _types.SimpleNamespace:
    """Invoke the backend and preserve the prior result-shaped adapter.
    """
    timeout_label = f"aog-self-critic post-worker ({op_name})"
    envelope = _backend.dispatch(
        "aog-self-critic",
        prompt,
        kind="skill",
        timeout=SELF_CRITIC_POST_WORKER_TIMEOUT_SEC,
    )
    if envelope.raw_envelope.get("timed_out"):
        raise subprocess.TimeoutExpired(timeout_label, SELF_CRITIC_POST_WORKER_TIMEOUT_SEC)
    return _types.SimpleNamespace(
        returncode=envelope.raw_envelope.get("returncode"),
        stdout=envelope.output_text or "",
        stderr=envelope.raw_envelope.get("stderr") or "",
    )


def _materialize_post_worker_audit(
        audit_doc: Path,
        self_critic_report: Path,
        report_mtime_before: float,
        op_name: str,
        result: _types.SimpleNamespace) -> None:
    """Keep the self-critic's file, promote its fallback, or synthesize evidence.
    """
    if audit_doc.exists():
        log.info("audit: post-worker self-critic doc written")
        return

    if self_critic_report.exists() and self_critic_report.stat().st_mtime > report_mtime_before:
        audit_doc.write_text(
            f"# Post-Worker Self-Critic Audit — {op_name}\n\n"
            f"_Auto-promoted from `self_critic_report.md` "
            f"(skill wrote canonical filename, orchestrator gate expects "
            f"`audit_self_critic_post_worker.md`)._\n\n"
            + self_critic_report.read_text()
        )
        log.info("audit: post-worker doc promoted from self_critic_report.md")
        return

    stdout_text = (result.stdout or "")[:8000]
    stderr_text = (result.stderr or "")[:2000]
    audit_doc.write_text(
        f"# Post-Worker Self-Critic Audit — {op_name}\n\n"
        f"_Synthesized from subprocess stdout — neither "
        f"`audit_self_critic_post_worker.md` nor a fresh "
        f"`self_critic_report.md` was produced by the skill._\n\n"
        f"## Subprocess returncode: {result.returncode}\n\n"
        f"## Subprocess stdout (truncated 8000)\n\n```\n{stdout_text}\n```\n\n"
        + (f"## Subprocess stderr (truncated 2000)\n\n```\n{stderr_text}\n```\n" if stderr_text else "")
    )
    log.info(
        f"audit: skill produced no audit doc; synthesized from "
        f"subprocess stdout (returncode={result.returncode})"
    )


def _refresh_post_worker_audit(workspace: Path) -> None:
    """Refresh the post-worker audit document when its inputs are newer.
    """
    if not audit_doc_needs_refresh(workspace):
        return

    audit_doc = workspace / "audit_self_critic_post_worker.md"
    if audit_doc.exists():
        _archive_stale_audit_doc(audit_doc)

    op_name = op_name_from_workspace(workspace)
    prompt = _post_worker_audit_prompt(workspace, op_name)
    self_critic_report = workspace / "self_critic_report.md"
    report_mtime_before = (
        self_critic_report.stat().st_mtime if self_critic_report.exists() else 0.0
    )
    try:
        log.info(f"audit: invoking /aog-self-critic post-worker for {op_name}...")
        result = _dispatch_post_worker_self_critic(op_name, prompt)
        _materialize_post_worker_audit(
            audit_doc,
            self_critic_report,
            report_mtime_before,
            op_name,
            result,
        )
    except subprocess.TimeoutExpired:
        log.info(
            f"audit: self-critic timed out after {SELF_CRITIC_POST_WORKER_TIMEOUT_SEC}s — finalize gate will block"
        )
    except Exception as error:
        log.info(f"audit: self-critic ERROR ({error}) — finalize gate will block")


def _ensure_audit_artifacts(workspace: Path, *, lane: int = 0) -> None:
    """Produce the delegation and post-worker audit artifacts the finalize gate requires.
    """
    try:
        from . import scan_delegation as _sd  # type: ignore[attr-defined]
    except Exception:
        _sd = None  # not required if scan_delegation_cheating.py is run as subprocess

    _refresh_delegation_scan_artifact(workspace)
    _refresh_post_worker_audit(workspace)


def _consume_applied_user_decision(workspace: Path) -> bool:
    """P0kk (2026-05-29): after the orchestrator routes FROM an already-present
    user_decision.md, rename it to .user_decision_consumed.md so the decision is
    applied exactly ONCE.

    Without this, a decision whose next_state loops back to await_user_decision
    (e.g. ``next_state: finalize`` while finalize keeps rolling back on the SAME
    O5 gate, with await_worker excluded for FA no-kw) re-advances the same stale
    decision every main-loop iteration — an infinite loop (observed 99× on
    FA-class 3_FusionAttention, ~hours of wall-clock burned). After consuming, a
    loop that returns to await_user_decision finds no file → genuine PAUSE for a
    fresh decision instead of infinite stale re-advance. A legitimate re-invoke
    (P0p) is unaffected — the state log already carries the advanced state.

    Returns True if a decision file was consumed, False if none was present.
    """
    src = workspace / "user_decision.md"
    if not src.exists():
        return False
    try:
        src.replace(workspace / ".user_decision_consumed.md")
        return True
    except OSError:
        return False


def _contains_kb_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether `text` contains one of the supplied classifier keywords.
    """
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def _structured_kb_distillation(content: str) -> Optional[tuple[dict[str, str], str, str]]:
    """Parse a structured kb_distillation block, returning None when absent.
    """
    distill_match = re.search(
        r"^kb_distillation:\s*$\n((?:^[ \t]+.+\n)+)", content, re.MULTILINE
    )
    if not distill_match:
        return None

    block_body = distill_match.group(1)
    sub_fields = {}
    for field in (
            "rule", "evidence", "applies_to", "anti_pattern_caught", "anti_pattern", "kb_target"):
        match = re.search(
            rf"^[ \t]+{re.escape(field)}:\s*[\"']?([^\"'\n]+)[\"']?\s*$",
            block_body,
            re.MULTILINE,
        )
        if match:
            sub_fields[field] = match.group(1).strip()
    anti_pattern = sub_fields.get("anti_pattern_caught") or sub_fields.get("anti_pattern", "")
    return sub_fields, anti_pattern, sub_fields.get("kb_target", "candidate")


def _heuristic_kb_distillation(content: str, op: str) -> tuple[dict[str, str], str, str]:
    """Classify an unstructured user decision while retaining its fallback contract.
    """
    reason_match = re.search(r"^reason:\s*\|?\s*\n((?:.+\n)+)", content, re.MULTILINE)
    reason_text = reason_match.group(1) if reason_match else content
    reason_lower = reason_text.lower()

    if _contains_kb_keyword(reason_lower, (
            "vendor", "not registered", "561000", "561003", "561004", "561103",
            "561108", "no binary", "kernel not registered", "tiling-key-no-match",
    )):
        kb_target = "PB"
    elif _contains_kb_keyword(reason_lower, (
            "precedent", "paradigm", "pattern", "approach", "swi_glu", "flat_quant",
            "similar to commit", "port pattern", "v220→v300", "v220->v300",
    )):
        kb_target = "OL"
    elif _contains_kb_keyword(reason_lower, (
            "build error", "compile fail", "compile error", "ec-", "错误码", "make fail",
            "linker error",
    )):
        kb_target = "EC"
    else:
        kb_target = "candidate"

    sub_fields = {
        "rule": (
            "(NEEDS_DISTILLATION — heuristic mode; main agent should rewrite as "
            "kb_distillation: structured block in user_decision.md)"
        ),
        "evidence": f"op={op}; full user_decision.md content embedded below for reviewer",
        "applies_to": "(NEEDS_DISTILLATION)",
    }
    return sub_fields, "(NEEDS_DISTILLATION)", kb_target


def _kb_draft_body(
        marker: str,
        op: str,
        timestamp: str,
        mode: str,
        sub_fields: dict[str, str],
        anti_pattern: str,
        kb_target: str,
        content: str) -> list[str]:
    """Render the KB-draft artifact with its established provenance text.
    """
    return [
        marker,
        "",
        f"Provenance: workspace/{op}/user_decision.md (extracted by orchestrator P0ff at {timestamp}; mode={mode})",
        "",
        f"## Candidate {kb_target} (auto-extracted)",
        "",
        f"- **rule**: {sub_fields.get('rule', '(missing)')}",
        f"- **evidence**: {sub_fields.get('evidence', '(missing)')}",
        f"- **applies_to**: {sub_fields.get('applies_to', '(missing)')}",
        f"- **anti_pattern_caught**: {anti_pattern or '(missing)'}",
        f"- **kb_target**: {kb_target} (suggested slot type — kb_manager will assign final ID)",
        "",
        "## Provenance directive (verbatim — for human auditor / kb_manager Mode 1)",
        "",
        "```",
        content[:8000],  # cap for kb_manager prompt size
        "```",
        "",
        "## Promotion rules",
        "",
        "- kb_manager Mode 1 MUST treat this file alongside knowledge_update.md.",
        "- Each candidate entry gets standard mechanical-scanner gates (C34a/C34b/C34c/C35).",
        "- Promoted entry MUST include `applies_to:` scope per `aog-knowledge-maintain/SKILL.md` step 2.5.",
        "- Promoted entry's source-anchor MUST cite `derived from {op}/user_decision.md session {date}`.",
        "- If mode=heuristic, kb_manager SHOULD reject promotion until main agent "
        "rewrites with structured kb_distillation block (raises a non-blocking "
        "warning to KB_USAGE_LOG.md).",
    ]


def _extract_kb_draft_from_user_decision(workspace: Path, op: str) -> Optional[Path]:
    """P0ff (2026-05-23, owner directive 20:48Z): when user_decision.md contains
    main-agent researcher content, extract it into a KB-draft file so
    aog-knowledge-maintain Mode 1 can promote the strategic insight into canonical
    KB entries — enabling customer-side cold-clone reproduction without needing
    the session-state user_decision.md file.

    The harness IS the product (per `feedback_no_patch_fix_harness_for_next_customer`).
    Strategic insight that only lives in workspace/<op>/user_decision.md is band-aid
    knowledge — it evaporates when the session ends. KB promotion is the only path
    to durable, customer-portable reproducibility.

    Two extraction modes (structured wins when present):
      A) Structured: user_decision.md has a `kb_distillation:` YAML block with
         rule / evidence / applies_to / anti_pattern_caught / kb_target fields.
         Parse the block, write a clean candidate entry.
      B) Heuristic fallback: scan the prose `reason:` body for keyword classifiers
         (vendor / API / 561xxx → PB; precedent / paradigm / pattern → OL; build
         error / compile / EC-N / 错误码 → EC; default → P-CAT candidate).

    Output: workspace/<op>/kb_draft_from_user_decision.md with provenance line +
    classified candidate section. aog-knowledge-maintain Mode 1 reads both this
    file AND knowledge_update.md when running KB promotion at finalize.

    Returns: Path to the draft file if extraction happened, None if user_decision.md
    is trivial (<100 bytes content) or already extracted (idempotency marker).
    """
    udec_path = workspace / "user_decision.md"
    if not udec_path.is_file():
        return None
    content = udec_path.read_text()
    if len(content.strip()) < 100:
        # trivial: just `next_state: X\nreason: <single line>` — not strategic
        return None

    draft_path = workspace / "kb_draft_from_user_decision.md"
    marker = f"# KB Draft Extracted From user_decision.md ({op})"
    if draft_path.is_file() and marker in draft_path.read_text():
        # already extracted — idempotency guard prevents re-append on every
        # orchestrator re-invoke that re-consumes user_decision.md
        return draft_path

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    structured = _structured_kb_distillation(content)
    if structured:
        sub_fields, anti_pat, kb_target = structured
        mode = "structured"
    else:
        sub_fields, anti_pat, kb_target = _heuristic_kb_distillation(content, op)
        mode = "heuristic"

    body = _kb_draft_body(
        marker,
        op,
        timestamp,
        mode,
        sub_fields,
        anti_pat,
        kb_target,
        content,
    )
    draft_path.write_text("\n".join(body) + "\n")
    log.info(
        f"P0ff: extracted kb_draft_from_user_decision.md "
        f"(mode={mode}, kb_target={kb_target}, {draft_path.stat().st_size} bytes)"
    )
    return draft_path

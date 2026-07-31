# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize pipeline — shared pure leaf helpers (DEBT-201, 2026-07-06).

Extracted verbatim from finalize_pipeline.py. These helpers + their marker
constants are PURE (no dependency on any finalize_pipeline module-level symbol
beyond stdlib) and are imported by BOTH finalize_pipeline AND its finalize_checks_*
sibling / finalize_candidates modules. Housing them here BREAKS the pre-existing
finalize_pipeline ↔ finalize_checks_* import cycle for these symbols (previously
the siblings did `from finalize_pipeline import _is_negative_assertion_window,
_benchmark_case_count, _has_profiler_csv_method, ...`, forcing finalize_pipeline
to be import-complete before them). None are monkeypatched. Behaviour is
byte-identical to the pre-split definitions.

Clusters:
  * verification hash              — _verification_hash
  * negative-assertion classifier  — _is_negative_assertion / _is_negative_assertion_window
  * kernel-caused context          — _is_kernel_caused_context_window
  * infra paper-over phrase tables — _INFRA_PAPER_OVER_PHRASES / _NPU_BASELINE_ERROR_CODES
  * benchmark case count           — _benchmark_case_count
  * kb writeup body length         — _kb_writeup_body_len
  * harness-internal path          — _is_harness_internal
  * v220 EC41 output-pad exemption — _is_v220_ec41_output_pad_exempt
  * profiler CSV method detection  — _has_profiler_csv_method
"""
from __future__ import annotations
import logging

import hashlib
import json
from pathlib import Path
from typing import Optional


# ── verification hash (drives .finalized-<hash> idempotency marker) ─────────
def _verification_hash(workspace: Path) -> Optional[str]:
    """sha256 of verification.json contents — drives the .finalized-<hash>
    marker so re-finalize on identical state is a no-op."""
    vp = workspace / "verification.json"
    if not vp.exists():
        return None
    return hashlib.sha256(vp.read_bytes()).hexdigest()[:16]


# ── negative-assertion markers + classifier ─────────────────────────────────


_NEGATIVE_ASSERTION_MARKERS = (
    "did not", "didn't", "did_not",
    "did not attempt", "didn't attempt",
    "do not", "don't",
    "declined to", "decline to",
    "no manual", "no replace", "no bypass", "no merge",
    "not to ", "never ",
    "without ",
    "did not perform", "didn't perform",
    "without performing", "without resorting to",
    "instead of", "rather than",
    "would not", "wouldn't",
    "deny", "denied",
    "0 hits", "zero hits", "none present",
    "did not bypass", "did not replace", "did not merge",
    "did not skip", "did not patch", "did not hand-edit",
    # DEBT-194 (2026-07-03): contrast/citation framing — the worker CITES an
    # anti-pattern (to DISTINGUISH a legitimate action from it, or to document
    # what was NOT done), not performs it. P96 false-positived on a real gate-#2
    # run's honest line: "distinguishing legitimate recovery … from the actual
    # P9 anti-patterns (replace .so / bypass --pkg / merge binary / …)".
    "distinguish", "the actual p9", "actual anti-pattern", "actual p9 anti",
    "not a p9 anti-pattern", "not a p9", "legitimate recovery",
)


def _is_negative_assertion(context_text: str) -> bool:
    """DEBT-98: True if context_text contains a negative-assertion marker."""
    for marker in _NEGATIVE_ASSERTION_MARKERS:
        if marker in context_text:
            return True
    return False


def _is_negative_assertion_window(text_lower: str, match_start: int, match_end: int) -> bool:
    """DEBT-98 (2026-05-24, GMSQ_v2 P96 false-positive loop): True if the
    keyword match at [match_start, match_end) is in a negative-assertion
    context — worker documenting P9 anti-pressure discipline (they DID NOT
    perform the paper-over). Checks 200 chars BEFORE the match AND 200 chars
    AFTER (negation can appear on either side: "did NOT replace .so" before,
    or "`replace .so` ... **NONE present**" pattern after).

    Also detects backtick-quoted-keyword pattern: when the keyword is wrapped
    in backticks (`...`), it's being CITED as a phrase-to-audit, not used.
    """
    win_before_start = max(0, match_start - 200)
    win_after_end = min(len(text_lower), match_end + 200)
    context_before = text_lower[win_before_start:match_start]
    context_after = text_lower[match_end:win_after_end]

    # Negation marker in either window → P9 negative assertion
    if _is_negative_assertion(context_before):
        return True
    if _is_negative_assertion(context_after):
        return True

    # Backtick-quoted phrase pattern: keyword inside `...` typically means
    # citation (worker listing keywords-to-audit), not action
    if match_start > 0 and match_end < len(text_lower):
        if text_lower[match_start - 1] == '`' and text_lower[match_end] == '`':
            return True

    # DEBT-194 (2026-07-03): citation-in-an-enumeration. When the phrase sits
    # inside a parenthetical slash-list of >=2 separators (e.g. "(replace .so /
    # bypass --pkg / merge binary / retry)"), the worker is ENUMERATING the
    # anti-patterns (to cite/distinguish), not performing them — you don't
    # "do" a slash-separated list. A real paper-over reads "I bypassed --pkg",
    # not a 3-item slash-list; so this is structurally safe (no evasion hole).
    open_paren = text_lower.rfind("(", 0, match_start)
    close_paren = text_lower.find(")", match_end)
    if 0 <= open_paren and 0 <= close_paren and (close_paren - open_paren) < 300:
        enclosed = text_lower[open_paren:close_paren]
        if enclosed.count(" / ") >= 2:
            return True

    return False


# ── kernel-caused context markers + classifier ──────────────────────────────


_KERNEL_CAUSED_CONTEXT_MARKERS = (
    "kernel code bug", "kernel bug", "kernel-caused", "kernel caused",
    "kernel-execution exception", "kernel execution exception",
    "kernel code", "in-kernel fix", "in kernel fix", "kernel fix",
    "not infra", "not an infra", "non-infra", "npu is healthy",
    "env is healthy", "env baseline is healthy", "baseline is healthy",
    "env probe", "ub bug", "ub-budget", "ub budget", "popstackbuffer",
    "vector-core exception", "aivec", "sharedtmpbuffer",
)


def _is_kernel_caused_context_window(text_lower: str, match_start: int, match_end: int) -> bool:
    """P96 self-defeat fix (2026-06-16): True if an NPU baseline error-code
    citation at [match_start, match_end) sits in a KERNEL-CAUSED context — the
    docs within +/-300 chars give evidence it was the worker's own kernel bug
    (env-health proof + kernel-bug/kernel-fix markers), NOT an infra baseline
    violation requiring INFRA_BASELINE_VIOLATED escalation. The cause analysis +
    env-health proof can sit on either side of the code mention, hence the
    two-sided window. Mirrors `_is_negative_assertion_window` (DEBT-98), which
    the paper-over phrase-check already honors. Fail-honest: a citation lacking
    any such marker is still treated as a (conservative) infra violation.
    """
    win_before = max(0, match_start - 300)
    win_after = min(len(text_lower), match_end + 300)
    context = text_lower[win_before:win_after]
    return any(marker in context for marker in _KERNEL_CAUSED_CONTEXT_MARKERS)


# ── infra paper-over phrases + npu baseline error codes ─────────────────────


_INFRA_PAPER_OVER_PHRASES = (
    # Critical .so replacement (worker mucked with install tree libs)
    "replace libophost", "replaced libophost", "replacing libophost",
    "replace libopapi", "replaced libopapi", "replacing libopapi",
    "libophost.*rollback", "rollback libophost",
    "libopapi.*rollback", "rollback libopapi",
    # Manual install bypassing ops-nn-port --pkg
    "manual install", "manually install",
    "bypass --pkg", "bypassing --pkg", "bypass the --pkg",
    "bypass build pipeline", "bypassing build pipeline",
    # binary_info_config.json hand-edit
    "merge binary_info_config", "patch binary_info_config",
    "hand-edit binary_info_config", "hand edit binary_info_config",
    # NPU error code in PROGRESS without INFRA_BASELINE_VIOLATED handoff
    # (checked separately — these phrases are bare-mention triggers)
)


_NPU_BASELINE_ERROR_CODES = ("507008", "507033", "507035", "561103")


# ── benchmark case count ────────────────────────────────────────────────────


def _benchmark_case_count(workspace: Path) -> Optional[int]:
    """Count cases in workspace's <op>.json (one JSON-line per case).

    Returns None if no .json file is found or it isn't readable.
    """
    op = workspace.name
    candidates = [
        workspace / f"{op}.json",
        workspace / "model.json",
    ]
    # Also look for any single .json with op-name prefix (e.g.
    # "3_FusionAttention.json" when workspace is "3_FusionAttention").
    for child in workspace.glob(f"{op.split('_')[0]}_*.json"):
        if child not in candidates:
            candidates.append(child)
    for path in candidates:
        if not path.exists():
            continue
        skip_current_item = False
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        # Each non-blank JSON object is one declared case.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # Sanity: each line should parse as JSON with an "inputs" field.
        try:
            valid = sum(
                1 for ln in lines
                if json.loads(ln).get("inputs") is not None
            )
        except Exception:
            # Malformed; fall back to non-blank line count if it looks
            # like the benchmark format (first char is `{`).
            valid = sum(1 for ln in lines if ln.startswith("{"))
        if valid > 0:
            return valid
    return None


# ── kb writeup body length ──────────────────────────────────────────────────


def _kb_writeup_body_len(workspace: Path) -> int:
    """Helper for the eligible-success message — re-reads knowledge_update.md
    length (the inline code reused the `body` variable; this keeps the success
    message identical without threading state through the gate iterator)."""
    ku = workspace / "knowledge_update.md"
    if not ku.exists():
        ku = workspace / ".harness" / "knowledge_update.md"
    if not ku.exists():
        return 0
    try:
        return len(ku.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return 0


# ── harness-internal path classifier ────────────────────────────────────────


# User directive 2026-05-16 02:00Z: 算子 archive 根目录只放给客户用的文件,
# harness 内部状态/审计/KB-反馈文件统一放到 .harness/ 子目录。
_HARNESS_INTERNAL_FILES: frozenset[str] = frozenset({
    # State machine + orchestrator audit
    "state_transitions.jsonl",
    "state_transitions.pre-fo-recovery.jsonl",
    "orchestrator_events.jsonl",
    # Self-critic / audit outputs
    "audit_self_critic_post_worker.md",
    "self_critic_report.md",
    # KB-feedback / harness-only docs
    "knowledge_update.md",
    "cpu_truth_template.md",
    "prior_art_learn.md",
    # Workflow decisions (not customer-facing)
    "user_decision.md",
    # Project meta (harness uses for routing)
    "op_classification.json",
    "a3_reference_runnable.json",
    # State files (already filtered by leading-`.` rule, but safety)
    ".opgen_state.json",
    ".kb_merged",
})


def _is_harness_internal(rel: str) -> bool:
    """Should this workspace-relative file land in archive's .harness/?

    Matches: top-level files in _HARNESS_INTERNAL_FILES, all
    `audit_self_critic_*.md` variants (STALE / kw3 / etc.), all
    `.finalized-*` markers (no extension after the hex hash),
    all `.cc_stream_log_<agent>_<idx>.jsonl` per-agent stream-json
    traces (Meta-Harness training-data; see ROADMAP §F7).
    """
    name = rel.split("/")[-1]
    if name in _HARNESS_INTERNAL_FILES:
        return True
    # audit_self_critic_post_worker.STALE_*.md variants
    if name.startswith("audit_self_critic_post_worker.") and name.endswith(".md"):
        return True
    # .finalized-<hash> markers (with or without extension)
    if name.startswith(".finalized-"):
        return True
    # .cc_stream_log_<agent>_<idx>.jsonl — per-agent stream-json full trace
    # (2026-05-19, F7 Meta-Harness data preservation). These are the
    # richest training inputs the harness produces; without archive-time
    # copy they get lost when workspace/ is reset on next cold-start.
    if name.startswith(".cc_stream_log_") and name.endswith(".jsonl"):
        return True
    return False


# ── v220 EC41 output-pad exemption ──────────────────────────────────────────


def _is_v220_ec41_output_pad_exempt(workspace: Path, vj: dict) -> bool:
    """P149 Pattern-2 STRUCTURAL carve-out (2026-06-03, main-authorized w/ hard
    constraints + main pre-merge review; back-agent backward L1-L3 sweep, gelu_grad/
    mul_grad anchor).

    The `torch::empty(numel+PAD)` + `.narrow(0,0,numel)` output-pad pattern is
    HARDWARE-FORCED on V220 (arch22), NOT host-compute masking a bug: `DataCopyPad`
    UB→GM CRASHES on V220 (EC-23) and OL-120 mandates 3-arg `DataCopy` with a
    32B-aligned `count`, so a non-32B-aligned tail must be written by an aligned
    `DataCopy` that overruns into a pad-scratch buffer, then cropped by `.narrow`
    (EC-41 / OL-181). The valid region is the kernel's aligned write, precision-verified.

    Exempt ONLY when ALL hold (structural — NOT a target-flag blanket):
      1. V220 / arch22 target (where DataCopyPad is EC-23-forbidden).
      2. the kernel (op_kernel/kernel *.h/*.cpp, comments stripped) ACTUALLY uses an
         aligned 3-arg `DataCopy(... AlignUp/Align32/RoundUp ...)` AND has NO real
         `DataCopyPad(` call — proving the pad is the V220-forced aligned-overrun
         scratch, not an arbitrary host cleanup.
      3. precision.status == PASS / PASS_WITHIN_TOLERANCE (valid region bit-correct).
    This NEVER exempts Pattern-1 (CPU offload) / Pattern-3 (cat/stack) / Pattern-5
    (model_new_ascendc.py compute) — only the Pattern-2 output-pad, only on V220, only
    with the structural aligned-DataCopy + no-DataCopyPad signal.
    """
    import re as _re
    # (1) V220 / arch22
    arch = str(vj.get("arch") or "").lower()
    target = str(vj.get("target") or "").lower()
    is_v220 = ("arch22" in arch) or ("v220" in arch) or (target == "a3")
    if not is_v220:
        return False
    # (3) precision verified bit-correct on the valid region — by the INDEPENDENT oracle.
    # NOT the worker's self-declared status: require that the precision came from the
    # self-contained autograd verify (verify_<op>.py vs fp64 torch.autograd.grad — an oracle
    # computed independently of the worker's kernel logic, AND re-measured by phase_o5
    # backward_verify_runner BEFORE finalize is reached). A worker cannot earn the exemption
    # by self-asserting "PASS" without the autograd-oracle method signature.
    prec = vj.get("precision") or {}
    if prec.get("status") not in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return False
    pa_method = str((prec.get("pass_a") or {}).get("method") or "").lower()
    if "autograd" not in pa_method:
        return False  # exemption requires independent autograd-oracle verification evidence
    # (2) structural kernel signal: aligned 3-arg DataCopy + NO real DataCopyPad call
    ktext = ""
    for d in (workspace / "kernel", workspace / "op_kernel", workspace):
        if d.is_dir():
            for f in list(d.glob("*.h")) + list(d.glob("*.cpp")):
                if f.name == "pybind11.cpp":
                    continue
                try:
                    ktext += f.read_text(errors="replace") + "\n"
                except Exception as error:
                    logging.getLogger(__name__).debug(
                        "Recoverable operation failed.", exc_info=error
                    )
    if not ktext:
        return False
    # strip comments so the "no DataCopyPad on V220" doc-comment isn't mistaken for a call
    knc = _re.sub(r"//[^\n]*", "", ktext)
    knc = _re.sub(r"/\*.*?\*/", "", knc, flags=_re.S)
    if _re.search(r"\bDataCopyPad\s*\(", knc):
        return False  # real DataCopyPad call → not the V220-forced 3-arg path
    has_aligned_3arg_datacopy = bool(
        _re.search(r"\bDataCopy\s*\([^;]*(?:AlignUp|Align32|ALIGN|RoundUp)", knc)
    )
    return has_aligned_3arg_datacopy


# ── profiler CSV method detection ───────────────────────────────────────────


_PROFILER_CSV_TOKENS = ("operator_details", "kernel_details")


_DEVICE_DURATION_TOKENS = ("device self duration", "device_self_duration")


def _has_profiler_csv_method(method_low: str) -> bool:
    """torch_npu.profiler symmetric device-side measurement detection.

    Shared between P141 (_check_perf_methodology) and P0ee
    (_check_methodology_declaration) — both gates need the same positive
    signal for the same purpose (gate-accept the profiler-CSV path as a
    legitimate symmetric methodology). Extracting into helper avoids the
    god-module pattern Zheng codified 2026-05-26 16:13Z (duplicated check
    sites = future extension requires N edits instead of 1).

    Accepts any of `_PROFILER_CSV_TOKENS` as the CSV source. Both CSVs are
    torch_npu.profiler outputs; either is a symmetric device-side
    measurement source. The operator-vs-kernel API distinction is
    irrelevant for the gate's purpose.
    """
    return (
        "torch_npu.profiler" in method_low
        and any(t in method_low for t in _PROFILER_CSV_TOKENS)
        and any(t in method_low for t in _DEVICE_DURATION_TOKENS)
    )

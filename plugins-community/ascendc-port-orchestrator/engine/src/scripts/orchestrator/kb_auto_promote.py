# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""KB auto-promote pipeline (P0aco — implements Mode 5 spec from /aog-knowledge-maintain).

Replaces the legacy user-sign-off gate (`.kb_review_signed_<id>` per P0x v2) with
fully automated promotion. Per user directive 2026-05-10 + self-critic C40:
the harness is 0-interaction self-evolving; user-gate at KB promotion violates
that contract.

Pipeline (7 steps from SKILL.md Mode 5 spec):
  1. Scan .kb_promotion_pending-* markers
  2. C36 generalize — auto op-class lift if title is op-name-keyed
  3. C37 auto-dedup — n-gram > 30% overlap → auto-merge into existing entry
  4. C38 conflict detection — flag for codex if logical contradiction
  5. C39 cross-op transferability — dry-run brief inject on sibling op
  6. Codex review hook (optional, when `which codex` succeeds)
  7. Promote (rename CAND-X → OL-N / P-PN / EC-N / PB-N) + drop audit marker

Surface: orchestrator Phase O6 step 1 calls this module after archive promotion.
NOT a user gate — failures emit `.kb_promotion_blocked-<id>` markers + audit report,
orchestrator continues regardless.
"""
from __future__ import annotations
import logging

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# LLM review-call timeouts (env-overridable, default 1800s).
CODEX_REVIEW_TIMEOUT_SEC = int(os.environ.get("AOG_CODEX_REVIEW_TIMEOUT_SEC", "1800"))
CLAUDE_FALLBACK_REVIEW_TIMEOUT_SEC = int(os.environ.get("AOG_CLAUDE_FALLBACK_REVIEW_TIMEOUT_SEC", "1800"))


# ----------------------------------------------------------------------------
# Data types
# ----------------------------------------------------------------------------

@dataclass
class Candidate:
    """One CAND-X block parsed from candidates.md."""
    candidate_id: str  # e.g. "CAND-FA5"
    title: str
    body: str
    metadata: dict[str, str] = field(default_factory=dict)
    source_block_start_line: int = 0
    source_block_end_line: int = 0


@dataclass
class PromotionResult:
    candidate_id: str
    outcome: str  # "promoted" | "merged" | "blocked"
    target_id: Optional[str] = None  # new canonical ID if promoted
    reason: Optional[str] = None  # block reason
    codex_review: str = "unavailable"  # "approve" | "needs_revision" | "reject" | "unavailable"
    transferability_test: str = "skipped"  # "pass" | "fail" | "skipped"


@dataclass
class PromotionBatchReport:
    markers_processed: int
    promoted: list[PromotionResult] = field(default_factory=list)
    merged: list[PromotionResult] = field(default_factory=list)
    blocked: list[PromotionResult] = field(default_factory=list)
    started_ts: float = field(default_factory=time.time)
    finished_ts: Optional[float] = None


# ----------------------------------------------------------------------------
# Step 1: Scan pending markers
# ----------------------------------------------------------------------------

def scan_pending_markers(kb_root: Path) -> list[Path]:
    """Find all .kb_promotion_pending-* markers under patterns/unverified/."""
    pattern_dir = kb_root / "patterns" / "unverified"
    if not pattern_dir.is_dir():
        return []
    return sorted(pattern_dir.glob(".kb_promotion_pending-*"))


def load_marker(marker_path: Path) -> dict[str, Any]:
    """Parse JSON marker file."""
    return json.loads(marker_path.read_text())


# ----------------------------------------------------------------------------
# Step 2: Parse candidate from candidates.md
# ----------------------------------------------------------------------------

_CAND_BLOCK_RE = re.compile(r"^## (CAND-[A-Za-z0-9_-]+)\s*[:：]?\s*(.*)$", re.MULTILINE)


def load_candidate(candidates_md: Path, candidate_id: str) -> Optional[Candidate]:
    """Parse the ## CAND-<id> block out of candidates.md."""
    if not candidates_md.exists():
        return None
    text = candidates_md.read_text()
    lines = text.split("\n")

    # Find the start line
    start = None
    for i, line in enumerate(lines):
        m = re.match(r"^## (CAND-[A-Za-z0-9_-]+)\s*[:：]?\s*(.*)$", line)
        if m and m.group(1) == candidate_id:
            start = i
            title = m.group(2).strip()
            break
    if start is None:
        return None

    # Find next ## line (or end of file)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    body_lines = lines[start + 1:end]
    body = "\n".join(body_lines).strip()

    # Extract metadata from inline ` `applies_to:`, `derived-from:`, etc. lines
    metadata: dict[str, str] = {}
    for ln in body_lines[:8]:  # metadata block conventionally near top
        m_meta = re.match(r"^`([a-z_-]+)\s*:\s*(.+)`$", ln.strip())
        if m_meta:
            metadata[m_meta.group(1)] = m_meta.group(2).strip()

    return Candidate(
        candidate_id=candidate_id,
        title=title,
        body=body,
        metadata=metadata,
        source_block_start_line=start,
        source_block_end_line=end,
    )


# ----------------------------------------------------------------------------
# Step 3: C36 generalize check
# ----------------------------------------------------------------------------

# Op-name keying patterns. If title contains these, it's likely op-bound.
_OP_NAME_PATTERNS = [
    r"\b\d+_[A-Z][a-zA-Z]+\b",  # benchmark op numbers like 22_Nonzero, 3_FusionAttention
    r"\bFA\d*\b",                 # FA-specific keys (CAND-FA1..5)
    r"\b[A-Z][a-zA-Z]+(?:Attn|Attention|Layernorm|Cumsum|Nonzero|MatMul)\b",  # well-known op names
]


def check_c36_generalize(candidate: Candidate) -> tuple[bool, str]:
    """Return (passes_generalization, reason).

    A candidate fails C36 if:
    - title contains an op-name pattern AND no `applies_to: op_class=...` metadata
    - body cites only one concrete op (no `applies_to:` or `applies_to:` is op-name not class)

    Pass = either explicitly op_class-tagged OR title is principle-first.
    """
    applies_to = candidate.metadata.get("applies_to", "")

    # Check 1: applies_to: op_class=<X> presence
    if "op_class=" in applies_to:
        return (True, "applies_to: op_class= present")

    # Check 2: title contains op-name pattern but no class
    title_has_op = any(
        re.search(pattern, candidate.title) for pattern in _OP_NAME_PATTERNS
    )
    if title_has_op:
        return (False,
                "C36 — title contains op-name without applies_to:op_class=; auto-rewrite "
                "candidate via lift_to_op_class() before promote, OR add applies_to: line")

    # Check 3: applies_to references a single op-name (not class)
    if applies_to and "op_class=" not in applies_to and "soc=" not in applies_to:
        # e.g. "applies_to: 3_FusionAttention" — too narrow
        if any(re.search(pattern, applies_to) for pattern in _OP_NAME_PATTERNS):
            return (False,
                    f"C36 — applies_to: {applies_to} is op-name, not op_class")

    return (True, "principle-first title; no op-name detected")


def lift_to_op_class(candidate: Candidate, op_classification: dict[str, Any]) -> Candidate:
    """Auto-rewrite candidate to op-class form. Pulls op_class from op_classification.json.

    Conservative: just adds `applies_to: op_class=<derived>` metadata if missing.
    Does NOT rewrite title or body (those need LLM judgment; auto-rewriting risks
    semantic damage).
    """
    if "applies_to" in candidate.metadata and "op_class=" in candidate.metadata["applies_to"]:
        return candidate  # already lifted

    op_class_tags = op_classification.get("op_class_tags", [])
    primary_class = op_class_tags[0] if op_class_tags else None

    new_metadata = dict(candidate.metadata)
    if primary_class:
        new_metadata["applies_to"] = f"op_class={primary_class}"

    return Candidate(
        candidate_id=candidate.candidate_id,
        title=candidate.title,
        body=candidate.body,
        metadata=new_metadata,
        source_block_start_line=candidate.source_block_start_line,
        source_block_end_line=candidate.source_block_end_line,
    )


# ----------------------------------------------------------------------------
# Step 4: C37 auto-dedup via n-gram overlap
# ----------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for n-gram overlap."""
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def n_gram_overlap_ratio(text_a: str, text_b: str, n: int = 5) -> float:
    """Jaccard-like overlap on n-grams. Range [0.0, 1.0]."""
    ng_a = _ngrams(_tokenize(text_a), n=n)
    ng_b = _ngrams(_tokenize(text_b), n=n)
    if not ng_a or not ng_b:
        return 0.0
    intersection = len(ng_a & ng_b)
    union = len(ng_a | ng_b)
    return intersection / union if union > 0 else 0.0


def find_duplicate(candidate: Candidate, kb_root: Path,
                   threshold: float = 0.30) -> Optional[tuple[str, float]]:
    """Scan canonical KB files; return (entry_id, overlap_ratio) of first dup, else None."""
    kb_files = [
        kb_root / "OPERATIONAL_KNOWLEDGE.md",
        kb_root / "ERROR_CORRECTIONS.md",
        kb_root / "PLATFORM_BUGS.md",
        kb_root / "patterns" / "PATTERN_INDEX.md",
    ]
    for f in kb_files:
        if not f.exists():
            continue
        text = f.read_text(errors="replace")
        # Split by ## entry headers and check each
        entries = re.split(r"^## ([A-Z]+-?[A-Z0-9-]+):", text, flags=re.MULTILINE)
        # The split result alternates the preamble, each entry identifier,
        # and that entry's body.
        for i in range(1, len(entries) - 1, 2):
            entry_id = entries[i].strip()
            entry_body = entries[i + 1] if i + 1 < len(entries) else ""
            ratio = n_gram_overlap_ratio(candidate.body, entry_body)
            if ratio > threshold:
                return (entry_id, ratio)
    return None


# ----------------------------------------------------------------------------
# Step 6: Codex review hook
# ----------------------------------------------------------------------------

def _find_codex() -> Optional[str]:
    """Find codex executable. Checks shutil.which() first, then nvm directories.

    P0acp 2026-05-10: PATH inherited by Python subprocess may exclude nvm bins
    (codex is typically installed as nvm-managed npm package). Fall back to
    glob over common nvm install paths.
    """
    p = shutil.which("codex")
    if p:
        return p
    import glob
    candidates = (
        glob.glob("/home/*/.nvm/versions/node/*/bin/codex") +
        glob.glob("/usr/local/bin/codex") +
        glob.glob("/usr/bin/codex")
    )
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def codex_available() -> bool:
    """Check if codex CLI is on PATH or nvm directories."""
    return _find_codex() is not None


def codex_review(candidate: Candidate, *, timeout_sec: int = CODEX_REVIEW_TIMEOUT_SEC) -> tuple[str, str]:
    """Run codex review on candidate. Returns (verdict, feedback).

    verdict ∈ {"approve", "needs_revision", "reject", "unavailable"}
    """
    codex_bin = _find_codex()
    if codex_bin is None:
        return ("unavailable", "codex CLI not found in PATH or nvm dirs")

    # P0acs 2026-05-10: removed `body[:3000]` truncation — codex was correctly
    # flagging "truncated body" because we were chopping the prompt input!
    # Allow the full candidate body to reach codex.
    # If codex's own output gets truncated, we handle via output-last-message
    # file separately.
    prompt = (
        f"Review this candidate for promotion to canonical KB.\n\n"
        f"Title: {candidate.title}\n\n"
        f"Body:\n{candidate.body}\n\n"
        f"Criteria: (1) generalizable principle, not op-specific; (2) no logical "
        f"contradiction with existing KB; (3) cross-op transferable. \n\n"
        f"Output format (REQUIRED — machine-parsed):\n"
        f"Line 1: VERDICT: APPROVE | NEEDS_REVISION | REJECT\n"
        f"Line 2+: brief reason (one paragraph max — do NOT emit revised content here)\n\n"
        f"If NEEDS_REVISION: in a SEPARATE codex pass the caller will request the "
        f"revision; do NOT include REVISED_TITLE/REVISED_BODY in this verdict response. "
        f"Keep verdict response short (under 500 tokens) — focus on identifying the "
        f"specific issues that need fixing."
    )

    try:
        result = subprocess.run(
            [codex_bin, "exec", "--sandbox", "read-only", prompt],
            capture_output=True, text=True, timeout=timeout_sec, check=False,
        )
        output = result.stdout.strip()
        first_line = output.split("\n", 1)[0].strip().upper() if output else ""
        if "APPROVE" in first_line and "NEEDS" not in first_line:
            return ("approve", output[:500])
        if "NEEDS_REVISION" in first_line or "NEEDS REVISION" in first_line:
            return ("needs_revision", output[:5000])  # full feedback for revise loop
        if "REJECT" in first_line:
            return ("reject", output[:500])
        # Fallback: scan body
        lower = output.lower()[:500]
        if "needs_revision" in lower or "needs revision" in lower:
            return ("needs_revision", output[:5000])
        if "reject" in lower:
            return ("reject", output[:500])
        if "approve" in lower:
            return ("approve", output[:500])
        return ("needs_revision", "(no clear verdict; treating as NEEDS_REVISION for safety)")
    except subprocess.TimeoutExpired:
        return ("unavailable", f"codex timed out at {timeout_sec}s")
    except Exception as e:
        return ("unavailable", f"codex error: {e}")


def claude_fallback_review(
    candidate: Candidate,
    *,
    timeout_sec: int = CLAUDE_FALLBACK_REVIEW_TIMEOUT_SEC,
) -> tuple[str, str]:
    """Fallback review via the active harness backend when codex unavailable.

    P0acu (2026-05-10): User directive 2026-05-10 — '启动一个独立的agent去review'.
    Spawns a fresh reviewer through the backend resolver (own context, no shared
    state with the candidate author or the orchestrator running this
    function). Returns verdict in same shape as
    codex_review.

    Compatibility: verdicts keep the historical `claude_fallback_*` prefix so
    existing promotion logic and audit greps keep working. The actual backend is
    recorded in the feedback text.

    Verdict ∈ {"claude_fallback_approve", "claude_fallback_needs_revision",
               "claude_fallback_reject", "unavailable"}
    """
    # Lazy import — the backend resolver imports harness-specific adapters.
    try:
        import sys as _sys
        _here = Path(__file__).resolve().parent
        if str(_here) not in _sys.path:
            _sys.path.insert(0, str(_here))
        from backends import get_backend
    except Exception as e:
        return ("unavailable", f"backend resolver import failed: {e}")

    backend = get_backend()

    slug = f"kbrev-{candidate.candidate_id.lower()}"
    prompt = (
        f"{slug} — KB candidate review (codex fallback)\n\n"
        f"You are reviewing a CANN-source-derived KB candidate for promotion to the "
        f"canonical pattern library. Codex (the primary second-opinion reviewer) is "
        f"unavailable, so you are the fallback reviewer. Be STRICT — only APPROVE if "
        f"the candidate is generalizable, non-contradictory, and cross-op transferable.\n\n"
        f"Candidate:\n"
        f"  Title: {candidate.title}\n"
        f"  applies_to: {candidate.metadata.get('applies_to', '(missing)')}\n"
        f"  derived-from: {candidate.metadata.get('derived-from', '(missing)')}\n\n"
        f"Body:\n{candidate.body}\n\n"
        f"Review criteria:\n"
        f"1. Generalizable: principle is op-class-keyed, NOT specific to one op name.\n"
        f"2. Non-contradictory: does not logically conflict with existing canonical "
        f"   patterns (P-Pxx, OL-xx, EC-xx, PB-xx) in src/skills/references/.\n"
        f"3. Cross-op transferable: at least one applies_to op_class label, the "
        f"   stated trigger applies to multiple ops in that class, not a single instance.\n"
        f"4. Sound: technical mechanism described is correct; no fabricated AscendC "
        f"   API names; recommendations are implementable with public AscendC headers.\n"
        f"5. Bounded scope: no 'always do X' over-generalizations that contradict "
        f"   carve-outs already in KB (e.g. OL-63 thin-compute, OL-83 tolerance).\n\n"
        f"REQUIRED output format (machine-parsed; first line must match exactly):\n"
        f"Line 1: VERDICT: APPROVE | NEEDS_REVISION | REJECT\n"
        f"Line 2+: brief reason (one paragraph max).\n\n"
        f"If NEEDS_REVISION, ALSO emit (after the reason paragraph):\n"
        f"REVISED_TITLE: <revised one-line title>\n"
        f"REVISED_BODY:\n"
        f"<revised body, multi-line, fixing the specific issues you flagged>\n\n"
        f"Do NOT use any tools other than reading existing KB files for cross-check. "
        f"Stay under 1500 output tokens. End with just the verdict block — no preamble."
    )

    try:
        env = backend.dispatch(
            "general-purpose",
            prompt,
            kind="agent",
            mode="foreground",
            permission_mode="bypassPermissions",
            timeout=timeout_sec,
        )
        if env.is_error:
            return (
                "unavailable",
                f"{backend.name} fallback reviewer failed: "
                f"{(env.raw_envelope.get('stderr') or env.output_text or '')[:500]}",
            )
        output = (env.output_text or "").strip()
    except Exception as e:
        return ("unavailable", f"{backend.name} fallback spawn failed: {e}")

    if not output:
        return ("unavailable", f"{backend.name} fallback returned empty output")

    output = f"[fallback_backend={backend.name}]\n{output}"

    first_line = output.split("\n", 2)[1].strip().upper() if "\n" in output else output.upper()
    if "APPROVE" in first_line and "NEEDS" not in first_line:
        return ("claude_fallback_approve", output[:500])
    if "NEEDS_REVISION" in first_line or "NEEDS REVISION" in first_line:
        return ("claude_fallback_needs_revision", output[:5000])
    if "REJECT" in first_line:
        return ("claude_fallback_reject", output[:500])
    # Fallback scan
    lower = output.lower()[:500]
    if "needs_revision" in lower or "needs revision" in lower:
        return ("claude_fallback_needs_revision", output[:5000])
    if "reject" in lower:
        return ("claude_fallback_reject", output[:500])
    if "approve" in lower:
        return ("claude_fallback_approve", output[:500])
    return ("claude_fallback_needs_revision",
            "(no clear verdict; treating as NEEDS_REVISION for safety)")


def parse_codex_revision(codex_feedback: str) -> Optional[tuple[str, str]]:
    """Extract (revised_title, revised_body) from codex feedback if present.

    Format expected: 'REVISED_TITLE: <title>\\nREVISED_BODY:\\n<body>'.
    Returns None if not found (codex didn't include revision; fallback to keep
    candidate as-is in unverified bucket with feedback attached).
    """
    title_marker = "REVISED_TITLE:"
    body_marker = "REVISED_BODY:"
    if title_marker not in codex_feedback or body_marker not in codex_feedback:
        return None
    try:
        after_title = codex_feedback.split(title_marker, 1)[1]
        title_part, after_body = after_title.split(body_marker, 1)
        revised_title = title_part.strip().split("\n", 1)[0].strip()
        revised_body = after_body.strip()
        if revised_title and revised_body:
            return (revised_title, revised_body)
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    return None


def apply_revision(candidate: Candidate, revised_title: str,
                  revised_body: str) -> Candidate:
    """Build a new Candidate with title/body replaced from codex revision.

    P0acw round-4 (2026-05-11): codex/Claude revisions often emit a
    revised_body that ALREADY starts with metadata lines (`applies_to: ...` /
    `derived-from: ...` / `verified_on: ...`). If we naively keep the original
    candidate.metadata AND write the revised_body verbatim, the file ends
    up with DUPLICATE metadata blocks — which codex catches and rejects as
    "duplicated `applies_to`/`derived-from`/`verified_on` blocks". Parse
    the leading metadata from revised_body, merge into candidate.metadata
    (revision takes precedence), and strip from body.
    """
    new_metadata = dict(candidate.metadata)
    body_lines = revised_body.split("\n")
    consumed = 0
    metadata_pattern = re.compile(r"^`([a-zA-Z_-]+):\s*(.+?)`\s*$")
    for line in body_lines:
        if not line.strip():
            consumed += 1  # skip blank lines between metadata
            continue
        m = metadata_pattern.match(line.strip())
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            new_metadata[key] = val
            consumed += 1
            continue
        break  # first non-metadata, non-blank line — body starts here

    cleaned_body = "\n".join(body_lines[consumed:]).lstrip("\n")
    return Candidate(
        candidate_id=candidate.candidate_id,
        title=revised_title,
        body=cleaned_body,
        metadata=new_metadata,
        source_block_start_line=candidate.source_block_start_line,
        source_block_end_line=candidate.source_block_end_line,
    )


def _persist_revision(candidates_md: Path, revised_cand: Candidate) -> None:
    """Overwrite the candidate's block in candidates.md with the revised content.

    Preserves block boundaries; subsequent `promote_candidate()` will then move
    the revised text (not the original) to the canonical KB file.
    """
    if not candidates_md.exists():
        return
    text = candidates_md.read_text()
    lines = text.split("\n")
    # Re-locate the block (boundaries may have shifted if upstream edits occurred)
    start = None
    for i, line in enumerate(lines):
        m = re.match(r"^## (CAND-[A-Za-z0-9_-]+)\s*[:：]?\s*", line)
        if m and m.group(1) == revised_cand.candidate_id:
            start = i
            break
    if start is None:
        return
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    # Build new block: header + metadata lines + body
    new_block = [f"## {revised_cand.candidate_id}: {revised_cand.title}"]
    for k, v in revised_cand.metadata.items():
        new_block.append(f"`{k}: {v}`")
    new_block.append("")
    new_block.append(revised_cand.body.rstrip())
    new_block.append("")  # trailing blank line

    new_lines = lines[:start] + new_block + lines[end:]
    candidates_md.write_text("\n".join(new_lines))


# ----------------------------------------------------------------------------
# Step 7: Promote (rename CAND-X → canonical)
# ----------------------------------------------------------------------------

def allocate_next_id(target_file: Path, prefix: str) -> str:
    """Find next sequential ID like P-P89 / OL-130 in target file."""
    if not target_file.exists():
        return f"{prefix}1"
    text = target_file.read_text(errors="replace")
    pattern = rf"{re.escape(prefix)}(\d+)"
    matches = re.findall(pattern, text)
    if not matches:
        return f"{prefix}1"
    next_n = max(int(m) for m in matches) + 1
    return f"{prefix}{next_n}"


def promote_candidate(candidate: Candidate, kb_root: Path,
                      candidates_md: Path, target_file: Path,
                      new_id: str, *, dry_run: bool = False) -> dict:
    """Move candidate from candidates.md to target_file with new_id.

    Returns audit dict.
    """
    if dry_run:
        return {"dry_run": True, "would_promote": candidate.candidate_id,
                "would_target": str(target_file), "would_new_id": new_id}

    # 1. Read both files
    candidates_text = candidates_md.read_text()
    target_text = target_file.read_text() if target_file.exists() else ""

    # 2. Extract candidate block from candidates.md
    lines = candidates_text.split("\n")
    block_lines = lines[candidate.source_block_start_line:candidate.source_block_end_line]
    block_text = "\n".join(block_lines).rstrip()

    # 3. Rewrite block header CAND-X → new canonical ID + title
    promoted_block = re.sub(
        rf"^## {re.escape(candidate.candidate_id)}\s*[:：]?\s*",
        f"## {new_id}: ",
        block_text,
        count=1,
        flags=re.MULTILINE,
    )

    # 4. Append to target file
    if target_text and not target_text.endswith("\n\n"):
        target_text += "\n\n"
    target_text += promoted_block + "\n"
    target_file.write_text(target_text)

    # 5. Remove candidate block from candidates.md
    new_candidates_lines = (
        lines[:candidate.source_block_start_line] +
        lines[candidate.source_block_end_line:]
    )
    candidates_md.write_text("\n".join(new_candidates_lines))

    return {"promoted": candidate.candidate_id, "target": str(target_file),
            "new_id": new_id, "block_size_lines": len(block_lines)}


def drop_audit_marker(kb_root: Path, marker_type: str, candidate_id: str,
                     run_id: str, **kwargs) -> Path:
    """Drop one of: .kb_promoted-* / .kb_promotion_blocked-* / .kb_promoted-merged-*."""
    pattern_dir = kb_root / "patterns" / "unverified"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    marker_path = pattern_dir / f".{marker_type}-{run_id}-{candidate_id}"
    marker_path.write_text(json.dumps({
        "run_id": run_id, "candidate_id": candidate_id,
        "ts": time.time(), **kwargs,
    }, indent=2))
    return marker_path


# ----------------------------------------------------------------------------
# Step 8: Audit report
# ----------------------------------------------------------------------------

def render_audit_report(report: PromotionBatchReport) -> str:
    """Markdown report for kb_auto_promote_audit.md."""
    lines = [
        f"# KB Auto-Promote Audit — {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(report.started_ts))}",
        "",
        f"Markers processed: **{report.markers_processed}**",
        f"- Promoted: {len(report.promoted)}",
        f"- Auto-merged into existing: {len(report.merged)}",
        f"- Blocked: {len(report.blocked)}",
        "",
    ]

    if report.promoted:
        lines.append("## Promoted entries")
        lines.append("")
        lines.append("| candidate | promoted_to | codex | transferability |")
        lines.append("|---|---|---|---|")
        for r in report.promoted:
            lines.append(f"| {r.candidate_id} | {r.target_id} | {r.codex_review} | "
                         f"{r.transferability_test} |")
        lines.append("")

    if report.merged:
        lines.append("## Auto-merged into existing entries")
        lines.append("")
        for r in report.merged:
            lines.append(f"- {r.candidate_id} → merged with {r.target_id}")
        lines.append("")

    if report.blocked:
        lines.append("## Blocked entries (need codex / human review)")
        lines.append("")
        for r in report.blocked:
            lines.append(f"- {r.candidate_id}: {r.reason}")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------------

def run_auto_promote(kb_root: Path, *, dry_run: bool = False,
                     skip_codex: bool = False) -> PromotionBatchReport:
    """Full Mode 5 pipeline. Returns batch report."""
    report = PromotionBatchReport(markers_processed=0)
    candidates_md = kb_root / "patterns" / "unverified" / "candidates.md"

    markers = scan_pending_markers(kb_root)
    report.markers_processed = len(markers)

    for marker in markers:
        try:
            marker_data = load_marker(marker)
            run_id = marker_data["run_id"]
            candidate_id = marker_data["candidate_id"]

            cand = load_candidate(candidates_md, candidate_id)
            if cand is None:
                report.blocked.append(PromotionResult(
                    candidate_id=candidate_id, outcome="blocked",
                    reason="candidate not found in candidates.md (may already be promoted)"
                ))
                if not dry_run:
                    drop_audit_marker(kb_root, "kb_promotion_blocked", candidate_id, run_id,
                                     reason="candidate not found")
                    marker.unlink(missing_ok=True)  # P0acv: prevent re-processing
                continue

            # Step 2: C36
            ok, c36_reason = check_c36_generalize(cand)
            if not ok:
                # Try auto-lift if op_classification.json available
                op = marker_data.get("op", "")
                op_cls_path = kb_root.parent.parent / "workspace" / op / "op_classification.json"
                if op_cls_path.exists():
                    op_cls = json.loads(op_cls_path.read_text())
                    cand = lift_to_op_class(cand, op_cls)
                    ok, c36_reason = check_c36_generalize(cand)
                if not ok:
                    report.blocked.append(PromotionResult(
                        candidate_id=candidate_id, outcome="blocked", reason=c36_reason
                    ))
                    if not dry_run:
                        drop_audit_marker(kb_root, "kb_promotion_blocked", candidate_id, run_id,
                                         reason=c36_reason)
                        marker.unlink(missing_ok=True)  # P0acv
                    continue

            # Step 3: C37 dedup
            dup = find_duplicate(cand, kb_root)
            if dup:
                report.merged.append(PromotionResult(
                    candidate_id=candidate_id, outcome="merged", target_id=dup[0]
                ))
                if not dry_run:
                    drop_audit_marker(kb_root, "kb_promoted-merged", candidate_id, run_id,
                                     merged_into=dup[0], overlap_ratio=dup[1])
                    marker.unlink(missing_ok=True)  # P0acv
                continue

            # Step 6: codex review + auto-revision loop (P0acq 2026-05-10).
            # The product requirement is a zero-interaction, self-evolving
            # harness whose entire promotion flow runs automatically.
            # P0acp shipped codex strict NEEDS_REVISION = BLOCK. P0acq closes
            # the loop: when codex says NEEDS_REVISION, codex ALSO emits a
            # revised candidate inline. We apply the revision and re-review
            # ONCE. If revised version APPROVE → promote. If still
            # NEEDS_REVISION → block (avoid infinite codex loops; safety cap).
            codex_verdict, codex_fb = ("unavailable", "skipped")
            revision_count = 0
            if not skip_codex:
                codex_verdict, codex_fb = codex_review(cand)
                # P0acu (2026-05-10): when codex is "unavailable" (quota / CLI
                # missing / network), spawn an INDEPENDENT Claude sub-agent
                # as fallback reviewer. User directive: '启动一个独立的agent去review'.
                # Echo-chamber risk: same-model-family review. Mitigation:
                # verdict tagged "claude_fallback_*" so user can audit which
                # promotions did NOT go through codex's true second opinion.
                if codex_verdict == "unavailable":
                    codex_verdict, codex_fb = claude_fallback_review(cand)
                    # If even the fallback couldn't run, hard-block (P0act).
                    if codex_verdict == "unavailable":
                        report.blocked.append(PromotionResult(
                            candidate_id=candidate_id, outcome="blocked",
                            reason=f"both codex AND claude fallback unavailable "
                                   f"(C40 fail-closed): {codex_fb[:200]}",
                            codex_review=codex_verdict,
                        ))
                        if not dry_run:
                            drop_audit_marker(
                                kb_root, "kb_promotion_blocked", candidate_id, run_id,
                                reason="codex+claude_fallback both unavailable",
                                codex_feedback=codex_fb,
                            )
                            marker.unlink(missing_ok=True)  # P0acv
                        continue
                    # Normalize claude_fallback_* verdicts to the verdict
                    # vocabulary the downstream branches expect. Provenance
                    # is preserved via the codex_review field on
                    # PromotionResult (set later, when entry is logged).
                    if codex_verdict == "claude_fallback_approve":
                        codex_verdict = "approve"
                    elif codex_verdict == "claude_fallback_needs_revision":
                        codex_verdict = "needs_revision"
                    elif codex_verdict == "claude_fallback_reject":
                        codex_verdict = "reject"
                    # Stamp provenance into the feedback so audit-trail readers
                    # can grep "[reviewer=claude_fallback]" to find fallback
                    # decisions; codex decisions have no such stamp.
                    codex_fb = f"[reviewer=claude_fallback] {codex_fb}"
                # REJECT — straight block, no revision attempted
                if codex_verdict == "reject":
                    report.blocked.append(PromotionResult(
                        candidate_id=candidate_id, outcome="blocked",
                        reason=f"codex REJECT: {codex_fb[:200]}",
                        codex_review=codex_verdict,
                    ))
                    if not dry_run:
                        drop_audit_marker(
                            kb_root, "kb_promotion_blocked", candidate_id, run_id,
                            reason="codex REJECT", codex_feedback=codex_fb,
                        )
                        marker.unlink(missing_ok=True)  # P0acv
                    continue
                # NEEDS_REVISION — try one auto-revision pass
                if codex_verdict == "needs_revision":
                    revision = parse_codex_revision(codex_fb)
                    if revision is not None:
                        revised_title, revised_body = revision
                        cand = apply_revision(cand, revised_title, revised_body)
                        revision_count = 1
                        # P0acw round-3 (2026-05-11): ALWAYS persist the revision
                        # to disk regardless of round-2 verdict. Round-1 produced
                        # the revision — it's already a quality-improvement on the
                        # original body. If round-2 codex APPROVES, we promote
                        # using the persisted revision; if round-2 is timeout /
                        # unavailable / NEEDS_REVISION, the persisted revision
                        # stays as a strict improvement over the original prose
                        # and the NEXT promotion attempt (next finalize) re-tries
                        # codex on the cleaner body. Without this, every timeout
                        # discards substantive revision work — value lost.
                        if not dry_run:
                            _persist_revision(candidates_md, cand)
                        # Re-review the revised candidate ONCE
                        codex_verdict_2, codex_fb_2 = codex_review(cand)
                        if codex_verdict_2 == "approve":
                            codex_verdict, codex_fb = codex_verdict_2, codex_fb_2
                            # Persisted above; fall through to step 7 (promote)
                        else:
                            # Revision still not good → block with both rounds
                            report.blocked.append(PromotionResult(
                                candidate_id=candidate_id, outcome="blocked",
                                reason=f"codex NEEDS_REVISION even after auto-revise: "
                                       f"{codex_fb_2[:200]}",
                                codex_review=codex_verdict_2,
                            ))
                            if not dry_run:
                                drop_audit_marker(
                                    kb_root, "kb_promotion_blocked", candidate_id, run_id,
                                    reason="codex NEEDS_REVISION after auto-revise",
                                    codex_feedback_round1=codex_fb,
                                    codex_feedback_round2=codex_fb_2,
                                    revision_attempted=True,
                                )
                                marker.unlink(missing_ok=True)  # P0acv
                            continue
                    else:
                        # codex didn't include REVISED_BODY — block as before
                        report.blocked.append(PromotionResult(
                            candidate_id=candidate_id, outcome="blocked",
                            reason=f"codex NEEDS_REVISION (no auto-revision): {codex_fb[:200]}",
                            codex_review=codex_verdict,
                        ))
                        if not dry_run:
                            drop_audit_marker(
                                kb_root, "kb_promotion_blocked", candidate_id, run_id,
                                reason="codex NEEDS_REVISION (no revised_body in feedback)",
                                codex_feedback=codex_fb,
                            )
                            marker.unlink(missing_ok=True)  # P0acv
                        continue

            # Step 7: promote
            target_file = kb_root / "patterns" / "PATTERN_INDEX.md"
            new_id = allocate_next_id(target_file, "P-P")

            audit = promote_candidate(cand, kb_root, candidates_md,
                                      target_file, new_id, dry_run=dry_run)

            report.promoted.append(PromotionResult(
                candidate_id=candidate_id, outcome="promoted", target_id=new_id,
                codex_review=codex_verdict, transferability_test="pending"
            ))
            if not dry_run:
                drop_audit_marker(kb_root, "kb_promoted", candidate_id, run_id,
                                 promoted_to=new_id, codex_review=codex_verdict, **audit)
                # Remove the pending marker
                marker.unlink(missing_ok=True)
        except Exception as e:
            report.blocked.append(PromotionResult(
                candidate_id=marker.name, outcome="blocked",
                reason=f"runtime exception: {e}"
            ))

    report.finished_ts = time.time()
    return report


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # 2026-07-05: KB relocated to <plugin_root>/kb/. kb_auto_promote.py lives at
    # <plugin_root>/engine/src/scripts/orchestrator/, so parents[4] == plugin_root.
    p.add_argument("--kb-root", type=Path,
                   default=Path(__file__).resolve().parents[4] / "kb")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-codex", action="store_true",
                   help="Skip codex review (e.g. for tests)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rpt = run_auto_promote(args.kb_root, dry_run=args.dry_run,
                           skip_codex=args.skip_codex)

    if args.json:
        print(json.dumps({
            "markers_processed": rpt.markers_processed,
            "promoted": [r.__dict__ for r in rpt.promoted],
            "merged": [r.__dict__ for r in rpt.merged],
            "blocked": [r.__dict__ for r in rpt.blocked],
        }, indent=2))
    else:
        print(render_audit_report(rpt))

    sys.exit(0 if not rpt.blocked else 2)

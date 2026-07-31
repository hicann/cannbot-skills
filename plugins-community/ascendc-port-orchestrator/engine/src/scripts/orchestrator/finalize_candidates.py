# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize pipeline — KB-candidate verified-on tracking + archive-op-name
resolution (DEBT-201, 2026-07-06).

Extracted verbatim from finalize_pipeline.py: the candidate-reference scan,
`verified_on:` prose patcher, and consumed-candidate bookkeeping, plus the
archive op-name resolver. `_verification_hash` (used by
update_verified_on_for_consumed_candidates) now lives in finalize_shared —
imported from there, NOT from finalize_pipeline, so this module does not add to
the finalize_pipeline import cycle. None monkeypatched. finalize_pipeline
re-imports these names (bottom shim) so `finalize_pipeline.update_verified_on_
for_consumed_candidates` / `_resolve_archive_op_name` and call-sites stay valid.
Behaviour is byte-identical.
"""
from __future__ import annotations
import logging

import json
import re
from pathlib import Path

from finalize_shared import _verification_hash

# cannbot KB-relocation adaptation: candidates.md lives under <plugin_root>/kb/,
# resolved via kb_paths.kb_root(). Imported here (NOT from finalize_pipeline) to
# keep this module off the finalize_pipeline import cycle. Tests redirect the KB
# root by monkeypatching finalize_candidates._kb_root.
try:
    from kb_paths import kb_root as _kb_root
except ImportError:  # pragma: no cover — fallback if orchestrator/ not on sys.path
    def _kb_root():  # type: ignore
        return Path(__file__).resolve().parent.parent.parent.parent.parent / "kb"


_CAND_TOKEN_RE = re.compile(r"\bCAND-[A-Z0-9][A-Z0-9-]*\b")


def _resolve_archive_op_name(op: str, archive_root: Path) -> str:
    """Map workspace op-name to archive dir name. Workspaces use lowercase
    or mixed (`5_Cumsum`, `10_layernorm`); archives use `<N>_<CamelCase>`
    (`5_Cumsum`, `10_LayerNorm`).

    P0dd v3 (2026-05-05): the previous prefix-only fuzzy match was a
    data-corruption bug — workspace `10_layernorm` matched archive
    `10_SwigluQuant` because both start with `10_`, causing files to be
    written to the wrong archive. Match strategies (in order):

      1. Exact match: workspace name == archive dir name.
      2. Case-insensitive exact match: lowercase normalization equals.
      3. Number-prefix + suffix match: split on `_`, both must have the
         same number AND the post-number part must match
         case-insensitively after stripping non-alphanumerics.
      4. Otherwise: return the workspace name (creates new archive dir).

    Strategy 3 must NEVER reduce to "any sibling with same number prefix" —
    that's the bug class. The post-number part match is required.
    """
    if not archive_root.exists():
        return op

    op_lower = op.lower()

    # 1. exact. Do not use ``(archive_root / op).exists()`` here: on a
    # case-insensitive filesystem that also succeeds for a differently-cased
    # archive and loses the canonical directory spelling.
    children = [child for child in archive_root.iterdir() if child.is_dir()]
    for child in children:
        if child.name == op:
            return child.name

    # 2. case-insensitive exact
    for child in children:
        if child.name.lower() == op_lower:
            return child.name

    # 3. number-prefix + suffix match
    if "_" in op:
        prefix, _, op_rest = op.partition("_")
        if prefix.isdigit():
            op_rest_norm = "".join(c for c in op_rest.lower() if c.isalnum())
            if not op_rest_norm:
                return op  # weird name like `10_` — give up
            for child in children:
                if "_" not in child.name:
                    continue
                arch_prefix, _, arch_rest = child.name.partition("_")
                if arch_prefix != prefix:
                    continue
                arch_rest_norm = "".join(c for c in arch_rest.lower() if c.isalnum())
                # Require the rest parts to match (or one is contained in the
                # other) — `layernorm` ≈ `LayerNorm`, but `layernorm` ≠
                # `SwigluQuant`. Pure prefix-match is forbidden.
                if op_rest_norm == arch_rest_norm:
                    return child.name
                # Allow abbreviation: workspace `12_kvrms` matches archive
                # `12_KvRmsnormRopeCache` (op_rest_norm prefix of arch_rest_norm)
                # OR workspace `14_adaptive_instance_norm_bwd` matches archive
                # `14_AdaptiveInstanceNormalization2DBackward` (longer-prefix).
                # Require the SHORTER one to be a prefix of the LONGER one,
                # AND require shared prefix length ≥ 4 (to reject `10_l`
                # matching `10_layernorm` purely by accident).
                shorter, longer = sorted([op_rest_norm, arch_rest_norm], key=len)
                if len(shorter) >= 4 and longer.startswith(shorter):
                    return child.name

    # 4. give up — return workspace name (new archive dir)
    return op


def _scan_workspace_for_candidate_refs(workspace: Path) -> set[str]:
    """Scan workspace artifacts that kw / probe / optimizer write for
    CAND-X token references. Returns the unique set of candidate IDs cited.

    We scan exactly the documents where kw/ko/pp would cite a candidate:
    analysis.md (kw architectural rationale), optimization_log.md (ko strategy
    citations), probe_report.md (pp investigation cross-refs), and
    knowledge_update.md (the kw / ko emitted new-findings file). We do NOT
    scan kernel/*.h or pybind — code references would be circumstantial.
    """
    cited: set[str] = set()
    files_to_scan = [
        "analysis.md",
        "optimization_log.md",
        "probe_report.md",
        "knowledge_update.md",
        "fused_analysis.md",
    ]
    for fname in files_to_scan:
        f = workspace / fname
        if not f.exists():
            continue
        skip_current_item = False
        try:
            text = f.read_text()
            cited.update(_CAND_TOKEN_RE.findall(text))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    return cited


_PROSE_PATCH_PATTERNS = [
    # "- a5_ops has no currently-shipped X kernel; the pattern is unverified on this codebase"
    (
        re.compile(
            r"^(\s*-?\s*)a5_ops\s+has\s+no\s+"
            r"(currently[-\s]shipped|shipped)\s+.*?"
            r"(unverified.*?codebase|unverified.*$)",
            re.IGNORECASE,
        ),
        r"\1a5_ops now has shipped evidence ({token}); see verified_on metadata above",
    ),
    # "...unverified on a5_ops..." or "pattern is unverified on a5_ops"
    (
        re.compile(
            r"^(\s*-?\s*.*?)\bunverified\s+on\s+a5_ops\b(.*)$",
            re.IGNORECASE,
        ),
        r"\1previously unverified on a5_ops; now verified ({token})\2",
    ),
    # "Source-structure verification only — promotion to P-P requires an a5_ops kernel"
    (
        re.compile(
            r"^(\s*-?\s*)Source-structure\s+verification\s+only\s*"
            r"[—-]\s*promotion\s+to\s+P-P\s+requires\s+an\s+"
            r"a5_ops\s+kernel.*$",
            re.IGNORECASE,
        ),
        r"\1Promotion gate satisfied: a5_ops kernel evidence recorded ({token})",
    ),
    # Matches a "Promote when" sentence that requires an a5_ops kernel to ship.
    (
        re.compile(
            r"(\*\*Promote when\*\*\s*:\s*)an\s+a5_ops\s+(.+?)\s+ships\s+(.+)$",
            re.IGNORECASE,
        ),
        r"\1a5_ops evidence recorded ({token}); remaining criteria — \3",
    ),
    # "promotion to canonical KB requires ... AND a5_ops shipped" variants
    (
        re.compile(
            r"(promotion to canonical (?:KB )?requires.*?and\s+)an?\s+a5_ops\s+kernel(.*?)(?=[.;]|$)",
            re.IGNORECASE,
        ),
        r"\1a5_ops evidence recorded ({token})\2",
    ),
]


def _patch_evidence_prose(block: str, evidence_token: str) -> str:
    """Rewrite stale evidence-claim prose in a candidate block to match the
    fact that verified_on now includes a5_ops shipped evidence.

    Each pattern in _PROSE_PATCH_PATTERNS matches a line and rewrites only
    that line (multi-line surgery would be too brittle; we accept that some
    extended prose paragraphs may still have stale phrasing — later review
    catches those rather than this hook).
    """
    out_lines: list[str] = []
    for line in block.split("\n"):
        replaced = line
        for pat, repl_template in _PROSE_PATCH_PATTERNS:
            repl = repl_template.replace("{token}", evidence_token)
            new_line, n = pat.subn(repl, replaced, count=1)
            if n > 0:
                replaced = new_line
                break  # one patch per line max
        out_lines.append(replaced)
    return "\n".join(out_lines)


def _append_verified_on(candidates_md: Path, candidate_id: str,
                        evidence_token: str) -> bool:
    """Append `verified_on: a5_ops:<op>:<case>` to the candidate's metadata
    block in candidates.md and remove `a5_ops` from any `unverified_on:` line.

    Returns True if the file was modified, False if the candidate wasn't
    found or the evidence token was already present (idempotent).
    """
    if not candidates_md.exists():
        return False
    text = candidates_md.read_text()
    # Find candidate block (## CAND-X: ... up to next ## or EOF)
    header_marker = f"## {candidate_id}:"
    start = text.find(header_marker)
    if start == -1:
        return False
    next_header = text.find("\n## ", start + 1)
    end = next_header if next_header != -1 else len(text)
    block = text[start:end]

    # P0acz (2026-05-11): respect existing refuted_on:<same_op> markers.
    # If the candidate has a `refuted_on: a5_ops:<op>:...` line for the
    # SAME op we're about to add verified_on for, the candidate has been
    # manually marked as a negative-evidence case (e.g. by codex strict
    # review catching a hard-do-not-apply violation, then human/orchestrator
    # converting auto-verified_on to explicit refuted_on). Auto-adding
    # verified_on AGAIN would re-create the contradiction. Skip.
    op_marker = evidence_token.split(":")[1] if ":" in evidence_token else ""
    refute_pattern = re.compile(
        rf"^`refuted_on:\s*a5_ops:{re.escape(op_marker)}\b",
        re.MULTILINE,
    )
    if op_marker and refute_pattern.search(block):
        # Don't add verified_on; this op was explicitly marked as negative.
        # Still patch other stale-evidence prose if any.
        patched_block = _patch_evidence_prose(block, evidence_token)
        if patched_block != block:
            new_text = text[:start] + patched_block + text[end:]
            candidates_md.write_text(new_text)
            return True
        return False

    # Idempotency: skip metadata APPEND if this exact evidence token is
    # already recorded. The prose patch still runs (it's idempotent
    # by construction — already-patched lines no longer match the
    # stale-evidence regexes).
    token_already_present = evidence_token in block
    if token_already_present:
        # Still patch prose if any stale-evidence lines remain.
        patched_block = _patch_evidence_prose(block, evidence_token)
        if patched_block != block:
            new_text = text[:start] + patched_block + text[end:]
            candidates_md.write_text(new_text)
            return True  # we modified the file (prose patch even if no metadata change)
        return False

    # Walk metadata lines (backtick-delimited at top of block).
    # P0acz (2026-05-11): metadata block is HEADER + CONSECUTIVE backtick-only
    # lines, stopped by first blank line or non-backtick-only line. Inline
    # `code-span` mid-body must NOT count as metadata (or `verified_on:` gets
    # inserted in the middle of body prose — observed on CAND-FA2 case_ad1de4ec).
    lines = block.split("\n")
    out_lines: list[str] = []
    appended = False
    last_metadata_idx = -1
    in_metadata_block = True  # starts True; flips False on first non-metadata-shape line
    for line in lines:
        # Modify unverified_on: remove a5_ops if listed
        if line.startswith("`unverified_on:"):
            content = line[len("`unverified_on:"):].rstrip("`").strip()
            parts = [p.strip() for p in re.split(r"[;,]", content) if p.strip()]
            parts_kept = [p for p in parts if "a5_ops" not in p.lower()]
            if not parts_kept:
                continue
            else:
                line = f"`unverified_on: {'; '.join(parts_kept)}`"
        out_lines.append(line)
        # Track last metadata-line index ONLY while still in top metadata block.
        # Header line counts as in-block; backtick-only lines count; blank /
        # non-backtick-only line ends the block.
        if in_metadata_block:
            stripped = line.rstrip()
            is_header = stripped.startswith("## ")
            is_metadata_line = (
                stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 1
            )
            if is_header:
                pass  # header is in the block but doesn't count as last_metadata_idx target
            elif is_metadata_line:
                last_metadata_idx = len(out_lines) - 1
            else:
                # Blank line, prose, code fence — metadata block has ended
                in_metadata_block = False

    # Append a new verified_on line right after the last existing metadata line
    new_verified_line = f"`verified_on: {evidence_token}`"
    if last_metadata_idx >= 0:
        out_lines.insert(last_metadata_idx + 1, new_verified_line)
        appended = True
    else:
        # No metadata lines? Insert after the header
        out_lines.insert(1, new_verified_line)
        appended = True

    new_block = "\n".join(out_lines)
    # P0acw round-2 (2026-05-11): also patch prose body for evidence-claim
    # contradictions. Candidate bodies may still claim that a5_ops has no
    # shipped kernel, or make promotion conditional on one being shipped;
    # either claim goes stale once a5_ops ships a kernel using the pattern.
    # Codex correctly catches the contradiction between metadata (verified_on
    # now includes a5_ops) and prose (still says unverified), and blocks
    # promotion. Patch the prose to match the new metadata state.
    new_block = _patch_evidence_prose(new_block, evidence_token)
    new_text = text[:start] + new_block + text[end:]
    candidates_md.write_text(new_text)
    return appended


def update_verified_on_for_consumed_candidates(
    workspace: Path, op: str, project_root: Path
) -> dict[str, bool]:
    """Scan workspace artifacts for CAND-X citations; for each cited
    candidate that exists in patterns/unverified/candidates.md, append a
    `verified_on: a5_ops:{op}:case_<hash>` marker.

    Gate: ONLY fires when verification.json precision.status == "PASS"
    (or PASS_WITHIN_TOLERANCE). Citations from FAILED kernels are NOT
    evidence — they'd promote a bad pattern.

    Returns: dict {candidate_id: was_appended}
    """
    ver = workspace / "verification.json"
    if not ver.exists():
        return {}
    try:
        v = json.loads(ver.read_text())
    except Exception:
        return {}
    prec = v.get("precision", {}) or {}
    prec_status = prec.get("status", "")
    # Accept clean PASS, AND PARTIAL when there is positive in-scope evidence
    # (tier1_pass >= 1 or pass_a.tier1_pass >= 1). PARTIAL with NO in-scope
    # passing cases is just failure dressed up — not evidence.
    if prec_status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        pass  # full PASS — evidence accepted
    elif prec_status in ("PARTIAL", "PARTIAL_PERSIST"):
        # Look for tier1_pass at top-level or under pass_a (canonical schema)
        t1 = prec.get("tier1_pass", 0)
        if not t1:
            pa = prec.get("pass_a", {}) or {}
            t1 = pa.get("tier1_pass", 0)
        if not t1 or int(t1) < 1:
            return {}  # PARTIAL with no positive evidence — skip
        # A positive count is accepted because at least one in-scope case
        # demonstrated that the kernel pattern works.
    else:
        return {}  # FAIL / EVAL_ERR / etc — no evidence

    cited = _scan_workspace_for_candidate_refs(workspace)
    if not cited:
        return {}

    candidates_md = (
        _kb_root()
        / "patterns" / "unverified" / "candidates.md"
    )
    if not candidates_md.exists():
        return {}

    # Use precision hash as a stable case identifier so re-runs don't
    # produce duplicate verified_on lines for the same evidence.
    h = _verification_hash(workspace) or "unknown"
    evidence_token = f"a5_ops:{op}:case_{h[:8]}"

    results: dict[str, bool] = {}
    for cand_id in sorted(cited):
        try:
            modified = _append_verified_on(candidates_md, cand_id, evidence_token)
            results[cand_id] = modified
        except Exception:
            results[cand_id] = False
    return results

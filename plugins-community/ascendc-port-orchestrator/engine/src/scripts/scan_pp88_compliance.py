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

"""scan_pp88_compliance.py — P0abi (2026-05-08): enforce P-P88 sigmoid-form
remediation when kernel uses primitives with bimodal precision floors
(PB-24 Tanh, PB-25 Sigmoid).

Background: 1_GELU regressed 50/50 (May-4 archive PASS_WITHIN_TOLERANCE) →
44/50 (May-8 cold-start PARTIAL) because cold-start kw-1 cited P-P88 as
diagnosis ("Tanh<fp32> bimodal floor") but didn't apply the prescribed
sigmoid-form remediation. Different tile size (4096→6144) changed the
SIMD path through Tanh's internal poly evaluation, exposing different
last-bit rounding on small-x inputs. KB had P-P88 but it was advisory.

Fix: scan kernel/*.h + kernel/*.cpp for risky primitive calls. If found
+ benchmark has small-value cases + op-class is transcendental →
require knowledge_update.md to contain a structured `p_p88:` YAML block
with status ∈ {applied, exempt, not_applicable} + concrete evidence.
Without that, the workspace is non-compliant and finalize_pipeline
rejects the handoff.

Codex + DS reviewed (2026-05-08); both approved. DS suggested adding a
positive compliance marker — sigmoid-form rewrite pattern (`Exp...Div`
or `1/(1+exp`) — so ops that already correctly applied P-P88 don't
need to re-cite via YAML. This scanner implements that:

  - "compliant by code": kernel has sigmoid-form rewrite + no Tanh()/Sigmoid()
    primitive calls → PASS without YAML required
  - "compliant by declaration": kernel has Tanh()/Sigmoid() call AND
    knowledge_update.md has valid p_p88: block → PASS
  - "non-compliant": Tanh()/Sigmoid() call + no YAML or invalid YAML → FAIL

Exit codes (CLI):
  0  PASS (compliant or no risky primitive)
  1  FAIL (non-compliant — caller propagates as REJECT)
  2  usage error
"""
from __future__ import annotations
import logging

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------

# Risky primitives — gate-mandatory subset. Currently scoped to PB-24 only:
# `Tanh<fp32>` has a BIMODAL precision floor (clean 2 ULP for |x|≥0.1,
# catastrophic small-x failure: 1599 ULP at x≈1.7e-4) — the failure mode
# that produced the 1_GELU 50/50 → 44/50 cold-start regression. PB-25
# `Sigmoid<fp32>` has a clean UNIFORM 2-ULP floor (no bimodal cliff, no
# schedule-sensitivity to tile-size choices), so doesn't reproduce the
# regression mode. P-P88 still recommends sigmoid-form rewrite for
# vendor-source-alignment, but it's not a structural mandate.
#
# Each entry: (pattern, primitive name, KB ref).
# Pattern matches the call form `<Primitive>(...)` allowing optional namespace
# prefix (`AscendC::`) and template arguments (`<float>`).
#
# Provenance (P0abi narrowing 2026-05-08, post-DS portfolio scan): adding
# Sigmoid to this list flagged 11_DequantSwigluQuant as FAIL even though
# Sigmoid uniform 2-ULP doesn't reproduce the regression mode. DS-flagged.
RISKY_PRIMITIVES: list[tuple[str, str, str]] = [
    (r'\b(?:AscendC::)?Tanh\s*(?:<[^>]*>)?\s*\(', "Tanh", "PB-24"),
]

# P-P88 compliance markers — kernel patterns that indicate sigmoid-form
# rewrite was applied (DS suggestion 2026-05-08). Presence of these +
# absence of risky primitive = compliant by code, no YAML required.
#
# Match conservatively — require multiple primitives in close proximity
# to avoid false positives on incidental Exp/Div uses.
COMPLIANCE_MARKERS: list[tuple[str, str]] = [
    # Cephes-form sigmoid: 1/(1+exp(-y)) literal pattern. Allow C/C++ float
    # literal suffixes (`1.0f`, `1.0`, `1`) on either constant.
    (r'1(?:\.\d*)?f?\s*/\s*\(\s*1(?:\.\d*)?f?\s*\+\s*(?:AscendC::)?Exp',
     "explicit 1/(1+exp(...)) sigmoid form"),
    # Tanh-via-sigmoid: 1 - 2/(exp(2y)+1) — standard P-P88 Tanh rewrite
    (r'1(?:\.\d*)?f?\s*-\s*2(?:\.\d*)?f?\s*/\s*\(\s*(?:AscendC::)?Exp',
     "1 - 2/(exp(2y)+1) tanh-via-sigmoid"),
    # Hand-rolled DAG marker: AddDeqRelu | Exp + Reciprocal close together
    # (looser; only applied when the explicit forms above don't match).
]


# ---------------------------------------------------------------------------
# C++ comment + string stripping (reuse delegation scanner pattern)
# ---------------------------------------------------------------------------
def _strip_comments_and_strings(text: str) -> str:
    """Remove C++ // line comments, /* block comments */, and string literals
    so token scanning doesn't false-positive on prose."""
    # Block comments first
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r'//[^\n]*', '', text)
    # String literals (handle escaped quotes)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
    text = re.sub(r"'(?:[^'\\]|\\.)*'", "''", text)
    return text


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class PrimitiveHit:
    file: str
    line: int
    primitive: str
    kb_ref: str
    text: str


@dataclass
class ComplianceMarker:
    file: str
    line: int
    description: str
    text: str


@dataclass
class P0abiReport:
    workspace: str
    risky_hits: list[PrimitiveHit] = field(default_factory=list)
    compliance_markers: list[ComplianceMarker] = field(default_factory=list)
    p_p88_yaml_block: Optional[dict] = None
    p_p88_yaml_error: Optional[str] = None
    knowledge_update_present: bool = False
    is_transcendental: bool = False
    has_smallvalue_cases: bool = False  # heuristic from benchmark JSON
    verdict: str = "PASS"  # PASS | FAIL | NOT_APPLICABLE
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "is_transcendental": self.is_transcendental,
            "has_smallvalue_cases": self.has_smallvalue_cases,
            "knowledge_update_present": self.knowledge_update_present,
            "p_p88_yaml_block": self.p_p88_yaml_block,
            "p_p88_yaml_error": self.p_p88_yaml_error,
            "risky_hits": [vars(h) for h in self.risky_hits],
            "compliance_markers": [vars(m) for m in self.compliance_markers],
        }


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
def scan_kernel_files(workspace: Path) -> tuple[list[PrimitiveHit], list[ComplianceMarker]]:
    """Walk kernel/* + scan for risky primitive calls + compliance markers."""
    risky: list[PrimitiveHit] = []
    markers: list[ComplianceMarker] = []
    kernel_dir = workspace / "kernel"
    if not kernel_dir.is_dir():
        return risky, markers

    for f in sorted(kernel_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix not in {".h", ".hpp", ".cpp", ".cc", ".cxx"}:
            continue
        skip_current_item = False
        try:
            raw = f.read_text(errors="ignore")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        cleaned = _strip_comments_and_strings(raw)
        # Per-line scan for line numbers; raw lines for diagnostic text
        cleaned_lines = cleaned.splitlines()
        raw_lines = raw.splitlines()
        for lineno, (cline, rline) in enumerate(
            zip(cleaned_lines, raw_lines), start=1,
        ):
            for pat, name, ref in RISKY_PRIMITIVES:
                if re.search(pat, cline):
                    risky.append(PrimitiveHit(
                        file=str(f.relative_to(workspace)),
                        line=lineno,
                        primitive=name,
                        kb_ref=ref,
                        text=rline.strip()[:160],
                    ))
                    break
            for pat, desc in COMPLIANCE_MARKERS:
                if re.search(pat, cline):
                    markers.append(ComplianceMarker(
                        file=str(f.relative_to(workspace)),
                        line=lineno,
                        description=desc,
                        text=rline.strip()[:160],
                    ))
                    break
    return risky, markers


def parse_p_p88_yaml_block(knowledge_update_path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Extract structured `p_p88:` block from knowledge_update.md.

    The block is expected to look like:

      ```yaml
      p_p88:
        status: applied | exempt | not_applicable
        primitives_detected: [Tanh, Sigmoid]
        evidence:
          files: [kernel/gelu_kernel.h:42-67]
          rationale: "<concrete prose>"
        diff_refs: [<line range or commit>]
      ```

    Returns (parsed_dict, error_string). On success error is None.
    """
    if not knowledge_update_path.exists():
        return None, "knowledge_update.md missing"
    text = knowledge_update_path.read_text(errors="ignore")
    # Look for fenced yaml block containing `p_p88:`
    match = re.search(
        r"```ya?ml\s*\n(.*?p_p88\s*:.*?)\n```",
        text, flags=re.DOTALL,
    )
    if not match:
        # Also accept inline (no fence) `p_p88:` block — extract from key to
        # next blank line after a contiguous YAML-shaped region. Simpler: just
        # search for `p_p88:` and take the next ~30 lines until a blank line.
        idx = text.find("p_p88:")
        if idx == -1:
            return None, "no p_p88: block in knowledge_update.md"
        block_text = text[idx:].split("\n\n")[0]
    else:
        block_text = match.group(1)

    # Lazy YAML parser (avoid pyyaml dep): tokenize key:value + lists.
    # Required structure shallow enough to handle without external dep.
    try:
        parsed = _parse_simple_yaml(block_text)
    except Exception as e:
        return None, f"p_p88 block parse error: {e}"
    if "p_p88" not in parsed:
        return None, "p_p88 key not found in parsed block"
    return parsed["p_p88"], None


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for p_p88: structure. Only handles:
      key: value
      key: [a, b, c]
      key:
        sub-key: value
    """
    out: dict = {}
    stack: list[tuple[int, dict]] = [(-1, out)]
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        # Pop stack to current indent
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack.append((-1, out))
        parent = stack[-1][1]
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # nested dict
                child: dict = {}
                parent[key] = child
                stack.append((indent, child))
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                items = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
                parent[key] = items
            else:
                parent[key] = val.strip('"').strip("'")
    return out


def has_smallvalue_cases(workspace: Path) -> bool:
    """Heuristic — does the benchmark JSON have at least one case with
    likely small-magnitude inputs? For now, we treat all op runs as
    "potentially small-value" unless explicit evidence otherwise. The
    finalize gate path will use this to decide whether to require P-P88.

    Conservative default: True (require P-P88 when risky primitive present).
    Future: parse benchmark JSON cases for shapes/dtypes likely to produce
    small-magnitude post-input outputs (e.g., transcendentals near zero).
    """
    bench_json = next(workspace.glob("*.json"), None)
    if bench_json is None or bench_json.name == "verification.json":
        return True  # conservative: assume yes if uncertain
    return True  # always conservative for v1; refine if false-positives seen


def is_transcendental_op(workspace: Path) -> bool:
    """Detect whether the op is transcendental (uses Tanh/Sigmoid/Erf/Exp
    etc. in math) by scanning model.py forward() body for known
    transcendental keywords. Conservative: returns True if any match.
    """
    model_py = workspace / "model.py"
    if not model_py.exists():
        return False
    text = model_py.read_text(errors="ignore").lower()
    # Strip comments + docstrings to reduce false positives
    text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
    text_no_strings = re.sub(r"#[^\n]*", "", text_no_strings)
    keywords = ["gelu", "silu", "tanh(", "sigmoid(", "softmax", "softplus",
                "swish", "logsumexp", "log_softmax", "erf("]
    return any(kw in text_no_strings for kw in keywords)


def evaluate_compliance(report: P0abiReport) -> P0abiReport:
    """Apply the gate rules — fill in verdict + rationale."""
    # No risky primitive → not applicable
    if not report.risky_hits:
        report.verdict = "NOT_APPLICABLE"
        report.rationale = (
            "No AscendC::Tanh / Sigmoid (or PB-24/25-listed primitive) calls "
            "detected in kernel/*.h | *.cpp; P-P88 enforcement does not apply."
        )
        return report

    # Risky primitive present, but op is NOT transcendental — relax (the rule
    # is targeted at transcendental ops where small-value inputs amplify
    # primitive precision floor).
    if not report.is_transcendental:
        report.verdict = "NOT_APPLICABLE"
        report.rationale = (
            f"Risky primitive(s) detected ({len(report.risky_hits)}) but op-class "
            f"is not transcendental (model.py.forward has no gelu/silu/sigmoid/"
            f"tanh/softmax/erf calls). P-P88 enforcement scoped to transcendental "
            f"ops only."
        )
        return report

    # Risky primitive + transcendental + small-value cases → require evidence

    # First check: kernel has explicit sigmoid-form compliance markers
    # but ALSO has risky-primitive calls. That's contradictory — kw applied
    # the rewrite in one path but kept the primitive in another. Reject.
    if report.compliance_markers and report.risky_hits:
        report.verdict = "FAIL"
        report.rationale = (
            f"Mixed kernel: {len(report.compliance_markers)} sigmoid-form "
            f"compliance marker(s) AND {len(report.risky_hits)} risky-primitive "
            f"call(s) coexist. P-P88 must be applied uniformly across the kernel "
            f"(or exempted explicitly via knowledge_update.md `p_p88:` block). "
            f"Rewrite the remaining Tanh/Sigmoid sites OR document why those "
            f"sites are exempt."
        )
        return report

    # Risky primitive + no compliance markers → require YAML evidence
    if report.p_p88_yaml_block is None:
        report.verdict = "FAIL"
        report.rationale = (
            f"{len(report.risky_hits)} risky-primitive call(s) detected "
            f"({', '.join(set(h.primitive for h in report.risky_hits))}) "
            f"+ op is transcendental. P-P88 sigmoid-form remediation NOT "
            f"applied (no compliance marker in kernel) AND no `p_p88:` "
            f"YAML block in knowledge_update.md. "
            f"Required: either rewrite to sigmoid-form (Exp + Reciprocal + Add) "
            f"per P-P88, OR add structured exemption block. "
            f"YAML parse error: {report.p_p88_yaml_error or 'block missing'}."
        )
        return report

    # Validate YAML block fields
    block = report.p_p88_yaml_block
    status = (block.get("status") or "").lower()
    if status not in {"applied", "exempt", "not_applicable"}:
        report.verdict = "FAIL"
        report.rationale = (
            f"`p_p88:` block has invalid status={status!r}. Required: "
            f"applied | exempt | not_applicable."
        )
        return report
    evidence = block.get("evidence") or {}
    if not isinstance(evidence, dict) or not evidence.get("rationale"):
        report.verdict = "FAIL"
        report.rationale = (
            f"`p_p88:` block missing required field `evidence.rationale`. "
            f"Add concrete prose explaining the {status} decision."
        )
        return report
    if status == "applied" and not block.get("diff_refs"):
        report.verdict = "FAIL"
        report.rationale = (
            "`p_p88: status=applied` requires non-empty `diff_refs` "
            "(e.g., `[kernel/gelu_kernel.h:42-67]`) showing where the "
            "sigmoid-form rewrite lives."
        )
        return report
    if status == "exempt":
        # Stricter — exempt path requires isolated_primitive_measurements
        meas = evidence.get("isolated_primitive_measurements") or evidence.get("measurements")
        if not meas:
            report.verdict = "FAIL"
            report.rationale = (
                "`p_p88: status=exempt` requires `evidence.isolated_primitive_measurements` "
                "(or `evidence.measurements`) — concrete data showing why this "
                "specific op is exempt despite primitive presence."
            )
            return report

    report.verdict = "PASS"
    report.rationale = (
        f"P-P88 compliance validated: status={status}, "
        f"{len(report.risky_hits)} primitive call(s), "
        f"YAML evidence present."
    )
    return report


def scan_workspace(workspace: Path) -> P0abiReport:
    """Top-level: scan a workspace dir → P0abiReport."""
    report = P0abiReport(workspace=str(workspace))
    report.is_transcendental = is_transcendental_op(workspace)
    report.has_smallvalue_cases = has_smallvalue_cases(workspace)
    report.knowledge_update_present = (workspace / "knowledge_update.md").exists()

    risky, markers = scan_kernel_files(workspace)
    report.risky_hits = risky
    report.compliance_markers = markers

    if report.knowledge_update_present:
        block, err = parse_p_p88_yaml_block(workspace / "knowledge_update.md")
        report.p_p88_yaml_block = block
        report.p_p88_yaml_error = err

    return evaluate_compliance(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if not args.workspace.is_dir():
        print(f"workspace not found: {args.workspace}", file=sys.stderr)
        return 2

    rep = scan_workspace(args.workspace)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(f"=== P-P88 compliance scan: {rep.workspace} ===")
        print(f"verdict: {rep.verdict}")
        print(f"rationale: {rep.rationale}")
        print(f"transcendental: {rep.is_transcendental}")
        print(f"risky primitive hits: {len(rep.risky_hits)}")
        for h in rep.risky_hits:
            print(f"  - {h.file}:{h.line} {h.primitive} ({h.kb_ref}) :: {h.text}")
        print(f"compliance markers: {len(rep.compliance_markers)}")
        for m in rep.compliance_markers:
            print(f"  + {m.file}:{m.line} {m.description}")
        if rep.p_p88_yaml_block is not None:
            print(f"p_p88 block: {rep.p_p88_yaml_block}")
        elif rep.p_p88_yaml_error:
            print(f"p_p88 yaml error: {rep.p_p88_yaml_error}")

    if rep.verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

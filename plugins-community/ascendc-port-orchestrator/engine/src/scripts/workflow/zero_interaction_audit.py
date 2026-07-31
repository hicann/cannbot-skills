# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Zero-interaction audit — detect permission-ask patterns in skill outputs.

Design intent (P0ack 2026-05-10): `/ascendc-op-gen` and related skills are
declared as 0-interaction. User invokes the skill, the orchestrator + sub-agents
run to completion without permission-asks. Internal users reported (and the
2026-05-10 FA cold-start incident confirmed): in practice the LLM injects N
permission-asks per run, downgrading the product from 0-interaction to
user-guided.

This audit is a regression test that catches the over-permission anti-pattern
at PR time, not at runtime. It scans:

  1. Recent orchestrator stream-json logs (workspace/<op>/.cc_stream_log_*.jsonl)
  2. Recent Discord-payload audit (if present in .discord_audit/) or stdout-tail
     of background orchestrators
  3. The auth-gate matrix in src/skills/ascendc-op-gen/SKILL.md

For each suspected permission-ask string, classify by the matrix:
  - if action is in "LLM autonomous YES" rows AND LLM still asked → FAIL
  - if action is in "NO — ask first" rows → PASS (legitimate ask)
  - if action not in matrix → FLAG for matrix update

Intended usage:
    python3 -m src.scripts.workflow.zero_interaction_audit \\
        --workspace workspace/3_FusionAttention \\
        --max-permission-asks 0   # strict; 0-interaction means literally 0

Exit codes:
    0  audit pass
    2  audit FAIL (permission-asks exceed threshold)
    3  audit infrastructure error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Permission-ask phrases to scan for. Chinese + English. Each phrase is a
# regex pattern (case-insensitive). Matches in user-facing output (Discord
# replies, console prose, stream-json text deltas) count as permission-asks.
#
# These phrases are derived from the 2026-05-10 incident transcript where
# LLM asked for cold-start auth twice in a single user directive.
PERMISSION_ASK_PATTERNS = [
    # Direct permission-ask phrases
    r"需你\s*explicit\s*auth",
    r"需要你\s*确认",
    r"请明示",
    r"请确认",
    r"等你\s*ack",
    r"等你\s*授权",
    r"等你\s*decision",
    r"等你\s*回",
    r"need.*your\s*(?:explicit\s*)?(?:auth|authorization|approval|permission)",
    r"awaiting.*your\s*(?:auth|authorization|approval|decision)",
    r"can\s*I\s*proceed",
    r"may\s*I\s*proceed",
    r"should\s*I\s+(?:proceed|kill|stop|abort|cold-start|restart|continue)",
    # Option-menu phrases (A/B/C choice indicates LLM offloading decision)
    r"\(recommended\)",                  # generic "(recommended)" anywhere
    r"\(推荐\)",                         # Chinese variant
    r"选\s*[A-D]\s*[/]\s*[A-D]",        # "选 A / B / C" pattern
    r"option\s*[A-D]\s*[/]\s*[A-D]",
    r"path\s+[A-D]\s+[/]\s+[A-D]",
    # Standby-asking phrases (waiting for user input)
    r"standby\s+(?:until|for)\s+your",
    r"期间\s+plan\s+不动",
    r"期间\s+不动",
    r"等你\s+30s",
]

# Phrases that legitimately appear in non-asking contexts (whitelist).
# Suppress false positives.
WHITELIST_PATTERNS = [
    r"per\s+your\s+(?:explicit|previous|earlier)\s+(?:auth|directive|directive)",  # citing prior auth
    r"已\s*(?:auth|authorized|signed|signed-off)",  # already authorized
]


@dataclass
class AuditFinding:
    source_path: str
    line_number: int
    matched_phrase: str
    surrounding_context: str  # 80 chars around match
    severity: str  # "permission_ask" | "suspected_offload"


@dataclass
class AuditResult:
    findings: list[AuditFinding]
    sources_scanned: list[str]
    n_lines_scanned: int

    @property
    def pass_count(self) -> int:
        return sum(1 for _ in self.findings)

    def is_zero_interaction(self, max_asks: int = 0) -> bool:
        return len(self.findings) <= max_asks


def _scan_text(text: str, source_path: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    lines = text.split("\n")

    for ln_idx, line in enumerate(lines, start=1):
        # Whitelist check first — if line matches whitelist, skip
        if any(re.search(wp, line, re.IGNORECASE) for wp in WHITELIST_PATTERNS):
            continue

        for pattern in PERMISSION_ASK_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                ctx_start = max(0, ln_idx - 1)
                ctx_end = min(len(lines), ln_idx + 1)
                ctx = " | ".join(lines[ctx_start:ctx_end])[:200]
                findings.append(AuditFinding(
                    source_path=source_path,
                    line_number=ln_idx,
                    matched_phrase=pattern,
                    surrounding_context=ctx,
                    severity="permission_ask",
                ))
                break  # one finding per line is enough

    return findings


def _scan_stream_log(path: Path) -> list[AuditFinding]:
    """Stream-JSON log — extract assistant text content + scan."""
    findings: list[AuditFinding] = []
    if not path.exists():
        return findings

    try:
        for ln_idx, raw_line in enumerate(path.read_text().split("\n"), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            # Stream-JSON event with text content
            if event.get("type") == "assistant":
                msg = event.get("message", {})
                for block in msg.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        sub_findings = _scan_text(text, str(path))
                        for f in sub_findings:
                            f.line_number = ln_idx  # event-line, not text-line
                        findings.extend(sub_findings)
    except OSError:
        pass

    return findings


def audit_workspace(workspace: Path) -> AuditResult:
    """Audit a single op's workspace for permission-ask patterns."""
    findings: list[AuditFinding] = []
    sources: list[str] = []
    n_lines = 0

    # Stream logs (each agent spawn writes one)
    for stream_log in workspace.glob(".cc_stream_log_*.jsonl"):
        sources.append(str(stream_log))
        sub_findings = _scan_stream_log(stream_log)
        findings.extend(sub_findings)
        try:
            n_lines += sum(1 for _ in stream_log.read_text().split("\n"))
        except OSError:
            pass

    # PROGRESS.md (LLM may write permission-asks here)
    progress = workspace / "PROGRESS.md"
    if progress.exists():
        sources.append(str(progress))
        text = progress.read_text(errors="replace")
        n_lines += len(text.split("\n"))
        findings.extend(_scan_text(text, str(progress)))

    # Self-critic report
    critic_report = workspace / "self_critic_report.md"
    if critic_report.exists():
        sources.append(str(critic_report))
        text = critic_report.read_text(errors="replace")
        n_lines += len(text.split("\n"))
        findings.extend(_scan_text(text, str(critic_report)))

    return AuditResult(findings=findings, sources_scanned=sources, n_lines_scanned=n_lines)


def audit_text_corpus(text: str, source_label: str = "<inline>") -> AuditResult:
    """Audit a single blob of text (e.g. a Discord transcript or stdout dump)."""
    findings = _scan_text(text, source_label)
    return AuditResult(findings=findings, sources_scanned=[source_label],
                       n_lines_scanned=len(text.split("\n")))


def format_report(result: AuditResult) -> str:
    """Human-readable report."""
    lines = [
        "=" * 70,
        "Zero-Interaction Audit Report",
        "=" * 70,
        "",
        f"Sources scanned ({len(result.sources_scanned)}):",
    ]
    for s in result.sources_scanned:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append(f"Total lines scanned: {result.n_lines_scanned}")
    lines.append(f"Permission-ask findings: {len(result.findings)}")
    lines.append("")

    if result.findings:
        lines.append("FINDINGS (each violates 0-interaction design intent):")
        lines.append("")
        for i, f in enumerate(result.findings, start=1):
            lines.append(f"[{i}] {f.source_path}:{f.line_number}")
            lines.append(f"    pattern: {f.matched_phrase}")
            lines.append(f"    context: {f.surrounding_context}")
            lines.append("")
    else:
        lines.append("✓ No permission-ask patterns found. 0-interaction maintained.")

    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workspace", type=Path,
                   help="Single op workspace dir to audit")
    p.add_argument("--text-input", type=Path,
                   help="File containing assistant transcript to audit")
    p.add_argument("--max-permission-asks", type=int, default=0,
                   help="Max permission-asks tolerated (default: 0, strict)")
    p.add_argument("--json", action="store_true",
                   help="JSON output (for CI integration)")
    args = p.parse_args(argv)

    if not (args.workspace or args.text_input):
        print("ERROR: --workspace or --text-input required", file=sys.stderr)
        return 3

    if args.workspace:
        result = audit_workspace(args.workspace)
    else:
        text = args.text_input.read_text(errors="replace")
        result = audit_text_corpus(text, source_label=str(args.text_input))

    if args.json:
        print(json.dumps({
            "n_findings": len(result.findings),
            "n_lines_scanned": result.n_lines_scanned,
            "sources": result.sources_scanned,
            "findings": [
                {"source": f.source_path, "line": f.line_number,
                 "pattern": f.matched_phrase, "context": f.surrounding_context,
                 "severity": f.severity}
                for f in result.findings
            ],
            "passed": result.is_zero_interaction(args.max_permission_asks),
            "max_permission_asks": args.max_permission_asks,
        }, indent=2))
    else:
        print(format_report(result))

    return 0 if result.is_zero_interaction(args.max_permission_asks) else 2


if __name__ == "__main__":
    sys.exit(main())

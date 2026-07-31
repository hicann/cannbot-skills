# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""KB load audit (V2 #4) — verify agent actually read mandatory KB sections.

Background: brief tells agent to load specific KB sections; agent might skip
them. Today we have no enforcement. This module:

  1. Extracts the "Required reading" list from a kb_manifest brief block
  2. Cross-references against the agent's tool_use trace (V2 #3) for Read
     calls hitting those paths
  3. Reports coverage: covered / partial / missing

Usage:
  audit = audit_kb_load(brief_text, tool_uses_list)
  audit.coverage_pct  # 0-100
  audit.missing       # list of paths declared but not read
  audit.unexpected    # list of paths read but not declared

Note: this is a *post-hoc* audit, not a hard gate. We don't block the agent
from running when KB load is incomplete (that would conflict with how
agents legitimately skim — reading a section's TOC line counts as
'consulted', and we can't tell from envelope.permission_denials alone).
We surface coverage as a forensic signal in diagnose.py.
"""
from __future__ import annotations
import logging

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KbLoadAudit:
    declared_sections: list[str] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def coverage_pct(self) -> int:
        if not self.declared_sections:
            return 100
        return int(len(self.covered) * 100 / len(self.declared_sections))


# Match brief lines like "  - src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md"
# or "  - src/skills/references/target/ascendc/patterns/..."
_KB_PATH_PAT = re.compile(
    r"^\s*-\s*(src/skills/references/\S+\.md(?:#[A-Za-z0-9\-_]+)?)\s*$",
    re.MULTILINE,
)


def extract_declared_sections(brief: str) -> list[str]:
    """Pull the kb_manifest_block's required-reading paths from a brief string.

    Looks at the `# KB MANIFEST` block specifically (so we don't match
    incidental KB references in other blocks).
    """
    if "# KB MANIFEST" not in brief:
        return []
    # Slice from `# KB MANIFEST` to the next `^# ` heading
    after = brief.split("# KB MANIFEST", 1)[1]
    # Stop at next top-level header
    next_h = re.search(r"\n# ", after)
    section = after[: next_h.start()] if next_h else after
    paths = _KB_PATH_PAT.findall(section)
    return [p.split("#")[0] for p in paths]  # strip anchor


def extract_read_paths(tool_uses: list[dict]) -> list[str]:
    """From a tool-use trace, return list of paths that were Read by agent.

    Recognized tools: Read (file_path), Glob (pattern matching KB paths).
    Bash with `cat <kb-path>` is harder to match deterministically; skipped.
    """
    out: list[str] = []
    for tu in tool_uses or []:
        name = tu.get("tool_name")
        inp = tu.get("input") or {}
        if name == "Read":
            fp = inp.get("file_path")
            if fp and "src/skills/references/" in fp:
                # Normalize to start at "src/skills/references/..."
                idx = fp.find("src/skills/references/")
                out.append(fp[idx:])
        elif name == "Grep":
            # Grep with path under references/ is also acceptable evidence
            path = inp.get("path") or inp.get("glob") or ""
            if isinstance(path, str) and "src/skills/references/" in path:
                idx = path.find("src/skills/references/")
                out.append(path[idx:])
    return out


def audit_kb_load(brief: str, tool_uses: list[dict]) -> KbLoadAudit:
    """Cross-reference declared sections vs actually read paths.

    A declared section X.md is considered "covered" if the agent's tool_use
    trace shows a Read or Grep targeting that exact path or a subpath.
    """
    declared = extract_declared_sections(brief)
    read = extract_read_paths(tool_uses)
    declared_set = set(declared)
    read_set = set(read)

    covered: list[str] = []
    missing: list[str] = []
    for sec in declared:
        # Considered covered if any read path == sec, or a subpath under sec's dir
        if sec in read_set:
            covered.append(sec)
            continue
        # Subpath match: e.g. declared "src/skills/references/target/ascendc/patterns/" + read
        # "src/skills/references/target/ascendc/patterns/ascend950pr.md"
        sec_norm = sec.rstrip("/")
        if any(r.startswith(sec_norm) for r in read):
            covered.append(sec)
            continue
        missing.append(sec)

    unexpected: list[str] = []
    for read_path in read_set:
        if read_path in declared_set:
            continue
        if any(declared_path.rstrip("/") in read_path for declared_path in declared):
            continue
        unexpected.append(read_path)

    return KbLoadAudit(
        declared_sections=declared,
        read_paths=sorted(read_set),
        covered=covered,
        missing=missing,
        unexpected=sorted(unexpected),
    )


# ---------------------------------------------------------------------------
# CLI for forensics on a completed op
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Audit KB load coverage for an op")
    ap.add_argument("--brief", type=Path, required=True,
                    help="path to brief text (use agent_dispatch --print-only)")
    ap.add_argument("--tool-uses", type=Path, required=True,
                    help="path to JSONL with one tool_use entry per line "
                         "(extracted from .cc_envelope_log.jsonl etc.)")
    args = ap.parse_args()

    brief = args.brief.read_text()
    tool_uses = []
    for line in args.tool_uses.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        skip_current_item = False
        try:
            tool_uses.append(json.loads(line))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue

    audit = audit_kb_load(brief, tool_uses)
    print(json.dumps({
        "declared": audit.declared_sections,
        "read": audit.read_paths,
        "covered": audit.covered,
        "missing": audit.missing,
        "unexpected": audit.unexpected,
        "coverage_pct": audit.coverage_pct,
    }, indent=2))
    sys.exit(0 if audit.coverage_pct >= 80 else 2)

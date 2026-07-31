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

"""terminology_gate.py — block the `V220` / `V300` arch-naming error at commit time.

Why this exists
---------------
"V220" / "V300" is an INVENTED vocabulary that matches none of the three legitimate
ones, and one of the two tokens actively names the WRONG chip:

  * arch (the `__NPU_ARCH__` token):  `220x` (our a3),  `351x` (our a5)
  * SoC:      `Ascend910_9382` / `Ascend910_9392` / `Ascend950PR_9579`
  * product:  `Atlas A3 训练系列产品` / `Atlas A3 推理系列产品` / ...
  * (our target aliases `a3` / `a5` are fine and are NOT policed here.)

Per the vendor docs on disk
(`docs/references/.../编程指南__硬件实现__架构规格__NPU架构版本{220x,300x,351x}.md`):
  * `220x` → "Atlas A3 训练系列产品/Atlas A3 推理系列产品" — this is our **a3**.
  * `351x` → Ascend950PR — our **a5**.
  * `300x` → "Atlas 200I/500 A2 推理产品" — a DIFFERENT product. So every "V300" we
    write names someone else's chip. "V220" is at best a phantom fourth vocabulary.

The error is self-reproducing: it lives in files re-injected into every agent every
session (CLAUDE.md, KB_INDEX.md), while the owner's chat corrections evaporate at each
compaction. This gate stops the BLEEDING — it fails a commit that ADDS a new bare
`V220`/`V300`. It deliberately does NOT police the ~3.7k pre-existing sites (a
whole-tree gate would fail every commit and get disabled within a day); those are a
separate, staged rename.

What fires
----------
A bare `V220` / `V300` / `v220` / `v300` as a WHOLE WORD, on an ADDED line of the
staged diff only.

What does NOT fire (verified false-positive sources)
----------------------------------------------------
  * `V300x` / `V351x` / `V220x` — real hiascend arch tokens carry a trailing `x`; the
    whole-word lookahead excludes them (and `V351`/`351x` are never a policed token).
  * `220x` / `351x` — no `V` prefix, never matches.
  * existing KB ids `CAND-V300SYNC-1`, `CAND-V220-V300-FA-DIFF-1` — cross-referenced by
    id; renaming them is out of scope. `CAND-<...>` tokens are stripped before scanning,
    so touching such a line never blocks the commit.
  * a line carrying the allow-marker `terminology-ok` — used by the rule's OWN
    documentation (CLAUDE.md) and by legitimate historical quotes. Documenting the rule
    must not trip the rule.

Escape hatches
--------------
  * per-line marker `terminology-ok` (an HTML comment `<!-- terminology-ok -->` keeps it
    invisible in rendered markdown).
  * env `TERMINOLOGY_GATE_OFF=1` — disables the gate wholesale, for the Phase-2 bulk
    rename work and for bulk historical imports.

Usage
-----
  python3 src/scripts/terminology_gate.py          # scan staged diff, exit 1 on violation
  python3 src/scripts/terminology_gate.py --staged  # explicit; same as default

The pure predicate (`scan_line` / `scan_added_lines`) is import-testable and carries no
git state; `main()` is the only part that shells out to `git diff --cached`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Bare V220 / V300 (either case) as a whole word: not preceded or followed by another
# identifier char. The trailing lookahead is what lets `V300x` / `V300SYNC` through — a
# following word char means it is NOT the bare token.
_VIOLATION_RE = re.compile(r"(?<![A-Za-z0-9_])[Vv](?:220|300)(?![A-Za-z0-9_])")

# KB candidate ids are cross-referenced by literal id and are out of scope for the
# rename. Strip whole `CAND-...` tokens before scanning so an embedded `-V220-` inside an
# id never counts, while a bare `V220` elsewhere on the same line still does.
_CAND_ID_RE = re.compile(r"CAND-[A-Za-z0-9-]+")

# Per-line escape hatch (see module docstring). Substring match, case-insensitive.
_ALLOW_MARKER = "terminology-ok"

_ENV_OFF = "TERMINOLOGY_GATE_OFF"

# The gate's own machinery legitimately carries the literal tokens — the detector, its
# tests, and the hook that wires it all DOCUMENT and IMPLEMENT the rule. Scanning them
# would self-flag the definition, exactly as scrub_on_export.py exempts `_SELF_NAME`.
# Matched on the EXACT repo-relative path (not a suffix) so no unrelated file named
# `pre-commit` elsewhere gets a free pass. CLAUDE.md is deliberately NOT here: it stays
# scanned, and its rule prose uses the per-line `terminology-ok` marker instead, so a
# stray NEW violation added to CLAUDE.md is still caught.
_SELF_EXEMPT = frozenset({
    "src/scripts/terminology_gate.py",
    "src/scripts/tests/test_terminology_gate.py",
    ".githooks/pre-commit",
})


def scan_line(line: str) -> list[str]:
    """Return the list of offending bare tokens on a single line (empty = clean).

    Pure: no git, no I/O. This is the whole predicate.
    """
    if _ALLOW_MARKER in line.lower():
        return []
    cleaned = _CAND_ID_RE.sub("", line)
    return [m.group(0) for m in _VIOLATION_RE.finditer(cleaned)]


def scan_added_lines(added: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Scan added lines.

    `added` is a list of (path, line_text). Returns (path, token, line_text) per
    violation.
    """
    out: list[tuple[str, str, str]] = []
    for path, text in added:
        if path in _SELF_EXEMPT:
            continue
        for token in scan_line(text):
            out.append((path, token, text))
    return out


def iter_staged_added_lines() -> list[tuple[str, str]]:
    """Parse `git diff --cached -U0` into (path, added_line_text) pairs.

    Reads the real staged index (partial commits use a temp index) — so, unlike the
    scrub gate, this MUST run with git's hook env intact.
    """
    git_executable = str(Path(shutil.which("git") or "git").resolve())
    diff = subprocess.run(
        [git_executable, "diff", "--cached", "-U0", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=True,
    ).stdout
    added: list[tuple[str, str]] = []
    path = "?"
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("+++ "):
            path = raw[4:]
        elif raw.startswith("+") and not raw.startswith("+++"):
            added.append((path, raw[1:]))
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", default=True,
                        help="scan the staged diff (default)")
    parser.parse_args(argv)

    if os.environ.get(_ENV_OFF):
        print(f"[terminology-gate] {_ENV_OFF} set; gate disabled")
        return 0

    violations = scan_added_lines(iter_staged_added_lines())
    if not violations:
        return 0

    print("[terminology-gate] FAIL: this commit ADDS the invented arch token "
          "'V220'/'V300'.", file=sys.stderr)
    print("", file=sys.stderr)
    for path, token, text in violations:
        print(f"  {path}: {token}   |{text.strip()}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Use the real vocabulary instead of the phantom 'V' one:", file=sys.stderr)
    print("  * arch token:  V220 -> 220x   (our a3) ;  V300 -> 351x   (our a5)",
          file=sys.stderr)
    print("  * SoC:         Ascend910_9382 / Ascend910_9392 / Ascend950PR_9579",
          file=sys.stderr)
    print("  * product:     Atlas A3 训练系列产品 / ...   (our aliases a3 / a5 are fine)",
          file=sys.stderr)
    print("  WHY: 'V300' collides with 300x = 'Atlas 200I/500 A2 推理产品', a DIFFERENT",
          file=sys.stderr)
    print("       chip — so every 'V300' names someone else's product.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Escape hatch (rename work / historical quotes): add the marker "
          "'terminology-ok'", file=sys.stderr)
    print(f"to the line, or set {_ENV_OFF}=1. See src/scripts/terminology_gate.py.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

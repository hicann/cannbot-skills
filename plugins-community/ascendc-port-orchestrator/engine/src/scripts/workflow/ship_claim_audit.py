#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ship-claim audit — PreToolUse hook on outbound communication tools.

Blocks the agent from sending Discord/PR/Slack messages that contain
"win"-style language (✅, PASS, "搞定", "smart workaround", "peak X×",
etc.) unless the same message also cites at least one commit SHA that
is **reachable from origin/main** in the project's git repository.

Why this hook exists
====================

The op-gen orchestrator already has multiple defensive layers:

- KB (TR-OL / TR-EC / TR-PB) loaded into worker context, constraining
  op-gen decisions.
- Self-critic catalog (C1..C26) firing at Phase O5/O6 to gate archive
  commits.
- `workflow_critic.py` PreToolUse hook gating Edit/Write/Bash inside
  the orchestrator pipeline (rules G1..G7).

None of those layers can stop the *main-session agent* from writing
"✅ {op} N/N PASS" in a Discord status update when the underlying op
actually hit a phase-gap, iter-cap, or transient API error mid-pipeline.
Empirically the agent self-reports partial worker output as "win"
repeatedly under load.

This hook closes that gap with a mechanical defense on the outbound
communication tool surface — the only layer that actually reaches the
human reader.

Backing requirement: a verified SHA on origin/main
==================================================

Earlier revisions of this hook used grep-level checks (regex for a
fixed `output/...` path and a `pass_a ... PASS` substring). Both were
wrong:

- The `output/<project>/` directory tree is user-configurable. The
  first-level name under `output/` is the project name (not a fixed
  mode tag), and skill users may redirect output entirely elsewhere
  via skill argument. Any path regex baked into the hook is brittle.
- A `pass_a ... PASS` substring is just text — the agent can include
  it in a message that has nothing to do with a real archive.

The robust signal is a commit SHA that the hook itself verifies against
`origin/main` of the project repo (located at $CLAUDE_PROJECT_DIR).
A real merge into main is the only artifact the agent cannot fabricate
inside a Discord message.

Decision rule
=============

1. If `tool_name` not in `MONITORED_TOOLS` → exit 0 (allow).
2. Extract `text`. If empty → exit 0.
3. Scan for any token in `FORBIDDEN_WIN_TOKENS`. If none → exit 0.
4. Find all 7-40-char hex strings in `text`. For each, ask git:
   "is this an ancestor of origin/main?" If at least one passes → exit 0.
5. Otherwise → exit 2 (block, surface stderr).

Failure modes
=============

- `git` binary missing or `$CLAUDE_PROJECT_DIR` unset / not a git repo
  → block, with a clear message ("hook cannot verify; rewrite without
  win-language or fix environment"). Fail-closed for win-claims here
  because a falsely-allowed claim is worse than a falsely-blocked one.
- Malformed JSON payload on stdin → exit 0 (fail-open) — hook bug must
  not block legitimate work.
- `origin/main` ref not present locally → block, instruct user to
  `git fetch origin main`.

Privacy / portability
=====================

- Hook reads only `tool_input.text` (or `body`/`message`/`content`
  fallback). It never reads, logs, prints, or persists chat_id,
  message_id, user_id, attachment metadata, or other channel
  identifiers. A successful run produces no output.
- Hook ships in the a5_ops repo. Customers running `bash src/deploy.sh`
  inherit it via project-local `.claude/settings.json`. The hook is
  NOT installed via marketplace skill (would require mutating
  user-global settings).

Codification trigger
====================

Group Discord 2026-05-17 22:18Z–23:03Z, after:
1. A sweep session where the agent reported 4 partial op runs as
   ✅-framed status updates while zero ops closed E2E on origin/main.
2. The v1 hook (with hardcoded project-specific output paths)
   path) was correctly criticized for baking in a project-specific
   path assumption — see Discord msg 1505707288, quote:
   "我们不能假设 output 下的输出 ... 用户可能会在使用 skills 的时候
   指定 output 以外的其他目录作为输出。"

The v2 redesign drops every path/substring assumption and instead
performs a real git verification.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys


# --- Win-token list (heuristic, additive over time) -----------------------
FORBIDDEN_WIN_TOKENS: list[str] = [
    "✅",
    "PASS",
    "搞定",
    "完成 ",
    "smart workaround",
    "smart approach",
    "beating auto",
    "peak ",
    "× perf 提升",
    "× perf gain",
    "win!",
    "wins!",
    "victory",
]

# Hex SHA pattern (7..40 chars, word-bounded).
_SHA_PATTERN = re.compile(r"\b[0-9a-f]{7,40}\b")

# Verification-artifact substring (a5 PR review 2026-05-17 23:07Z): even
# with a verified SHA on origin/main, require some textual reference to
# the actual product artifact so a random unrelated main-commit cite is
# rejected. EITHER `verification.json` (the canonical artifact filename
# of every op archive) OR a `pass_a ... PASS` substring (the schema
# marker for explicit-PASS verdict). NO PATH ASSUMPTIONS — strings only.
_VERIFICATION_ARTIFACT_PATTERN = re.compile(
    r"verification\.json|pass_a.{0,30}PASS",
    re.IGNORECASE,
)

# --- Tools whose primary purpose is outbound communication ----------------
MONITORED_TOOLS: set[str] = {
    "mcp__plugin_discord_discord__reply",
    "mcp__plugin_discord_discord__edit_message",
}


def _extract_text(tool_input: dict) -> str:
    for k in ("text", "body", "message", "content"):
        v = tool_input.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _project_dir() -> str | None:
    """Resolve the project repo path.

    Priority:
      1. $CLAUDE_PROJECT_DIR (set by Claude Code when invoking project hooks)
      2. $SHIP_CLAIM_AUDIT_PROJECT_DIR (test/override hook)
    """
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get(
        "SHIP_CLAIM_AUDIT_PROJECT_DIR"
    )


def _verify_sha_on_main(sha: str, project_dir: str) -> bool:
    """Return True iff `sha` is an ancestor of `origin/main` in `project_dir`.

    Uses `git merge-base --is-ancestor <sha> origin/main` which exits 0
    when reachable, 1 when not reachable, 128 when sha doesn't exist.
    Returns False on any non-zero exit or any subprocess error.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        return False
    try:
        result = subprocess.run(
            [git_executable, "-C", project_dir, "merge-base", "--is-ancestor",
             sha, "origin/main"],
            capture_output=True,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"[ship-claim-audit] payload parse failed: {e}\n")
        return 0  # fail-open on bad input

    tool_name = payload.get("tool_name", "")
    if tool_name not in MONITORED_TOOLS:
        return 0

    text = _extract_text(payload.get("tool_input", {}) or {})
    if not text:
        return 0

    hits = [t for t in FORBIDDEN_WIN_TOKENS if t in text]
    if not hits:
        return 0

    # Win-language present → require a verified SHA on origin/main.
    project_dir = _project_dir()
    if not project_dir or not os.path.isdir(project_dir):
        sys.stderr.write(
            "[ship-claim-audit] BLOCKED: outgoing message contains win-words "
            f"{hits!r} but $CLAUDE_PROJECT_DIR is not set or not a directory "
            f"(got: {project_dir!r}). Hook cannot verify the claim.\n"
            "  Either set $CLAUDE_PROJECT_DIR to the project repo path, "
            "or rewrite without win-language.\n"
            "  See docs/design/SHIP_CLAIM_AUDIT_DESIGN.md.\n"
        )
        return 2

    candidate_shas = set(_SHA_PATTERN.findall(text))
    verified_shas = [
        sha for sha in candidate_shas
        if _verify_sha_on_main(sha, project_dir)
    ]
    has_artifact_ref = _VERIFICATION_ARTIFACT_PATTERN.search(text) is not None

    if verified_shas and has_artifact_ref:
        return 0

    # Build a precise failure message: which of the two requirements failed.
    fails: list[str] = []
    if not verified_shas:
        fails.append(
            f"no commit SHA reachable from origin/main in "
            f"{project_dir!r}; hex-like candidates: "
            f"{sorted(candidate_shas) or 'none'}"
        )
    if not has_artifact_ref:
        fails.append(
            "no `verification.json` or `pass_a ... PASS` substring "
            "(the canonical archive artifact reference)"
        )

    sys.stderr.write(
        "[ship-claim-audit] BLOCKED: outgoing message contains win-words "
        f"{hits!r} but failed the backing checks.\n"
        "  Required (both):\n"
        "    1. a commit SHA reachable from origin/main, verified via\n"
        "       git -C $CLAUDE_PROJECT_DIR merge-base --is-ancestor <sha> "
        "origin/main\n"
        "    2. the text mentions `verification.json` OR `pass_a ... PASS` "
        "(NO path assumption — substring only).\n"
        f"  Failures:\n    - " + "\n    - ".join(fails) + "\n"
        "  Rewrite:\n"
        "    - If pipeline truly closed on main → cite a commit SHA from "
        "`git log origin/main` AND mention `verification.json` / "
        "`pass_a PASS`.\n"
        "    - If pipeline NOT closed → replace ✅/PASS/win with "
        "'partial — state=X, not E2E closed'.\n"
        "    - If origin/main isn't fetched locally → `git fetch origin main` "
        "first.\n"
        "  See docs/design/SHIP_CLAIM_AUDIT_DESIGN.md.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

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

"""output_read_guard — PreToolUse hook: own-dir-only anti-output-read defense.

DEBT-HERMETIC-PROBE (owner-directed 2026-05-30, multi-SoC escalation). Design:
docs/design/HERMETIC_REPRODUCTION_PROBE_DESIGN.md §"Owner-escalated: own-dir-only".

The problem (owner): with MULTIPLE SoC ports in `output/` (arch22/A3 + arch35/A5 +
many ops), an op-gen agent generating op X for SoC Y can read X's *other-SoC* port
or a sibling op's archive at runtime → cross-contamination / cross-SoC cheat (reading
OUR OWN finished answers instead of generating).

The rule (owner): **own-dir-only**. During generation an agent reads ONLY its own op's
working dir. `--resume`/`--optimize` read that *same* op dir = still own-dir-only. There
is NO valid cross-op / cross-SoC `output/` read. So: allow own-op-dir under output/, DENY
every other `output/` path. (CANN source + input source are OUTSIDE `output/` → never
touched by this guard — owner confirmed 2026-05-30.)

Matcher (settings.json PreToolUse): Read | Grep | Glob | Bash. Catches ALL access methods
(per OL-160 name-coupling lesson: a safety net that misses a path/method is useless) —
Read.file_path, Grep/Glob.path, and Bash commands containing cat/head/tail/sed/awk/less/
python-open/grep against an output/ path.

Protocol: reads PreToolUse JSON on stdin {tool_name, tool_input, cwd?}; on a denied
cross-dir output read, prints reason to stderr + exit 2 (blocks the tool call). Allowed
reads → exit 0 (silent). Fail-OPEN on parse/unknown (exit 0) so the guard can never wedge
a legit run — it is a deny-specific guard, not a gate.

NOTE (impl status 2026-05-30): VERIFIED empirically (probe hook + spawned subagent) —
PreToolUse hooks DO fire for spawned-agent (worker/designer/translator) tool calls, and
the payload carries `agent_id` (present ONLY in subagent context; ABSENT for the main/
coordinator agent) plus `agent_type`. So this guard scopes its deny to SUBAGENT context
(the generation workers, where the cross-dir / cross-SoC cheat-read risk lives); the main/
coordinator agent is unaffected — it legitimately reads output/ archives for PR review +
verification. See design doc HERMETIC_REPRODUCTION_PROBE_DESIGN.md §"Owner-escalated".
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[3]))
_OUTPUT = "output/"
# Bash sub-commands that read file content (must scan their args for output/ paths).
_BASH_READERS = re.compile(r"\b(cat|head|tail|less|more|sed|awk|grep|rg|xxd|od|nl|cut|python3?\s+-c)\b")
# answer-bearing leaf dirs/files inside an op archive (informational — the rule is dir-scoped,
# but these sharpen the deny message).
_ANSWER_HINT = re.compile(r"/(op_kernel|kernel|model_new_\w+\.py|tested_\w+|pass_[ab]_summary)")


def _opgen_state() -> dict:
    """The active run's .opgen_state.json (op / mode / build_archive_dir).

    Resolution: $ASCENDC_WORKSPACE/.opgen_state.json, else the most-recently-modified
    workspace/<op>/.opgen_state.json."""
    cands: list[Path] = []
    ws_env = os.environ.get("ASCENDC_WORKSPACE")
    if ws_env:
        cands.append(Path(ws_env) / ".opgen_state.json")
    wsroot = PROJECT_ROOT / "workspace"
    if wsroot.is_dir():
        cands += sorted((p for p in wsroot.glob("*/.opgen_state.json") if p.exists()),
                        key=lambda p: -p.stat().st_mtime)
    for c in cands:
        try:
            return json.loads(c.read_text())
        except Exception:
            continue
    return {}


# resume/optimize iterate on THIS op's own archive → legit own-dir read. Any other
# mode = fresh/cold generation → the agent works in workspace/, so NO output/ read is
# legitimate (own-dir-only collapses to "no output/ read at all" during generation).
_ITERATE_MODES = {"resume", "optimize", "reoptimize"}


def _output_paths(token: str) -> list[str]:
    """Return output/<...> path fragments referenced by a token (a path or a bash cmd)."""
    hits = []
    for m in re.finditer(r"(?:^|[\s='\"(])((?:\./|/[^\s'\")]*?/)?output/[^\s'\")]+)", token):
        hits.append(m.group(1))
    # also bare 'output/...' at token start
    if token.lstrip().startswith(_OUTPUT) and token.strip() not in hits:
        hits.append(token.strip())
    return hits


def _should_deny(path_frag: str, mode: str, own_archive: str) -> bool:
    """Deny an output/ read unless it is THIS run's own archive on resume/optimize.

    Scoped by the FULL own-archive path (build_archive_dir), NOT op-name — op-name alone
    would allow the same op's OTHER-SoC port (output/<other-project>/.../<op>/), which is
    exactly the cross-SoC cheat owner's rule targets. During fresh/cold generation the
    agent works in workspace/, so no output/ read is legitimate at all."""
    norm = path_frag.replace("\\", "/")
    if _OUTPUT not in norm:
        return False
    if mode in _ITERATE_MODES and own_archive:
        oa = own_archive.replace("\\", "/").rstrip("/")
        # allow only paths under the own archive dir (resume/optimize iterating on it)
        if oa and (oa in norm or norm.split(_OUTPUT, 1)[-1].startswith(oa.split(_OUTPUT, 1)[-1])):
            return False
    return True  # fresh/cold (any output/ read) OR cross-archive on resume → DENY


# git READ verbs that DUMP FILE CONTENT (not just metadata). These can resolve the prior
# delivered kernel from history even after the working-tree archive is blinded — which is the
# gap A1's non-git export closes at the source; this is the durable defense-in-depth guard.
# NOTE `git log`/`git whatchanged` only dump content with -p/-u/--patch (bare log = messages).
_GIT_CONTENT_VERBS = {"show", "cat-file", "diff", "grep", "log"}
_GIT_GLOBAL_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def _git_verb(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Return (subcommand, args-after-subcommand) for a git invocation, else (None, []).

    Skips leading env-var assignments (FOO=bar git ...) and git global options
    (-C dir / -c k=v / --git-dir=...). Fail-safe: returns (None, []) for anything not
    recognizably `git <verb>` so the caller falls through to normal handling (allow)."""
    gi = None
    for i, t in enumerate(tokens):
        base = t.rsplit("/", 1)[-1]
        if base == "git":
            gi = i
            break
        if "=" in t and not t.startswith("-"):
            continue  # leading env assignment
        break  # a non-git, non-env leading token → not a simple git command
    if gi is None:
        return None, []
    rest = tokens[gi + 1:]
    j = 0
    while j < len(rest):
        t = rest[j]
        if t in _GIT_GLOBAL_OPTS_WITH_ARG:
            j += 2
            continue
        if t.startswith("-"):
            j += 1
            continue
        return t, rest[j + 1:]
    return None, []


def _git_has_targeted_safe_path(args: list[str]) -> bool:
    """True if the git read explicitly restricts to a NON-output pathspec — i.e. a
    `<commitish>:<path>` where <path> is not under output/, or a `-- <pathspec>` list all
    outside output/. Such a read (e.g. `git show <sha>:src/scripts/foo.py`) is harness code,
    not our finished answer → allow."""
    for a in args:
        if a.startswith("-"):
            continue
        if ":" in a:
            after = a.split(":", 1)[1].replace("\\", "/")
            if after and _OUTPUT not in after:
                return True
    if "--" in args:
        specs = args[args.index("--") + 1:]
        if specs and all(_OUTPUT not in s.replace("\\", "/") for s in specs):
            return True
    return False


def _git_read_verdict(cmd: str, mode: str, own_archive: str) -> str | None:
    """Decide a git Bash command. Returns "DENY" to block, or None to defer/allow.

    DENY (SUBAGENT + fresh/cold) when a git content-read verb (show|cat-file|log -p|diff|grep):
      - references an output/** path not allowed by own-dir-only (via _should_deny), OR
      - is a BARE commit-ish content dump (no non-output pathspec) — could expose the answer.
    ALLOW (return None) for: non-content verbs (status, rev-parse, ls-files, …), bare `git log`
    without -p, a `<sha>:src/...` / `-- src/...` targeted read, and own-archive reads on
    resume/optimize. Fail-OPEN (None) on any parse trouble — never wedge a legit run."""
    try:
        tokens = shlex.split(cmd.strip())
    except Exception:
        return None  # fail-open
    if not tokens:
        return None
    verb, vargs = _git_verb(tokens)
    if verb is None or verb not in _GIT_CONTENT_VERBS:
        return None  # not a git content read → not our concern
    if verb == "log":
        if not any(a in ("-p", "-u", "--patch") or a.startswith("--patch") for a in vargs):
            return None  # `git log` without patch = commit messages only → allow

    # 1. Explicit output/ references anywhere in the command → own-dir-only rule.
    out_frags = _output_paths(cmd)
    if out_frags:
        for frag in out_frags:
            if _should_deny(frag, mode, own_archive):
                return "DENY"
        return None  # all referenced output/ paths are the own archive on resume → allow

    # 2. No output ref, but a targeted non-output pathspec (harness code) → allow.
    if _git_has_targeted_safe_path(vargs):
        return None

    # 3. Bare content dump (no path restriction). In a fresh/cold generation subagent there is
    #    no legit reason to dump arbitrary history/tree content → deny. Resume/optimize (already
    #    trusted to iterate on the own archive) is left alone to avoid over-firing.
    if mode not in _ITERATE_MODES:
        return "DENY"
    return None


def main() -> int:
    try:
        ev = json.load(sys.stdin)
    except Exception:
        return 0  # fail-open

    # SUBAGENT-SCOPED guard. This restricts ONLY generation subagents (worker/designer/
    # translator) — where the cross-dir / cross-SoC cheat-read risk lives. The main/
    # coordinator agent (agent_id absent in payload) legitimately reads output/ archives
    # for PR review + verification, so it is NEVER restricted. Verified 2026-05-30:
    # `agent_id` is present in subagent PreToolUse payloads, absent for the main agent.
    if not (ev.get("agent_id") or ""):
        return 0  # main/coordinator agent — not subject to the own-dir-only guard

    tool = ev.get("tool_name", "")
    ti = ev.get("tool_input", {}) or {}

    st = _opgen_state()
    op = st.get("op")
    # .opgen_state.json stores the mode under `opgen_mode` (for example,
    # "port_a3_to_a5" or "backward"); the legacy `mode` key was never written, so this read
    # always defaulted to "default" → _ITERATE_MODES gating never fired →
    # an in-scope op's own archive .so was unreadable by the verify
    # subagent (worker had to cp it to a neutral path). Read opgen_mode first.
    mode = st.get("opgen_mode") or st.get("mode", "default")
    own_archive = st.get("build_archive_dir", "") or ""

    tokens: list[str] = []
    if tool == "Read":
        tokens.append(str(ti.get("file_path", "")))
    elif tool in ("Grep", "Glob"):
        tokens.append(str(ti.get("path", "")))
        tokens.append(str(ti.get("pattern", "")) if tool == "Glob" else "")
    elif tool == "Bash":
        cmd = str(ti.get("command", ""))
        # git READ handling (A2): git verbs are NOT in _BASH_READERS, so a `git show
        # <sha>:output/.../<op>/model_new_*.py` (or bare `git show <sha>` / `git log -p`)
        # would otherwise slip past. Evaluate the git-specific verdict first.
        if _git_read_verdict(cmd, mode, own_archive) == "DENY":
            sys.stderr.write(
                f"[output-read-guard] BLOCKED: git history/content read during "
                f"fresh/cold generation: {cmd}\n"
                f"  own-dir-only rule (DEBT-HERMETIC-PROBE): this run owns op "
                f"'{op or '<unknown>'}' (mode={mode}). A git content-read (show/cat-file/"
                f"log -p/diff/grep) that reaches an output/ archive OR dumps a bare commit-ish "
                f"can resolve a prior delivered kernel from history = our own finished answer = "
                f"cheat, not generation.\n"
                f"  Allowed: `git status`, `git rev-parse`, `git log` (no -p), and a targeted "
                f"non-output read like `git show <sha>:src/...`. Sanctioned reference (CANN "
                f"source) is OUTSIDE output/ and unaffected. See docs/design/"
                f"HERMETIC_REPRODUCTION_PROBE_DESIGN.md.\n"
            )
            return 2  # block
        if not _BASH_READERS.search(cmd):
            return 0  # a non-git Bash that doesn't read content → not our concern
        tokens.append(cmd)
    else:
        return 0

    for tok in tokens:
        if not tok or _OUTPUT not in tok:
            continue
        for frag in _output_paths(tok):
            if _should_deny(frag, mode, own_archive):
                ans = " (answer-bearing)" if _ANSWER_HINT.search(frag) else ""
                allow_note = (
                    f"own archive = {own_archive}" if own_archive
                    else "no own-archive recorded (fresh/cold gen → work in workspace/, not output/)"
                )
                sys.stderr.write(
                    f"[output-read-guard] BLOCKED: cross-dir output/ read{ans}: {frag}\n"
                    f"  own-dir-only rule (DEBT-HERMETIC-PROBE): this run owns op "
                    f"'{op or '<unknown>'}' (mode={mode}); {allow_note}.\n"
                    f"  Reading another op's archive, OR this op's port for a DIFFERENT SoC, OR "
                    f"any output/ during fresh/cold generation = our own finished answer = cheat, "
                    f"not generation. No valid cross-op/cross-SoC output/ read exists.\n"
                    f"  Sanctioned reference (CANN source for your target's arch / cv_agent "
                    f"tile2asc) is OUTSIDE output/ and unaffected. See docs/design/"
                    f"HERMETIC_REPRODUCTION_PROBE_DESIGN.md.\n"
                )
                return 2  # block
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

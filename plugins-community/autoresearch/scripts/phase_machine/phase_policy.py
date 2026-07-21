# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase-aware Bash and Edit/Write gates.

Bash gate is three layers, each with one job:

  1. CLASSIFIER (`classify`) — pure function: command string → CommandShape.
       AR(name)   canonical `autoresearch/scripts/<name>.py` invocation
       LIFECYCLE  AR but lifecycle (dashboard / parse_args / scaffold /
                  resume), allowed in every phase
       READONLY   every chain segment is read-only (ls / cat / git read /
                  echo / pwd / ...). NO file redirects, NO mutations.
       OTHER      anything else (ad-hoc bash, malformed AR, writes)
     Consults command text only — no phase, no filesystem.

  2. PHASE TABLE — `_AR_ALLOWED_BY_PHASE`. Static dict.
     LIFECYCLE / READONLY are implicitly allowed everywhere.
     OTHER is never allowed (writes go through Edit/Write tool gated by
     check_edit, state changes go through AR scripts) — a single
     ownership invariant for .ar_state/ across both tools.

  3. `check_bash` — global string bans, classify(), table lookup.

The Edit/Write gate (`check_edit`) lives below.  Business transitions are
selected by ``workflow.transition`` and committed by the transaction that
owns each event.
"""
import re
import shlex
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .state_store import (
    BASELINE,
    DIAGNOSE,
    EDIT,
    FINISH,
    INIT,
    PLAN,
    PLAN_FILE,
    PLAN_ITEMS_FILE,
    REPLAN,
)

# === LAYER 1: CLASSIFIER ===============================================

# Match only Python invocations that execute an allowed AR script.
# Absolute paths and the optional engine/ subdirectory are supported.
# Common Python flags allowed by both `py` and `python*` (don't affect
# script execution semantics; just runtime tweaks).
_COMMON_PY_FLAGS = (
    r'-[OuBdESIqs]+'                              # combinable singles
    r'|-X(?:\s+\S+|\S+)'                          # -X dev or -Xdev
    r'|-W(?:\s+\S+|\S+)'                          # -W default or -Wdefault
)

_CANONICAL_AR_RE = re.compile(
    r'\A\s*'
    r'(?:[A-Za-z_]\w*=\S+\s+)*'                  # env-var prefixes
    r'(?:'
    # py launcher (Windows): accepts version flags `-3`/`-3.10` AND
    # the common Python flags. The launcher then forwards everything
    # past the version to the actual interpreter.
    r'py(?:\s+(?:-\d+(?:\.\d+)?|' + _COMMON_PY_FLAGS + r'))*'
    r'|'
    # python / python3 / python3.10: NO version flag (Python rejects
    # `-3`/`-3.10` with "Unknown option"). Only the common flags.
    r'python(?:\d+(?:\.\d+)?)?(?:\s+(?:' + _COMMON_PY_FLAGS + r'))*'
    r')'
    # Canonical form: bare scripts/ relative to the project root; ./ tolerated.
    r'\s+(?:\./)?scripts/'
    r'(engine/)?'                                # group 1: engine/ presence
    r'([A-Za-z_]\w*\.py)\b'                      # group 2 = basename
    r'(?:\s+(?:'                                 # script args
    # Quoted strings: backtick and `$(` are forbidden (caught by the
    # pre-check in `classify`); `$VAR`/`${VAR}` are still fine.
    r'"[^"]*"|\'[^\']*\'|[^\s&|;()<>`][^\s&|;()<>`]*'
    r'))*'
    r'(?:\s+(?:'                                 # FD redirections
    r'\d?>>?\s*\S+|\d?<\s*\S+|2>&1|1>&2|&>>?\s*\S+'
    r'))*'
    r'\s*\Z'
)

# Canonical location per script. Used by classify to reject AR
# invocations that match the regex shape but point at the wrong
# directory (e.g., `python scripts/pipeline.py` —
# regex-OK but pipeline.py lives at engine/pipeline.py, so the bash
# would 404 and post_bash would still announce "Pipeline complete").
# The classifier rejects the wrong location before that can happen.
#
# Invariant: this set MUST equal decide._BLESSED_SCRIPTS (the agent-neutral
# decision layer; formerly hooks/guard_bash._BLESSED_SCRIPTS). The pre_tool
# Bash gate rejects un-blessed names before this map is consulted, so
# any name here that isn't blessed is unreachable. Internal subprocesses
# (e.g. pipeline.py → eval_kernel.py) don't go through the Bash tool
# and therefore don't appear in this map.
_CANONICAL_LOCATION = {
    # engine/ — AR scripts (subprocess pipeline) and parse_args lifecycle.
    "baseline.py": "engine",
    "create_plan.py": "engine",
    "parse_args.py": "engine",
    "pipeline.py": "engine",
    # scripts/ root — LIFECYCLE scripts. report.py is auto-invoked by
    # pipeline.py at FINISH (not by the agent), so it's omitted here
    # and from guard_bash._BLESSED_SCRIPTS for the same reason.
    "dashboard.py": "root",
    "resume.py": "root",
    "scaffold.py": "root",
}

# Tokenize arguments before enforcing the per-command read-only policy.
# Explicit output flags and effectful find actions remain denied.

_READONLY_HEAD_SINGLE = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "echo", "pwd",
})
_READONLY_GIT_OPS = frozenset({
    "log", "diff", "status", "show", "rev-parse", "blame",
})
_FIND_EFFECT_ACTIONS = frozenset({
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprint0", "-fprintf", "-fls",
})
_GIT_BRANCH_LISTING_FLAGS = frozenset({
    "--list", "--show-current", "-a", "-r", "-v", "-vv",
})


def _is_safe_readonly_arg(arg: str) -> bool:
    """Reject `--output` and `--output=...` (file-writing flag).
    Accepts `--output-format`, `--output-something`, etc. — those are
    different flags.
    """
    return arg != "--output" and not arg.startswith("--output=")


def _is_safe_find_arg(arg: str) -> bool:
    return not any(
        arg == action or arg.startswith(action + "=")
        for action in _FIND_EFFECT_ACTIONS
    )


def _has_unquoted_redirect(s: str) -> bool:
    """True iff `s` contains an unquoted `<` or `>`. shlex tokenization
    can't see these as redirection — `cat foo > log` becomes
    `["cat", "foo", ">", "log"]` with all tokens looking like normal
    args. Pre-scan with quote tracking so the readonly check can
    reject the whole segment before tokenizing.
    """
    in_s = in_d = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and not in_s and i + 1 < n:
            i += 2
            continue
        if c == "'" and not in_d:
            in_s = not in_s
            i += 1
            continue
        if c == '"' and not in_s:
            in_d = not in_d
            i += 1
            continue
        if not in_s and not in_d and c in "<>":
            return True
        i += 1
    return False


def _is_readonly_segment(seg_text: str) -> bool:
    """True iff `seg_text` is a single read-only command. Tokenizes
    via shlex so quoted args go through the same checks as bare args:
    `git diff "--output=patch"` and `git diff --output=patch` both
    fail by the same rule.

    shlex parse error (unbalanced quotes etc.) → False; the caller's
    READONLY claim cannot be made about a malformed segment.
    """
    s = seg_text.strip()
    if not s:
        return False
    if _has_unquoted_redirect(s):
        return False  # `cat foo > log`, `cat foo 2>&1`, etc.
    try:
        tokens = shlex.split(s, posix=True, comments=False)
    except ValueError:
        return False
    if not tokens:
        return False

    head, args = tokens[0], tokens[1:]

    if head == "git":
        if not args:
            return False
        sub, rest = args[0], args[1:]
        if sub == "branch":
            return all(a in _GIT_BRANCH_LISTING_FLAGS for a in rest)
        if sub in _READONLY_GIT_OPS:
            return all(_is_safe_readonly_arg(a) for a in rest)
        return False

    if head == "export" and args:
        # `export AR_TASK_DIR=<value>` — shlex unquotes the value into
        # the same token, so `AR_TASK_DIR="..."` arrives as one piece.
        return args[0].startswith("AR_TASK_DIR=") and len(args) == 1

    if head == "find":
        return all(_is_safe_find_arg(a) for a in args)

    if head in _READONLY_HEAD_SINGLE:
        return all(_is_safe_readonly_arg(a) for a in args)

    return False


_LIFECYCLE_SCRIPTS = frozenset({
    "dashboard.py", "parse_args.py", "scaffold.py", "resume.py",
})


@dataclass(frozen=True)
class CommandShape:
    """Output of `classify`. Pure data — no methods, no phase awareness."""
    klass: str                     # "AR" | "LIFECYCLE" | "READONLY" | "OTHER"
    name: Optional[str] = None     # for AR/LIFECYCLE: the .py basename


def _normalize(command: str) -> str:
    """Forward-slash all backslashes so Windows path forms hit the same
    grammar as POSIX forms.
    """
    return command.replace("\\", "/")


def _split_chain(command: str) -> List[str]:
    """Split on bash chain operators (&& || ; | bare-&) outside quotes,
    keeping `&` adjacent to `>`/`<` literal as FD redirection. Used by
    `classify` to walk segments for the READONLY check.

    Quote tracking and the redirection-aware `&` rule are inlined here
    because this is the only consumer; AR shapes are vetted in one pass
    by `_CANONICAL_AR_RE` and don't need segmenting.
    """
    segments: List[str] = []
    cur: List[str] = []
    in_s = in_d = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c == "\\" and not in_s and i + 1 < n:
            cur.extend((c, command[i + 1]))
            i += 2
            continue
        if c == "'" and not in_d:
            in_s = not in_s
            cur.append(c)
            i += 1
            continue
        if c == '"' and not in_s:
            in_d = not in_d
            cur.append(c)
            i += 1
            continue
        if in_s or in_d:
            cur.append(c)
            i += 1
            continue
        if i + 1 < n and command[i:i + 2] in ("&&", "||"):
            segments.append("".join(cur))
            cur = []
            i += 2
            continue
        if c in (";", "|"):
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        if c == "&":
            prev = next((ch for ch in reversed(cur) if not ch.isspace()), None)
            nxt = command[i + 1] if i + 1 < n else None
            if prev in (">", "<") or nxt == ">":
                cur.append(c)
                i += 1
                continue  # FD redirect, not chain
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    segments.append("".join(cur))
    return segments


def classify(command: str) -> CommandShape:
    """Sole authority on 'what shape is this command?'. Pure function;
    consults only `command`. Does NOT know phase, does NOT advance state.

    Decision order:
      1. Command-substitution pre-check (`$(...)` and backticks). These
         execute arbitrary commands at parse time — never canonical AR,
         never READONLY. Caught here so neither the AR regex's quoted-
         arg branch nor the READONLY segment grammar has to defend
         individually. `$VAR` / `${VAR}` (variable expansion) is fine.
      2. Canonical AR regex on the normalized command.
      3. Per-segment READONLY check. Read-only heads (cat/git diff/...)
         don't execute their args, so AR script paths in args
         (`cat autoresearch/scripts/X.py`,
         `git diff -- autoresearch/scripts/X.py`) are pure references,
         not invocations — safe to allow. Pipes / chains / redirects
         to non-readonly heads still fail the grammar so smuggle attempts
         like `cat .../X.py | bash` are rejected at the second segment.
      4. AR-mention non-canonical and not readonly → malformed AR shape
         (wrappers like `nohup python .../X.py &`, `bash -lc "..."`, etc.).
         Returns OTHER; check_bash rejects in every phase.
      5. Else OTHER.
    """
    if "$(" in command or "`" in command:
        return CommandShape("OTHER")

    normalized = _normalize(command)

    m = _CANONICAL_AR_RE.match(normalized)
    if m:
        has_engine = m.group(1) is not None
        name = m.group(2)
        canonical = _CANONICAL_LOCATION.get(name)
        # Location must match the canonical mapping. Unknown basenames
        # AND wrong-directory invocations fall through to the AR-mention
        # non-canonical → OTHER path below, where check_bash rejects with
        # the canonical-form message.
        if canonical is not None and (canonical == "engine") == has_engine:
            klass = "LIFECYCLE" if name in _LIFECYCLE_SCRIPTS else "AR"
            return CommandShape(klass, name)

    segments = [s for s in _split_chain(command) if s.strip()]
    if segments and all(_is_readonly_segment(s) for s in segments):
        return CommandShape("READONLY")

    if "autoresearch/scripts/" in normalized:
        return CommandShape("OTHER")  # malformed AR shape

    return CommandShape("OTHER")


# Thin views over `classify` — kept for hook callers that want a
# one-liner answer to a specific question.

def parse_canonical_ar(command: str) -> Optional[str]:
    """Return the AR script basename if `command` classifies as AR or
    LIFECYCLE, else None.
    """
    shape = classify(command)
    return shape.name if shape.klass in ("AR", "LIFECYCLE") else None


def parse_invoked_ar_script(command: str) -> Optional[str]:
    """Basename of the AR script invocation, or None. Used by
    hooks/post_bash for routing on baseline.py / pipeline.py /
    create_plan.py.
    """
    return parse_canonical_ar(command)


_SCRIPT_SHAPE_RE = re.compile(
    r'\bscripts/((?:engine/)?[A-Za-z_]\w*\.py)\b'
)
# Match the python launcher as a whole word so unrelated tokens like
# `python_helper` don't trigger the unknown-script hint on read-only
# references (`cat .../python_helper.py`). `\bpy\b` is intentionally
# omitted — it false-matches `.py` extensions in cat/git diff args.
# Windows `py` launcher users should write `python` explicitly.
_PY_LAUNCHER_RE = re.compile(r'\bpython(?:\d+(?:\.\d+)?)?\b')


def parse_script_names(command: str) -> List[Tuple[str, str]]:
    """Return [(path, basename)] for every `scripts/X.py`
    reference in a python-invocation command, regardless of canonical-
    location validity. Used by hooks/guard_bash's hallucinated/library/
    unknown-name pre-check: that pass wants the basename even when the
    invocation is non-canonical (wrong directory, malformed shape), so
    it can give a targeted hint before check_bash falls through to the
    generic canonical-form rejection.

    Returns [] when the command isn't a python invocation — read-only
    references (`cat .../X.py`, `git diff -- .../X.py`) and unrelated
    chains don't get unknown-script hints.

    Path is the slash-form sub-path under `scripts/`
    (e.g. `engine/eval.py` or `scaffold.py`); basename is the trailing
    file name. Consumers today only read the basename.
    """
    normalized = _normalize(command)
    if not _PY_LAUNCHER_RE.search(normalized):
        return []
    out: List[Tuple[str, str]] = []
    for sub in _SCRIPT_SHAPE_RE.findall(normalized):
        basename = sub.rsplit("/", 1)[-1]
        out.append((f"scripts/{sub}", basename))
    return out


def is_single_foreground_ar_invocation(command: str, *, script: str) -> tuple:
    """Recovery-gate predicate: is `command` exactly one foreground
    invocation of `<autoresearch/scripts/script>`?
    """
    invoked = parse_canonical_ar(command)
    if invoked is None:
        return False, _CANONICAL_FORM_REJECTION
    if invoked != script:
        return False, f"expected {script!r}, got {invoked!r}"
    return True, ""


# === LAYER 2: PHASE TABLE ==============================================

# Per-phase: which AR script names may run.
# LIFECYCLE scripts are always allowed (handled separately, not duplicated).
# Subprocess-only scripts (in guard_bash._LIBRARY_NOT_CLI) are blocked
# at the LNC layer before reaching here.
# EDIT-recovery (create_plan.py while state.pending_settle is set) is
# layered on top by hooks/guard_bash; this table reflects the normal path.
_AR_ALLOWED_BY_PHASE = {
    INIT: frozenset(),
    BASELINE: frozenset({"baseline.py"}),
    DIAGNOSE: frozenset({"create_plan.py"}),
    PLAN: frozenset({"create_plan.py"}),
    REPLAN: frozenset({"create_plan.py"}),
    EDIT: frozenset({"pipeline.py"}),
    FINISH: frozenset(),
}

# Edit/Write rules: which file classes may be written per phase.
#   "editable" — anything in task.yaml:editable_files
# plan.md is never in any set — it's machine-generated.
# reference.py is fixed at scaffold time and not editable thereafter.
_EDIT_RULES = {
    EDIT: {"editable"},
}


_CANONICAL_FORM_REJECTION = (
    "AR scripts must be invoked directly: "
    "`python scripts/engine/<name>.py <task_dir> [args...]` "
    "for blessed CLIs (pipeline, baseline, create_plan, parse_args), "
    "or `python scripts/<name>.py` for top-level scripts "
    "(scaffold, resume, dashboard). "
    "Allowed alongside: env-var assignments, real Python flags "
    "(`-O`, `-u`, `-X dev`, ...), and FD redirection (`> log`, `2>&1`). "
    "Not supported: short-circuit flags (`--version`, `-c`, `-m`), "
    "wrappers (`nohup`, `bash -lc`, subshells, `$(…)`), chains "
    "(`&&`, `||`, `;`, `|`), and backgrounding (`&`). Run multiple AR "
    "scripts in separate Bash calls; use the Read tool to inspect "
    "script source. (This is an LLM workflow guardrail, not a Bash "
    "sandbox.)"
)


# === LAYER 3: check_bash + check_edit ==================================

def check_bash(phase: str, command: str) -> tuple:
    """Return (allowed: bool, reason: str) for a Bash command at `phase`.

    Three layers, in order:
      1. `git commit` substring ban (must never run raw, even inside
         a permissive phase).
      2. classify() — pure command-shape decision.
      3. Phase table lookup — (phase, class) → allowed/blocked. The
         subprocess-only AR-script check fires only when classify
         returns AR(name) for one of those names (not a substring
         match — that would falsely block READONLY mentions).

    AR-mention-but-not-canonical is rejected in EVERY phase to keep
    the canonical-form contract crisp; without that, permissive phases
    would accept shapes like `nohup python scripts/X.py
    &` as 'ad-hoc shell'.
    """
    if "git commit" in command:
        return False, ("manual 'git commit' forbidden — commits are "
                       "produced by pipeline.py via workflow.record_round")

    shape = classify(command)

    if shape.klass == "LIFECYCLE":
        return True, ""

    if shape.klass == "READONLY":
        return True, ""

    if shape.klass == "AR":
        # Subprocess-only AR scripts (quick_check.py, eval_kernel.py)
        # are caught earlier by guard_bash._LIBRARY_NOT_CLI before
        # reaching this layer — they never appear in BS so classify()
        # wouldn't return AR(name) for them at all.
        allowed = _AR_ALLOWED_BY_PHASE.get(phase)
        if allowed is None:
            return False, f"unknown phase {phase!r}"
        if shape.name in allowed:
            return True, ""
        allowed_txt = sorted(allowed) or "(no AR scripts allowed in this phase)"
        return False, (f"phase {phase}: AR script {shape.name!r} not "
                       f"allowed; allowed = {allowed_txt}")

    # OTHER. AR-mention here means a malformed AR shape (chain, wrapper,
    # backgrounded, --version, etc.) — reject with the canonical-form
    # explanation. All other ad-hoc shell is rejected too: bash writes
    # need to come through guard_edit's whitelist (Edit/Write tool) or an
    # AR script, never raw `>` / `sed -i` / `cp`, because those bypass
    # the .ar_state ownership invariant guard_edit enforces.
    if "scripts/" in _normalize(command):
        return False, _CANONICAL_FORM_REJECTION

    return False, (f"phase {phase}: only AR scripts, lifecycle scripts, "
                   f"and read-only commands are allowed. Use the Edit / "
                   f"Write tool (gated by phase) for file changes, or an "
                   f"AR script for state changes — never ad-hoc bash.")


_DIAGNOSE_ARTIFACT_RE = re.compile(r"^\.ar_state/diagnose_v\d+\.md$")


def _check_plan_input_edit(
    phase: str,
    diagnose_action: Optional[str],
    pending_settle: bool,
) -> tuple:
    if phase in (PLAN, REPLAN):
        return True, ""
    if phase == DIAGNOSE:
        if diagnose_action in (
            "DIAGNOSIS_READY",
            "MANUAL_FALLBACK",
        ):
            return True, ""
        return False, (
            f".ar_state/{PLAN_ITEMS_FILE} is locked until the diagnosis "
            "artifact validates; run the ar-diagnosis subagent first. "
            f"(current diagnose_action={diagnose_action!r})"
        )
    if phase == EDIT and pending_settle:
        return True, ""
    return False, (
        f".ar_state/{PLAN_ITEMS_FILE} is writable only when create_plan.py "
        "is legal (PLAN, REPLAN, ready DIAGNOSE, or pending-settle EDIT); "
        f"current phase={phase}."
    )


def _check_state_edit(
    phase: str,
    rel_path: str,
    diagnose_action: Optional[str],
    pending_settle: bool,
) -> tuple:
    if rel_path == f".ar_state/{PLAN_ITEMS_FILE}":
        return _check_plan_input_edit(
            phase, diagnose_action, pending_settle
        )
    if _DIAGNOSE_ARTIFACT_RE.match(rel_path):
        if phase == DIAGNOSE:
            return True, ""
        return False, (
            f"{rel_path!r} is writable only while phase=DIAGNOSE."
        )
    if rel_path == f".ar_state/{PLAN_FILE}":
        return False, (
            "plan.md is machine-generated; write <items>...</items> to "
            f".ar_state/{PLAN_ITEMS_FILE}, then run create_plan.py."
        )
    return False, (
        f"{rel_path!r} is machine-maintained state. Only "
        f".ar_state/{PLAN_ITEMS_FILE} and .ar_state/diagnose_v<N>.md "
        "are agent-writable."
    )


def check_edit(
    phase: str,
    rel_path: str,
    editable_files,
    *,
    diagnose_action: Optional[str] = None,
    pending_settle: bool = False,
) -> tuple:
    """Return whether the current phase permits writing a relative path."""
    if rel_path.startswith(".ar_state/"):
        return _check_state_edit(
            phase, rel_path, diagnose_action, pending_settle
        )
    allowed_classes = _EDIT_RULES.get(phase, set())
    editable = set(editable_files or ())
    if "editable" in allowed_classes and rel_path in editable:
        return True, ""
    return False, f"phase {phase} does not allow writing {rel_path!r}"

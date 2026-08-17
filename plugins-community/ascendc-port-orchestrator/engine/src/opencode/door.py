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

"""opencode adapter — the JS->Python door for the adapter's own inline guards.

``src/opencode/a5_ops_hooks.mjs`` is supposed to be pure translation: opencode hook event ->
normalised payload -> canonical Python checkers under ``src/scripts/workflow/`` -> decision.
``src/scripts/orchestrator/backends/base.py`` states the invariant: a backend only WIRES the
harness, it NEVER re-implements a gate. Three adapter functions violated that by carrying real
policy — their own regexes, their own thrown denials — in JavaScript:

    runInlineAccessGuard        -> access_guard()
    runBuildArtifactGuard       -> build_artifact_guard()
    runInlineGeneratedCodeGuard -> generated_code_guard()

This module is where that policy now lives, and it is the single crossing point the .mjs uses
to reach it. The port is a RELOCATION, not a redesign: every branch, every regex and every
message string is preserved verbatim (including the ``[a5_ops opencode hook]`` message prefix,
so log greps written against the JS keep working, and including messages that name
``aog-kernel-worker`` even when the executing agent is ``aog-kernel-optimizer``).

Invocation (argv carries a base64-encoded JSON payload, so no stdin piping or shell-escaping is
needed from Bun's $ — same call shape as ``plugins-community/autoresearch/.opencode/door.py``):

    python3 door.py check <base64(json payload)>

The payload is the dict the .mjs already builds: ``hook_event_name``, ``tool_name``,
``tool_input``, ``session_id``, ``call_id``, ``agent_id``, ``agent_type``, ``cwd`` (plus an
optional ``project_root``; see ``_resolve_project_root``).

Output (stdout): a JSON decision::

    {"blocked": bool, "reason": str}

Exit status is a SEPARATE channel from the decision:

  * 0  — a decision was computed. ``blocked`` may be true (policy denial) or false.
  * 2  — the door itself failed (bad argv, undecodable payload, unreadable workspace, bug).
         stdout still carries ``{"blocked": true, ...}`` so a caller that only parses stdout
         also fails closed.

**Fail CLOSED — the caller must treat a door failure as a denial.** This is the opposite of
``plugins-community/autoresearch/.opencode/door.py``, which returns a safe ALLOW on internal
error. The difference is what each door guards. autoresearch's door guards a workflow-phase
nudge: a broken hook that denies would wedge the agent, and the worst case of a stray allow is
a mis-timed reminder. This door guards an anti-cheating boundary (cross-workspace reads of
another op's answers, unbuildable/hallucinated generated kernels, deploy invocations that mask
exit status). Here a silent allow is the dangerous direction: it lets exactly the tool call the
guard exists to stop through, unlogged and unattributed. A false denial is loud, recoverable
and diagnosable; a false allow is neither.

The one place the port is not message-identical is exactly this path: when a guard cannot read
what it is judging (``kernel/`` missing, a directory where a ``.cpp`` was expected), the JS let
the raw ``fs`` error escape and the plugin's catch turned it into a deny carrying that error's
text. Here it becomes ``blocked: true`` with an ``[a5_ops opencode door] internal error`` reason
and exit 2. Same direction — deny — different wording.

**Event gating stays with the caller.** These functions do not inspect ``hook_event_name``,
exactly as the JS originals did not: the .mjs called them from ``permission.ask`` and
``tool.execute.before`` only (never from ``tool.execute.after``). Wiring the door into a new
event would therefore newly apply these rules — that is a caller-side decision.

PARITY SCOPE — READ BEFORE EXTENDING
------------------------------------
These rules apply ONLY to the opencode harness. They are NOT enforced on the Claude Code path.
They are opencode-only policy, not a second copy of a canonical gate: none of them exists in
``src/scripts/workflow/workflow_critic.py`` or ``src/scripts/workflow/output_read_guard.py``
(verified 2026-08-13 — cross-workspace / project-wide / deploy_to_npu / unpiped / op_host /
pybind11.cpp / kernels.cpp all return zero hits in both). So an ``aog-kernel-worker`` running
under Claude Code is subject to strictly fewer rules than the same agent under opencode.

That is a known HARNESS-PARITY GAP, recorded here rather than papered over. Promoting these
rules to canonical (one implementation, applied to both harnesses) is deliberately OUT OF SCOPE
for this move, because it would add new gates to the Claude Code path — a behaviour change for
every existing Claude Code run, which is not something a relocation gets to do silently. Do not
read this module's existence under ``src/opencode/`` as parity: if you need these rules on both
harnesses, promote them into the canonical checkers as an explicit, separately reviewed change.

RESIDUAL LIMITATIONS
--------------------
The Bash/Glob/Grep/Read access rules above are TEXT-SCANNING tripwires: they inspect the tool
arguments, not the process tree. A worker with a shell can express an equivalent access through
indirection (variables, symlinks) that the scan does not see, and the nested-harness rule in
``_access_guard_bash`` (deny ``opencode run`` / ``claude -...`` / ``codex ...`` inside Bash)
closes the direct "strip the identity labels and re-enter the harness" route, not every
conceivable re-entry. The actual execution boundary remains the migration-mode graybox sandbox
wired in ``agent_dispatch.py`` — treat these rules as an anti-cheating tripwire with audit logs,
never as the confinement layer.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys

# ---- fidelity notes on the JS->Python regex translation ------------------------------------
# * JS `/…/s` -> re.DOTALL, `/…/m` -> re.MULTILINE, `/…/i` -> re.IGNORECASE.
# * JS `$` (no `m` flag) matches only at end of input, while Python `$` also matches before a
#   trailing newline. Every such `$` is translated as `\Z` to keep the JS meaning.
# * `\b`/`\w` are ASCII-only in JS but Unicode-aware in Python. Left as Python defaults: the
#   divergence needs a non-ASCII identifier neighbour to show up, and generated C/C++ carrying
#   non-ASCII is independently rejected by the "non-ASCII text in generated C/C++ source" rule.

KERNEL_AUTHOR_AGENTS = frozenset({"aog-kernel-worker", "aog-kernel-optimizer"})


class GuardDenied(Exception):
    """A policy denial. Carries the operator-facing message verbatim from the JS original."""


# ---- small helpers (direct ports of the .mjs helpers) ---------------------------------------

def _js_basename(p) -> str:
    """Port of the adapter's own `basename()` — `String(p).split(/[\\/]/).pop()`.

    NOT the same as Node's `path.basename`: this one returns "" for a trailing separator. The
    JS used BOTH functions, so both are reproduced (see `_node_basename`).
    """
    return re.split(r"[\\/]", str(p or ""))[-1]


def _node_basename(p) -> str:
    """Port of Node's `path.basename` (POSIX): trailing separators are ignored."""
    s = str(p or "").rstrip("/")
    return s.split("/")[-1]


def _lines(content) -> list:
    return re.split(r"\r?\n", str(content or ""))


def _is_kernel_author(payload: dict) -> bool:
    return str(payload.get("agent_type") or "") in KERNEL_AUTHOR_AGENTS


def _tool_input(payload: dict) -> dict:
    value = payload.get("tool_input")
    return value if isinstance(value, dict) else {}


def _expected_pybind_module_name(file_path, workspace) -> str:
    from_workspace = str(workspace or os.environ.get("ASCENDC_WORKSPACE", "") or "")
    if from_workspace:
        return "_%s_ext" % _node_basename(from_workspace)
    normalized = str(file_path or "").replace("\\", "/")
    match = re.search(r"(?:^|/)workspace/([^/]+)/kernel/pybind11\.cpp\Z", normalized)
    return "_%s_ext" % match.group(1) if match else ""


def _generated_kernel_path(p) -> bool:
    s = str(p or "")
    b = _js_basename(s)
    if re.search(r"(^|/)(op_host|op_kernel)/", s):
        return True
    if b in ("model_new_ascendc.py", "pybind11.cpp", "kernels.cpp", "kernel.h"):
        return True
    return bool(re.search(r"(^|/)kernel/[^/]+\.(h|hpp|cpp|cc)\Z", s))


def _content_from_tool_input(input_: dict) -> str:
    if not isinstance(input_, dict):
        return ""
    chunks = []
    for key in ("content", "file_text", "new_string", "replacement"):
        if isinstance(input_.get(key), str):
            chunks.append(input_[key])
    edits = input_.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict) and isinstance(edit.get("new_string"), str):
                chunks.append(edit["new_string"])
            if isinstance(edit, dict) and isinstance(edit.get("replacement"), str):
                chunks.append(edit["replacement"])
    return "\n".join(chunks)


def _extract_workspace_from_command(command) -> str:
    match = re.search(
        r"""(?:^|\s)ASCENDC_WORKSPACE=(?:"([^"]+)"|'([^']+)'|(\S+))""", str(command or ""))
    if not match:
        return ""
    return match.group(1) or match.group(2) or match.group(3) or ""


def _find_short_init_buffer_line(content) -> str:
    for line in _lines(content):
        if not re.search(r"\bInitBuffer\s*\(", line):
            continue
        args = re.search(r"\bInitBuffer\s*\(([^)]*)\)", line)
        if not args:
            continue
        if args.group(1).count(",") < 2:
            return line.strip()
    return ""


def _find_dynamic_init_buffer_line(content) -> str:
    for line in _lines(content):
        if not re.search(r"\bInitBuffer\s*\(", line):
            continue
        args = re.search(r"\bInitBuffer\s*\(([^)]*)\)", line)
        if not args:
            continue
        parts = [p.strip() for p in args.group(1).split(",")]
        bytes_expr = ",".join(parts[2:]) if len(parts) >= 3 else ""
        if re.search(
            r"\b(totalElems?|totalSize|blockSize|count|numel|shapeSize|nElems?)\b",
            bytes_expr, re.IGNORECASE,
        ):
            return line.strip()
    return ""


_KERNEL_DEFINITION_RE = re.compile(
    r"\bextern\s+\"C\"\s+__global__\s+__aicore__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{",
    re.DOTALL,
)
_KERNEL_DECLARATION_RE = re.compile(
    r"\bextern\s+\"C\"\s+__global__\s+__aicore__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*;",
    re.DOTALL,
)


def _has_ascendc_kernel_definition(content) -> bool:
    return bool(_KERNEL_DEFINITION_RE.search(str(content or "")))


def _has_ascendc_kernel_declaration_only(content) -> bool:
    s = str(content or "")
    return bool(_KERNEL_DECLARATION_RE.search(s)) and not _has_ascendc_kernel_definition(s)


def _has_pybind_kernel_launch(content) -> bool:
    s = str(content or "")
    return bool(re.search(r"\bACLRT_LAUNCH_KERNEL\s*\(", s)
                or re.search(r"\baclrtlaunch_[A-Za-z0-9_]+\s*\(", s))


def _find_short_set_global_buffer_line(content) -> str:
    for line in _lines(content):
        if not re.search(r"\bSetGlobalBuffer\s*\(", line):
            continue
        idx = line.find("SetGlobalBuffer")
        call_text = line[idx:] if idx >= 0 else line
        if "," not in call_text:
            return line.strip()
    return ""


def _uses_unqualified_ascend_symbols_without_namespace(content) -> bool:
    s = str(content or "")
    if not re.search(r"\b(?:TPipe|TQue|GlobalTensor|LocalTensor)\b", s):
        return False
    if re.search(r"\busing\s+namespace\s+AscendC\s*;", s):
        return False
    if re.search(r"\bAscendC::(?:TPipe|TQue|GlobalTensor|LocalTensor)\b", s):
        return False
    return True


def _find_non_ascii_line(content) -> str:
    for line in _lines(content):
        if re.search(r"[^\x00-\x7F]", line):
            return line.strip()
    return ""


def _find_overlapping_aligned_block_data_copy(content) -> str:
    s = str(content or "")
    remainder_block_split = (
        bool(re.search(r"\bbase\s*=\s*\(?\s*total_?\s*/\s*blockNum\s*\)?\s*;", s))
        and bool(re.search(r"\bremainder\s*=\s*total_?\s*%\s*blockNum\s*;", s))
        and bool(re.search(r"\bstart\s*=\s*blockIdx\s*\*\s*base\b", s))
        and bool(re.search(r"\bcount\s*=\s*base\s*\+", s))
    )
    ceil_block_size_split = (
        bool(re.search(r"\bblockSize\s*=\s*\(\s*total_?\s*\+\s*blockNum\s*-\s*1\s*\)\s*/\s*blockNum\s*;", s))
        and bool(re.search(r"\bstart\s*=\s*blockIdx\s*\*\s*blockSize\s*;", s))
        and bool(re.search(
            r"\bend\s*=\s*\([^;]*start\s*\+\s*blockSize[^;]*\)\s*\?[^;]*total_?[^;]*:[^;]*start\s*\+\s*blockSize[^;]*;",
            s))
        and bool(re.search(r"\bcount\s*=\s*end\s*-\s*start\s*;", s))
    )
    scalar_block_split = remainder_block_split or ceil_block_size_split
    if not scalar_block_split:
        return ""
    aligned_from_count = (
        bool(re.search(r"\balignedCount\s*=\s*\([^;]*\bcount\b[^;]*\+["
            r"^;]*(?:7|kFP32BlockElems\s*-\s*1)[^;]*\)[^;]*;", s))
        or bool(re.search(r"\balignedCount\s*=.*\bAlign(?:Up|UP)?[^;]*\bcount\b", s, re.DOTALL))
        or bool(re.search(r"\btileLenAligned\s*=\s*\([^;]*\btileLen\b[^;]*"
            r"\+[^;]*(?:7|kFP32BlockElems\s*-\s*1)[^;]*\)[^;]*;", s))
    )
    if not aligned_from_count:
        return ""
    copies_aligned_tile_from_scalar_start = (
        bool(re.search(r"\bDataCopy\s*\([^;]*\[\s*start\s*\+\s*offset\s*\][^;]*,\s*[^;]*\btileSize\b[^;]*\)\s*;", s))
        or bool(re.search(r"\bDataCopy\s*\([^;]*,\s*[^;]*\[\s*start\s*\+\s*"
            r"offset\s*\][^;]*,\s*[^;]*\btileSize\b[^;]*\)\s*;", s))
        or bool(re.search(r"\bDataCopy\s*\([^;]*\[\s*start\s*\+\s*offset\s"
            r"*\][^;]*,\s*[^;]*\btileLenAligned\b[^;]*\)\s*;", s))
        or bool(re.search(r"\bDataCopy\s*\([^;]*,\s*[^;]*\[\s*start\s*\+\s*off"
            r"set\s*\][^;]*,\s*[^;]*\btileLenAligned\b[^;]*\)\s*;", s))
    )
    return ("scalar block split with count-rounded DataCopy at start+offset"
            if copies_aligned_tile_from_scalar_start else "")


def _is_project_wide_recursive_glob(pattern, search_path, project_root: str) -> bool:
    p = str(pattern or "")
    if "**/" not in p:
        return False
    root = str(search_path or "").strip()
    if not root or root == "." or root == project_root:
        return True
    return os.path.abspath(root) == project_root


def _active_workspace_root(project_root: str = "") -> str:
    """The single op workspace the cross-workspace rules scope themselves to.

    A value that is not a STRICT descendant of `<project_root>/workspace/` is not an op
    workspace. The case that matters is an inherited `ASCENDC_WORKSPACE` pointing one level
    too high, at the workspace ROOT: honouring it puts every sibling op *inside* the active
    workspace, so `_references_other_workspace_path` stops matching and every cross-workspace
    rule silently turns off. Measured: with the leaf value a kernel-worker read of another
    op is DENIED; with the root value it is ALLOWED — i.e. that one variable was strictly
    more dangerous than leaving it unset, which keeps the rules armed.

    So an implausible value is discarded rather than trusted, and the guards fall back to the
    unset behaviour, which is the armed one.
    """
    ws = os.environ.get("ASCENDC_WORKSPACE") or os.environ.get("CLAUDE_ACTIVE_WORKSPACE") or ""
    if not ws:
        return ""
    active = os.path.abspath(ws)
    if project_root:
        workspace_root = os.path.join(os.path.abspath(project_root), "workspace")
        if not active.startswith(workspace_root + os.sep):
            return ""
    return active


def _references_other_workspace_path(value, project_root: str) -> str:
    text = str(value or "")
    root = os.path.abspath(project_root)
    active = _active_workspace_root(root)
    workspace_root = os.path.join(root, "workspace")
    patterns = [
        re.compile(r"""(?:^|[\s'"`(])((?:\./)?workspace/[^\s'"`)]+)"""),
        re.compile(r"""(?:^|[\s'"`(])(""" + re.escape(workspace_root) + r"""/[^\s'"`)]+)"""),
    ]
    for regex in patterns:
        for match in regex.finditer(text):
            raw = match.group(1) or ""
            resolved = os.path.abspath(os.path.join(root, raw))
            if not resolved.startswith(workspace_root + os.sep):
                continue
            if active and (resolved == active or resolved.startswith(active + os.sep)):
                continue
            return raw
    return ""


def _read_text(path: str) -> str:
    # JS `fs.readFileSync(p, "utf8")` substitutes U+FFFD for undecodable bytes rather than
    # raising; `errors="replace"` keeps a stray binary file from turning a policy decision
    # into a door failure.
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


# ---- guard 1: runInlineAccessGuard ----------------------------------------------------------

_ACCESS_DENY = "[a5_ops opencode hook] access guard blocked "


def _access_guard_glob(project_root: str, input_: dict) -> None:
    pattern = str(input_.get("pattern") or input_.get("file_path") or "")
    search_path = str(input_.get("path") or "")
    other_workspace = _references_other_workspace_path(
        "%s %s" % (search_path, pattern), project_root)
    if other_workspace:
        raise GuardDenied(
            _ACCESS_DENY + "cross-workspace glob by aog-kernel-worker: %s" % other_workspace)
    if re.search(r"(^|/)output/", pattern) or re.search(r"(^|/)output/", search_path):
        raise GuardDenied(
            _ACCESS_DENY + "kernel-worker glob over output/ archives; use current workspace, "
            "KB, SDK headers, and source inputs")
    if _is_project_wide_recursive_glob(pattern, search_path, project_root):
        raise GuardDenied(
            _ACCESS_DENY + "project-wide recursive glob '%s' from aog-kernel-worker; scope Glob "
            "to ASCENDC_WORKSPACE or an explicit source directory" % pattern)
    if re.search(
        r"\b(pass_[ab]_runner|verification\.json|tested_[A-Za-z0-9_]+|model_new_[A-Za-z0-9_]+\.py)\b",
        pattern,
    ) and not search_path:
        raise GuardDenied(
            _ACCESS_DENY + "answer-bearing glob '%s' without an explicit non-output search path"
            % pattern)


def _access_guard_grep(project_root: str, input_: dict) -> None:
    search_path = str(input_.get("path") or input_.get("file_path") or "")
    other_workspace = _references_other_workspace_path(search_path, project_root)
    if other_workspace:
        raise GuardDenied(
            _ACCESS_DENY + "cross-workspace grep by aog-kernel-worker: %s" % other_workspace)
    if re.search(r"(^|/)output/", search_path):
        raise GuardDenied(_ACCESS_DENY + "kernel-worker grep over output/ archives")


def _access_guard_read(project_root: str, input_: dict) -> None:
    file_path = str(input_.get("file_path") or input_.get("path") or "")
    other_workspace = _references_other_workspace_path(file_path, project_root)
    if other_workspace:
        raise GuardDenied(
            _ACCESS_DENY + "cross-workspace read by aog-kernel-worker: %s" % other_workspace)


def _access_guard_bash(project_root: str, input_: dict) -> None:
    command = str(input_.get("command") or "")
    other_workspace = _references_other_workspace_path(command, project_root)
    if other_workspace:
        raise GuardDenied(
            _ACCESS_DENY + "cross-workspace Bash access by aog-kernel-worker: %s"
            % other_workspace)
    if re.search(r"""(^|[\s'"`])(?:\./|/[^\s'"`]*)?(op_host|op_kernel)(?:/|[\s'"`]|\Z)""", command):
        raise GuardDenied(
            _ACCESS_DENY + "op_host/op_kernel Bash access in direct-pybind kernel-worker mode")
    if re.search(r"a5_exec\.py\b[\s\S]*\bdocker\s+exec\b", command):
        raise GuardDenied(
            "[a5_ops opencode hook] runtime guard blocked nested docker exec through "
            "a5_exec.py; a5_exec.py already runs inside the configured A5 container")
    if re.search(
        r"""(^|[\s;&|'"([])(?:opencode\s+run\b|claude(?=\s+-)|codex(?=\s+(?:exec\b|-)))""",
        command,
        flags=re.MULTILINE,
    ):
        raise GuardDenied(
            _ACCESS_DENY + "nested harness invocation by aog-kernel-worker; a nested run can "
            "strip the dispatch identity (AOG_HOOK_AGENT_*) and silently disarm the safety net")


_ACCESS_GUARDS_BY_TOOL = {
    "Glob": _access_guard_glob,
    "Grep": _access_guard_grep,
    "Read": _access_guard_read,
    "Bash": _access_guard_bash,
}


def access_guard(project_root: str, payload: dict) -> None:
    """Port of `runInlineAccessGuard` — 10 rules, kernel-author agents only.

    Dispatch by tool rather than a chain of `if tool == ...` blocks: the original ran to 64
    statements, and the per-tool rule sets are disjoint, so a table keeps each set readable on
    its own. A tool absent from the table has no access rules — the same as before, when its
    `if` simply did not match.
    """
    if not _is_kernel_author(payload):
        return
    # str(): a non-string tool_name (a JSON array/object) is unhashable and would make
    # dict.get raise, where the old `if tool == "Glob"` chain simply fell through to allow.
    # Unreachable in production — a5_ops_hooks.mjs::titleToolName always hands over a string —
    # but keeping the two paths literally equivalent costs one call.
    handler = _ACCESS_GUARDS_BY_TOOL.get(str(payload.get("tool_name") or ""))
    if handler is not None:
        handler(project_root, _tool_input(payload))


# ---- guard 2: runBuildArtifactGuard ---------------------------------------------------------

_BUILD_DENY = "[a5_ops opencode hook] build guard blocked deploy: "


def _guard_deploy_command_hygiene(command: str) -> None:
    """Rules about HOW deploy_to_npu*.sh is invoked, independent of the workspace."""
    if not re.search(r"\bdeploy_to_npu(?:_lane)?\.sh\b", command):
        return
    if re.search(r"\|\s*(tail|head|grep|sed|awk|tee|less|more|cat)\b", command):
        raise GuardDenied(
            _BUILD_DENY + "do not pipe deploy_to_npu*.sh output; pipes mask exit status and can "
            "hang post-build sync")
    if re.search(r"\bunpiped\b", command):
        raise GuardDenied(
            _BUILD_DENY + "run deploy_to_npu*.sh with direct output; do not append unsupported "
            "marker words such as unpiped")


def _guard_one_kernel_source(file: str, content: str) -> None:
    """AscendC-surface rules that apply to every kernel/*.{h,hpp,cpp,cc}."""
    if re.search(r"\bcoreCoord_t\b", content):
        raise GuardDenied(
            _BUILD_DENY + "%s uses unsupported coreCoord_t; use GetBlockIdx()/GetBlockNum() "
            "scalars" % file)
    if re.search(r"\b(?:IN_QUE_NUM|OUT_QUE_NUM)\b", content):
        raise GuardDenied(
            _BUILD_DENY + "%s uses undefined IN_QUE_NUM/OUT_QUE_NUM queue constants" % file)
    if re.search(r"\bWaitAllDone\s*\(", content):
        raise GuardDenied(_BUILD_DENY + "%s uses unsupported TQue::WaitAllDone()" % file)
    if re.search(r"\bGlobalTensor\s*<", content) and not re.search(r"\bSetGlobalBuffer\s*\(", content):
        raise GuardDenied(
            _BUILD_DENY + "%s declares GlobalTensor but never calls SetGlobalBuffer" % file)
    if re.search(r"^\s*TQue<[^;]+>\s+g_[A-Za-z_][A-Za-z0-9_]*\s*;", content, re.MULTILINE):
        raise GuardDenied(
            _BUILD_DENY + "%s declares file-scope TQue queues; keep queues inside the kernel "
            "operator object" % file)
    short_gm = _find_short_set_global_buffer_line(content)
    if short_gm:
        raise GuardDenied(
            _BUILD_DENY + "%s calls GlobalTensor::SetGlobalBuffer without an element-count "
            "argument: %s" % (file, short_gm))
    short_init = _find_short_init_buffer_line(content)
    if short_init:
        raise GuardDenied(
            _BUILD_DENY + "%s calls TPipe::InitBuffer without a byte-size argument: %s"
            % (file, short_init))
    dynamic_init = _find_dynamic_init_buffer_line(content)
    if dynamic_init:
        raise GuardDenied(
            _BUILD_DENY + "%s allocates UB buffer from dynamic full input size; use a fixed tile "
            "byte-size and loop over chunks: %s" % (file, dynamic_init))
    if re.search(r"\bpipe\s*\.\s*Barrier\s*\(", content):
        raise GuardDenied(
            _BUILD_DENY + "%s calls unsupported pipe.Barrier(); use TQue EnQue/DeQue or "
            "documented PipeBarrier APIs" % file)
    if _uses_unqualified_ascend_symbols_without_namespace(content):
        raise GuardDenied(
            _BUILD_DENY + "%s uses unqualified AscendC symbols without using namespace AscendC "
            "or AscendC:: qualification" % file)


def _guard_kernel_translation_unit(file: str, content: str, kernel_dir: str) -> bool:
    """Rules for a non-pybind .cpp/.cc. Returns True when it defines the kernel body."""
    defines_kernel = _has_ascendc_kernel_definition(content)
    if _has_ascendc_kernel_declaration_only(content):
        raise GuardDenied(
            _BUILD_DENY + "%s declares an AscendC kernel but does not define its body" % file)
    for include_name in re.findall(r'#include\s+"([^"]+)"', content):
        if include_name == "kernel_operator.h" or include_name.startswith("aclrtlaunch_"):
            continue
        if not os.path.exists(os.path.join(kernel_dir, include_name)):
            raise GuardDenied(
                _BUILD_DENY + "%s includes missing local header %s" % (file, include_name))
    return defines_kernel


def _guard_kernel_dir(kernel_dir: str) -> None:
    # `fs.readdirSync` and `os.listdir` are both in unspecified filesystem order; sorted() only
    # pins WHICH violation is reported first when a file has several, never whether one is.
    kernel_files = []
    for name in sorted(os.listdir(kernel_dir)):
        if re.search(r"\.(h|hpp|cpp|cc)\Z", name):
            kernel_files.append(os.path.join(kernel_dir, name))
    kernel_definition_seen = False
    for file in kernel_files:
        name = _node_basename(file)
        content = _read_text(file)
        _guard_one_kernel_source(file, content)
        if re.search(r"\.(cpp|cc)\Z", name) and name != "pybind11.cpp":
            if _guard_kernel_translation_unit(file, content, kernel_dir):
                kernel_definition_seen = True
    if not kernel_definition_seen:
        raise GuardDenied(
            _BUILD_DENY + "kernel/*.cpp lacks an extern \"C\" __global__ __aicore__ kernel "
            "definition")


def _guard_pybind_module_name(pybind: str, pybind_path: str, workspace: str) -> str:
    module_match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", pybind)
    if not module_match:
        raise GuardDenied(
            _BUILD_DENY + "%s lacks literal PYBIND11_MODULE(_<op>_ext, m)" % pybind_path)
    module_name = module_match.group(1)
    expected_module = _expected_pybind_module_name(pybind_path, workspace)
    shaped_like_pybind_module = module_name.startswith("_") and module_name.endswith("_ext")
    matches_this_op = not expected_module or module_name == expected_module
    if not shaped_like_pybind_module or not matches_this_op:
        raise GuardDenied(
            _BUILD_DENY + "pybind module %s must use exact _<op>_ext naming%s"
            % (module_name, (" (%s)" % expected_module) if expected_module else ""))
    return module_name


def _guard_model_new_wiring(model: str, model_path: str, module_name: str) -> None:
    if ("kernel" not in model or "build" not in model
            or not re.search(r"sys\.path\.(insert|append)\s*\(", model)):
        raise GuardDenied(
            _BUILD_DENY + "%s must add workspace/<op>/kernel/build to sys.path before importing "
            "the extension" % model_path)
    if re.search(r"\bfrom\s+kernel\s+import\b", model):
        raise GuardDenied(
            _BUILD_DENY + "%s must import %s from kernel/build, not from package kernel"
            % (model_path, module_name))
    escaped_module = re.escape(module_name)
    import_pattern = r"\bimport\s+%s\b" % escaped_module
    from_import_pattern = r"\bfrom\s+%s\s+import\s+" % escaped_module
    if not re.search(import_pattern, model) and not re.search(from_import_pattern, model):
        raise GuardDenied(
            _BUILD_DENY + "%s must import %s, matching PYBIND11_MODULE"
            % (model_path, module_name))
    calls_module_wrapper = bool(re.search(r"\.run_[A-Za-z0-9_]+\s*\(", model)) or (
        bool(re.search(from_import_pattern, model))
        and bool(re.search(r"\brun_[A-Za-z0-9_]+\s*\(", model)))
    if not calls_module_wrapper:
        raise GuardDenied(
            _BUILD_DENY + "ModelNew.forward must call the pybind run_<op> wrapper")


def build_artifact_guard(payload: dict) -> None:
    """Port of `runBuildArtifactGuard` — 24 rules, kernel-author agents on Bash only.

    The rules are grouped into helpers rather than inlined: as one function this ran to 143
    statements at cyclomatic complexity 40, which CodeCheck rejects and which makes the check
    ORDER — load-bearing here, since the first denial is the one the model sees — impossible
    to read off. The sequence below is that order, unchanged.
    """
    if not _is_kernel_author(payload):
        return
    if payload.get("tool_name") != "Bash":
        return
    command = str(_tool_input(payload).get("command") or "")
    _guard_deploy_command_hygiene(command)
    if not re.search(r"deploy_to_npu_lane\.sh\b", command) or not re.search(r"--build\b", command):
        return

    workspace = _extract_workspace_from_command(command) or os.environ.get("ASCENDC_WORKSPACE", "") or ""
    if not workspace:
        raise GuardDenied(
            _BUILD_DENY + "ASCENDC_WORKSPACE is required for op workspace validation")
    pybind_path = os.path.join(workspace, "kernel", "pybind11.cpp")
    model_path = os.path.join(workspace, "model_new_ascendc.py")
    if not os.path.exists(pybind_path):
        raise GuardDenied(
            _BUILD_DENY + "missing %s; build_ascendc.py does not auto-generate pybind11.cpp"
            % pybind_path)
    if not os.path.exists(model_path):
        raise GuardDenied(_BUILD_DENY + "missing %s" % model_path)

    pybind = _read_text(pybind_path)
    model = _read_text(model_path)
    _guard_kernel_dir(os.path.join(workspace, "kernel"))
    module_name = _guard_pybind_module_name(pybind, pybind_path, workspace)
    _guard_model_new_wiring(model, model_path, module_name)


# ---- guard 3: runInlineGeneratedCodeGuard ---------------------------------------------------
#
# The JS built one `checks` array of [pattern|predicate, reason] pairs and evaluated it LAST,
# after the pybind11.cpp PYBIND11_MODULE block threw its own denials. Both the composition
# order and that evaluation order are preserved: they decide WHICH message an offending file
# gets when it trips more than one rule.

_MODEL_NEW_CHECKS = (
    (re.compile(r"return\s+[A-Za-z_]\w*\s*[-+*/]\s*[A-Za-z_]\w*", re.MULTILINE),
     "model_new_ascendc.py host arithmetic fallback"),
    (re.compile(r"\btorch\.(add|sub|mul|div|matmul|mm|bmm|sum|mean|max|min|sort|topk|where)\s*\("),
     "model_new_ascendc.py torch compute fallback"),
    (re.compile(r"torch\.nn\.functional\."),
     "model_new_ascendc.py torch functional fallback"),
    (re.compile(r"\bnumpy\b|\bnp\."),
     "model_new_ascendc.py numpy fallback"),
)

_CPP_CHECKS = (
    (re.compile(r"\baclrtLaunchKernel\s*\("),
     "direct aclrtLaunchKernel call; use auto-generated aclrtlaunch_* wrapper via ACLRT_LAUNCH_KERNEL"),
    (re.compile(r"#include\s*<torch_npu/csrc/aten/common/ACLRT(?:Launch|Lauch)Kernel\.h>"),
     "non-portable torch_npu ACLRT macro header; use generated aclrtl"
         "aunch_<kernel>.h or an explicit extern aclrtlaunch_<kernel> stub"),
    (re.compile(r"#include\s*<pybind11/strict_rcward\.h>"),
     "invalid pybind11 strict_rcward header"),
    (re.compile(r"#include\s*<torch/npu\.h>"),
     "non-project torch/npu.h include; use torch_npu NPUStream header for current stream"),
    (re.compile(r"\bpy::object\b"),
     "pybind wrapper uses py::object; use torch::Tensor or at::Tensor for NPU tensors"),
    (re.compile(r"\bpy::tensor\b"),
     "pybind wrapper uses py::tensor; use torch::Tensor or at::Tensor for NPU tensors"),
    (re.compile(r"\bpy::array_t\b"),
     "CPU pybind array fallback"),
    (re.compile(r"\btorch::kCPU\b|\bc10::DeviceType::CPU\b|\.device\s*\(\s*torch::kCPU\s*\)"),
     "CPU tensor allocation/device in generated pybind; output must stay on NPU"),
    (re.compile(r"\b(?:static\s+)?uint32_t\s+run_[A-Za-z0-9_]*\s*\("),
     "pybind run_<op> wrapper returns launch status instead of output tensor"),
    (re.compile(r"\bretistory\b|\blaunchRetistory\b|\bstatusistory\b"),
     "pybind launch-status check references an invented status variable"),
    (re.compile(r"reinterpret_cast\s*<\s*GM_ADDR\s*>\s*\(\s*&"),
     "host stack pointer passed as GM_ADDR tiling/workspace"),
    (re.compile(r"\b(?:uint64_t|int64_t|uint32_t|int32_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]+\]\s*=\s*"
        r"\{[^;]*\}[\s\S]*reinterpret_cast\s*<\s*uint64_t\s*>\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)"),
     "host stack array passed as GM tiling/workspace; create a NPU tensor and pass its data_ptr"),
    (re.compile(r"\baclrtlaunch_[A-Za-z0-9_]+\s*\([^;{)]*\buint64_t\b[^;{)]*\)\s*;", re.DOTALL),
     "aclrtlaunch_<kernel> host stub uses uint64_t GM addresses; generated stubs take void* tensor data_ptr arguments"),
    (re.compile(r"(\bkernel_module_t\b|\bKernelAddParams\b|\bKERNEL_STATUS_SUCCESS\b)"),
     "OPP/PR4778 op_kernel registration scaffold in direct pybind path"),
    (re.compile(r"std::vector<[^>]+>\s+\w+_h\b|for\s*\([^)]*\)\s*\{[^{}]*(out|result)[^{}]*=", re.DOTALL),
     "host-side compute loop in generated binding"),
    (re.compile(r"\btorch::(add|sub|mul|div|matmul|mm|bmm|sum|mean|max|min|sort|topk)\s*\("),
     "torch C++ compute fallback"),
)

_PYBIND_CPP_EXTRA_CHECKS = (
    (re.compile(r"\b__gm__\b"),
     "pybind host wrapper must not use device-side __gm__ pointer qualifiers"
         "; use void*, uint8_t*, or ordinary host pointer types for launch stubs"),
)

_KERNELS_CPP_EXTRA_CHECKS = (
    (re.compile(r"\bPYBIND11_MODULE\s*\("),
     "kernels.cpp must hold AscendC kernel/source glue, not a pybind module"),
    (_has_ascendc_kernel_declaration_only,
     "kernels.cpp declares an AscendC kernel but does not define its body"),
)

# Presence/absence checks for a pybind11.cpp that actually defines a PYBIND11_MODULE. NOTE the
# inverted polarity carried over verbatim from the JS: every entry is REQUIRED-PRESENT except
# the one whose reason starts with "pybind exposes", which is required-absent.
_PYBIND_MODULE_CHECKS = (
    (re.compile(r"\bm\.def\s*\([^;]*&\s*ACLRT_LAUNCH_KERNEL\s*\(", re.DOTALL),
     "pybind exposes ACLRT_LAUNCH_KERNEL directly; wrap it in a run_<op> function"),
    (_has_pybind_kernel_launch,
     "missing aclrtlaunch_<kernel> or ACLRT_LAUNCH_KERNEL launch in pybind wrapper"),
    (re.compile(r"\bgetCurrentNPUStream\s*\("),
     "missing c10_npu::getCurrentNPUStream() stream handoff"),
    (re.compile(r"""#include\s*[<"]torch_npu/csrc/core/npu/NPUStream\.h[>"]"""),
     "missing torch_npu NPUStream header for c10_npu::getCurrentNPUStream()"),
    (re.compile(r"\b(?:torch|at)::empty(?:_like)?\s*\("),
     "missing NPU output allocation before launch"),
    (re.compile(r"""\bm\.def\s*\(\s*["']run_"""),
     "pybind module must expose a run_<op> wrapper function"),
)

_SOURCE_CHECKS = (
    (re.compile(r"\bOPENVINO_HIDDEN\b"), "OpenVINO token in AscendC kernel"),
    (re.compile(r"\b__opencl__\b"), "OpenCL kernel qualifier in AscendC kernel"),
    (re.compile(r"\bKernelTensor\b"), "non-project KernelTensor API in AscendC kernel"),
    (re.compile(r"reinterpret_cast\s*<\s*__gm__\s+\w+\s*\*\s*>\s*\(\s*(offset|idx|index)\s*\)"),
     "fake GM pointer reconstructed from numeric offset instead of saved GM_ADDR base"),
    (re.compile(r"""#include\s+" +ascendc/"""), "malformed ascendc include path"),
    (_find_non_ascii_line, "non-ASCII text in generated C/C++ source"),
    (re.compile(r"//\s*\.\.\.|/\*\s*\.\.\.|write process logic|TODO(?:\b|_)", re.IGNORECASE),
     "placeholder/TODO left in generated C/C++ source"),
    (re.compile(r"\b\w+\s*--\s*\)"),
     "post-decrement expression in tile/count calculation; use explicit arithmetic"),
    (re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:onge|istory|apse)\b"),
     "hallucinated identifier suffix in generated C/C++ source"),
    (re.compile(r"\bpipe_\s*\.\s*(?:EnQue|DeQue)\s*\("),
     "TPipe has no EnQue/DeQue queue operations; use TQue::EnQue/DeQue on LocalTensor values"),
    (re.compile(r"\bGetTPipe\s*\("),
     "unsupported GetTPipe() in generated kernel; keep a TPipe member "
         "inside the kernel operator object and call pipe_.InitBuffer(...)"),
    (re.compile(r"\bextern\s+\"C\"\s+__global__\s+__aicore__\s+void"
        r"\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*__gm__\s+\w+\s*\*", re.DOTALL),
     "kernel entry parameters must use GM_ADDR; cast to __gm__ pointers inside the operator Init"),
    (re.compile(r"\bperBlock\s*=\s*\(\s*total_?\s*\+\s*blockNum\s*-\s*1\s*\)\s*/\s*blockNum\s*;"),
     "unaligned per-block ceil division for fp32 DataCopy; use blockDim=8 for the simp"
         "le smoke or round per-block/tile counts to 8 elements and handle tails explicitly"),
    (_find_overlapping_aligned_block_data_copy,
     "overlapping aligned DataCopy block partition; do not scalar-split "
         "total/blockNum and then round each block count to 8 at start+offset"),
    (re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*DeQue\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)"),
     "TQue::DeQue takes no LocalTensor argument; store `queue.DeQue<T>()` exactly once"),
    (re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*EnQue\s*\(\s*([A-Za-z_]["
        r"A-Za-z0-9_]*)\s*\)\s*;\s*\1\s*\.\s*EnQue\s*\(\s*\2\s*\)\s*;", re.DOTALL),
     "same LocalTensor enqueued twice without an intervening DeQue"),
    (re.compile(r"\bDataCopy\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*,", re.IGNORECASE),
     "DataCopy destination must be LocalTensor, not a TQue queue"),
    (re.compile(r"\bDataCopy\s*\([^,]+,\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*,", re.IGNORECASE),
     "DataCopy source must be LocalTensor/GlobalTensor, not a TQue queue"),
    (re.compile(r"\bFreeTensor\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*\.\s*DeQue\s*<"),
     "do not DeQue inside FreeTensor; store the DeQue result once and free that LocalTensor"),
    (re.compile(r"\bInitBuffer\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Tbuf_\s*,\s*\d+\s*,", re.IGNORECASE),
     "TBuf used as a queue buffer in TPipe::InitBuffer; initialize TQue queues for the DataCopy/Add pipeline"),
    (re.compile(r"\bepilogue_len\b|\b(?:Add|Sub|Mul|Div|Muls|Adds)\s*\([^;]*,\s"
        r"*[A-Za-z_][A-Za-z0-9_]*\s*[-+*/]\s*[A-Za-z_][A-Za-z0-9_]*\s*\)"),
     "vector intrinsic count uses invented tail arithmetic; pass the current tile count directly"),
    (re.compile(r"\busing\s+namespace\s+ascendc\b"), "lowercase ascendc namespace"),
    (re.compile(r"#ifndef\s+(?:__)?KERNEL_OPERATOR_H(?:__)?\b"),
     "kernel_operator.h header guard collision"),
    (re.compile(r"^\s*TQue<[^;]+>\s+g_[A-Za-z_][A-Za-z0-9_]*\s*;", re.MULTILINE),
     "file-scope TQue queues; keep queues inside the kernel operator object"),
    (re.compile(r"\bpipe\s*\.\s*Barrier\s*\("),
     "unsupported pipe.Barrier(); use TQue EnQue/DeQue or documented PipeBarrier APIs"),
    (_find_short_set_global_buffer_line,
     "GlobalTensor::SetGlobalBuffer without an element-count argument"),
    (_uses_unqualified_ascend_symbols_without_namespace,
     "unqualified AscendC symbols without using namespace AscendC or AscendC:: qualification"),
    (_find_short_init_buffer_line, "TPipe::InitBuffer without a byte-size argument"),
    (_find_dynamic_init_buffer_line,
     "TPipe::InitBuffer uses dynamic full-input size; use fixed tile byte-size and loop over chunks"),
)

_HEADER_EXTRA_CHECKS = (
    (re.compile(r"\bextern\s+\"C\"\s+__global__\s+__aicore__\s+void\b"),
     "kernel entry definition/declaration belongs in kernels.cpp, not kernel.h"),
    (re.compile(r"}\s*//\s*namespace\s+AscendC"),
     "do not close or define namespace AscendC in generated kerne"
         "l.h; use `using namespace AscendC;` or explicit `AscendC::`"),
)


def _matches(matcher, content):
    """`typeof pattern === "function" ? pattern(content) : pattern.test(content)`."""
    if callable(matcher):
        return bool(matcher(content))
    return bool(matcher.search(content))


_GENERATED_DENY = "[a5_ops opencode hook] generated-code guard blocked %s: %s"


def _checks_for_generated_file(b: str, file_path) -> list:
    """The matcher list that applies to this file, in the original accumulation order."""
    checks = []
    if b == "model_new_ascendc.py":
        checks.extend(_MODEL_NEW_CHECKS)
    if b in ("pybind11.cpp", "kernels.cpp") or re.search(r"(^|/)kernel/[^/]+\.(cpp|cc)\Z", str(file_path)):
        checks.extend(_CPP_CHECKS)
        if b == "pybind11.cpp":
            checks.extend(_PYBIND_CPP_EXTRA_CHECKS)
        if b == "kernels.cpp":
            checks.extend(_KERNELS_CPP_EXTRA_CHECKS)
    if re.search(r"\.(h|hpp|cpp|cc)\Z", b) or re.search(r"(^|/)kernel/[^/]+\.(h|hpp|cpp|cc)\Z", str(file_path)):
        checks.extend(_SOURCE_CHECKS)
        if re.search(r"\.(h|hpp)\Z", b):
            checks.extend(_HEADER_EXTRA_CHECKS)
    return checks


def _guard_pybind_module_body(file_path, content: str) -> None:
    """Rules that only apply inside a real PYBIND11_MODULE translation unit."""
    if re.search(r"\bpy::tensor\b", content):
        raise GuardDenied(_GENERATED_DENY % (
            file_path, "pybind wrapper uses py::tensor; use torch::Tensor or at::Tensor for "
            "NPU tensors"))
    if re.search(r"\bpy::object\b", content):
        raise GuardDenied(_GENERATED_DENY % (
            file_path, "pybind wrapper uses py::object; use torch::Tensor or at::Tensor for "
            "NPU tensors"))
    if re.search(r"#include\s*<torch/pybind\.h>", content):
        raise GuardDenied(_GENERATED_DENY % (
            file_path, "invalid torch/pybind.h header; use torch/extension.h"))
    module_match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    expected_module = _expected_pybind_module_name(file_path, "")
    if module_match and expected_module and module_match.group(1) != expected_module:
        raise GuardDenied(_GENERATED_DENY % (
            file_path, "pybind module %s must be %s" % (module_match.group(1), expected_module)))
    for matcher, reason in _PYBIND_MODULE_CHECKS:
        present = _matches(matcher, content)
        # These entries are a mix of must-NOT-appear and must-appear rules, told apart by the
        # reason text — kept as-is from the .mjs so the two lists stay diffable.
        must_reject = present if reason.startswith("pybind exposes") else not present
        if must_reject:
            raise GuardDenied(_GENERATED_DENY % (file_path, reason))


def generated_code_guard(payload: dict) -> None:
    """Port of `runInlineGeneratedCodeGuard` — 66 rules, kernel-author writes only."""
    if not _is_kernel_author(payload):
        return
    if payload.get("tool_name") not in ("Write", "Edit", "MultiEdit"):
        return
    input_ = _tool_input(payload)
    file_path = input_.get("file_path") or input_.get("path") or ""
    if not _generated_kernel_path(file_path):
        return
    b = _js_basename(file_path)
    if re.search(r"(^|/)(op_host|op_kernel)/", str(file_path)):
        raise GuardDenied(
            "[a5_ops opencode hook] generated-code guard blocked %s: direct pybind benchmark "
            "tasks must not create op_host/ or op_kernel/ scaffold" % file_path)
    if b in ("kernel.h", "kernels.cpp", "pybind11.cpp") and not re.search(r"(^|/)kernel/", str(file_path)):
        raise GuardDenied(
            "[a5_ops opencode hook] generated-code guard blocked %s: kernel sources must live "
            "under workspace/<op>/kernel/ for deploy sync" % file_path)
    content = _content_from_tool_input(input_)
    if not content:
        return

    if b == "pybind11.cpp" and re.search(r"PYBIND11_MODULE\s*\(", content):
        _guard_pybind_module_body(file_path, content)
    for matcher, reason in _checks_for_generated_file(b, file_path):
        if _matches(matcher, content):
            raise GuardDenied(_GENERATED_DENY % (file_path, reason))


# ---- door ------------------------------------------------------------------------------------

def _resolve_project_root(payload: dict) -> str:
    """Mirror the .mjs `projectRoot` resolution: explicit > env > the run's cwd.

    The adapter computes `path.resolve(options.projectRoot || AOG_PROJECT_ROOT || ctx.directory)`
    once at plugin construction. The door accepts the same value under an optional
    `project_root` key, falls back to the env the adapter already exports, and finally to the
    payload's `cwd` (which the .mjs fills from `process.cwd()`).
    """
    raw = (payload.get("project_root") or os.environ.get("AOG_PROJECT_ROOT")
           or payload.get("cwd") or os.getcwd())
    return os.path.abspath(str(raw))


def check(payload: dict) -> None:
    """Run the three relocated guards in the .mjs `runGuardSet` order. Raises GuardDenied."""
    project_root = _resolve_project_root(payload)
    access_guard(project_root, payload)
    build_artifact_guard(payload)
    generated_code_guard(payload)


def _emit(blocked: bool, reason: str) -> None:
    """Write the verdict on fd 1. This is the door's PROTOCOL channel, not logging.

    The .mjs adapter parses exactly this line to decide whether the tool call proceeds, so it
    cannot be routed to a logger (a logger writes to stderr, is formattable by config, and can
    be silenced — any of which would turn a denial into an allow).
    Unbuffered on purpose: `sys.stdout.write` buffers when fd 1 is a pipe, which is exactly how
    the adapter runs this, so a process killed between the write and the flush would lose the
    verdict. `os.write` leaves no window.
    """
    payload = json.dumps({"blocked": bool(blocked), "reason": str(reason)}) + "\n"
    os.write(1, payload.encode("utf-8"))


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        mode = argv[1] if len(argv) > 1 else ""
        if mode != "check":
            raise ValueError("unknown mode %r; expected 'check'" % mode)
        if len(argv) < 3:
            raise ValueError("missing base64 payload argument")
        payload = json.loads(base64.b64decode(argv[2]).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object, got %s" % type(payload).__name__)
        check(payload)
    except GuardDenied as denial:
        _emit(True, str(denial))
        return 0
    except Exception as exc:  # noqa: BLE001 — fail CLOSED; see module docstring
        _emit(True, "[a5_ops opencode door] internal error (blocking): %s: %s"
              % (type(exc).__name__, exc))
        return 2
    _emit(False, "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

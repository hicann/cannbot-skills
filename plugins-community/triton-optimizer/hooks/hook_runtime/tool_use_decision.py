#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Shared tool-use guard policy for triton-agent optimize runs.

This module contains the backend-agnostic decision logic used by Codex and
Claude `PreToolUse` wrappers. It decides whether a tool invocation should be
denied and returns a denial reason string, or `None` when the tool use is
allowed.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, cast


READ_COMMANDS = {
    "awk",
    "cat",
    "find",
    "grep",
    "head",
    "less",
    "ls",
    "more",
    "rg",
    "sed",
    "stat",
    "tail",
    "tree",
}

EDIT_TOOLS = {"Edit", "MultiEdit", "Write"}
READ_TOOL_PATH_KEYS = ("file_path", "filePath")
EDIT_TOOL_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path", "notebookPath")
SHELL_WRAPPER_FLAGS = {"-c", "-lc"}
SHELL_WRAPPERS = {"bash", "sh", "zsh"}
PROTECTED_RELATIVE_PATH_PREFIXES = (
    ".triton-agent/",
    "triton-agent-logs/",
)
WORKFLOW_STATE_RELATIVE_PATH = Path(".triton-agent") / "state.json"
ROUND_ACTIVE_ALLOWED_TOP_LEVEL_FILES = frozenset(
    {
        "opt-note.md",
        "learned_lessons.md",
        "supervisor-report.md",
    }
)

WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/].+")


class PathAccessContext:
    __slots__ = ("workspace_root", "cwd", "allow_read_roots", "deny_read_globs", "deny_message")

    def __init__(
        self,
        *,
        workspace_root: Path,
        cwd: Path,
        allow_read_roots: tuple[Path, ...],
        deny_read_globs: tuple[str, ...],
        deny_message: str,
    ) -> None:
        self.workspace_root = workspace_root
        self.cwd = cwd
        self.allow_read_roots = allow_read_roots
        self.deny_read_globs = deny_read_globs
        self.deny_message = deny_message


def deny_reason_for_tool_use(policy: dict[str, Any], payload: dict[str, Any]) -> str | None:
    guard_policy = _guard_policy(policy)
    if guard_policy.get("enabled") is False:
        return None

    tool_name = payload.get("tool_name")
    raw_tool_input = payload.get("tool_input")
    if not isinstance(raw_tool_input, dict):
        return None
    tool_input = cast(dict[str, Any], raw_tool_input)

    context = _build_path_access_context(policy, guard_policy, payload, tool_input)
    if context is None:
        return None

    if tool_name in EDIT_TOOLS:
        path_text = _first_path_text(tool_input, EDIT_TOOL_PATH_KEYS)
        if path_text is None:
            return None
        return _deny_reason_for_built_in_edit_path(path_text, context)

    if tool_name == "Read":
        path_text = _first_path_text(tool_input, READ_TOOL_PATH_KEYS)
        if path_text is None:
            return None
        return _deny_reason_for_path_access(path_text, context)

    if tool_name != "Bash":
        return None

    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    return _deny_reason_for_bash_command(command, context)


def _guard_policy(policy: dict[str, Any]) -> dict[str, Any]:
    guard_policy = policy.get("guard")
    if isinstance(guard_policy, dict):
        return cast(dict[str, Any], guard_policy)
    return policy


def _build_path_access_context(
    policy: dict[str, Any],
    guard_policy: dict[str, Any],
    payload: dict[str, Any],
    tool_input: dict[str, Any],
) -> PathAccessContext | None:
    workspace_root = _resolve_policy_path(policy.get("workspace_root"))
    if workspace_root is None:
        return None

    cwd = _resolve_cwd(tool_input.get("cwd") or payload.get("cwd"), workspace_root)
    allow_read_roots = tuple(_allow_read_roots(guard_policy, workspace_root))
    deny_read_globs = tuple(
        str(item)
        for item in guard_policy.get("deny_read_globs", [])
        if isinstance(item, str)
    )
    deny_message = str(guard_policy.get("deny_message") or "This read is blocked by workspace policy.")
    return PathAccessContext(
        workspace_root=workspace_root,
        cwd=cwd,
        allow_read_roots=allow_read_roots,
        deny_read_globs=deny_read_globs,
        deny_message=deny_message,
    )


def _deny_reason_for_bash_command(command: str, context: PathAccessContext) -> str | None:
    for path_text in _collect_command_path_references(command):
        reason = _deny_reason_for_path_access(path_text, context)
        if reason is not None:
            return reason
    return None


def _deny_reason_for_path_access(
    path_text: str,
    context: PathAccessContext,
) -> str | None:
    resolved_path = _resolve_path_text(path_text, context.cwd, context.workspace_root)
    if resolved_path is None:
        return None
    if not _is_under_any_root(resolved_path, list(context.allow_read_roots)):
        return context.deny_message
    if _matches_any_glob(resolved_path, list(context.deny_read_globs)):
        return context.deny_message
    return None


def _deny_reason_for_built_in_edit_path(path_text: str, context: PathAccessContext) -> str | None:
    resolved_path = _resolve_path_text(path_text, context.cwd, context.workspace_root)
    if resolved_path is None:
        return None
    if not _is_relative_to(resolved_path, context.workspace_root):
        return _built_in_edit_outside_workspace_denial()

    workspace_relative_path = resolved_path.relative_to(context.workspace_root).as_posix()
    if _is_protected_runtime_edit_path(workspace_relative_path):
        return _protected_runtime_edit_denial(workspace_relative_path)

    workflow_state = _workflow_state_or_none(context.workspace_root)
    if workflow_state is None:
        return None

    phase = _require_state_string(workflow_state, "phase")

    if phase == "baseline":
        return None

    if phase == "awaiting_round_start":
        return _awaiting_round_start_built_in_edit_denial()

    if phase == "round_active":
        active_round_dir = _active_round_dir(workflow_state)
        if active_round_dir is None:
            return _built_in_edit_missing_state_denial()
        if _is_allowed_round_active_edit_path(workspace_relative_path, active_round_dir):
            return None
        return _round_active_built_in_edit_denial(active_round_dir)

    return _built_in_edit_missing_state_denial()


def _workflow_state_or_none(workspace_root: Path) -> dict[str, Any] | None:
    state_path = workspace_root / WORKFLOW_STATE_RELATIVE_PATH
    try:
        state = _load_json(state_path)
        _require_state_string(state, "phase")
    except (FileNotFoundError, ValueError, TypeError):
        return None
    return state


def _require_state_string(state: dict[str, Any], key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"workflow state {key} must be a non-empty string")
    return value


def _active_round_dir(state: dict[str, Any]) -> str | None:
    current_round = state.get("current_round")
    raw_rounds = state.get("rounds")
    if not isinstance(current_round, int) or not isinstance(raw_rounds, dict):
        return None
    rounds = cast(dict[str, Any], raw_rounds)
    round_entry = rounds.get(str(current_round))
    if not isinstance(round_entry, dict):
        return None
    round_entry_dict = cast(dict[str, Any], round_entry)
    status = round_entry_dict.get("status")
    round_dir = round_entry_dict.get("round_dir")
    if status != "active" or not isinstance(round_dir, str) or not round_dir:
        return None
    return round_dir


def _is_protected_runtime_edit_path(workspace_relative_path: str) -> bool:
    if workspace_relative_path == ".triton-agent" or workspace_relative_path.startswith(".triton-agent/"):
        return True
    if workspace_relative_path == "triton-agent-logs" or workspace_relative_path.startswith("triton-agent-logs/"):
        return True
    if workspace_relative_path.startswith(".codex/skills/") and "/scripts/" in workspace_relative_path:
        return True
    if workspace_relative_path.startswith(".claude/skills/") and "/scripts/" in workspace_relative_path:
        return True
    if workspace_relative_path.startswith(".opencode/skills/") and "/scripts/" in workspace_relative_path:
        return True
    if workspace_relative_path.startswith(".codex/triton-agent-hooks/"):
        return True
    if workspace_relative_path.startswith(".claude/triton-agent-hooks/"):
        return True
    if workspace_relative_path.startswith(".opencode/triton-agent-hooks/"):
        return True
    return False


def _is_allowed_round_active_edit_path(workspace_relative_path: str, active_round_dir: str) -> bool:
    if workspace_relative_path == active_round_dir or workspace_relative_path.startswith(f"{active_round_dir}/"):
        return True
    if "/" not in workspace_relative_path and workspace_relative_path in ROUND_ACTIVE_ALLOWED_TOP_LEVEL_FILES:
        return True
    return False


def _built_in_edit_missing_state_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "The temporary optimize workflow state is missing or invalid. "
        "Ask the runner to restart the optimize session so workflow state can be rebuilt."
    )


def _built_in_edit_outside_workspace_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Keep built-in edits inside the current optimize workspace."
    )


def _protected_runtime_edit_denial(workspace_relative_path: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        f"`{workspace_relative_path}` is a protected internal runtime path. "
        "Do not edit `.triton-agent/`, `triton-agent-logs/`, or backend-managed staged hook/skill implementation files."
    )


def _awaiting_round_start_built_in_edit_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Current phase is awaiting_round_start, so no optimize round is active yet. "
        "Use `triton-npu-optimize` `optimize-state start-round` to open the next `opt-round-N/` before editing."
    )


def _round_active_built_in_edit_denial(round_dir: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        f"Current active round is {round_dir}. "
        f"Built-in edits must stay inside `{round_dir}/`. "
        "Edit the round-local snapshot and round artifacts instead of top-level workspace files. "
        "If the round's intent or required evidence depth changes mid-round, use "
        "`triton-npu-optimize` `optimize-state set-current-round-state` "
        "before continuing edits. "
        "When this round is ready, use `triton-npu-optimize` "
        "`optimize-state submit-round` to submit it before moving on."
    )


def _first_path_text(tool_input: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _collect_command_path_references(command: str) -> list[str]:
    return _collect_command_path_references_inner(command, seen_commands=set())


def _collect_command_path_references_inner(
    command: str,
    *,
    seen_commands: set[str],
) -> list[str]:
    if command in seen_commands:
        return []

    scan_command = _strip_heredoc_payload(command)
    first_command = _first_simple_command_text(scan_command)
    tokens = _split_command(first_command)
    if not tokens:
        return []

    next_seen_commands = seen_commands | {command}
    nested_command = _first_shell_wrapper_command(tokens)
    if nested_command is not None:
        return _collect_command_path_references_inner(nested_command, seen_commands=next_seen_commands)

    scan_tokens = _filter_tokens_for_read_scan(tokens)
    if not scan_tokens or not _is_read_command_token(scan_tokens[0]):
        return []

    path_texts: list[str] = []
    for token in scan_tokens[1:]:
        if _looks_like_path(token):
            path_texts.append(token)
    return path_texts


def _first_shell_wrapper_command(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    if Path(tokens[0]).name not in SHELL_WRAPPERS:
        return None
    if len(tokens) < 3:
        return None
    if tokens[1] not in SHELL_WRAPPER_FLAGS:
        return None
    return tokens[2]


def _strip_heredoc_payload(command: str) -> str:
    if "<<" not in command or "\n" not in command:
        return command
    return command.splitlines()[0]


def _first_simple_command_text(command: str) -> str:
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in {";", "\n", "|", "&"}:
            return command[:index]
        index += 1
    return command


def _filter_tokens_for_read_scan(tokens: list[str]) -> list[str]:
    filtered: list[str] = []
    index = 0
    while index < len(tokens):
        kept, consumed = _filter_read_scan_token(tokens, index)
        if kept is not None:
            filtered.append(kept)
        index += consumed
    return filtered


def _filter_read_scan_token(tokens: list[str], index: int) -> tuple[str | None, int]:
    token = tokens[index]
    if _is_heredoc_operator_token(token):
        return None, 2 if token in {"<<", "<<-"} and index + 1 < len(tokens) else 1
    output_target = _output_redirection_target(token)
    if output_target is not None:
        return None, 2 if output_target == "" else 1
    input_target = _input_redirection_target(token)
    if input_target is not None:
        if input_target == "" and index + 1 < len(tokens):
            return tokens[index + 1], 2
        return input_target, 1
    return token, 1


def _is_heredoc_operator_token(token: str) -> bool:
    return token in {"<<", "<<-"} or token.startswith("<<")


def _output_redirection_target(token: str) -> str | None:
    match = re.match(r"^(?:(?:\d+)?>>|(?:\d+)?>\||(?:\d+)?>|&>>|&>)(.*)$", token)
    if match is None:
        return None
    return match.group(1)


def _input_redirection_target(token: str) -> str | None:
    if token.startswith("<<"):
        return None
    match = re.match(r"^(?:(?:\d+)?<>|(?:\d+)?<)(.*)$", token)
    if match is None:
        return None
    return match.group(1)


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return []


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _looks_like_path(token: str) -> bool:
    if not token:
        return False
    if token.startswith("-"):
        return False
    return (
        token == ".triton-agent"
        or token == "triton-agent-logs"
        or token == "~"
        or token.startswith("~/")
        or token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith(".codex/")
        or token.startswith(".claude/")
        or token.startswith(".opencode/")
        or token.startswith(PROTECTED_RELATIVE_PATH_PREFIXES)
        or bool(WINDOWS_ABSOLUTE_PATH_RE.match(token))
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], data)


def _resolve_policy_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def _resolve_cwd(value: object, workspace_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        return workspace_root
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _allow_read_roots(policy: dict[str, Any], workspace_root: Path) -> list[Path]:
    roots = [workspace_root]
    for raw_root in policy.get("allow_read_roots", []):
        root = _resolve_policy_path(raw_root)
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def _contains_glob_chars(path_text: str) -> bool:
    return "*" in path_text or "?" in path_text or "{" in path_text or "}" in path_text


def _resolve_path_text(path_text: str, cwd: Path, workspace_root: Path) -> Path | None:
    if _contains_glob_chars(path_text):
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path_text
    try:
        return path.resolve()
    except OSError:
        return (workspace_root / path_text).resolve()


def _is_under_any_root(path: Path, roots: list[Path]) -> bool:
    return any(_is_relative_to(path, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _matches_any_glob(path: Path, patterns: list[str]) -> bool:
    raw_path = str(path)
    return any(fnmatch.fnmatch(raw_path, pattern) for pattern in patterns)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""opencode CLI backend plugin."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging

import os
import json
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .base import Backend, Envelope
from .skill_context import load_skill_context

_OPENCODE_RUN_MARKERS = ("opencode run", "opencode  run")
_OP_SLUG_RE = re.compile(r"\b([a-z][a-z0-9_]*)-(?:kw|pp|ko|fo|ar|da|bs|td|tt|tpo|cl)-\d+")


@dataclass
class _StreamState:
    """Mutable state collected while consuming an opencode event stream."""

    last_output_at: float
    invalid_tool_limit: int
    raw_lines: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    session_id: str | None = None
    total_cost: float | None = None
    num_turns: int | None = None
    invalid_tool_count: int = 0
    invalid_tool_event: bool = False
    pending: bytes = b""


class OpencodeBackend(Backend):
    name = "opencode"

    def __init__(self, opencode_bin: str | None = None):
        self.opencode_bin = opencode_bin or os.environ.get("AOG_OPENCODE_BIN", "opencode")

    def dispatch(self, target: str, prompt: str, *, kind: str = "agent", mode: str = "foreground",
                 settings: Optional[str] = None, session: Optional[str] = None,
                 timeout: Optional[float] = None, sandbox_prefix: Optional[list] = None,
                 progress_callback=None, tee_path=None, silence_timeout: Optional[int] = None,
                 extra_args: Optional[list] = None, output_file=None, cwd=None,
                 permission_mode: str = "bypassPermissions", output_format: Optional[str] = "json",
                 stdin_prompt: bool = False) -> Envelope:
        if mode == "background":
            if output_file is None:
                raise ValueError("dispatch(mode='background') requires output_file")

        formatted = self._format_prompt(target, prompt, kind=kind)
        use_json_events = mode == "streaming" or (
            kind == "skill" and os.environ.get("AOG_OPENCODE_SKILL_FORMAT", "").lower() == "json"
        )
        # `opencode run` AUTO-REJECTS every permission request unless --auto is passed
        # (measured on 1.18.18: `permission requested: ...; auto-rejecting`). That is the
        # analogue of CC's bypassPermissions, so honour permission_mode by default and let
        # AOG_OPENCODE_AUTO override explicitly.
        auto_env = os.environ.get("AOG_OPENCODE_AUTO")
        auto = (auto_env == "1") if auto_env is not None else (
            permission_mode == "bypassPermissions"
        )
        cmd = self._build_run_cmd(
            session_id=session,
            auto=auto,
            cwd=str(cwd) if cwd is not None else None,
            extra_args=extra_args,
            format_json=use_json_events,
            model=self._select_model(target, kind),
            agent=self._select_agent(target, kind),
            variant=self._select_variant(target, kind),
        )
        if sandbox_prefix:
            cmd = list(sandbox_prefix) + cmd
        env = self.build_env(target, formatted, kind=kind,
                             cwd=str(cwd) if cwd is not None else None)
        if mode == "background":
            return self._dispatch_background(cmd, formatted, env, output_file)
        if mode == "streaming" or use_json_events:
            return self._run_streaming(
                cmd,
                formatted,
                kind=kind,
                mode=mode,
                env=env,
                timeout=timeout,
                silence_timeout=silence_timeout,
                progress_callback=progress_callback,
                tee_path=tee_path,
            )
        return self._dispatch_foreground(cmd, formatted, env, timeout, tee_path, kind, mode)

    @staticmethod
    def opencode_config(extra_agent: str | None = None) -> dict:
        """Public view of the process-private opencode config, parsed into a dict.

        Preflight callers (phase_o0 hook registration check) inspect the same config
        the backend injects at dispatch time without reaching into private members.
        """
        content = OpencodeBackend._opencode_config_content(extra_agent=extra_agent)
        if not content:
            return {}
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def build_env(target: str, prompt: str | None = None, *, kind: str = "agent",
                  cwd: str | None = None) -> dict:
        env = os.environ.copy()
        # NOT setdefault: a stale value inherited from an outer dispatch would otherwise win
        # and mislabel this process's identity for every hook payload it emits.
        env["AOG_HOOK_AGENT_ID"] = f"opencode:{target}"
        env["AOG_HOOK_AGENT_TYPE"] = target
        # opencode selects the PROJECT config from PWD; Popen(cwd=)/`--dir` do NOT update an
        # explicitly inherited PWD (documented in autoresearch/.opencode/run_loop.py:223).
        # Without this the child discovers config from the orchestrator's cwd (engine/).
        if cwd:
            env["PWD"] = str(cwd)
        # The agent definitions address the knowledge base as ${CLAUDE_PLUGIN_ROOT}/kb/...
        # (agents/aog-kernel-worker.md). Claude Code sets that variable; opencode does not,
        # so without this the worker receives an UNEXPANDED literal and the entire KB — the
        # always-loaded rules, the kernel-authoring guards, the error catalogue — is
        # unaddressable for it. The name is Claude-Code-flavoured but it is simply the
        # plugin-root contract the agent prompts are written against, so the opencode
        # backend must satisfy it too.
        env.setdefault("CLAUDE_PLUGIN_ROOT", str(OpencodeBackend._plugin_root()))
        # Regenerated unconditionally: an inherited OPENCODE_CONFIG_CONTENT is user input,
        # and it carries the safety-net adapter registration (cfg["plugin"]) for every
        # dispatch kind — trusting an inherited value would run sub-agents with the whole
        # door, identity guard and output guard silently absent.
        content = OpencodeBackend._opencode_config_content(extra_agent=target)
        if content:
            env["OPENCODE_CONFIG_CONTENT"] = content
        workspace = OpencodeBackend._extract_workspace_from_prompt(prompt or "")
        if workspace:
            # Still setdefault rather than assignment, for the reason it always was:
            # `workspace` is SCRAPED FROM THE PROMPT with a regex and is therefore influenced
            # by brief content, so an explicit value from the caller stays the more
            # trustworthy of the two and keeps winning.
            #
            # What changed: an inherited value is no longer trusted *blindly*. One pointing at
            # the workspace ROOT instead of an op directory disarms every cross-workspace rule
            # in the door (each sibling op then reads as "inside" the active workspace), so it
            # is the one inherited value that is more dangerous than no value at all. Such a
            # value is discarded here in favour of the dispatch's own workspace. The door
            # applies the same test independently — this is the near side of a fix that has to
            # hold even when the operator, not the orchestrator, exported the variable.
            for name in ("ASCENDC_WORKSPACE", "CLAUDE_ACTIVE_WORKSPACE"):
                if not OpencodeBackend._agrees_with_dispatch(env.get(name), workspace):
                    env[name] = workspace
        return env


    @staticmethod
    def _dispatch_background(cmd: list, formatted: str, env: dict, output_file) -> subprocess.Popen:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Popen duplicates the descriptor for the child; close the
        # parent-side handle as soon as the process has started.
        with out_path.open("w") as out_f:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        if proc.stdin is not None:
            proc.stdin.write(formatted)
            proc.stdin.close()
        return proc

    def _dispatch_foreground(self, cmd: list, formatted: str, env: dict, timeout: Optional[float], tee_path,
                             kind: str, mode: str) -> Envelope:
        try:
            completed = subprocess.run(
                cmd,
                input=formatted,
                capture_output=True,
                text=True,
                timeout=int(timeout) if timeout else None,
                env=env,
            )
            if tee_path:
                Path(tee_path).parent.mkdir(parents=True, exist_ok=True)
                Path(tee_path).write_text(completed.stdout or "")
            return Envelope(
                is_error=completed.returncode != 0,
                output_text=completed.stdout or "",
                raw_envelope={
                    "returncode": completed.returncode,
                    "stderr": completed.stderr or "",
                    "cmd_kind": kind,
                    "backend": self.name,
                    "mode": mode,
                },
            )
        except subprocess.TimeoutExpired:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"timed_out": True, "backend": self.name, "cmd_kind": kind},
            )
        except FileNotFoundError as e:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"not_found": True, "stderr": str(e), "backend": self.name},
            )

    def _build_run_cmd(self, prompt: str | None = None, *, session_id: str | None = None,
                       auto: bool = False, cwd: str | None = None,
                       extra_args: Optional[list] = None, format_json: bool = False,
                       model: str | None = None, agent: str | None = None,
                       variant: str | None = None) -> list:
        # Backward-compatible wrapper for tests/callers from v3.13.0. The
        # prompt parameter is intentionally ignored: opencode run receives the
        # full a5_ops brief on stdin, not as a positional argv message.
        cmd = [self.opencode_bin, "run"]
        if session_id:
            cmd += ["--session", session_id]
        if auto:
            cmd += ["--auto"]
        if cwd:
            cmd += ["--dir", cwd]
        env_model = model if model is not None else os.environ.get("AOG_OPENCODE_MODEL")
        if env_model and not self._has_model_arg(extra_args or []):
            cmd += ["--model", env_model]
        env_agent = agent if agent is not None else os.environ.get("AOG_OPENCODE_AGENT")
        if env_agent and not self._has_agent_arg(extra_args or []):
            cmd += ["--agent", env_agent]
        env_variant = variant if variant is not None else os.environ.get("AOG_OPENCODE_VARIANT")
        if env_variant and not self._has_variant_arg(extra_args or []):
            cmd += ["--variant", env_variant]
        if format_json and not self._has_format_arg(extra_args or []):
            cmd += ["--format", "json"]
        if extra_args:
            cmd += list(extra_args)
        return cmd

    @staticmethod
    def _has_model_arg(extra_args: list) -> bool:
        return any(arg == "--model" or arg == "-m" or str(arg).startswith("--model=") for arg in extra_args)

    @staticmethod
    def _has_format_arg(extra_args: list) -> bool:
        return any(arg == "--format" or str(arg).startswith("--format=") for arg in extra_args)

    @staticmethod
    def _has_agent_arg(extra_args: list) -> bool:
        return any(arg == "--agent" or str(arg).startswith("--agent=") for arg in extra_args)

    @staticmethod
    def _has_variant_arg(extra_args: list) -> bool:
        return any(arg == "--variant" or str(arg).startswith("--variant=") for arg in extra_args)

    @staticmethod
    def _select_model(target: str, kind: str) -> str | None:
        return OpencodeBackend._select_env_value("MODEL", target, kind)

    @staticmethod
    def _select_agent(target: str, kind: str) -> str | None:
        """Resolve the `--agent` value for this dispatch.

        `kind="agent"` MUST bind to the dispatched target: without it opencode runs its
        default agent while the hook payload still claims the worker's identity, so gates
        judge as the worker while a different agent executes.

        `kind="skill"` MUST NOT pass `--agent`: the skill funnel dispatches skill names
        (aog-knowledge-maintain / aog-a3-author / aog-op-classify), which are not registered
        agents. Passing them would select a nonexistent agent. Skill instructions are
        inlined into the prompt by `load_skill_context()` instead.
        """
        explicit = OpencodeBackend._select_env_value("AGENT", target, kind)
        if explicit:
            return explicit
        # Skills are registered as primary agents alongside the aog-* agents (see
        # _opencode_config_content), so binding them is both possible and REQUIRED: without
        # `--agent` the run falls back to opencode's default agent while the hook env still
        # claims the skill's name, and the identity guard then refuses every tool call.
        # Every kind, including "resume": the target is auto-registered in the generated
        # config when it is not a plugin agent/skill, so it is always bindable. Leaving a
        # kind unbound means running the default agent while the hook env claims a different
        # identity, which the identity guard correctly refuses — turning the whole dispatch
        # into a silent, exit-0 failure.
        if target:
            return target
        return None

    @staticmethod
    def _select_variant(target: str, kind: str) -> str | None:
        return OpencodeBackend._select_env_value("VARIANT", target, kind)

    @staticmethod
    def _select_env_value(name: str, target: str, kind: str) -> str | None:
        safe_target = re.sub(r"[^A-Za-z0-9]+", "_", target).upper().strip("_")
        if safe_target:
            specific = os.environ.get(f"AOG_OPENCODE_{name}_{safe_target}")
            if specific:
                return specific
        kind_specific = os.environ.get(f"AOG_OPENCODE_{kind.upper()}_{name}")
        if kind_specific:
            return kind_specific
        return os.environ.get(f"AOG_OPENCODE_{name}")

    @staticmethod
    def _agrees_with_dispatch(candidate, dispatch_workspace) -> bool:
        """True only when the inherited value names the SAME op this dispatch is for.

        An earlier version accepted anything under the workspace root, which let a stale
        SIBLING through — and a sibling is the worst case of all, because the door scopes its
        cross-workspace rules to this value. Measured with the real door, dispatching to opA
        while `ASCENDC_WORKSPACE` still said opB: the worker could read opB's
        `verification.json` — another operator's answers, exactly what the anti-cheating layer
        exists to stop — and was refused its OWN files. The guard was not merely off, it was
        pointed backwards.

        There is no reading of a disagreement that favours the inherited value: the dispatch
        knows which op it is running. So agreement is the whole test, and anything else is
        replaced.
        """
        if not candidate:
            return False
        return os.path.abspath(str(candidate)) == os.path.abspath(str(dispatch_workspace))

    # ---- process-private opencode config (agents + skill discovery) --------------------
    # opencode 1.18.18's own tool set, as reported by `opencode debug agent <name>`. Used to
    # restore whitelist semantics: opencode's per-agent `tools` record is additive, so every
    # tool NOT granted by the Claude Code definition has to be denied explicitly.
    # NOTE `patch` is deliberately absent: opencode treats edit/write/patch as ALIASES of one
    # write-side group, so emitting `patch: false` switches the whole group off and an agent
    # that Claude Code grants Edit to ends up unable to write at all. Measured consequence of
    # getting this wrong: the worker loses its write tool, falls back to `bash printf > file`,
    # and every generated-code rule — which only fires on Write/Edit/MultiEdit — is bypassed.
    _OPENCODE_TOOLS = {
        "bash", "read", "glob", "grep", "edit", "write", "task", "webfetch",
        "todowrite", "skill", "question", "invalid",
    }

    _CC_TO_OPENCODE_TOOL = {
        "read": "read", "write": "write", "edit": "edit", "multiedit": "edit",
        "grep": "grep", "glob": "glob", "bash": "bash", "webfetch": "webfetch",
        "skill": "skill", "task": "task",
    }

    @staticmethod
    def _plugin_root() -> Path:
        # backends/ -> orchestrator/ -> scripts/ -> src/ -> engine/ -> <plugin root>
        return Path(__file__).resolve().parents[5]

    @staticmethod
    def _engine_root() -> Path:
        # backends/ -> orchestrator/ -> scripts/ -> src/ -> engine/
        return Path(__file__).resolve().parents[4]

    @staticmethod
    def _neutralize_file_macro(text: str) -> str:
        """Defuse opencode's `{file:...}` config macro in prose.

        opencode expands `{file:PATH}` inside config strings and hard-fails the whole config
        when PATH does not exist. Prose that merely mentions the token must not be able to
        break the harness, so insert a zero-width-safe space after the brace.
        """
        return text.replace("{file:", "{ file:")

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Return (frontmatter_dict, body). Minimal + dependency-free."""
        if not text.startswith("---"):
            return {}, text
        end = text.find("\n---", 3)
        if end == -1:
            return {}, text
        raw = text[3:end].lstrip("\n")
        body = text[end + 4:].lstrip("\n")
        try:
            import yaml  # noqa: PLC0415
            data = yaml.safe_load(raw) or {}
            if isinstance(data, dict):
                return data, body
        except Exception as exc:
            # Deliberate fallback, not a swallowed error: PyYAML is optional here, and a
            # frontmatter block it rejects is still parseable by the minimal key: value
            # reader below. Raising would make an unparseable agent file abort every
            # dispatch, which is a strictly worse outcome than a slightly poorer parse.
            logging.getLogger(__name__).debug(
                "PyYAML frontmatter parse failed; using the minimal reader: %s", exc
            )
        return _parse_frontmatter_minimal(raw), body

    @staticmethod
    def _parse_frontmatter_minimal(raw: str) -> dict:
        """Dependency-free key: value reader used when PyYAML is absent or rejects the block."""
        data: dict = {}
        key = None
        for line in raw.splitlines():
            if line.startswith("  - ") and key:
                data.setdefault(key, [])
                if isinstance(data[key], list):
                    data[key].append(line[4:].strip())
                continue
            if ":" in line and not line.startswith((" ", "\t")):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                data[key] = val if val else []
        return data

    @staticmethod
    def _tools_record(fm: dict) -> dict | None:
        """Translate a Claude Code `tools:` whitelist into an opencode `tools` record.

        Returns None when the agent declares no list (opencode then keeps its defaults, which
        is what the absent-frontmatter case meant before).
        """
        tools = fm.get("tools")
        if not (isinstance(tools, list) and tools):
            return None
        allowed = set()
        for t in tools:
            oc = OpencodeBackend._CC_TO_OPENCODE_TOOL.get(str(t).strip().lower())
            if oc:
                allowed.add(oc)
        # opencode cannot express "Edit but not Write": edit/write/patch are one alias group
        # and resolve together. Measured on 1.18.18 — {"edit": true, "write": false} resolves
        # to edit=False write=False, and {"write": false, "edit": true} resolves to BOTH true.
        # So an agent that Claude Code grants Edit to will have Write as well, whatever we
        # emit. Make that explicit rather than emitting `write: false` and believing a
        # restriction the harness silently discards: aog-kernel-optimizer / aog-fused-optimizer
        # / aog-researcher are Edit-only in CC and DO resolve write=true here. This is a
        # harness-parity gap, not a whitelist.
        if allowed & {"edit", "write"}:
            allowed |= {"edit", "write"}
        if not allowed:
            return None
        # A Claude Code `tools:` list is a WHITELIST — anything absent is denied. opencode's
        # `tools` record is ADDITIVE: unlisted tools keep their default (enabled). Emitting
        # only the allowed ones therefore silently WIDENS every agent's surface — measured: an
        # analyzer-only agent whose CC definition grants Read/Grep/Glob/Bash/WebFetch/Skill
        # came back with edit/write/task enabled. That matters beyond tidiness: door.py's
        # write-side guards only apply to kernel-author agents, so a widened analyzer could
        # write kernel sources with none of the generated-code rules judging it. Deny every
        # other known tool explicitly to restore whitelist semantics.
        #
        # Order matters and must be DETERMINISTIC. opencode collapses edit/write/patch into
        # one write-side group and the LAST key for that group wins, so a set-comprehension
        # here made each agent's write capability depend on Python's per-process string hash
        # seed (measured: kernel-worker kept `edit` in 4 runs out of 10). Emit denials first
        # and grants last, from sorted lists, so the grant always wins and the config is
        # byte-stable across processes. `tool_name`, never `name`: `name` is the AGENT name
        # the entry is keyed by, and rebinding it here silently filed every agent under the
        # last tool it was granted.
        record = {tool_name: False for tool_name in sorted(OpencodeBackend._OPENCODE_TOOLS - allowed)}
        for tool_name in sorted(allowed):
            record[tool_name] = True
        return record

    @staticmethod
    def _agent_prompt_value(root, bodies_dir, name: str, body: str) -> str:
        """Materialise an agent body and return the value to put in `prompt`.

        The bodies address the KB as ${CLAUDE_PLUGIN_ROOT}/kb/... . That is PROMPT TEXT, not
        shell: the model passes it to the read tool verbatim and opencode does not expand
        environment variables in tool arguments, so the worker tries to open a literal
        "${CLAUDE_PLUGIN_ROOT}/kb/..." and gives up (measured: READ_OK=no on the very first
        mandatory KB file). Exporting the variable to the child does NOT help, for the same
        reason. Expand it while materialising instead.
        """
        body = body.replace("${CLAUDE_PLUGIN_ROOT}", str(root))
        body = body.replace("$CLAUDE_PLUGIN_ROOT", str(root))
        prompt_value = OpencodeBackend._neutralize_file_macro(body)
        if bodies_dir is not None:
            body_path = bodies_dir / f"{name}.md"
            try:
                body_path.write_text(body)
                prompt_value = "{file:%s}" % body_path
            except OSError as exc:
                # Body file is an optimisation: keep the inlined prompt text on failure.
                logging.getLogger(__name__).debug(
                    "cannot persist agent body file %s: %s", body_path, exc
                )
        return prompt_value

    @staticmethod
    def _agents_from_markdown(root, agents_dir, bodies_dir) -> dict:
        agents: dict = {}
        if not agents_dir.is_dir():
            return agents
        for md in sorted(agents_dir.glob("*.md")):
            try:
                fm, body = OpencodeBackend._parse_frontmatter(md.read_text(errors="replace"))
            except OSError:
                continue
            name = str(fm.get("name") or md.stem).strip()
            if not name:
                continue
            entry: dict = {
                "description": OpencodeBackend._neutralize_file_macro(
                    str(fm.get("description") or name).strip()
                ),
                # primary: `opencode run --agent` refuses to bind a subagent and silently
                # falls back to the default agent.
                "mode": "primary",
                "prompt": OpencodeBackend._agent_prompt_value(root, bodies_dir, name, body),
            }
            record = OpencodeBackend._tools_record(fm)
            if record is not None:
                entry["tools"] = record
            agents[name] = entry
        return agents

    @staticmethod
    def _register_skill_agents(skills_dir, agents: dict) -> None:
        """Register the plugin's SKILLS as primary agents too.

        Skill dispatches used to run with no `--agent`, i.e. as opencode's default `build`
        agent, while the hook env still announced AOG_HOOK_AGENT_TYPE=<skill name>. Once the
        adapter was loaded for skill dispatches as well, those two signals contradicted each
        other and the identity guard refused EVERY tool call of every skill run — and because
        `opencode run` still exits 0, the orchestrator recorded the refusal prose as a
        successful skill result. Registering the skill under its own name makes the observed
        agent match the dispatched one, so identity is consistent instead of merely silent.
        """
        if not skills_dir.is_dir():
            return
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            try:
                fm, _body = OpencodeBackend._parse_frontmatter(
                    skill_md.read_text(errors="replace")
                )
            except OSError:
                continue
            name = str(fm.get("name") or skill_md.parent.name).strip()
            if not name or name in agents:
                continue
            # MINIMAL prompt on purpose — do NOT preload the SKILL.md body.
            #
            # Registering a skill exists solely so `--agent <skill>` binds and the observed
            # agent matches the announced identity. Skill CONTENT is delivered by the caller:
            # inlined via the load_skill_context helper, or — for skills in
            # _PROMPT_MANAGED_SKILLS — by a slim purpose-built prompt that tells the model what
            # to read. Front-loading the body as a system prompt duplicates that and is
            # expensive: aog-self-critic's SKILL.md is ~150 KB, and preloading it pushed the
            # pre-spawn critic past its deliberately short 180 s opencode budget
            # (critic_invoke._default_prespawn_critic_timeout_sec). Measured in an end-to-end
            # run: critic timed out with no report, where the same call had completed before by
            # reading the file on demand.
            agents[name] = {
                "description": OpencodeBackend._neutralize_file_macro(
                    str(fm.get("description") or name).strip()
                ),
                "mode": "primary",
                "prompt": (
                    f"You are running the a5_ops skill {name!r}. Its instructions are "
                    f"either included in the prompt below or readable at {skill_md}. "
                    "Follow the orchestrator brief and return only what it asks for."
                ),
            }

    @staticmethod
    def _opencode_config_content(extra_agent: str | None = None) -> str | None:
        """Build the JSON passed via OPENCODE_CONFIG_CONTENT.

        Converts the plugin's CC-format `agents/*.md` into opencode `agent{}` entries with
        `mode: "primary"` and points skill discovery at the plugin's own skills dir.
        """
        root = OpencodeBackend._plugin_root()
        agents_dir = root / "agents"
        skills_dir = root / "skills"
        # opencode expands `{file:...}` macros inside config STRINGS. An agent body that
        # merely mentions `{file:line, before/after}` (aog-precision-probe.md:285) makes the
        # whole config invalid and opencode refuses to start. Referencing the body as
        # `{file:<abs>}` is the supported form, and the referenced file's own content is NOT
        # re-expanded (measured), so arbitrary prose stays safe.
        bodies_dir = OpencodeBackend._engine_root() / "workspace" / ".opencode-agents"
        if agents_dir.is_dir():
            try:
                bodies_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                bodies_dir = None
        agents = OpencodeBackend._agents_from_markdown(root, agents_dir, bodies_dir)
        OpencodeBackend._register_skill_agents(skills_dir, agents)

        # Any dispatch target that is neither an agent nor a skill of this plugin still has
        # to be BINDABLE, or the run silently degrades: `opencode run --agent <unknown>`
        # only warns and falls back to the default agent (exit code unaffected), while
        # build_env still announces AOG_HOOK_AGENT_TYPE=<unknown> — so the identity guard
        # then refuses every tool call of that run and the orchestrator, seeing exit 0,
        # files the refusal prose as a successful result. Live examples that hit this:
        # kb_auto_promote dispatching Claude Code's built-in "general-purpose", and
        # resume() dispatching the pseudo-target "resume".
        if extra_agent and extra_agent not in agents:
            agents[extra_agent] = {
                "description": f"a5_ops dispatch target {extra_agent}",
                "mode": "primary",
                "prompt": (
                    f"You are running as the a5_ops dispatch target {extra_agent!r}. "
                    "Follow the orchestrator brief supplied on stdin."
                ),
            }

        cfg: dict = {}
        if agents:
            cfg["agent"] = agents
        if skills_dir.is_dir():
            cfg["skills"] = {"paths": [str(skills_dir)]}
        # Safety-net adapter. Registering it here (rather than copying a file into the
        # user's config dir) keeps the install footprint at zero: the tuple form delivers
        # `options.projectRoot` to A5OpsHooksPlugin(ctx, options), which is how the adapter
        # locates the canonical Python checkers under engine/src/scripts/workflow/.
        engine = OpencodeBackend._engine_root()
        adapter = engine / "src" / "opencode" / "a5_ops_hooks.mjs"
        if adapter.is_file():
            cfg["plugin"] = [[f"file://{adapter}", {"projectRoot": str(engine)}]]
        if not cfg:
            return None
        return json.dumps(cfg)

    @staticmethod
    def _extract_workspace_from_prompt(prompt: str) -> str | None:
        patterns = (
            r"\bASCENDC_WORKSPACE:\s*(/[^\s`]+)",
            r"\bworkspace\s*[:=]\s*(/[^\s`]+)",
            r"\bworkspace\s+(/[^\s`]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1).rstrip(".,)")
        return None

    def _run_streaming(self, cmd: list, prompt: str, *, kind: str, mode: str, env: dict,
                       timeout: Optional[float], silence_timeout: Optional[int] = None,
                       progress_callback=None,
                       tee_path=None) -> Envelope:
        started = time.monotonic()
        state, timeout_sec, deadline, silence_timeout_sec = self._new_stream_state(
            started, timeout, silence_timeout
        )
        tee = None
        proc: subprocess.Popen | None = None
        sel: selectors.BaseSelector | None = None
        try:
            tee = self._open_tee(tee_path)
            proc, sel, stdout_fd = self._open_stream_process(cmd, prompt, env)
            timed_out, silence_timed_out = self._monitor_stream(
                proc,
                sel,
                stdout_fd,
                state,
                deadline=deadline,
                silence_timeout_sec=silence_timeout_sec,
                tee=tee,
                progress_callback=progress_callback,
            )
            if timed_out or silence_timed_out or state.invalid_tool_event:
                self._terminate_process_group(proc)
                return self._stream_failure_envelope(
                    cmd,
                    state,
                    proc,
                    started=started,
                    timeout_sec=timeout_sec,
                    silence_timeout_sec=silence_timeout_sec,
                    silence_timed_out=silence_timed_out,
                    kind=kind,
                    mode=mode,
                    tee=tee,
                )
            rc = proc.wait()
            return self._stream_success_envelope(cmd, state, rc, started, kind, mode)
        except FileNotFoundError as e:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"not_found": True, "stderr": str(e), "backend": self.name},
            )
        finally:
            self._close_stream_resources(sel, tee)

    def _new_stream_state(self, started: float, timeout: Optional[float], silence_timeout: Optional[int]
                          ) -> tuple[_StreamState, int | None, float | None, int | None]:
        timeout_sec = int(timeout) if timeout else None
        return (
            _StreamState(last_output_at=started, invalid_tool_limit=self._invalid_tool_limit()),
            timeout_sec,
            started + timeout_sec if timeout_sec else None,
            self._stream_silence_timeout(silence_timeout),
        )

    @staticmethod
    def _open_tee(tee_path):
        if not tee_path:
            return None
        tee_file = Path(tee_path)
        tee_file.parent.mkdir(parents=True, exist_ok=True)
        return tee_file.open("w")

    @staticmethod
    def _close_stream_resources(selector: selectors.BaseSelector | None, tee) -> None:
        if selector is not None:
            selector.close()
        if tee:
            tee.close()

    @staticmethod
    def _open_stream_process(cmd: list, prompt: str, env: dict) -> tuple[subprocess.Popen, selectors.BaseSelector, int]:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
            bufsize=0,
            start_new_session=True,
        )
        if proc.stdin is not None:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        selector = selectors.DefaultSelector()
        if proc.stdout is None:
            raise RuntimeError("opencode backend did not provide a stdout pipe")
        stdout_fd = proc.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        selector.register(proc.stdout, selectors.EVENT_READ)
        return proc, selector, stdout_fd

    def _monitor_stream(self, proc: subprocess.Popen, selector: selectors.BaseSelector, stdout_fd: int,
                        state: _StreamState, *, deadline: float | None, silence_timeout_sec: int | None,
                        tee, progress_callback) -> tuple[bool, bool]:
        while True:
            timed_out, silence_timed_out = self._stream_timeout_state(
                state.last_output_at, deadline, silence_timeout_sec
            )
            if timed_out or silence_timed_out:
                return timed_out, silence_timed_out
            if proc.poll() is not None:
                self._drain_stream_stdout(state, stdout_fd, tee, progress_callback)
                if state.pending and not state.invalid_tool_event:
                    self._record_stream_line(
                        state, state.pending.decode("utf-8", errors="replace"),
                        tee, progress_callback,
                    )
                    state.pending = b""
                return False, False
            if selector.select(self._stream_wait_interval(state.last_output_at, deadline, silence_timeout_sec)):
                self._drain_stream_stdout(state, stdout_fd, tee, progress_callback)
                if state.invalid_tool_event:
                    return False, False

    @staticmethod
    def _stream_timeout_state(last_output_at: float, deadline: float | None,
                              silence_timeout_sec: int | None) -> tuple[bool, bool]:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return True, False
        if silence_timeout_sec is not None and now - last_output_at >= silence_timeout_sec:
            return False, True
        return False, False

    @staticmethod
    def _stream_wait_interval(last_output_at: float, deadline: float | None,
                              silence_timeout_sec: int | None) -> float:
        wait = 0.5
        now = time.monotonic()
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - now))
        if silence_timeout_sec is not None:
            wait = min(wait, max(0.0, silence_timeout_sec - (now - last_output_at)))
        return wait

    def _drain_stream_stdout(self, state: _StreamState, stdout_fd: int, tee, progress_callback) -> None:
        while True:
            try:
                chunk = os.read(stdout_fd, 65536)
            except BlockingIOError:
                return
            if not chunk:
                return
            self._process_stream_chunk(state, chunk, tee, progress_callback)
            if state.invalid_tool_event:
                return

    def _process_stream_chunk(self, state: _StreamState, chunk: bytes, tee, progress_callback) -> None:
        # Buffer BYTES and decode only complete lines: a multi-byte UTF-8 sequence split
        # across two os.read() chunks would otherwise decode to U+FFFD on both sides and
        # silently corrupt (then drop) that JSON event line.
        state.pending += chunk
        while b"\n" in state.pending:
            raw_line, state.pending = state.pending.split(b"\n", 1)
            self._record_stream_line(
                state, raw_line.decode("utf-8", errors="replace") + "\n", tee, progress_callback
            )
            if state.invalid_tool_event:
                return

    def _record_stream_line(self, state: _StreamState, line: str, tee, progress_callback) -> None:
        state.last_output_at = time.monotonic()
        session_id, cost, turns, invalid = self._consume_stream_line(
            line, state.raw_lines, state.text_parts, tee, progress_callback
        )
        state.session_id = session_id or state.session_id
        state.total_cost = cost if cost is not None else state.total_cost
        state.num_turns = turns if turns is not None else state.num_turns
        if invalid:
            state.invalid_tool_count += 1
            state.invalid_tool_event = state.invalid_tool_count >= state.invalid_tool_limit

    def _stream_failure_envelope(self, cmd: list, state: _StreamState, proc: subprocess.Popen, *,
                                 started: float, timeout_sec: int | None, silence_timeout_sec: int | None,
                                 silence_timed_out: bool, kind: str, mode: str, tee) -> Envelope:
        duration_ms = int((time.monotonic() - started) * 1000)
        output_text = self._joined_output(state.text_parts)
        raw_tail = "".join(state.raw_lines)[-4000:]
        if tee:
            tee.write(
                json.dumps({
                    "type": "a5_ops_timeout" if not state.invalid_tool_event else "a5_ops_invalid_tool_event",
                    "timeout_sec": timeout_sec,
                    "silence_timeout_sec": silence_timeout_sec,
                    "invalid_tool_event": state.invalid_tool_event,
                    "partial_output_tail": output_text[-1000:],
                }) + "\n"
            )
        return Envelope(
            is_error=True,
            output_text=output_text,
            session_id=state.session_id,
            duration_ms=duration_ms,
            raw_envelope={
                "timed_out": True,
                "timeout_sec": timeout_sec,
                "silence_timed_out": silence_timed_out,
                "silence_timeout_sec": silence_timeout_sec,
                "invalid_tool_event": state.invalid_tool_event,
                "invalid_tool_limit": state.invalid_tool_limit,
                "backend": self.name,
                "cmd_kind": kind,
                "mode": mode,
                "event_format": "json",
                "returncode": proc.returncode,
                "stdout_tail": raw_tail,
                "cmd": self._redacted_cmd(cmd),
            },
        )

    def _stream_success_envelope(self, cmd: list, state: _StreamState, returncode: int, started: float,
                                 kind: str, mode: str) -> Envelope:
        output_text = self._joined_output(state.text_parts) or "".join(state.raw_lines)
        return Envelope(
            is_error=returncode != 0,
            output_text=output_text,
            session_id=state.session_id,
            total_cost_usd=state.total_cost,
            num_turns=state.num_turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            raw_envelope={
                "returncode": returncode,
                "stdout_tail": "".join(state.raw_lines)[-4000:],
                "cmd_kind": kind,
                "backend": self.name,
                "mode": mode,
                "event_format": "json",
                "cmd": self._redacted_cmd(cmd),
            },
        )

    def _consume_stream_line(self, line: str, raw_lines: list[str], text_parts: list[str],
                             tee, progress_callback) -> tuple[str | None, float | None, int | None, bool]:
        raw_lines.append(line)
        if len(raw_lines) > 1000:
            del raw_lines[: len(raw_lines) - 1000]
        if tee:
            tee.write(line)
            tee.flush()
        session_id = None
        total_cost = None
        num_turns = None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None, None, None, False
        session_id = event.get("sessionID") or event.get("session_id")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if event.get("type") == "text" or part.get("type") == "text":
            text = part.get("text") or event.get("text") or ""
            if text:
                text_parts.append(text)
                self._emit_text_progress(progress_callback, text)
        tool_name = (
            part.get("tool")
            or part.get("toolName")
            or part.get("name")
            or event.get("tool")
            or event.get("toolName")
        )
        if tool_name:
            tool_input = part.get("input") or part.get("args") or event.get("input") or {}
            self._emit_tool_progress(progress_callback, str(tool_name), tool_input)
        invalid_tool = str(tool_name).lower() == "invalid"
        if event.get("type") == "step_finish" or part.get("type") == "step-finish":
            if isinstance(part.get("cost"), (int, float)):
                total_cost = float(part["cost"])
            tokens = part.get("tokens")
            if isinstance(tokens, dict) and isinstance(tokens.get("total"), int):
                num_turns = tokens["total"]
        return session_id, total_cost, num_turns, invalid_tool

    @staticmethod
    def _stream_silence_timeout(silence_timeout: Optional[int]) -> int | None:
        value = silence_timeout
        if value is None:
            env_value = os.environ.get("AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC")
            if env_value:
                try:
                    value = int(env_value)
                except ValueError:
                    value = None
        if value is None or value <= 0:
            return None
        return int(value)

    @staticmethod
    def _invalid_tool_limit() -> int:
        raw = os.environ.get("AOG_OPENCODE_INVALID_TOOL_LIMIT", "1")
        try:
            value = int(raw)
        except ValueError:
            value = 1
        return max(1, value)

    @staticmethod
    def _joined_output(parts: list[str]) -> str:
        return "\n".join(part.strip("\n") for part in parts if part is not None)

    @staticmethod
    def _emit_text_progress(progress_callback, text: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    @staticmethod
    def _emit_tool_progress(progress_callback, name: str, tool_input: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input or {}}]},
            })
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            proc.wait(timeout=5)

    @staticmethod
    def _redacted_cmd(cmd: list) -> list:
        return [str(part) for part in cmd]

    def _format_prompt(self, target: str, prompt: str, *, kind: str) -> str:
        skill_context = load_skill_context(target) if kind == "skill" else None
        skill_block = f"\n\n{skill_context}" if skill_context else ""
        # No semantic guidance here: canonical KB rules are shared by both harnesses, and
        # backends/base.py forbids a backend owning semantics.
        return (
            f"You are running as the a5_ops harness backend target {target!r} "
            f"(kind={kind!r}) through opencode CLI.\n\n"
            "Follow the a5_ops orchestrator brief below. Return only the final "
            "worker/skill result text expected by the orchestrator."
            f"{skill_block}\n\n"
            "----- a5_ops prompt -----\n"
            f"{prompt}"
        )

    def normalize(self, raw: Any) -> Envelope:
        if isinstance(raw, Envelope):
            return raw
        if isinstance(raw, dict):
            return Envelope(
                is_error=bool(raw.get("is_error")),
                output_text=raw.get("result") or raw.get("output_text") or "",
                api_error_status=raw.get("api_error_status"),
                session_id=raw.get("session_id"),
                raw_envelope=raw,
            )
        return Envelope(is_error=False, output_text=str(raw), raw_envelope={"backend": self.name})

    def format_agent(self, agent_def: dict) -> dict:
        rendered = dict(agent_def)
        rendered["harness_backend"] = self.name
        return rendered

    def wire_safety(self, checkers: list) -> dict:
        plugin_path = Path(__file__).resolve().parents[3] / "opencode" / "a5_ops_hooks.mjs"
        return {
            "backend": self.name,
            "kind": "host-hook-plugin",
            "plugin_path": str(plugin_path),
            "events": [
                "tool.execute.before",
                "tool.execute.after",
                "permission.ask",
            ],
            "checkers": list(checkers or []),
        }

    def resume(self, session_id: str, prompt: str) -> Envelope:
        return self.dispatch("resume", prompt, kind="resume", session=session_id)

    def identify_cmd(self, cmd: str) -> bool:
        return any(marker in cmd for marker in _OPENCODE_RUN_MARKERS)

    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        m = _OP_SLUG_RE.search(cmd)
        return m.group(1) if m else None

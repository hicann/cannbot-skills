// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// a5_ops opencode host-hook adapter.
//
// opencode exposes plugin hooks such as permission.ask and
// tool.execute.before/after. This file maps those events into the
// Claude-Code-style JSON payload consumed by the existing a5_ops hook scripts.
// The canonical check logic stays in Python.

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TOOL_MAP = new Map([
  ["agent", "Agent"],
  ["task", "Agent"],
  ["bash", "Bash"],
  ["shell", "Bash"],
  ["edit", "Edit"],
  ["write", "Write"],
  ["multiedit", "MultiEdit"],
  ["read", "Read"],
  ["grep", "Grep"],
  ["glob", "Glob"],
  ["webfetch", "WebFetch"],
  // --- opencode 1.18.18 tools that must reach the same canonical guards --------------
  // Verified against the installed binary before mapping: `websearch` occurs 220 times,
  // `codesearch` ZERO. A mapping for a tool the harness does not have is not coverage — it
  // is an untestable claim (the test for it had to invent its own argument shape), and it
  // hides the fact that an unmapped tool would be refused by the unknown-tool default
  // anyway, which is the safe direction. Map only what the harness actually exposes.
  ["websearch", "WebFetch"],
]);

// Tools that carry no file/command payload and therefore have nothing for the canonical
// guards to judge. Listed EXPLICITLY so that "not dangerous" is a decision on record
// rather than the accidental result of a missing map entry.
// Tools and permission prompts with no file, shell, or network side effect. They reach the
// canonical guards under names those guards do not know, so without this they would hit the
// default-deny below and break ordinary sessions.
//
// Measured against the installed opencode 1.18.18 rather than assumed — `opencode debug agent`
// reports exactly: bash, edit, glob, grep, invalid, question, read, skill, task, todowrite,
// webfetch, write. An earlier version of this list carried `todoread` and `lsp`, neither of
// which exists in that toolset, while omitting `question`, which does — so the real tool was
// denied and the two dead entries covered nothing. A name that the harness does not have is
// not coverage, it is an untestable claim.
const BENIGN_TOOLS = new Set(["todowrite", "skill", "invalid", "question"]);

// Permission prompts, which arrive through `permission.ask` named by PERMISSION rather than by
// tool (`external_directory` is the one that maps onto a real tool and is handled separately).
// These three are opencode's own session controls — the runaway-loop breaker and the two plan
// mode transitions — so denying them would wedge a session over a prompt that touches nothing.
const BENIGN_PERMISSIONS = new Set(["doom_loop", "plan_enter", "plan_exit"]);

// Write-class tools opencode may expose that this adapter cannot faithfully normalise into
// the {file_path, content} shape the canonical checkers expect. `apply_patch` is the live
// example: it carries a multi-file patch blob, and opencode REMOVES `edit`/`write` from the
// toolset when it enables apply_patch (gpt-class models), so an unmapped apply_patch would
// silently become the only write path — with every write guard bypassed. Refuse instead of
// guessing: a wrong path extraction would hand the guards the wrong file and "pass".
const UNSUPPORTED_WRITE_TOOLS = new Set(["apply_patch", "applypatch", "patch"]);

// The Claude-Code tool names the canonical guards know how to judge. Used to accept a
// permission event whose PERMISSION name has already been mapped onto one of these.
const KNOWN_GUARDED_TOOLS = new Set([...TOOL_MAP.values()]);

const AUTO_ALLOW_AFTER_GUARD = new Set([
  "Agent",
  "Bash",
  "Edit",
  "Write",
  "MultiEdit",
  "Read",
  "Grep",
  "Glob",
  "WebFetch",
]);

// ---- agent identity -------------------------------------------------------------------
// opencode's tool hooks carry only {tool, sessionID, callID} — no agent identity
// (@opencode-ai/plugin index.d.ts). Identity therefore has to come from `chat.params`,
// which DOES carry an authoritative `agent` keyed by sessionID, and is emitted before the
// turn's first tool call (measured on 1.18.18).
//
// Two traps this encodes:
//  1. opencode runs internal utility agents (title/summary/compaction) inside the SAME
//     sessionID, concurrently. A naive last-writer-wins can therefore attribute a worker's
//     tool call to `title`, which is not an aog-* agent, so the dispatcher would select an
//     empty gate set and ALLOW. Utility agents are filtered out of identity resolution.
//  2. Process env alone cannot identify a sub-agent: every task-tool child of one process
//     shares it. Env is used only to establish "this process is a plugin-driven run".
const UTILITY_AGENTS = new Set(["title", "summary", "compaction"]);
const SESSION_AGENTS = new Map();

function recordSessionAgent(sessionID, agent) {
  if (!sessionID || !agent) return;
  if (UTILITY_AGENTS.has(String(agent))) return;
  SESSION_AGENTS.set(String(sessionID), String(agent));
}

// Returns {agentType, agentId, unresolved}. `unresolved` is true only for a plugin-driven
// run whose identity could not be confirmed — callers must fail CLOSED on it rather than
// fall back to the empty-identity path, which the canonical read guard treats as the main
// agent and allows.
function resolveIdentity(sessionID) {
  const dispatched = process.env.AOG_HOOK_AGENT_TYPE || "";
  const observed = sessionID ? SESSION_AGENTS.get(String(sessionID)) || "" : "";
  if (!dispatched) {
    // Not a plugin dispatch (interactive opencode, or another project's session).
    // Do not impose this plugin's gates on an unrelated session.
    return { agentType: "", agentId: "", unresolved: false };
  }
  if (observed && observed !== dispatched) {
    // Two independent signals disagree about who is executing. The classic cause is
    // `run --agent <x>` silently falling back to opencode's default agent while our env
    // still claims <x>: gates would judge as <x> while something else runs. Refuse.
    return { agentType: observed, agentId: "", unresolved: true };
  }
  // Under Path A one `opencode run` process serves exactly one dispatched agent, so the
  // env label is a sound baseline. `chat.params` is used to CONTRADICT it, not to license
  // it — requiring an observation would make enforcement depend on hook ordering.
  return {
    agentType: dispatched,
    agentId: process.env.AOG_HOOK_AGENT_ID || `opencode:${dispatched}`,
    unresolved: false,
  };
}

// A NAMESPACED name (`mcp__<server>__bash`, `some.plugin.write`) is not the builtin it happens
// to end with, and must not inherit its identity. Matching on the leaf segment used to do
// exactly that: an MCP tool called `mcp__x__bash` became `Bash`, the guards then inspected an
// argument shape they do not understand (that server names its command field whatever it
// likes), found no command to judge, and allowed the call — while `assertToolIsGuardable` saw
// a KNOWN_GUARDED_TOOLS name and stood down. Guard-SHAPED but content-blind is strictly worse
// than unknown, because unknown fails closed in plugin-dispatched runs. So namespaced names
// are passed through verbatim and land on that default-deny path.
function titleToolName(raw) {
  const s = String(raw || "");
  if (isNamespacedToolName(s)) return s;
  return TOOL_MAP.get(s.toLowerCase()) || s;
}

function isNamespacedToolName(raw) {
  const s = String(raw || "");
  return s.includes("__") || s.includes(".");
}

function titlePermissionName(raw) {
  const s = String(raw || "");
  if (s === "external_directory") return "Read";
  return titleToolName(s);
}

function normalizeToolInput(toolName, args) {
  const input = args && typeof args === "object" ? { ...args } : {};
  if (toolName === "Bash" && input.command == null) {
    input.command = input.cmd || input.shell || "";
  }
  if ((toolName === "Edit" || toolName === "Write" || toolName === "MultiEdit") && input.file_path == null) {
    input.file_path = input.filePath || input.filepath || input.path || input.file || "";
  }
  // Read-class content tools must surface the path they touch under BOTH `file_path` and
  // `path`, because the canonical output guard inspects whichever field the Claude-Code
  // tool would have supplied and opencode names it differently per tool (filePath/path).
  // Grep is included because `codesearch` normalises to it and reads file CONTENT — if its
  // target never reaches the payload the archive guard sees nothing to judge and allows it.
  //
  // Glob is deliberately NOT included: its `pattern` is a match expression, not a path, and
  // copying it into `path` makes the inline guard read a project-wide glob as a scoped one
  // (it distinguishes them precisely by `path` being empty).
  if (toolName === "Read" || toolName === "Grep") {
    if (input.file_path == null) {
      // `pattern` is deliberately NOT a fallback, for the reason the Glob branch below spells
      // out: it is a match expression, not a path. Copying it here made a grep for a string
      // that merely CONTAINS "workspace/" or "output/" read as a scoped path and get refused,
      // while the case those rules are actually for — a grep with no `path`, i.e. the whole
      // project — still slipped past, because the rules were then judging the regex instead of
      // the absent scope.
      input.file_path = input.filePath || input.filepath || input.path || input.file || "";
    }
    if (input.path == null && input.file_path != null) {
      input.path = input.file_path;
    }
  }
  if ((toolName === "Grep" || toolName === "Glob") && input.path == null && input.file_path != null) {
    input.path = input.file_path;
  }
  return input;
}

function toolArgs(input, output) {
  const candidates = [
    output && output.args,
    input && input.args,
    input && input.input,
    input && input.toolInput,
    input && input.parameters,
    input && input.metadata,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === "object") {
      return candidate;
    }
  }
  return {};
}

function normalizePermissionInput(toolName, input) {
  const metadata = input && typeof input.metadata === "object" ? { ...input.metadata } : {};
  if (input && input.pattern != null && metadata.pattern == null) {
    metadata.pattern = input.pattern;
  }
  if (toolName === "Bash" && metadata.command == null) {
    metadata.command = metadata.cmd || metadata.shell || "";
  }
  if ((toolName === "Edit" || toolName === "Write" || toolName === "MultiEdit") && metadata.file_path == null) {
    metadata.file_path = metadata.filePath || metadata.filepath || metadata.path || metadata.file || "";
  }
  if ((toolName === "Read" || toolName === "Grep" || toolName === "Glob") && metadata.path == null) {
    metadata.path = metadata.filePath || metadata.filepath || metadata.file_path || metadata.parentDir || metadata.pattern || "";
  }
  return normalizeToolInput(toolName, metadata);
}

function runChecker(projectRoot, scriptRel, payload, timeoutMs) {
  const script = path.join(projectRoot, scriptRel);
  const env = {
    ...process.env,
    CLAUDE_PROJECT_DIR: projectRoot,
    AOG_HARNESS_BACKEND: "opencode",
  };
  const res = spawnSync("python3", [script], {
    input: JSON.stringify(payload),
    encoding: "utf8",
    timeout: timeoutMs,
    env,
  });
  // `status === null` means the child never exited normally — killed by a signal, which for
  // this call is overwhelmingly the `timeout` above (SIGTERM). `res.error` is NOT set in that
  // case, and `res.status && …` reads null as falsy, so a checker that was killed before it
  // could object used to be indistinguishable from one that approved. runDoor already fails
  // closed on this; the checker path did not.
  if (res.error || res.status === null) {
    const why = res.error ? res.error.message : `killed by signal ${res.signal || "unknown"}`;
    throw new Error(`[a5_ops opencode hook] ${scriptRel} failed: ${why}`);
  }
  if (res.status !== 0) {
    const stderr = (res.stderr || "").trim();
    const stdout = (res.stdout || "").trim();
    throw new Error(stderr || stdout || `[a5_ops opencode hook] ${scriptRel} exited ${res.status}`);
  }
}

// Default-deny for tools this adapter does not understand.
//
// The map above is a whitelist and `titleToolName` passes unmapped names through verbatim,
// so before this check any tool absent from the map matched nothing in runGuardSet and ran
// COMPLETELY unguarded. That turns "upstream added a tool" into a silent hole in the
// anti-cheating layer, and opencode's toolset is model-dependent (apply_patch appears only
// for gpt-class models), so the hole opens based on which model the operator picks.
//
// Only plugin-dispatched runs fail closed: an unrelated interactive opencode session must
// not be crippled by this plugin's presence.
function assertToolIsGuardable(payload, rawTool) {
  if (!process.env.AOG_HOOK_AGENT_TYPE) return;
  // Permission events are named by PERMISSION, not by tool (`external_directory`, …), and
  // titlePermissionName has already mapped them onto a Claude-Code tool name. If that
  // mapping produced a name the guards understand, the call is guardable — re-deriving from
  // the raw permission string would classify every permission as an unknown tool and deny
  // it, which broke the two permission-path tests when this check was first added here.
  if (KNOWN_GUARDED_TOOLS.has(payload.tool_name)) return;
  // Both allowlists below are keyed by BUILTIN name, so they may only be consulted for a name
  // that actually is one. A namespaced tool that merely ends in a benign leaf
  // (`mcp__x__todoread`) is a different tool from a different vendor and gets no free pass —
  // it falls through to the default-deny throw at the end.
  const raw = String(rawTool || "");
  const leaf = raw.toLowerCase();
  if (isNamespacedToolName(raw)) {
    throw new Error(
      `[a5_ops opencode hook] tool guard blocked namespaced tool '${raw}': this adapter can ` +
        "only normalise opencode's builtin tools for the canonical guards, so an MCP/plugin " +
        "tool would run unguarded inside a plugin-dispatched run.",
    );
  }
  if (BENIGN_TOOLS.has(leaf) || BENIGN_PERMISSIONS.has(leaf)) return;
  if (UNSUPPORTED_WRITE_TOOLS.has(leaf)) {
    throw new Error(
      `[a5_ops opencode hook] tool guard blocked '${leaf}': its multi-file payload cannot be ` +
        "normalised for the canonical write guards, and opencode drops edit/write when it is " +
        "enabled. Use edit/write (select a model whose toolset provides them).",
    );
  }
  if (!TOOL_MAP.has(leaf)) {
    throw new Error(
      `[a5_ops opencode hook] tool guard blocked unknown tool '${leaf}': this adapter has no ` +
        "mapping for it, so the canonical guards would not see it. Add an explicit mapping " +
        "(or list it as benign) before allowing it in a guarded run.",
    );
  }
}

// The JS↔Python door. Policy decisions are made in Python (src/opencode/door.py), next to
// the canonical checkers; this file only translates events and enforces the verdict.
//
// Before this existed, the access / build-artifact / generated-code rules were implemented
// HERE in JavaScript with their own regexes. That broke the boundary invariant in
// backends/base.py (a backend WIRES, it does not own gate logic) and made denials
// indistinguishable from canonical ones — a difference that hid a real bug: an O0 liveness
// canary stayed green with the canonical Python guard deleted, because the JS copy was
// answering instead.
//
// Fail CLOSED: unlike the autoresearch door (which safe-allows on internal error because it
// nudges workflow phases), this door guards an anti-cheating boundary, so "we could not
// decide" must mean refuse.
function runDoor(projectRoot, payload) {
  // The door ships WITH this adapter, so resolve it from this module's own location.
  // `projectRoot` points at the engine whose canonical checkers runChecker() invokes — a
  // different thing, and not necessarily a tree that contains the adapter bundle.
  const doorPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "door.py");
  if (!fs.existsSync(doorPath)) {
    throw new Error(
      `[a5_ops opencode hook] policy door missing at ${doorPath}; refusing rather than ` +
        "running the tool call unjudged",
    );
  }
  // project_root travels INSIDE the payload: the cross-workspace and project-wide-glob
  // rules are relative to the engine root, which the door cannot infer from cwd alone.
  const encoded = Buffer.from(
    JSON.stringify({ ...payload, project_root: projectRoot }),
    "utf8",
  ).toString("base64");
  const res = spawnSync("python3", [doorPath, "check", encoded], {
    cwd: projectRoot,
    encoding: "utf8",
    timeout: 30000,
    env: process.env,
  });
  if (res.error || res.status === null) {
    throw new Error(
      `[a5_ops opencode hook] policy door did not complete (${res.error || "killed/timeout"}); ` +
        "refusing rather than allowing an unjudged tool call",
    );
  }
  let verdict;
  try {
    verdict = JSON.parse(String(res.stdout || "").trim().split("\n").pop());
  } catch {
    throw new Error(
      "[a5_ops opencode hook] policy door returned an unreadable verdict: " +
        String(res.stderr || res.stdout || "").slice(0, 300),
    );
  }
  if (verdict && verdict.blocked) {
    throw new Error(verdict.reason || "[a5_ops opencode hook] blocked by policy door");
  }
}

function runGuardSet(projectRoot, payload) {
  runDoor(projectRoot, payload);
  const tool = payload.tool_name;
  if (["Agent", "Edit", "Write", "MultiEdit", "Bash", "WebFetch"].includes(tool)) {
    runChecker(projectRoot, "src/scripts/workflow/workflow_critic.py", payload, 30000);
  }
  if (["Read", "Grep", "Glob", "Bash"].includes(tool)) {
    runChecker(projectRoot, "src/scripts/workflow/output_read_guard.py", payload, 10000);
  }
  if (tool.startsWith("mcp__plugin_discord_discord__")) {
    runChecker(projectRoot, "src/scripts/workflow/ship_claim_audit.py", payload, 5000);
  }
}

function hookPayload(hookEventName, input, output) {
  const toolName = titleToolName(input.tool || input.type || input.permission || output.tool);
  const toolInput = normalizeToolInput(toolName, toolArgs(input, output));
  const ident = resolveIdentity(input.sessionID);
  return {
    hook_event_name: hookEventName,
    tool_name: toolName,
    tool_input: toolInput,
    session_id: input.sessionID,
    call_id: input.callID,
    agent_id: ident.agentId,
    agent_type: ident.agentType,
    __identity_unresolved: ident.unresolved,
    cwd: process.cwd(),
  };
}

// A plugin-driven run whose executing agent could not be confirmed must not reach the
// canonical guards with an empty agent_id: that is the main-agent path, which allows
// answer-bearing reads. Refuse the tool call instead.
function assertIdentityResolved(payload) {
  if (payload.__identity_unresolved) {
    throw new Error(
      "[a5_ops opencode hook] identity guard blocked " +
        `${payload.tool_name}: this is a plugin-dispatched run (expected ` +
        `'${process.env.AOG_HOOK_AGENT_TYPE}') but opencode has not reported a matching ` +
        "executing agent for this session; refusing rather than running unguarded",
    );
  }
}

function permissionPayload(input) {
  const toolName = titlePermissionName(input.type || input.permission);
  // Resolve identity the same way tool.execute.before does. Reading the env directly here
  // meant this path never set __identity_unresolved, so assertIdentityResolved() was a
  // no-op on it — and this is the branch that AUTO-ALLOWS after the guards pass, i.e. the
  // one place where a contradicted identity (`run --agent X` silently falling back to the
  // default agent) is most costly. It also skipped resolveIdentity's `opencode:<agent>`
  // fallback, so a missing AOG_HOOK_AGENT_ID produced an EMPTY agent_id — which the
  // canonical read guard treats as the main agent and allows.
  const ident = resolveIdentity(input.sessionID);
  return {
    hook_event_name: "PreToolUse",
    tool_name: toolName,
    tool_input: normalizePermissionInput(toolName, input),
    session_id: input.sessionID,
    call_id: input.callID,
    agent_id: ident.agentId,
    agent_type: ident.agentType,
    __identity_unresolved: ident.unresolved,
    cwd: process.cwd(),
  };
}

export async function A5OpsHooksPlugin(ctx, options = {}) {
  const projectRoot = path.resolve(String(options.projectRoot || process.env.AOG_PROJECT_ROOT || ctx.directory));
  return {
    // Identity source. opencode emits this per LLM turn with an authoritative `agent`,
    // keyed by sessionID; the tool hooks carry no identity of their own.
    async "chat.params"(input) {
      recordSessionAgent(input && input.sessionID, input && input.agent);
    },
    async "chat.message"(input) {
      recordSessionAgent(input && input.sessionID, input && input.agent);
    },

    async "permission.ask"(input, output) {
      try {
        const payload = permissionPayload(input);
        assertIdentityResolved(payload);
        // Same default-deny as the tool path: an unmapped tool reaching the guards under a
        // name they do not understand matches nothing and would then be AUTO-ALLOWED below.
        assertToolIsGuardable(payload, input.type || input.permission);
        runGuardSet(projectRoot, payload);
        if (AUTO_ALLOW_AFTER_GUARD.has(payload.tool_name)) {
          output.status = "allow";
        }
      } catch (err) {
        output.status = "deny";
        output.message = err instanceof Error ? err.message : String(err);
      }
    },

    async "tool.execute.before"(input, output) {
      const payload = hookPayload("PreToolUse", input, output);
      assertIdentityResolved(payload);
      assertToolIsGuardable(payload, input && input.tool);
      runGuardSet(projectRoot, payload);
    },

    async "tool.execute.after"(input, output) {
      const payload = hookPayload("PostToolUse", input, { args: output.args || {} });
      if (payload.tool_name === "Agent") {
        runChecker(projectRoot, "src/scripts/workflow/workflow_critic.py", payload, 30000);
      }
    },
  };
}

export default A5OpsHooksPlugin;

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Agent-identity contract for the opencode hook adapter.

opencode's tool hooks carry only ``{tool, sessionID, callID}`` — no agent identity — while
``chat.params`` does carry an authoritative ``agent`` keyed by sessionID. These tests pin the
resolution rules that keep the canonical guards from being handed the wrong identity:

* a plugin dispatch whose observed agent CONTRADICTS the dispatched one is refused, because
  ``run --agent`` silently falls back to opencode's default agent and the gates would then
  judge as the worker while something else executes;
* opencode's internal utility agents (title/summary/compaction) share the caller's sessionID
  and must never be mistaken for the executing agent;
* a session that is NOT a plugin dispatch keeps an empty identity so this plugin's gates are
  not imposed on an unrelated opencode session.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER = REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"


def _resolve(dispatched: str, observed_events: list[tuple[str, str]], session: str = "ses_1") -> dict:
    """Feed chat.params events, then report what tool.execute.before would carry."""
    script = f"""
import {{ A5OpsHooksPlugin }} from {json.dumps(str(ADAPTER))};
const hooks = await A5OpsHooksPlugin({{ directory: process.cwd() }}, {{ projectRoot: process.cwd() }});
for (const [sid, agent] of {json.dumps(observed_events)}) {{
  await hooks["chat.params"]({{ sessionID: sid, agent }});
}}
let verdict = {{ blocked: false, message: "" }};
try {{
  await hooks["tool.execute.before"](
    {{ tool: "read", sessionID: {json.dumps(session)}, callID: "c1" }},
    {{ args: {{ filePath: "/tmp/does-not-matter" }} }},
  );
}} catch (e) {{
  verdict = {{ blocked: true, message: String(e.message || e) }};
}}
console.log(JSON.stringify(verdict));
"""
    # Same runtime order the installer and phase O0 use: opencode ships as a
    # bun-compiled binary, so a machine can have bun and no node. Hard-coding node plus a
    # restricted PATH made these tests fail on nvm-only / bun-only machines.
    runtime = shutil.which("node") or shutil.which("bun")
    assert runtime, "neither node nor bun found — cannot drive the adapter"
    env = dict(os.environ)
    env["AOG_HOOK_AGENT_TYPE"] = dispatched
    env["AOG_HOOK_AGENT_ID"] = f"opencode:{dispatched}" if dispatched else ""
    proc = subprocess.run(
        [runtime, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_contradicting_observed_agent_is_refused():
    """`run --agent X` falling back to opencode's default must not run unguarded."""
    verdict = _resolve("aog-kernel-worker", [["ses_1", "build"]])
    assert verdict["blocked"], "a contradicted identity must fail closed, not proceed"
    assert "identity guard" in verdict["message"]


def test_matching_observed_agent_is_accepted():
    verdict = _resolve("aog-kernel-worker", [["ses_1", "aog-kernel-worker"]])
    assert not verdict["blocked"], verdict["message"]


def test_utility_agents_do_not_overwrite_the_executing_agent():
    """title/summary/compaction share the sessionID; they must not become the identity."""
    for utility in ("title", "summary", "compaction"):
        verdict = _resolve(
            "aog-kernel-worker",
            [["ses_1", "aog-kernel-worker"], ["ses_1", utility]],
        )
        assert not verdict["blocked"], f"{utility} wrongly displaced the worker identity"


def test_utility_agent_alone_does_not_contradict_dispatch():
    """A turn that has only produced a utility observation must not hard-fail the worker."""
    verdict = _resolve("aog-kernel-worker", [["ses_1", "title"]])
    assert not verdict["blocked"], verdict["message"]


def test_non_dispatch_session_keeps_empty_identity():
    """An interactive opencode session must not inherit this plugin's gates."""
    verdict = _resolve("", [["ses_1", "build"]])
    assert not verdict["blocked"], verdict["message"]


def _permission(dispatched: str, observed_events, perm_type: str = "read", session: str = "ses_1") -> dict:
    """Drive the permission.ask branch — the one that AUTO-ALLOWS after the guards pass."""
    script = f"""
import {{ A5OpsHooksPlugin }} from {json.dumps(str(ADAPTER))};
const hooks = await A5OpsHooksPlugin({{ directory: process.cwd() }}, {{ projectRoot: process.cwd() }});
for (const [sid, agent] of {json.dumps(observed_events)}) {{
  await hooks["chat.params"]({{ sessionID: sid, agent }});
}}
const output = {{ status: "ask" }};
await hooks["permission.ask"](
  {{ type: {json.dumps(perm_type)}, sessionID: {json.dumps(session)}, callID: "c1",
     filePath: "/tmp/does-not-matter" }},
  output,
);
console.log(JSON.stringify(output));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "AOG_HOOK_AGENT_TYPE": dispatched,
             "AOG_HOOK_AGENT_ID": f"opencode:{dispatched}" if dispatched else ""},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_permission_path_enforces_identity_too():
    """A contradicted identity must not reach the auto-allow branch.

    permission.ask read the env directly instead of resolving identity, so it never set the
    unresolved flag and assertIdentityResolved was a no-op there — on the very branch that
    sets status="allow" once the guards pass.
    """
    denied = _permission("aog-kernel-worker", [["ses_1", "build"]])
    assert denied["status"] == "deny", f"contradicted identity was allowed: {denied}"


def test_permission_path_allows_a_consistent_identity():
    """Paired control: an always-deny permission hook would satisfy the test above."""
    ok = _permission("aog-kernel-worker", [["ses_1", "aog-kernel-worker"]])
    assert ok["status"] == "allow", f"consistent identity was refused: {ok}"

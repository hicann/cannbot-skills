# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tool-coverage contract for the opencode hook adapter.

The adapter maps opencode tool names onto the Claude-Code names the canonical checkers
understand. That map is a whitelist, and unmapped names used to pass through verbatim and
match nothing in the guard router — so any tool the map did not know about ran completely
unguarded. opencode's toolset is model-dependent (``apply_patch`` is offered to gpt-class
models and, when offered, opencode REMOVES ``edit``/``write``), which means the hole could
open purely because of which model an operator selected.

These tests pin the default-deny behaviour, and are written so they fail if someone widens
the map without deciding what the new tool means for the guards.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER = REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"


def _call(tool: str, *, dispatched: str = "aog-kernel-worker", args: dict | None = None) -> dict:
    script = f"""
import {{ A5OpsHooksPlugin }} from {json.dumps(str(ADAPTER))};
const hooks = await A5OpsHooksPlugin({{ directory: process.cwd() }}, {{ projectRoot: process.cwd() }});
let verdict = {{ blocked: false, message: "" }};
try {{
  await hooks["tool.execute.before"](
    {{ tool: {json.dumps(tool)}, sessionID: "s1", callID: "c1" }},
    {{ args: {json.dumps(args or {"filePath": "/tmp/plain.txt"})} }},
  );
}} catch (e) {{ verdict = {{ blocked: true, message: String(e.message || e) }}; }}
console.log(JSON.stringify(verdict));
"""
    runtime = shutil.which("node") or shutil.which("bun")
    assert runtime, "neither node nor bun found — cannot drive the adapter"
    env = dict(os.environ)
    if dispatched:
        env["AOG_HOOK_AGENT_TYPE"] = dispatched
        env["AOG_HOOK_AGENT_ID"] = f"opencode:{dispatched}"
    proc = subprocess.run(
        [runtime, "--input-type=module", "-e", script],
        cwd=REPO_ROOT, text=True, capture_output=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_unknown_tool_fails_closed():
    """A tool the adapter has no mapping for must not reach the model unguarded."""
    verdict = _call("some_future_tool")
    assert verdict["blocked"], "unmapped tool ran unguarded"
    assert "unknown tool" in verdict["message"]


def test_apply_patch_is_refused_rather_than_guessed():
    """apply_patch displaces edit/write on gpt-class models; guessing its paths is unsafe."""
    verdict = _call("apply_patch", args={"patch": "*** Begin Patch\n*** End Patch\n"})
    assert verdict["blocked"], "apply_patch would have become an unguarded write path"
    assert "apply_patch" in verdict["message"]


def test_map_only_claims_tools_the_harness_actually_has():
    """A mapping for a tool opencode does not expose is an untestable claim.

    `codesearch` was mapped to the read class and covered by a test that invented its own
    argument shape — it occurs ZERO times in the installed 1.18.18 binary. An absent tool
    needs no mapping: the unknown-tool default already refuses it, which is the safe
    direction. `websearch` (220 occurrences) is real and stays mapped.
    """
    adapter_src = ADAPTER.read_text()
    assert '["websearch", "WebFetch"]' in adapter_src
    assert '["codesearch"' not in adapter_src, (
        "mapping re-added for a tool the harness does not expose; verify against the "
        "installed binary before claiming coverage"
    )
    # An absent tool must still be refused, not silently allowed.
    verdict = _call("codesearch", args={"filePath": "/tmp/x.txt"})
    assert verdict["blocked"] and "unknown tool" in verdict["message"]


@pytest.mark.parametrize("tool", ["todowrite", "skill", "invalid", "question"])
def test_benign_tools_are_allowed(tool):
    """Side-effect-free builtins must not be caught by the default-deny.

    The list is taken from what `opencode debug agent` reports on the installed 1.18.18 —
    bash, edit, glob, grep, invalid, question, read, skill, task, todowrite, webfetch, write —
    not from memory. The earlier version of this test asserted `lsp`, which is not in that
    toolset at all, so it proved nothing while `question`, which IS, went unlisted and was
    denied on sight. A benign-list entry for a tool the harness does not have cannot fail, and
    a test that only exercises such entries cannot either.
    """
    verdict = _call(tool, args={})
    assert not verdict["blocked"], f"{tool} should be allowed: {verdict['message']}"


@pytest.mark.parametrize("permission", ["doom_loop", "plan_enter", "plan_exit"])
def test_benign_permissions_are_not_denied(permission):
    """opencode's own session controls arrive by PERMISSION name and touch nothing.

    They reach the guards under names the canonical rules do not know, so the default-deny
    swallowed them — which wedges a session on a prompt that has no file, shell, or network
    effect. `doom_loop` is the runaway-loop breaker; denying it removes the escape hatch from
    exactly the situation it exists for.
    """
    script = f"""
import {{ A5OpsHooksPlugin }} from {json.dumps(str(ADAPTER))};
const hooks = await A5OpsHooksPlugin({{ directory: process.cwd() }}, {{ projectRoot: process.cwd() }});
const output = {{ status: "ask", message: "" }};
await hooks["permission.ask"]({{ type: {json.dumps(permission)}, sessionID: "s1" }}, output);
console.log(JSON.stringify(output));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "AOG_HOOK_AGENT_TYPE": "aog-kernel-worker",
             "AOG_HOOK_AGENT_ID": "opencode:aog-kernel-worker"},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    output = json.loads(proc.stdout.strip().splitlines()[-1])
    assert output["status"] != "deny", (
        f"{permission} was denied: {output.get('message', '')}"
    )


def test_non_dispatch_session_is_not_crippled():
    """An unrelated interactive opencode session must keep working."""
    verdict = _call("some_future_tool", dispatched="")
    assert not verdict["blocked"], verdict["message"]


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__evil__bash",   # leaf collides with the builtin shell
        "mcp__evil__read",
        "mcp__evil__write",
        "vendor.plugin.edit",
    ],
)
def test_namespaced_tool_does_not_inherit_a_builtin_identity(tool):
    """An MCP/plugin tool is not the builtin its name happens to end with.

    Matching on the trailing segment used to hand `mcp__evil__bash` the identity `Bash`. The
    guards then inspected an argument shape they do not understand — that server names its
    command field whatever it likes — found nothing to judge, and allowed the call, while the
    default-deny check stood down because it saw a known guarded name. Guard-SHAPED but
    content-blind is worse than unknown, because unknown fails closed.
    """
    verdict = _call(tool, args={"cmd": "cat workspace/other_op/verification.json"})

    assert verdict["blocked"], f"{tool} ran unguarded under a builtin's identity"
    assert "namespaced tool" in verdict["message"]

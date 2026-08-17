# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for src/opencode/door.py — the opencode adapter's JS->Python door.

The door carries three guards relocated verbatim from ``src/opencode/a5_ops_hooks.mjs``
(``runInlineAccessGuard`` / ``runBuildArtifactGuard`` / ``runInlineGeneratedCodeGuard``). These
tests pin the properties a relocation must not lose:

* **Each guard still BLOCKS what it blocked** — with the message wording unchanged, because
  those strings are what an operator greps for in a run log.
* **Each guard still ALLOWS what it allowed.** Every deny case here is paired with an allow
  case on purpose: a deny-only suite is worthless, since a door that denied everything would
  pass it while making the harness unusable.
* **Kernel-author scoping survives.** All three guards apply only to ``aog-kernel-worker`` /
  ``aog-kernel-optimizer``; any other agent type (or none) must sail through the same payload.
* **Failure is CLOSED.** A payload the door cannot even parse yields ``blocked: true`` — the
  opposite of the autoresearch door's safe-allow, because this door guards an anti-cheating
  boundary where a silent allow is the dangerous direction.

Exit status is a separate channel from the decision: 0 means "a decision was computed" (block
or allow), non-zero means "the door itself failed" — and even then stdout carries a blocked
decision so a stdout-only caller still fails closed.

The workspace fixtures below are deliberately CLEAN against every rule (they are what a passing
op looks like); each deny case mutates exactly one thing.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DOOR = Path(__file__).resolve().parents[2] / "opencode" / "door.py"

KERNEL_AUTHOR = "aog-kernel-worker"

KERNELS_CPP = '''#include "kernel_operator.h"
using namespace AscendC;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLen);
        zGm.SetGlobalBuffer((__gm__ float *)z, totalLen);
        pipe_.InitBuffer(inQueueX, 2, 1024 * sizeof(float));
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueueX;
    GlobalTensor<float> xGm;
    GlobalTensor<float> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen)
{
    KernelAdd op;
    op.Init(x, y, z, totalLen);
}
'''

PYBIND_CPP = '''#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclrtlaunch_add_custom.h"

at::Tensor run_vector_add(const at::Tensor &x, const at::Tensor &y)
{
    auto z = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    uint32_t blockDim = 8;
    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, x.data_ptr(), y.data_ptr(), z.data_ptr(), 1024);
    return z;
}

PYBIND11_MODULE(_vector_add_ext, m)
{
    m.def("run_vector_add", &run_vector_add, "vector add on NPU");
}
'''

MODEL_NEW = '''import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernel", "build"))

import _vector_add_ext


class ModelNew:
    def forward(self, x, y):
        return _vector_add_ext.run_vector_add(x, y)
'''


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project root holding one CLEAN op workspace: workspace/vector_add/."""
    root = tmp_path_factory.mktemp("proj")
    ws = root / "workspace" / "vector_add"
    (ws / "kernel").mkdir(parents=True)
    (ws / "kernel" / "kernels.cpp").write_text(KERNELS_CPP, encoding="utf-8")
    (ws / "kernel" / "pybind11.cpp").write_text(PYBIND_CPP, encoding="utf-8")
    (ws / "model_new_ascendc.py").write_text(MODEL_NEW, encoding="utf-8")
    return root


def _door(payload: dict, env: dict | None = None):
    """Invoke the real CLI: `door.py check <base64(json)>` -> (returncode, decision)."""
    blob = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    proc_env = dict(os.environ)
    for key in ("ASCENDC_WORKSPACE", "CLAUDE_ACTIVE_WORKSPACE", "AOG_PROJECT_ROOT"):
        proc_env.pop(key, None)
    proc_env.update(env or {})
    proc = subprocess.run([sys.executable, str(DOOR), "check", blob],
                          capture_output=True, text=True, env=proc_env)
    assert proc.stdout, f"door produced no decision; stderr={proc.stderr!r}"
    return proc.returncode, json.loads(proc.stdout)


def _payload(project, tool_name, tool_input, agent_type=KERNEL_AUTHOR):
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": "ses_1",
        "call_id": "call_1",
        "agent_id": f"opencode:{agent_type}" if agent_type else "",
        "agent_type": agent_type,
        "cwd": str(project),
        "project_root": str(project),
    }


def _own_ws_env(project):
    return {"ASCENDC_WORKSPACE": str(project / "workspace" / "vector_add")}


# ── guard 1: access guard (runInlineAccessGuard) ────────────────────────────

def test_access_guard_blocks_cross_workspace_read(project):
    rc, decision = _door(
        _payload(project, "Read", {"file_path": "workspace/other_op/kernel/kernels.cpp"}),
        _own_ws_env(project))
    assert rc == 0, "a policy denial is a decision, not a door failure"
    assert decision["blocked"] is True
    assert decision["reason"] == (
        "[a5_ops opencode hook] access guard blocked cross-workspace read by "
        "aog-kernel-worker: workspace/other_op/kernel/kernels.cpp")


def test_access_guard_allows_own_workspace_read(project):
    rc, decision = _door(
        _payload(project, "Read", {"file_path": "workspace/vector_add/kernel/kernels.cpp"}),
        _own_ws_env(project))
    assert rc == 0
    assert decision == {"blocked": False, "reason": ""}


@pytest.mark.parametrize("tool_input,fragment", [
    ({"pattern": "**/*.cpp"}, "project-wide recursive glob"),
    ({"pattern": "output/**/*.py"}, "glob over output/ archives"),
    ({"pattern": "verification.json"}, "answer-bearing glob"),
])
def test_access_guard_blocks_unscoped_globs(project, tool_input, fragment):
    _, decision = _door(_payload(project, "Glob", tool_input), _own_ws_env(project))
    assert decision["blocked"] is True
    assert fragment in decision["reason"]


def test_access_guard_allows_glob_scoped_to_own_workspace(project):
    ws = str(project / "workspace" / "vector_add")
    _, decision = _door(_payload(project, "Glob", {"pattern": "**/*.cpp", "path": ws}),
                        _own_ws_env(project))
    assert decision["blocked"] is False


def test_access_guard_blocks_nested_docker_exec(project):
    _, decision = _door(
        _payload(project, "Bash", {"command": "python3 a5_exec.py -- docker exec cann bash -c ls"}),
        _own_ws_env(project))
    assert decision["blocked"] is True
    assert decision["reason"].startswith(
        "[a5_ops opencode hook] runtime guard blocked nested docker exec through a5_exec.py")


def test_access_guard_allows_ordinary_bash(project):
    _, decision = _door(_payload(project, "Bash", {"command": "ls -la"}), _own_ws_env(project))
    assert decision["blocked"] is False


# ── guard 2: build artifact guard (runBuildArtifactGuard) ───────────────────

def test_build_guard_blocks_piped_deploy(project):
    rc, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu.sh | tail -20"}),
        _own_ws_env(project))
    assert rc == 0
    assert decision["blocked"] is True
    assert decision["reason"] == (
        "[a5_ops opencode hook] build guard blocked deploy: do not pipe deploy_to_npu*.sh "
        "output; pipes mask exit status and can hang post-build sync")


def test_build_guard_allows_clean_workspace_deploy(project):
    """The paired allow: a workspace that satisfies every artifact rule must build."""
    rc, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --build"}),
        _own_ws_env(project))
    assert rc == 0
    assert decision == {"blocked": False, "reason": ""}, decision["reason"]


def test_build_guard_blocks_deploy_without_workspace(project):
    _, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --build"}), {})
    assert decision["blocked"] is True
    assert decision["reason"] == (
        "[a5_ops opencode hook] build guard blocked deploy: ASCENDC_WORKSPACE is required for "
        "op workspace validation")


def test_build_guard_blocks_unsupported_kernel_api(project, tmp_path):
    """One mutated file in an otherwise-clean workspace is enough to stop the deploy."""
    ws = tmp_path / "workspace" / "coreCoord_op"
    (ws / "kernel").mkdir(parents=True)
    (ws / "kernel" / "kernels.cpp").write_text(
        KERNELS_CPP.replace("KernelAdd op;", "coreCoord_t c; KernelAdd op;"), encoding="utf-8")
    (ws / "kernel" / "pybind11.cpp").write_text(PYBIND_CPP, encoding="utf-8")
    (ws / "model_new_ascendc.py").write_text(MODEL_NEW, encoding="utf-8")
    _, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --build"}),
        {"ASCENDC_WORKSPACE": str(ws)})
    assert decision["blocked"] is True
    assert decision["reason"].endswith(
        "uses unsupported coreCoord_t; use GetBlockIdx()/GetBlockNum() scalars")


def test_build_guard_ignores_non_build_deploy(project):
    _, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --sync"}),
        _own_ws_env(project))
    assert decision["blocked"] is False


# ── guard 3: generated code guard (runInlineGeneratedCodeGuard) ─────────────

def test_generated_code_guard_blocks_torch_fallback(project):
    rc, decision = _door(
        _payload(project, "Write", {
            "file_path": "workspace/vector_add/model_new_ascendc.py",
            "content": "import torch\n\n\nclass ModelNew:\n    def forward(self, x, y):\n"
                       "        return torch.add(x, y)\n"}),
        _own_ws_env(project))
    assert rc == 0
    assert decision["blocked"] is True
    assert decision["reason"] == (
        "[a5_ops opencode hook] generated-code guard blocked "
        "workspace/vector_add/model_new_ascendc.py: model_new_ascendc.py torch compute fallback")


def test_generated_code_guard_allows_clean_model(project):
    _, decision = _door(
        _payload(project, "Write", {"file_path": "workspace/vector_add/model_new_ascendc.py",
                                    "content": MODEL_NEW}),
        _own_ws_env(project))
    assert decision["blocked"] is False, decision["reason"]


def test_generated_code_guard_allows_clean_kernel_and_pybind(project):
    """Both generated C++ artifacts of a passing op must survive the guard."""
    for name, content in (("kernels.cpp", KERNELS_CPP), ("pybind11.cpp", PYBIND_CPP)):
        _, decision = _door(
            _payload(project, "Write", {"file_path": f"workspace/vector_add/kernel/{name}",
                                        "content": content}),
            _own_ws_env(project))
        assert decision["blocked"] is False, f"{name}: {decision['reason']}"


def test_generated_code_guard_blocks_missing_pybind_stream_handoff(project):
    """The pybind rules are REQUIRED-PRESENT checks: dropping a required call must deny."""
    _, decision = _door(
        _payload(project, "Write", {
            "file_path": "workspace/vector_add/kernel/pybind11.cpp",
            "content": PYBIND_CPP.replace(
                "#include <torch_npu/csrc/core/npu/NPUStream.h>\n", "")}),
        _own_ws_env(project))
    assert decision["blocked"] is True
    assert decision["reason"].endswith(
        "missing torch_npu NPUStream header for c10_npu::getCurrentNPUStream()")


def test_generated_code_guard_blocks_op_host_scaffold(project):
    _, decision = _door(
        _payload(project, "Write", {"file_path": "workspace/vector_add/op_host/add_tiling.cpp",
                                    "content": "int x = 1;"}),
        _own_ws_env(project))
    assert decision["blocked"] is True
    assert decision["reason"].endswith(
        "direct pybind benchmark tasks must not create op_host/ or op_kernel/ scaffold")


def test_generated_code_guard_ignores_non_generated_paths(project):
    _, decision = _door(
        _payload(project, "Write", {"file_path": "workspace/vector_add/notes.md",
                                    "content": "torch.add(x, y) TODO"}),
        _own_ws_env(project))
    assert decision["blocked"] is False


# ── agent scoping: kernel-author-only rules ─────────────────────────────────

@pytest.mark.parametrize("tool_name,tool_input", [
    ("Read", {"file_path": "workspace/other_op/kernel/kernels.cpp"}),          # access guard
    ("Bash", {"command": "bash deploy_to_npu.sh | tail -20"}),                 # build guard
    ("Write", {"file_path": "workspace/vector_add/model_new_ascendc.py",       # generated-code
               "content": "import torch\nz = torch.add(x, y)\n"}),
])
@pytest.mark.parametrize("agent_type", ["aog-researcher", "general-purpose", ""])
def test_non_kernel_author_is_not_subject_to_kernel_author_rules(
        project, tool_name, tool_input, agent_type):
    """Same payloads that deny for a kernel author must pass for anyone else."""
    blocked_rc, blocked = _door(_payload(project, tool_name, tool_input), _own_ws_env(project))
    assert blocked_rc == 0 and blocked["blocked"] is True, "payload must deny for a kernel author"

    _, decision = _door(_payload(project, tool_name, tool_input, agent_type=agent_type),
                        _own_ws_env(project))
    assert decision["blocked"] is False, decision["reason"]


def test_kernel_optimizer_is_also_a_kernel_author(project):
    _, decision = _door(
        _payload(project, "Read", {"file_path": "workspace/other_op/kernel/kernels.cpp"},
                 agent_type="aog-kernel-optimizer"),
        _own_ws_env(project))
    assert decision["blocked"] is True


# ── fail-closed: a door that cannot decide must deny ────────────────────────

@pytest.mark.parametrize("argv_tail", [
    [],                                            # no mode, no payload
    ["check"],                                     # mode without payload
    ["check", "not-valid-base64!!"],               # undecodable argv
    ["check", base64.b64encode(b"{not json").decode("ascii")],       # malformed JSON
    ["check", base64.b64encode(b'["a", "list"]').decode("ascii")],   # JSON, wrong shape
    ["event", base64.b64encode(b"{}").decode("ascii")],              # wrong mode
])
def test_malformed_invocation_fails_closed(argv_tail):
    proc = subprocess.run([sys.executable, str(DOOR)] + argv_tail,
                          capture_output=True, text=True)
    decision = json.loads(proc.stdout)
    assert decision["blocked"] is True, "an undecidable payload must NOT become an allow"
    assert proc.returncode != 0, "door failure must also be visible in the exit status"
    assert "[a5_ops opencode door] internal error" in decision["reason"]


def test_missing_workspace_artifacts_deny(project, tmp_path):
    """An empty workspace is a policy denial (rc 0), not a door failure."""
    ws = tmp_path / "workspace" / "empty_op"
    (ws / "kernel").mkdir(parents=True)
    rc, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --build"}),
        {"ASCENDC_WORKSPACE": str(ws)})
    assert rc == 0
    assert decision["blocked"] is True
    assert decision["reason"].endswith(
        "; build_ascendc.py does not auto-generate pybind11.cpp")


def test_mid_guard_io_failure_fails_closed(project, tmp_path):
    """A guard that cannot read what it must judge denies — it never falls through to allow.

    Here `kernel/` holds a DIRECTORY named like a source file, so the artifact walk cannot read
    it. The JS turned the same situation into a deny carrying the raw fs error; the door turns
    it into a blocked decision plus a non-zero exit. Same direction, different message.
    """
    ws = tmp_path / "workspace" / "trap_op"
    (ws / "kernel" / "bogus.cpp").mkdir(parents=True)
    (ws / "kernel" / "kernels.cpp").write_text(KERNELS_CPP, encoding="utf-8")
    (ws / "kernel" / "pybind11.cpp").write_text(PYBIND_CPP, encoding="utf-8")
    (ws / "model_new_ascendc.py").write_text(MODEL_NEW, encoding="utf-8")
    rc, decision = _door(
        _payload(project, "Bash", {"command": "bash deploy_to_npu_lane.sh --build"}),
        {"ASCENDC_WORKSPACE": str(ws)})
    assert decision["blocked"] is True
    assert rc != 0
    assert "[a5_ops opencode door] internal error" in decision["reason"]


def test_allow_decision_is_exit_zero_and_empty_reason(project):
    rc, decision = _door(_payload(project, "Bash", {"command": "echo hi"}), _own_ws_env(project))
    assert rc == 0
    assert decision == {"blocked": False, "reason": ""}


@pytest.mark.parametrize("shape", ["root", "ancestor", "outside", "unset"])
def test_implausible_active_workspace_leaves_the_guards_armed(project, shape):
    """A mis-scoped active workspace must not be able to switch the guards off.

    The cross-workspace rules scope themselves to `ASCENDC_WORKSPACE`. Point it one level too
    high, at the workspace ROOT, and every sibling op reads as "inside" the active workspace,
    so the rules stop matching — measured before the fix: the leaf value DENIED a
    kernel-worker's read of another op, the root value ALLOWED it. That made the variable
    strictly more dangerous set than unset, since unset keeps the rules armed. An implausible
    value is therefore discarded rather than honoured.
    """
    env = {
        "root": {"ASCENDC_WORKSPACE": str(project / "workspace")},
        "ancestor": {"ASCENDC_WORKSPACE": str(project)},
        "outside": {"ASCENDC_WORKSPACE": "/somewhere/else"},
        "unset": {},
    }[shape]

    _, decision = _door(
        _payload(project, "Read", {"file_path": "workspace/other_op/kernel/kernels.cpp"}), env)

    assert decision["blocked"] is True, (
        f"a {shape} ASCENDC_WORKSPACE disarmed the cross-workspace read guard"
    )


def test_plausible_active_workspace_still_permits_its_own_files(project):
    """The armed-by-default behaviour above must not swallow the legitimate case."""
    _, decision = _door(
        _payload(project, "Read", {"file_path": "workspace/vector_add/kernel/kernels.cpp"}),
        _own_ws_env(project))

    assert decision["blocked"] is False, decision["reason"]

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_node_script_succeeds(script: str) -> None:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def _write_passing_workflow_critic(project: Path) -> None:
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "workflow_critic.py").write_text("import sys\nsys.exit(0)\n")


def _assert_worker_write_is_blocked(
    tmp_path: Path,
    file_path: str,
    content: str,
    expected_reason: str,
    scenario: str,
) -> None:
    project = tmp_path / "project"
    _write_passing_workflow_critic(project)
    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "write", sessionID: "ses_1", callID: "call_1"}},
          {{args: {{filePath: {json.dumps(file_path)}, content: {json.dumps(content)}}}}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes({json.dumps(expected_reason)})) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error({json.dumps("expected generated-code guard to block " + scenario)});
    """
    _assert_node_script_succeeds(script)






def _assert_worker_write_cases_are_blocked(project: Path) -> None:
    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      {BAD_PYBIND_SHAPE_CASES}
      for (const c of cases) {{
        let blocked = false;
        try {{
          await hooks["tool.execute.before"](
            {{tool: "write", sessionID: "ses_1", callID: "call_1"}},
            {{args: {{filePath: c.filePath, content: c.content}}}}
          );
        }} catch (err) {{
          if (!String(err.message || err).includes(c.reason)) {{
            throw err;
          }}
          blocked = true;
        }}
        if (!blocked) {{
          throw new Error(`expected generated-code guard to block ${{c.reason}}`);
        }}
      }}
    """
    _assert_node_script_succeeds(script)


def _prepare_payload_capture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    capture = tmp_path / "payload.json"
    (workflow / "output_read_guard.py").write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE_PAYLOAD']).write_text(sys.stdin.read())\n"
    )
    return project, capture


def _write_model_importing_op_ext(workspace: Path) -> None:
    (workspace / "model_new_ascendc.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'kernel' / 'build'))\n"
        "import _op_ext\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return _op_ext.run_op(x)\n"
    )


def _assert_worker_bash_is_blocked(
    project: Path,
    command: str,
    expected_reasons: tuple[str, ...],
    failure_message: str,
    workspace: Path | None = None,
) -> None:
    workspace_env = (
        ""
        if workspace is None
        else f"process.env.ASCENDC_WORKSPACE = {json.dumps(str(workspace))};"
    )
    reason_checks = " || ".join(
        f"!String(err.message || err).includes({json.dumps(reason)})"
        for reason in expected_reasons
    )
    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      {workspace_env}
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "bash", sessionID: "ses_1", callID: "call_1", args: {{command: {json.dumps(command)}}}}},
          {{}}
        );
      }} catch (err) {{
        if ({reason_checks}) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error({json.dumps(failure_message)});
    """
    _assert_node_script_succeeds(script)


def _assert_worker_bash_is_allowed(project: Path, workspace: Path, command: str) -> None:
    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.ASCENDC_WORKSPACE = {json.dumps(str(workspace))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      await hooks["tool.execute.before"](
          {{tool: "bash", sessionID: "ses_1", callID: "call_1", args: {{command: {json.dumps(command)}}}}},
        {{}}
      );
    """
    _assert_node_script_succeeds(script)


BAD_PYBIND_SHAPE_CASES = r'''
      const cases = [
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/pybind11.h>\n#include <torch/torch.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\n#include <torch_npu/csrc/aten/common/ACLRTLauchKernel.h>\n#include \"aclrtlaunch_add.h\"\nat::Tensor run_add(at::Tensor x) { auto stream = c10_npu::getCurrentNPUStream().stream(false); auto out = at::empty_like(x); ACLRT_LAUNCH_KERNEL(add)(1, stream, x.data_ptr(), out.data_ptr()); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "non-portable torch_npu ACLRT macro header",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nTQue<QuePosition::VECIN, 1> g_inQueueX;\nclass Add { TPipe pipe; };\n",
          reason: "file-scope TQue queues",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/pybind11.h>\n#include <torch/torch.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nextern \"C\" uint32_t aclrtlaunch_add(uint32_t, void*, void*, void*);\nstatic uint32_t run_add(at::Tensor x) { auto stream = c10_npu::getCurrentNPUStream().stream(false); auto out = at::empty_like(x); return aclrtlaunch_add(1, stream, x.data_ptr(), out.data_ptr()); }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "pybind run_<op> wrapper returns launch status",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { TPipe pipe; __aicore__ void Process() { pipe.Barrier(); } };\n",
          reason: "unsupported pipe.Barrier",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { GlobalTensor<float> x; __aicore__ void Init(GM_ADDR p) { x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p)); } };\n",
          reason: "GlobalTensor::SetGlobalBuffer without an element-count argument",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { __aicore__ void Init(int64_t total_, int32_t blockNum) { int64_t perBlock = (total_ + blockNum - 1) / blockNum; } };\n",
          reason: "unaligned per-block ceil division",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { GlobalTensor<float> x; GlobalTensor<float> y; GlobalTensor<float> z; __aicore__ void Process() { int64_t blockIdx = GetBlockIdx(); int64_t blockNum = GetBlockNum(); int64_t base = (total_ / blockNum); int64_t remainder = total_ % blockNum; int64_t start = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder); int64_t count = base + (blockIdx < remainder ? 1 : 0); int64_t alignedCount = (count + 7) / 8 * 8; for (int64_t offset = 0; offset < alignedCount; offset += 2048) { int64_t tileSize = alignedCount - offset; DataCopy(z[start + offset], q, tileSize); } } int64_t total_; LocalTensor<float> q; };\n",
          reason: "overlapping aligned DataCopy block partition",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { GlobalTensor<float> x; GlobalTensor<float> z; __aicore__ void Process() { int64_t blockIdx = GetBlockIdx(); int64_t blockNum = GetBlockNum(); int64_t total = total_; int64_t blockSize = (total + blockNum - 1) / blockNum; int64_t start = blockIdx * blockSize; int64_t end = (start + blockSize > total) ? total : start + blockSize; int64_t count = end - start; for (int64_t offset = 0; offset < count; offset += 4096) { int64_t tileLen = (offset + 4096 <= count) ? 4096 : (count - offset); int64_t tileLenAligned = (tileLen + 7) / 8 * 8; DataCopy(z[start + offset], q, tileLenAligned); } } int64_t total_; LocalTensor<float> q; };\n",
          reason: "overlapping aligned DataCopy block partition",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nclass Add { TPipe pipe; TQue<QuePosition::VECIN, 4> q; GlobalTensor<float> x; };\n",
          reason: "unqualified AscendC symbols without using namespace AscendC",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\n#include \"aclrtlaunch_add_kernel.h\"\nat::Tensor run_add(at::Tensor x) { auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU); auto out = torch::empty(x.sizes(), opts); auto stream = c10_npu::getCurrentNPUStream().stream(false); ACLRT_LAUNCH_KERNEL(add_kernel)(1, stream, x.data_ptr<float>(), out.data_ptr<float>()); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "CPU tensor allocation/device",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nextern \"C\" uint32_t aclrtlaunch_add(uint32_t, void*, uint64_t);\nat::Tensor run_add(at::Tensor x) { uint64_t tiling[2] = {1, 8}; auto out = at::empty_like(x); auto stream = c10_npu::getCurrentNPUStream().stream(false); aclrtlaunch_add(1, stream, reinterpret_cast<uint64_t>(tiling)); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "host stack array passed as GM tiling/workspace",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nextern \"C\" uint32_t aclrtlaunch_add(uint32_t, void*, uint64_t, uint64_t, uint64_t, int64_t);\nat::Tensor run_add(at::Tensor x) { auto out = at::empty_like(x); auto stream = c10_npu::getCurrentNPUStream().stream(false); aclrtlaunch_add(1, stream, reinterpret_cast<uint64_t>(x.data_ptr<float>()), reinterpret_cast<uint64_t>(x.data_ptr<float>()), reinterpret_cast<uint64_t>(out.data_ptr<float>()), x.numel()); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "aclrtlaunch_<kernel> host stub uses uint64_t GM addresses",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { TQue<QuePosition::VECIN, 2> q_; void f(LocalTensor<float> xLocal) { q_.DeQue(xLocal); } };\n",
          reason: "TQue::DeQue takes no LocalTensor argument",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/pybind11.h>\n#include <torch/torch.h>\npy::tensor run_add(py::tensor x) { return x; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "pybind wrapper uses py::tensor",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/pybind11.h>\n#include <torch/torch.h>\n#include \"aclrtlaunch_add_kernel.h\"\nnamespace py = pybind11;\npy::object run_add(py::object x) { auto stream = c10_npu::getCurrentNPUStream().stream(false); auto out = py::cast<at::Tensor>(x); ACLRT_LAUNCH_KERNEL(add_kernel)(1, stream, reinterpret_cast<GM_ADDR>(out.data_ptr<float>())); return py::cast(out); }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "pybind wrapper uses py::object",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/pybind11.h>\n#include <torch/torch.h>\n#include \"aclrtlaunch_add_kernel.h\"\nat::Tensor run_add(at::Tensor x) { auto stream = c10_npu::getCurrentNPUStream().stream(false); auto out = at::empty_like(x); ACLRT_LAUNCH_KERNEL(add_kernel)(1, stream, reinterpret_cast<GM_ADDR>(x.data_ptr<float>()), reinterpret_cast<GM_ADDR>(out.data_ptr<float>())); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_add\", &run_add); }\n",
          reason: "missing torch_npu NPUStream header",
        },
        {
          filePath: "workspace/op/kernel/kernels.cpp",
          content: "#include \"kernel.h\"\nextern \"C\" __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR out, int64_t n);\n",
          reason: "declares an AscendC kernel but does not define",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nusing namespace AscendC;\nclass Add { TPipe pipe_; TQue<QuePosition::VECIN, 4> q_; __aicore__ void Init() { pipe_.InitBuffer(q_, 4); } };\n",
          reason: "TPipe::InitBuffer without a byte-size argument",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <pybind11/strict_rcward.h>\n",
          reason: "invalid pybind11 strict_rcward header",
        },
        {
          filePath: "workspace/op/kernel/pybind11.cpp",
          content: "#include <torch/extension.h>\n#include \"aclrtlaunch_add_kernel.h\"\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nat::Tensor run_op(at::Tensor x) { auto out = at::empty_like(x); auto stream = c10_npu::getCurrentNPUStream().stream(false); int tiling = 1; ACLRT_LAUNCH_KERNEL(add_kernel)(1, stream, x.data_ptr<float>(), reinterpret_cast<GM_ADDR>(&tiling)); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", &run_op); }\n",
          reason: "host stack pointer passed as GM_ADDR",
        },
        {
          filePath: "workspace/op/kernel/kernels.cpp",
          content: "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(_op_ext, m) {}\n",
          reason: "kernels.cpp must hold AscendC",
        },
        {
          filePath: "workspace/op/kernel/kernel.h",
          content: "#include \"kernel_operator.h\"\nvoid f() { auto p = reinterpret_cast<__gm__ float*>(offset); }\n",
          reason: "fake GM pointer reconstructed",
        },
        {
          filePath: "workspace/op/kernel/kernels.cpp",
          content: "kernel_module_t AddLoad(void* param) { KernelAddParams* p; return KERNEL_STATUS_SUCCESS; }\n",
          reason: "OPP/PR4778 op_kernel registration scaffold",
        },
        {
          filePath: "workspace/op/op_kernel/arch35/op_apt.cpp",
          content: "int x = 0;\n",
          reason: "must not create op_host/ or op_kernel/ scaffold",
        },
      ];
'''

def test_opencode_permission_hook_denies_when_python_guard_rejects(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "workflow_critic.py").write_text(
        "import sys\nsys.stderr.write('blocked by fake guard')\nsys.exit(2)\n"
    )
    (workflow / "output_read_guard.py").write_text("import sys\nsys.exit(0)\n")

    script = f"""
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      const output = {{status: "ask"}};
      await hooks["permission.ask"](
        {{type: "bash", sessionID: "ses_1", callID: "call_1", metadata: {{command: "echo hi"}}}},
        output
      );
      if (output.status !== "deny") {{
        throw new Error(`expected deny, got ${{output.status}}`);
      }}
      if (!String(output.message || "").includes("blocked by fake guard")) {{
        throw new Error(`missing guard message: ${{output.message}}`);
      }}
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_payload_marks_agent_context(tmp_path: Path) -> None:
    project, capture = _prepare_payload_capture(tmp_path)

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.CAPTURE_PAYLOAD = {json.dumps(str(capture))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      await hooks["tool.execute.before"](
        {{tool: "read", sessionID: "ses_1", callID: "call_1"}},
        {{args: {{file_path: "output/a3_to_a5_port/src/kernels/other/kernel.h"}}}}
      );
    """
    _assert_node_script_succeeds(script)
    payload = json.loads(capture.read_text())
    assert payload["agent_id"] == "opencode:aog-kernel-worker"
    assert payload["agent_type"] == "aog-kernel-worker"
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"]["file_path"] == "output/a3_to_a5_port/src/kernels/other/kernel.h"
    assert payload["tool_input"]["path"] == "output/a3_to_a5_port/src/kernels/other/kernel.h"


def test_opencode_tool_hook_blocks_optimizer_nested_docker_exec_via_a5_exec(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src" / "scripts" / "workflow").mkdir(parents=True)

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-optimizer";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-optimizer";
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "bash", sessionID: "ses_1", callID: "call_1"}},
          {{args: {{command: "python ~/.claude/skills/a5_op/scripts/a5_exec.py \\"docker exec npu_container bash -lc 'echo bad'\\""}}}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes("nested docker exec through a5_exec.py")) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error("expected runtime guard to block nested docker exec");
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_reads_real_input_args_shape(tmp_path: Path) -> None:
    project, capture = _prepare_payload_capture(tmp_path)

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-self-critic";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-self-critic";
      process.env.CAPTURE_PAYLOAD = {json.dumps(str(capture))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      await hooks["tool.execute.before"](
        {{tool: "read", sessionID: "ses_1", callID: "call_1", args: {{path: "output/a3_to_a5_port/src/kernels/other/analysis.md"}}}},
        {{}}
      );
    """
    _assert_node_script_succeeds(script)
    payload = json.loads(capture.read_text())
    assert payload["agent_id"] == "opencode:aog-self-critic"
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"]["file_path"] == "output/a3_to_a5_port/src/kernels/other/analysis.md"
    assert payload["tool_input"]["path"] == "output/a3_to_a5_port/src/kernels/other/analysis.md"


def test_opencode_tool_hook_reads_real_output_filepath_shape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "output_read_guard.py").write_text(
        "import json, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "if payload['tool_input'].get('file_path', '').startswith('output/'):\n"
        "    sys.stderr.write('blocked output read')\n"
        "    sys.exit(2)\n"
    )

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-self-critic";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-self-critic";
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "read", sessionID: "ses_1", callID: "call_1"}},
          {{args: {{filePath: "output/a3_to_a5_port/src/kernels/other/verification.json"}}}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes("blocked output read")) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error("expected output read to be blocked");
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_blocks_kernel_worker_project_wide_glob(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "glob", sessionID: "ses_1", callID: "call_1", args: {{pattern: "**/pass_a_runner.py"}}}},
          {{}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes("project-wide recursive glob")) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error("expected kernel-worker project-wide glob to be blocked");
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_blocks_kernel_worker_op_host_bash(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    workspace = project / "workspace" / "op"

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.ASCENDC_WORKSPACE = {json.dumps(str(workspace))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "bash", sessionID: "ses_1", callID: "call_1", args: {{command: "mkdir -p {str(workspace / 'op_host')} {str(workspace / 'op_kernel' / 'arch35')}"}}}},
          {{}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes("op_host/op_kernel Bash access")) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error("expected kernel-worker op_host/op_kernel bash access to be blocked");
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_blocks_kernel_worker_cross_workspace_glob(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "workspace" / "current" / "kernel").mkdir(parents=True)
    (project / "workspace" / "old" / "kernel").mkdir(parents=True)

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.ASCENDC_WORKSPACE = {json.dumps(str(project / "workspace" / "current"))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      try {{
        await hooks["tool.execute.before"](
          {{tool: "glob", sessionID: "ses_1", callID: "call_1", args: {{pattern: "**/*.h", path: {json.dumps(str(project / "workspace" / "old"))}}}}},
          {{}}
        );
      }} catch (err) {{
        if (!String(err.message || err).includes("cross-workspace glob")) {{
          throw err;
        }}
        process.exit(0);
      }}
      throw new Error("expected kernel-worker cross-workspace glob to be blocked");
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_allows_active_workspace_glob(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    workspace = project / "workspace" / "current"
    (workspace / "kernel").mkdir(parents=True)
    capture = tmp_path / "payload.json"
    (workflow / "output_read_guard.py").write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE_PAYLOAD']).write_text(sys.stdin.read())\n"
    )

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.ASCENDC_WORKSPACE = {json.dumps(str(workspace))};
      process.env.CAPTURE_PAYLOAD = {json.dumps(str(capture))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      await hooks["tool.execute.before"](
        {{tool: "glob", sessionID: "ses_1", callID: "call_1", args: {{pattern: "**/*.h", path: {json.dumps(str(workspace))}}}}},
        {{}}
      );
    """
    _assert_node_script_succeeds(script)


def test_opencode_build_guard_blocks_missing_pybind(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workspace = project / "workspace" / "op"
    (workspace / "kernel").mkdir(parents=True)
    (workspace / "model_new_ascendc.py").write_text("import torch\n")

    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build",
        ("missing", "pybind11.cpp"),
        "expected build guard to block missing pybind11.cpp",
        workspace,
    )


def test_opencode_build_guard_blocks_bad_model_import(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workspace = project / "workspace" / "op"
    kernel = workspace / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "pybind11.cpp").write_text(
        "#include <pybind11/pybind11.h>\n"
        "PYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", [](){}); }\n"
    )
    (kernel / "kernels.cpp").write_text(
        'extern "C" __global__ __aicore__ void op_kernel(GM_ADDR x) {}\n'
    )
    (workspace / "model_new_ascendc.py").write_text(
        "from kernel import op_ext\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return op_ext.add(x)\n"
    )

    _assert_worker_bash_is_blocked(
        project,
        f"ASCENDC_WORKSPACE={workspace} bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build",
        ("kernel/build",),
        "expected build guard to block bad model import",
        workspace,
    )


def test_opencode_build_guard_allows_from_import_pybind_module(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "workflow_critic.py").write_text("import sys\nsys.exit(0)\n")
    (workflow / "output_read_guard.py").write_text("import sys\nsys.exit(0)\n")
    workspace = project / "workspace" / "op"
    kernel = workspace / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "kernel.h").write_text(
        '#include "kernel_operator.h"\n'
        "using namespace AscendC;\n"
        "class K { TPipe pipe_; GlobalTensor<float> x_; "
        "__aicore__ void Init(GM_ADDR x) { x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), 1); "
        "pipe_.InitBuffer(q_, 1, 1024 * sizeof(float)); } "
        "TQue<QuePosition::VECIN, 1> q_; };\n"
    )
    (kernel / "kernels.cpp").write_text(
        '#include "kernel.h"\n'
        'extern "C" __global__ __aicore__ void op_kernel(GM_ADDR x) { K k; }\n'
    )
    (kernel / "pybind11.cpp").write_text(
        "#include <torch/extension.h>\n"
        "at::Tensor run_op(at::Tensor x) { return x; }\n"
        "PYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", &run_op); }\n"
    )
    (workspace / "model_new_ascendc.py").write_text(
        "import sys\n"
        "sys.path.insert(0, 'kernel/build')\n"
        "from _op_ext import run_op\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return run_op(x)\n"
    )

    _assert_worker_bash_is_allowed(
        project,
        workspace,
        f"ASCENDC_WORKSPACE={workspace} bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build",
    )


def test_opencode_build_guard_blocks_kernel_declaration_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workspace = project / "workspace" / "op"
    kernel = workspace / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "kernel.h").write_text("#include \"kernel_operator.h\"\n")
    (kernel / "kernels.cpp").write_text(
        '#include "kernel.h"\n'
        'extern "C" __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR out, int64_t n);\n'
    )
    (kernel / "pybind11.cpp").write_text(
        "#include <pybind11/pybind11.h>\n"
        "PYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", [](){}); }\n"
    )
    _write_model_importing_op_ext(workspace)
    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build",
        ("declares an AscendC kernel but does not define",),
        "expected build guard to block declaration-only kernel",
        workspace,
    )


def test_opencode_build_guard_blocks_piped_build_command(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workspace = project / "workspace" / "op"
    kernel = workspace / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "kernels.cpp").write_text(
        'extern "C" __global__ __aicore__ void op_kernel(GM_ADDR x) {}\n'
    )
    (kernel / "pybind11.cpp").write_text(
        "#include <pybind11/pybind11.h>\n"
        "PYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", [](){}); }\n"
    )
    (workspace / "model_new_ascendc.py").write_text(
        "import sys\n"
        "sys.path.insert(0, 'kernel/build')\n"
        "import _op_ext\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return _op_ext.run_op(x)\n"
    )

    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build 2>&1 | tail -30",
        ("do not pipe deploy_to_npu*.sh output",),
        "expected build guard to block piped build command",
        workspace,
    )


def test_opencode_build_guard_blocks_piped_deploy_sync_command(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "workflow_critic.py").write_text("import sys\nsys.exit(0)\n")

    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 2>&1 | tail -20",
        ("do not pipe deploy_to_npu*.sh output",),
        "expected build guard to block piped deploy sync command",
    )


def test_opencode_build_guard_blocks_unpiped_marker_argument(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "workflow_critic.py").write_text("import sys\nsys.exit(0)\n")

    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build unpiped",
        ("do not append unsupported marker words such as unpiped",),
        "expected build guard to block unpiped marker argument",
    )


def test_opencode_build_guard_blocks_invalid_kernel_source_shape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workspace = project / "workspace" / "op"
    kernel = workspace / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "kernel.h").write_text(
        "#include \"kernel_operator.h\"\n"
        "class K { GlobalTensor<float> x; coreCoord_t c; void f(){ inQueue.Init(IN_QUE_NUM, 4); } };\n"
    )
    (kernel / "kernels.cpp").write_text("#include \"op_kernel_missing.h\"\n")
    (kernel / "pybind11.cpp").write_text(
        "#include <pybind11/pybind11.h>\n"
        "PYBIND11_MODULE(_op_ext, m) { m.def(\"run_op\", [](){}); }\n"
    )
    _write_model_importing_op_ext(workspace)
    _assert_worker_bash_is_blocked(
        project,
        "bash src/scripts/deploy_to_npu_lane.sh --lane 0 --build",
        ("coreCoord_t",),
        "expected build guard to block invalid kernel source shape",
        workspace,
    )


def test_opencode_permission_hook_normalizes_pattern_for_output_guard(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    capture = tmp_path / "payload.json"
    (workflow / "output_read_guard.py").write_text(
        "import json, os, pathlib, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "pathlib.Path(os.environ['CAPTURE_PAYLOAD']).write_text(json.dumps(payload))\n"
        "if payload['tool_input'].get('file_path', '').startswith('output/'):\n"
        "    sys.stderr.write('blocked output read')\n"
        "    sys.exit(2)\n"
    )

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-self-critic";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-self-critic";
      process.env.CAPTURE_PAYLOAD = {json.dumps(str(capture))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      const output = {{status: "ask"}};
      await hooks["permission.ask"](
        {{type: "external_directory", sessionID: "ses_1", callID: "call_1", pattern: "output/a3_to_a5_port/src/kernels/other/verification.json"}},
        output
      );
      if (output.status !== "deny") {{
        throw new Error(`expected deny, got ${{output.status}}`);
      }}
      if (!String(output.message || "").includes("blocked output read")) {{
        throw new Error(`missing guard message: ${{output.message}}`);
      }}
    """
    _assert_node_script_succeeds(script)
    payload = json.loads(capture.read_text())
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"]["file_path"] == "output/a3_to_a5_port/src/kernels/other/verification.json"
    assert payload["tool_input"]["path"] == "output/a3_to_a5_port/src/kernels/other/verification.json"


def test_opencode_permission_hook_allows_guarded_external_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)
    capture = tmp_path / "payload.json"
    (workflow / "output_read_guard.py").write_text(
        "import json, os, pathlib, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "pathlib.Path(os.environ['CAPTURE_PAYLOAD']).write_text(json.dumps(payload))\n"
    )

    script = f"""
      process.env.AOG_HOOK_AGENT_ID = "opencode:aog-kernel-worker";
      process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";
      process.env.CAPTURE_PAYLOAD = {json.dumps(str(capture))};
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      const output = {{status: "ask"}};
      await hooks["permission.ask"](
        {{type: "external_directory", sessionID: "ses_1", callID: "call_1", pattern: "/tmp/a5_opencode_e2e_deep/*"}},
        output
      );
      if (output.status !== "allow") {{
        throw new Error(`expected allow, got ${{output.status}}`);
      }}
    """
    _assert_node_script_succeeds(script)
    payload = json.loads(capture.read_text())
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"]["file_path"] == "/tmp/a5_opencode_e2e_deep/*"
    assert payload["tool_input"]["path"] == "/tmp/a5_opencode_e2e_deep/*"


def test_opencode_permission_hook_leaves_unknown_permission_as_ask(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflow = project / "src" / "scripts" / "workflow"
    workflow.mkdir(parents=True)

    script = f"""
      const mod = await import({json.dumps(str(REPO_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"))});
      const hooks = await mod.default({{directory: {json.dumps(str(project))}}}, {{projectRoot: {json.dumps(str(project))}}});
      const output = {{status: "ask"}};
      await hooks["permission.ask"](
        {{type: "question", sessionID: "ses_1", callID: "call_1", pattern: "*"}},
        output
      );
      if (output.status !== "ask") {{
        throw new Error(`expected ask, got ${{output.status}}`);
      }}
    """
    _assert_node_script_succeeds(script)


def test_opencode_tool_hook_blocks_model_new_host_fallback_write(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        "workspace/op/model_new_ascendc.py",
        "def forward(self, x, y):\n    return x + y\n",
        "host arithmetic fallback",
        "host fallback",
    )


def test_opencode_tool_hook_blocks_direct_aclrtlaunchkernel_write(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        'void f() { aclrtLaunchKernel(add_kernel, 0); }',
        'direct aclrtLaunchKernel call',
        'direct aclrtLaunchKernel',
    )


def test_opencode_tool_hook_blocks_torch_npu_header_write(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/npu.h>\n',
        'non-project torch/npu.h include',
        'torch/npu.h',
    )


def test_opencode_tool_hook_blocks_invalid_torch_pybind_header(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/pybind.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\n#include "aclrtlaunch_add.h"\nat::Tensor run_op(const at::Tensor& x) { auto y = at::empty_like(x); auto s = c10_npu::getCurrentNPUStream().stream(false); aclrtlaunch_add(1, s, x.data_ptr<float>(), y.data_ptr<float>(), x.numel()); return y; }\nPYBIND11_MODULE(_op_ext, m) { m.def("run_op", &run_op); }\n',
        'invalid torch/pybind.h header',
        'torch/pybind.h',
    )


def test_opencode_tool_hook_blocks_kernel_header_entry_definition(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nextern "C" __global__ __aicore__ void add(GM_ADDR x) {}\n',
        'kernel entry definition',
        'kernel.h entry definition',
    )


def test_opencode_tool_hook_blocks_kernel_operator_header_guard_collision(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#ifndef KERNEL_OPERATOR_H\n#define KERNEL_OPERATOR_H\n#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; };\n#endif\n',
        'kernel_operator.h header guard collision',
        'kernel_operator.h guard collision',
    )


def test_opencode_tool_hook_blocks_dynamic_full_input_initbuffer(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TQue<QuePosition::VECIN, 1> q_; __aicore__ void Init(uint64_t totalElems) { pipe_.InitBuffer(q_, 1, totalElems * sizeof(float)); } };\n',
        'dynamic full-input size',
        'dynamic InitBuffer size',
    )


def test_opencode_tool_hook_blocks_non_ascii_placeholder_kernel_write(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { void f() { pipe_.InitBuffer(q_, 4, kTileBytes在想); /* ... */ } };\n',
        'non-ASCII text',
        'non-ASCII placeholder kernel',
    )


def test_opencode_tool_hook_blocks_queue_api_misuse(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TQue<QuePosition::VECIN, 4> inQueueX_; void f() { pipe_.EnQue(inQueueX_, 16); DataCopy(inQueueX_, xGm_[0], 8); } };\n',
        'TPipe has no EnQue/DeQue',
        'queue API misuse',
    )


def test_opencode_tool_hook_blocks_tbuf_used_as_queue_buffer(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TBuf<QuePosition::VECIN> xTbuf_; void Init() { pipe_.InitBuffer(xTbuf_, 1, 1024 * sizeof(float)); } };\n',
        'TBuf used as a queue buffer',
        'TBuf queue InitBuffer',
    )


def test_opencode_tool_hook_blocks_invented_vector_tail_count(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TQue<QuePosition::VECIN, 4> inQueueX_; void f(LocalTensor<float> outLocal, LocalTensor<float> xDeque, LocalTensor<float> yDeque, uint32_t tileLen) { Add(outLocal, xDeque, yDeque, tileLen-epilogue_len); } };\n',
        'vector intrinsic count uses invented tail arithmetic',
        'invented vector tail count',
    )


def test_opencode_tool_hook_blocks_hallucinated_identifier_suffix(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TQue<QuePosition::VECIN, 2> q_; void f(LocalTensor<float> yDeque) { q_.FreeTensor(yDequeonge); } };\n',
        'hallucinated identifier suffix',
        'hallucinated identifier suffix',
    )


def test_opencode_tool_hook_blocks_hallucinated_float_suffix(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nconstexpr int32_t kTileBytes = 2048 * sizeof(floatapse;\n',
        'hallucinated identifier suffix',
        'hallucinated float suffix',
    )


def test_opencode_tool_hook_blocks_get_tpipe_kernel_pattern(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernels.cpp',
        '#include "kernel.h"\nextern "C" __global__ __aicore__ void add_kernel(GM_ADDR x) { KernelAdd op; op.Init(x, &(GetTPipe())); }\n',
        'unsupported GetTPipe()',
        'GetTPipe kernel pattern',
    )


def test_opencode_tool_hook_blocks_gm_pointer_kernel_entry(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernels.cpp',
        '#include "kernel.h"\nextern "C" __global__ __aicore__ void add_kernel(__gm__ float* x, __gm__ float* y, __gm__ float* out) {}\n',
        'kernel entry parameters must use GM_ADDR',
        '__gm__ pointer kernel entry',
    )


def test_opencode_tool_hook_blocks_duplicate_enque_same_tensor(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/kernel.h',
        '#include "kernel_operator.h"\nusing namespace AscendC;\nclass K { TPipe pipe_; TQue<QuePosition::VECOUT, 2> outQueue_; void f(LocalTensor<float> zLocal) { outQueue_.EnQue(zLocal); outQueue_.EnQue(zLocal); } };\n',
        'same LocalTensor enqueued twice',
        'duplicate EnQue',
    )


def test_opencode_tool_hook_blocks_direct_macro_pybind_export(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/extension.h>\n#include "aclrtlaunch_elementwise_add.h"\nPYBIND11_MODULE(_op_ext, m) { m.def("forward", &ACLRT_LAUNCH_KERNEL(elementwise_add)); }\n',
        'pybind exposes ACLRT_LAUNCH_KERNEL directly',
        'direct macro export',
    )


def test_opencode_tool_hook_blocks_incomplete_pybind_wrapper(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/extension.h>\nat::Tensor run_op(at::Tensor x) { return x; }\nPYBIND11_MODULE(_op_ext, m) { m.def("run_op", &run_op); }\n',
        'missing aclrtlaunch_<kernel> or ACLRT_LAUNCH_KERNEL launch',
        'incomplete pybind wrapper',
    )


def test_opencode_tool_hook_blocks_wrong_pybind_module_name(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/opencode_e2e_agent17_add_a3/kernel/pybind11.cpp',
        '#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\n#include "aclrtlaunch_add_kernel.h"\nat::Tensor run_add(const at::Tensor& x) { auto y = at::empty_like(x); auto s = c10_npu::getCurrentNPUStream().stream(false); aclrtlaunch_add_kernel(1, s, x.data_ptr<float>(), y.data_ptr<float>(), x.numel()); return y; }\nPYBIND11_MODULE(_add_ext, m) { m.def("run_add", &run_add); }\n',
        'must be _opencode_e2e_agent17_add_a3_ext',
        'wrong pybind module name',
    )


def test_opencode_tool_hook_blocks_invented_pybind_status_variable(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nextern "C" uint32_t aclrtlaunch_add_kernel(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* out, int32_t n);\ntorch::Tensor run_add(torch::Tensor x, torch::Tensor y) { auto out = torch::empty_like(x); auto stream = c10_npu::getCurrentNPUStream().stream(false); uint32_t ret = aclrtlaunch_add_kernel(56, stream, reinterpret_cast<uint8_t*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()), reinterpret_cast<uint8_t*>(out.data_ptr()), x.numel()); TORCH_CHECK(ret == 0, "failed", retistory); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def("run_add", &run_add); }\n',
        'invented status variable',
        'invented status variable',
    )


def test_opencode_tool_hook_blocks_gm_qualifier_in_pybind_wrapper(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel/pybind11.cpp',
        '#include <torch/extension.h>\n#include <torch_npu/csrc/core/npu/NPUStream.h>\nextern "C" uint32_t aclrtlaunch_op(uint32_t blockDim, void* stream, __gm__ float* x, __gm__ float* out, int32_t n);\ntorch::Tensor run_op(torch::Tensor x) { auto out = torch::empty_like(x); auto stream = c10_npu::getCurrentNPUStream().stream(false); uint32_t ret = aclrtlaunch_op(1, stream, x.data_ptr<float>(), out.data_ptr<float>(), x.numel()); TORCH_CHECK(ret == 0, "failed", ret); return out; }\nPYBIND11_MODULE(_op_ext, m) { m.def("run_op", &run_op); }\n',
        'pybind host wrapper must not use device-side __gm__',
        '__gm__ in pybind wrapper',
    )


def test_opencode_tool_hook_blocks_root_kernel_source_write(tmp_path: Path) -> None:
    _assert_worker_write_is_blocked(
        tmp_path,
        'workspace/op/kernel.h',
        '#include "kernel_operator.h"\n',
        'workspace/<op>/kernel/',
        'root kernel.h write',
    )


def test_opencode_tool_hook_blocks_real_bad_pybind_shapes_from_e2e(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_passing_workflow_critic(project)
    _assert_worker_write_cases_are_blocked(project)

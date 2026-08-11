# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Hook 入口集成测试 — 以子进程方式调用 pypto_op_lint.py，验证完整调用链。"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .helpers import build_stateless_op_dir, write_file

SCRIPT = str(Path(__file__).resolve().parents[1] / "pypto_op_lint.py")
PYTHON = sys.executable

_GOOD_MODULE_IMPL = """import pypto
_N = pypto.DYNAMIC
_M = pypto.STATIC
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([_N, _M], pypto.DT_BF16),
                         y: pypto.Tensor([_N, _M], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    tmp = pypto.cast(x, pypto.DataType.DT_FP32)
    y[:] = pypto.cast(tmp, pypto.DataType.DT_BF16)

def demo_module1_wrapper(x, y):
    return None
"""

_BAD_MODULE_IMPL_OL01 = """import pypto
def demo_module1_kernel(x, y):
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""

_STUB_INTEGRATED_IMPL = """# placeholder created by Stage 4 scaffolding.
# Stage 5 cleanup will replace this with the integrated kernel.
"""


def _run_hook(hook: str, payload: dict, env_extra: dict | None = None) -> tuple[int, str]:
    """以子进程执行 hook，返回 (exit_code, stdout)"""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [PYTHON, SCRIPT, "--hook", hook],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=15, env=env,
    )
    return result.returncode, result.stdout.strip()


def _write_standalone_module(op_dir: Path, content: str) -> str:
    (op_dir / "demo_impl.py").unlink()
    module_path = op_dir / "modules" / "demo_module1_impl.py"
    write_file(module_path, content)
    return str(module_path)


def _post_edit_decision(file_path: str) -> tuple[int, str]:
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": file_path}})
    return rc, json.loads(out)["hookSpecificOutput"]["decision"]


# ── post-edit: 非算子文件 → 无输出 ──

def test_post_edit_non_operator_file_silent():
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": "readme.md"}})
    assert rc == 0
    assert not out


# ── post-edit: 算子 impl 文件 → 返回结构化 JSON ──

def test_post_edit_impl_returns_hook_json(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl_path = str(op_dir / "demo_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0
    if out:
        data = json.loads(out)
        assert "hookSpecificOutput" in data
        hook_out = data["hookSpecificOutput"]
        assert hook_out.get("decision") in ("allow", "block")


# ── post-edit: S1 违规 → decision=block ──

def test_post_edit_blocks_on_s1_violation(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 写一个缺少 @jit 装饰器的 impl（违反 OL01）
    bad_impl = """import pypto
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", bad_impl)
    impl_path = str(op_dir / "demo_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0  # exit 0 even on block (structured output)
    data = json.loads(out)
    assert data["hookSpecificOutput"]["decision"] == "block"


# ── post-edit: modules/<op>_module*_impl.py（Stage 5 Phase M_k） ──


def test_post_edit_module_file_blocks_on_s1_violation(tmp_path: Path):
    """modules/<op>_module1_impl.py 上的 S1 违规应触发 block。

    回归 _infer_op_dir 的 modules/ → 父目录解析：若该解析缺失，规则全部
    会因 "无 impl 文件可供检查" 而 SKIP，决策错误地落到 allow。
    """
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "max_stage": 7,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    impl_path = _write_standalone_module(op_dir, _BAD_MODULE_IMPL_OL01)
    rc, decision = _post_edit_decision(impl_path)
    assert rc == 0
    assert decision == "block"


def test_post_edit_module_file_stateless_parent_blocks(tmp_path: Path):
    """无 .orchestrator_state.json 时, modules/ 也应能向上解析到 stateless op_dir。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl_path = _write_standalone_module(op_dir, _BAD_MODULE_IMPL_OL01)
    rc, decision = _post_edit_decision(impl_path)
    assert rc == 0
    assert decision == "block"


def test_post_edit_module_file_allow_when_clean(tmp_path: Path):
    """干净的 module impl 应通过 lint, decision=allow。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "max_stage": 7,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    write_file(op_dir / "modules" / "demo_module1_impl.py", _GOOD_MODULE_IMPL)
    impl_path = str(op_dir / "modules" / "demo_module1_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0
    if out:
        data = json.loads(out)
        # 干净情况下不应被 block。允许 allow / 无输出 / 仅含 WARN/INFO。
        assert data["hookSpecificOutput"].get("decision", "allow") != "block"


# ── post-bash: 非测试命令 → 无输出 ──

def test_post_bash_non_test_command_silent():
    rc, out = _run_hook("post-bash", {"tool_input": {"command": "ls -la"}})
    assert rc == 0
    assert not out


# ── post-bash: 测试命令 → 返回判定结果 ──

def test_post_bash_test_command_returns_verdict():
    payload = {
        "tool_input": {"command": "python3 test_demo.py"},
        "tool_result": {
            "stdout": "some output [PRECISION_PASS] done",
            "stderr": "",
            "exit_code": 0,
        },
    }
    rc, out = _run_hook("post-bash", payload)
    assert rc == 0
    data = json.loads(out)
    ctx = data["hookSpecificOutput"].get("additionalContext", "")
    assert "precision_pass" in ctx


# ── pre-edit-backup: 非 impl 文件 → 无输出 ──

def test_pre_edit_backup_non_impl_silent():
    rc, out = _run_hook("pre-edit-backup", {"tool_input": {"file_path": "readme.md"}})
    assert rc == 0
    assert not out


# ── stop: 非算子目录 → 无输出 ──

def test_stop_non_operator_dir_silent():
    rc, out = _run_hook("stop", {"cwd": "."})
    assert rc == 0
    assert not out


# ── stop: 算子目录有 S1 违规 → exit 2 ──

def test_stop_blocks_on_s1_violation(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 删除 impl 使 OL13 失败
    (op_dir / "demo_impl.py").unlink()
    state = {
        "operator_name": "demo",
        "max_stage": 8,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    rc, out = _run_hook("stop", {"cwd": str(op_dir)})
    assert rc == 2
    data = json.loads(out)
    assert data["hookSpecificOutput"]["decision"] == "block"


# ── 边界场景: 空 stdin → 安全退出 ──

def test_hook_empty_stdin():
    """所有 hook 在空输入时应安全退出"""
    for hook in ("post-edit", "post-bash", "pre-edit-backup", "stop"):
        rc, out = _run_hook(hook, {})
        assert rc == 0, f"{hook} failed with rc={rc}"


# ─── --check-phase-gate CLI: Stage 5 phase 限定门禁 ───


def _run_phase_gate(op_dir: Path, phase: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, SCRIPT, "--check-phase-gate",
         "--op-dir", str(op_dir), "--phase", phase],
        capture_output=True, text=True, timeout=15,
    )


def _write_memory_md(op_dir: Path, phase: str) -> None:
    content = f"""# Operator Memory

## Phase {phase} self-review

- [x] host_wrapper signature matches module_interfaces.yaml
- [x] All outputs written via pypto.assemble or `[:] =`
- [x] All pypto.view shape/offsets/valid_shape rank consistent
- [x] SPEC golden inventory cross-checked with impl line refs
- [x] No `for ... in range(...)` in Layer K
- [x] Layer K JIT call invoked exactly once
"""
    write_file(op_dir / "MEMORY.md", content)


def test_phase_gate_m1_ignores_stub_integrated_impl(tmp_path: Path):
    """Phase gate M1: 忽略顶层 <op>_impl.py stub 的 OL01 违规，只评估 M1 module impl。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_memory_md(op_dir, "M1")
    # 顶层 stub: 会触发 OL01，但与 phase gate 无关
    write_file(op_dir / "demo_impl.py", _STUB_INTEGRATED_IMPL)
    # M1 module: 干净
    write_file(op_dir / "modules" / "demo_module1_impl.py", _GOOD_MODULE_IMPL)
    r = _run_phase_gate(op_dir, "M1")
    assert r.returncode == 0, f"expected pass, got rc={r.returncode}\nstdout={r.stdout[:500]}"
    data = json.loads(r.stdout)
    # 确认不会报告顶层 stub 的 OL01 FAIL
    fails = [f for f in data["findings"] if f["status"] == "FAIL"]
    integrated_fails = [f for f in fails if f.get("file") == "demo_impl.py"]
    assert not integrated_fails, f"unexpected fail on integrated stub: {integrated_fails}"


def test_phase_gate_m1_blocks_on_module1_violation(tmp_path: Path):
    """Phase gate M1: module1 impl 自身有违规时 exit != 0 并 block。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "modules" / "demo_module1_impl.py", _BAD_MODULE_IMPL_OL01)
    r = _run_phase_gate(op_dir, "M1")
    assert r.returncode != 0, "expected fail (OL01), got rc=0"
    data = json.loads(r.stdout)
    fails = [f for f in data["findings"] if f["status"] == "FAIL"]
    ol01 = [f for f in fails if f["rule_id"] == "OL01"]
    assert ol01, f"OL01 should fail on module1; findings={fails}"
    # block 必须发生在 module1 impl 上
    assert any("module1_impl.py" in (f.get("file") or "") for f in ol01)


def test_phase_gate_m2_ignores_module1(tmp_path: Path):
    """Phase gate M2: 忽略 module1 impl 的违规，只评估 module12 impl。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_memory_md(op_dir, "M2")
    # module1 保持旧违规代码，相当于已 verified 的快照
    write_file(op_dir / "modules" / "demo_module1_impl.py", _BAD_MODULE_IMPL_OL01)
    # module12 干净，是 M2 phase 当前工作对象
    good_m12 = _GOOD_MODULE_IMPL.replace("demo_module1_kernel", "demo_module12_kernel") \
                                .replace("demo_module1_wrapper", "demo_module12_wrapper")
    write_file(op_dir / "modules" / "demo_module12_impl.py", good_m12)
    r = _run_phase_gate(op_dir, "M2")
    assert r.returncode == 0, (
        f"M2 phase gate should ignore module1 violations\n"
        f"rc={r.returncode}\nstdout={r.stdout[:500]}"
    )


def test_phase_gate_m2_blocks_on_module12_violation(tmp_path: Path):
    """Phase gate M2: module12 自身有违规时 block。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # M1 干净
    write_file(op_dir / "modules" / "demo_module1_impl.py", _GOOD_MODULE_IMPL)
    # M2 违规
    bad_m12 = _BAD_MODULE_IMPL_OL01.replace("demo_module1_kernel", "demo_module12_kernel") \
                                    .replace("demo_module1_wrapper", "demo_module12_wrapper")
    write_file(op_dir / "modules" / "demo_module12_impl.py", bad_m12)
    r = _run_phase_gate(op_dir, "M2")
    assert r.returncode != 0
    data = json.loads(r.stdout)
    fails = [f for f in data["findings"] if f["status"] == "FAIL"]
    ol01 = [f for f in fails if f["rule_id"] == "OL01"]
    assert any("module12_impl.py" in (f.get("file") or "") for f in ol01), (
        f"OL01 should fail specifically on module12: {fails}"
    )


def test_phase_gate_missing_phase_arg_errors(tmp_path: Path):
    """--phase 不指定时 argparse 报错。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    r = subprocess.run(
        [PYTHON, SCRIPT, "--check-phase-gate", "--op-dir", str(op_dir)],
        capture_output=True, text=True, timeout=15,
    )
    assert r.returncode != 0


def test_phase_gate_skips_test_and_gate_rules(tmp_path: Path):
    """Phase gate 不对 test_<op>.py 或 SPEC.md stub 触发规则。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_memory_md(op_dir, "M1")
    # SPEC.md 和 test_demo.py 已由 build_stateless_op_dir 创建。
    # 即使它们有违规，phase gate 也不应拦截。
    write_file(op_dir / "test_demo.py", "# stub, OL17/18/19 violations expected\n")
    write_file(op_dir / "modules" / "demo_module1_impl.py", _GOOD_MODULE_IMPL)
    r = _run_phase_gate(op_dir, "M1")
    assert r.returncode == 0, "phase gate should ignore test stub violations"
    data = json.loads(r.stdout)
    fails = [f for f in data["findings"] if f["status"] == "FAIL"]
    test_fails = [f for f in fails if "test_demo.py" in (f.get("file") or "")]
    assert not test_fails, f"phase gate should not run test rules: {test_fails}"


# ─── sidecar mechanism removed in Step 20 ───
# Previously a `.lint_retry_state.json` sidecar tracked per-file
# blocking_rules and consecutive_count to give the orchestrator a
# durable signal between Coder dispatches. That layer was dropped:
# the auto-trigger hook delivers `decision: block` in-band to the
# calling agent, and the complete_phase / complete_stage gates serve
# as the structural backstop. Tests that asserted sidecar write /
# clear / persistence semantics no longer apply.


# ─── post-edit hook file-scope (Step 13) ───
# post-edit hook should only lint the file just written, not other module stubs.


def test_post_edit_ignores_other_module_stubs(tmp_path: Path):
    """post-edit on module1: 不误报其他 module stub 的 OL01 等违规。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 5, "stage_status": {"5": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    # Remove top-level integrated impl so it can't interfere
    (op_dir / "demo_impl.py").unlink()
    # M2 stub (empty placeholder) — simulates verifier scaffolding artifact
    stub_m12 = """#!/usr/bin/env python3
# coding: utf-8
\"\"\"demo M12 stub - pending Stage 5 Phase M2.\"\"\"
pass
"""
    write_file(op_dir / "modules" / "demo_module12_impl.py", stub_m12)
    # M3 stub also exists (typical Stage 4 scaffolding output)
    write_file(op_dir / "modules" / "demo_module123_impl.py", stub_m12.replace("M12", "M123"))
    # Coder writes a clean M1 implementation
    good_m1 = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32),
                        y: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", good_m1)
    impl_path = str(op_dir / "modules" / "demo_module1_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0
    # Should NOT block: M1 is clean, M12/M123 stubs should not be scanned
    if out:
        data = json.loads(out)
        decision = data["hookSpecificOutput"].get("decision", "allow")
        reason = data["hookSpecificOutput"].get("reason", "")
        assert decision != "block", f"post-edit on clean M1 should not block; reason={reason[:400]}"
        # And the reason must not mention the other modules
        assert "module12_impl.py" not in reason
        assert "module123_impl.py" not in reason


def test_post_edit_blocks_only_on_written_file_violation(tmp_path: Path):
    """post-edit 即使 block 违规，也只限定在本次写入的 file_path。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 5, "stage_status": {"5": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    (op_dir / "demo_impl.py").unlink()
    # Other-module stub
    write_file(op_dir / "modules" / "demo_module12_impl.py", "# stub\n")
    # Bad M1 (no @jit)
    bad_m1 = """import pypto
def demo_module1_kernel(x, y):
    y[:] = x
def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", bad_m1)
    impl_path = str(op_dir / "modules" / "demo_module1_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    data = json.loads(out)
    assert data["hookSpecificOutput"]["decision"] == "block"
    reason = data["hookSpecificOutput"].get("reason", "")
    # Block message refers to M1, not to the M12 stub
    assert "module1_impl.py" in reason
    assert "module12_impl.py" not in reason


# ─── PENDING_STUB marker exclusion (Step 14) ───
# Files containing the PYPTO_PENDING_STUB marker are excluded from
# _impl_files_to_scan across all modes (file/phase/default).


_PENDING_STUB = """#!/usr/bin/env python3
# coding: utf-8
\"\"\"demo M12 stub — contract frozen, impl pending Phase M2.\"\"\"
# PYPTO_PENDING_STUB: phase=M2 — implementation pending
pass
"""


def test_pending_stub_excluded_from_default_scan(tmp_path: Path):
    """default mode (no scope): pending stubs filtered out of _impl_files_to_scan."""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 5, "stage_status": {"5": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    (op_dir / "demo_impl.py").unlink()
    # module1 是干净的 impl
    write_file(op_dir / "modules" / "demo_module1_impl.py", _GOOD_MODULE_IMPL)
    # module12 仍是待实现的 stub
    write_file(op_dir / "modules" / "demo_module12_impl.py", _PENDING_STUB)

    # Run --check-gate --stage 5 (default mode, scans all impl files)
    r = subprocess.run(
        [PYTHON, SCRIPT, "--check-gate", "--op-dir", str(op_dir), "--stage", "5"],
        capture_output=True, text=True, timeout=20)
    data = json.loads(r.stdout)
    fail_findings = [f for f in data["findings"] if f["status"] == "FAIL"]
    fails = [f for f in fail_findings if f["severity"] in ("S0", "S1")]
    # Pending stub should NOT trigger OL01/OL07/OL08 etc.
    stub_fails = [f for f in fails if "module12_impl.py" in (f.get("file") or "")]
    assert not stub_fails, f"pending stub should be skipped: {stub_fails}"


def test_pending_stub_excluded_from_phase_gate(tmp_path: Path):
    """phase gate: if its own phase file is pending stub → skip (no FAIL)。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_memory_md(op_dir, "M1")
    write_file(op_dir / "modules" / "demo_module1_impl.py", _PENDING_STUB)
    r = subprocess.run(
        [PYTHON, SCRIPT, "--check-phase-gate", "--op-dir", str(op_dir), "--phase", "M1"],
        capture_output=True, text=True, timeout=15)
    # phase file is pending → no impl files to scan → all D1 rules SKIP → exit 0
    assert r.returncode == 0


def test_pending_stub_excluded_from_post_edit(tmp_path: Path):
    """post-edit: if the just-written file is a pending stub → skip lint."""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 5, "stage_status": {"5": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    (op_dir / "demo_impl.py").unlink()
    # Designer commits a stub
    impl_path = op_dir / "modules" / "demo_module2_impl.py"
    write_file(impl_path, _PENDING_STUB)
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": str(impl_path)}})
    assert rc == 0
    # No block — stub is recognized and skipped
    if out:
        data = json.loads(out)
        assert data["hookSpecificOutput"].get("decision", "allow") != "block"


def test_pending_stub_marker_removed_then_lint_runs(tmp_path: Path):
    """Coder removes the marker → file is no longer a pending stub → lint runs."""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 5, "stage_status": {"5": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    (op_dir / "demo_impl.py").unlink()
    impl_path = op_dir / "modules" / "demo_module1_impl.py"
    # First: stub with marker
    write_file(impl_path, _PENDING_STUB)
    rc1, out1 = _run_hook("post-edit", {"tool_input": {"file_path": str(impl_path)}})
    # Then: Coder overwrites with bad impl (no marker) — should now be linted
    bad = """import pypto
def demo_module1_kernel(x, y):
    y[:] = x
def demo_module1_wrapper(x, y):
    return None
"""
    write_file(impl_path, bad)
    rc2, out2 = _run_hook("post-edit", {"tool_input": {"file_path": str(impl_path)}})
    assert rc2 == 0
    data = json.loads(out2)
    # Once marker is gone, OL01 (no @jit) should fire
    assert data["hookSpecificOutput"]["decision"] == "block"


def test_post_edit_top_level_impl_still_scoped(tmp_path: Path):
    """post-edit on top-level <op>_impl.py: scope to that file only (not module stubs)。"""
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {"operator_name": "demo", "max_stage": 7,
             "current_stage": 6, "stage_status": {"5": "completed", "6": "in_progress"}}
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    # module stub exists from Stage 4 scaffolding
    write_file(op_dir / "modules" / "demo_module1_impl.py", "# stub\n")
    # Integrated impl is clean (Stage 5 cleanup output)
    good = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32),
                y: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", good)
    impl_path = str(op_dir / "demo_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    if out:
        data = json.loads(out)
        decision = data["hookSpecificOutput"].get("decision", "allow")
        # Top-level clean → not block, module stub ignored
        assert decision != "block"


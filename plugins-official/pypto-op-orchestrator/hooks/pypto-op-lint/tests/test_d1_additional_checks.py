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

"""D1 框架约束合规规则（OL04-OL06, OL25, OL26）补充测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# ── OL04: 必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes ──


def test_ol04_pass_when_tile_shapes_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "PASS"


def test_ol04_fail_when_no_tile_shapes(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "FAIL"


def test_ol04_pass_when_tile_shapes_in_reachable_helper(tmp_path: Path):
    """Layer J can stay thin if it delegates to a helper that configures tiles."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto

def _demo_kernel_impl(x, y):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    _demo_kernel_impl(x, y)

def demo_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "PASS"


def test_ol04_ignores_unreachable_tile_shape_helper(tmp_path: Path):
    """A dead helper with tile config must not satisfy the JIT call-chain gate."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto

def _unused_tile_helper(x, y):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    y[:] = x

def demo_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "FAIL"


def test_ol04_fail_on_module_file_without_tile_shapes(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 module impl 漏配 tile shapes。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                        y: pypto.Tensor([1024], pypto.DT_FP32)):
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL05: kernel 张量参数必须有 pypto.Tensor 类型注解 ──

def test_ol05_pass_when_tensor_annotated(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL05")
    assert finding.status == "PASS"


def test_ol05_fail_when_no_annotation(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x, y):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL05")
    assert finding.status == "FAIL"


# ── OL06: kernel 内禁用 Python 原生 min()/max() ──

def test_ol06_pass_when_no_native_minmax(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL06")
    assert finding.status == "PASS"


def test_ol06_fail_when_native_max_used(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = max(x, 0)

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL06")
    assert finding.status == "FAIL"


def test_ol06_fail_on_module_file_with_native_max(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 module impl 中的原生 min/max。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                        y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = max(x, 0)

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL06")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL25: 禁止 pypto.Tensor() 空参数注解 ──

def test_ol25_pass_when_tensor_has_args(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL25")
    assert finding.status == "PASS"


def test_ol25_fail_when_tensor_empty_args(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor(), y: pypto.Tensor()):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL25")
    assert finding.status == "FAIL"


def test_ol25_fail_on_module_file_with_empty_bracket_annotation(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获空 [] 注解，避免拖到 Stage 6。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 删除 integrated impl，模拟仅有 module impl 的开发中状态
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL25")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL26: 张量参数必须在非张量参数之前 ──

def test_ol26_pass_when_order_correct(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL26")
    assert finding.status == "PASS"


def test_ol26_fail_when_scalar_before_tensor(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(n: int, x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(n, x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL26")
    assert finding.status == "FAIL"


def test_ol26_fail_on_module_file_with_scalar_before_tensor(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获参数顺序违规。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(n: int,
                        x: pypto.Tensor([1024], pypto.DT_FP32),
                        y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(n, x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL26")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message

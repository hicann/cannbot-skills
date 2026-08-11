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

"""D1 框架约束合规规则（OL01-OL08）核心测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def _module_impl(
    kernel_body: str,
    *,
    import_line: str = "import pypto",
    shape: str = "[1024]",
    with_jit: bool = True,
    with_wrapper: bool = True,
) -> str:
    body = "\n".join(
        f"    {line}" if line else "" for line in kernel_body.splitlines()
    )
    if with_jit:
        kernel = (
            "@pypto.frontend.jit\n"
            f"def demo_module1_kernel(x: pypto.Tensor({shape}, pypto.DT_FP32),\n"
            f"                        y: pypto.Tensor({shape}, pypto.DT_FP32)):\n"
            f"{body}\n"
        )
    else:
        kernel = f"def demo_module1_kernel(x, y):\n{body}\n"
    wrapper = (
        "\ndef demo_module1_wrapper(x, y):\n    return None\n"
        if with_wrapper else ""
    )
    return f"{import_line}\n{kernel}{wrapper}"


def _run_module_rule(
    tmp_path: Path,
    impl: str,
    rule_id: str,
):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    write_file(op_dir / "modules" / "demo_module1_impl.py", impl)
    return run_rule(mod, op_dir, rule_id)


def _two_arg_impl(kernel_body: str) -> str:
    body = "\n".join(
        f"    {line}" if line else "" for line in kernel_body.splitlines()
    )
    return f"""import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
{body}

def demo_wrapper(x, y):
    return None
"""


def test_ol01_pass_when_jit_decorator_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL01")
    assert finding.status == "PASS"


def test_ol01_fail_when_no_jit_decorator(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL01")
    assert finding.status == "FAIL"


def test_ol01_fail_when_multiple_jit_decorators(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel_a(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

@pypto.frontend.jit
def demo_kernel_b(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL01")
    assert finding.status == "FAIL"


def test_ol02_fail_on_bare_assignment(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y = x  # 违规: 应该用 y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL02")
    assert finding.status == "FAIL"


def test_ol02_pass_on_slice_assignment(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL02")
    assert finding.status == "PASS"


def test_ol02_fail_on_module_file_with_bare_assignment(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 module impl 中的写回违规。"""
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(32, 128)\n"
        "y = x  # 违规: 应该用 y[:] = x",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL02")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol03_fail_when_return_in_jit(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x
    return y  # 违规

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL03")
    assert finding.status == "FAIL"


def test_ol03_pass_when_no_return(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL03")
    assert finding.status == "PASS"


def test_ol07_fail_when_no_import_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import torch
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "FAIL"


def test_ol07_pass_when_import_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "PASS"


def test_ol07_fail_when_import_pypto_alias(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto as pt
@pt.frontend.jit
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "FAIL"
    assert "非正规 PyPTO 导入" in finding.message


def test_ol07_fail_when_import_pypto_frontend_alias_even_with_canonical_import(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import pypto.frontend as F

@pypto.frontend.jit
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "FAIL"
    assert "import pypto.frontend" in finding.message


def test_ol07_fail_when_from_pypto_import(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """from pypto import frontend
@frontend.jit
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "FAIL"
    assert "from pypto import" in finding.message


def test_ol15_fail_when_golden_imports_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    golden = """import pypto
def demo_golden(x, y):
    return x
"""
    write_file(op_dir / "demo_golden.py", golden)
    finding = run_rule(mod, op_dir, "OL15", stage=2)
    assert finding.status == "FAIL"


def test_ol15_pass_when_golden_clean(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL15", stage=2)
    assert finding.status == "PASS"


def test_ol15_fail_when_golden_uses_dot_t_attribute(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    golden = """import torch
def demo_golden(x, y):
    return x @ y.T
"""
    write_file(op_dir / "demo_golden.py", golden)
    finding = run_rule(mod, op_dir, "OL15", stage=2)
    assert finding.status == "FAIL"


def test_ol15_fail_when_golden_uses_dot_t_call(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    golden = """import torch
def demo_golden(x, y):
    return x @ y.t()
"""
    write_file(op_dir / "demo_golden.py", golden)
    finding = run_rule(mod, op_dir, "OL15", stage=2)
    assert finding.status == "FAIL"


# ── OL48: tile 参数必须编译期静态可知 ─────────────────────────────────────────


def test_ol48_pass_on_literal_int_args(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "PASS"


def test_ol48_pass_on_local_const_assign(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    D = 128
    pypto.set_vec_tile_shapes(1, D)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "PASS"


def test_ol48_pass_on_module_level_const(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
TILE_M = 32
TILE_N = 64
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(TILE_M, TILE_N)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "PASS"


def test_ol48_pass_on_cube_tile_with_literal_list(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 1024], pypto.DT_FP32), y: pypto.Tensor([1024, 1024], pypto.DT_FP32)):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(32, 32)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "PASS"


def test_ol48_fail_on_function_arg(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32), tile_m: int):
    pypto.set_vec_tile_shapes(tile_m, 64)
    y[:] = x

def demo_wrapper(x, y, tile_m):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL"
    assert "tile_m" in finding.message


def test_ol48_fail_on_tensor_shape_subscript(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32), y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(x.shape[0], 64)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL"
    assert "x.shape" in finding.message


def test_ol48_fail_on_symbolic_scalar_via_assign(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32), y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    B = x.shape[0]
    pypto.set_vec_tile_shapes(B, 64)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL"
    assert "B" in finding.message


# ── Module-file coverage: OL01/OL03/OL07/OL08 ──────────────────────────────
#
# 这些规则在 P3.2 之前仅扫描 <op>_impl.py，导致 Stage 5 阶段 module 文件
# 即使违反同一约束也能蒙混过关，要拖到 Stage 6 集成才暴露。下面 4 个测试
# 验证 module 文件违规会在 Stage 5 阶段就被捕获，且 finding.message 必须
# 包含 modules/<op>_module1_impl.py 路径以便定位。


def test_ol01_fail_on_module_file_without_jit(tmp_path: Path):
    module_impl = _module_impl("y[:] = x", with_jit=False)
    finding = _run_module_rule(tmp_path, module_impl, "OL01")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol03_fail_on_module_file_with_return(tmp_path: Path):
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(32, 128)\ny[:] = x\nreturn y",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL03")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol07_fail_on_module_file_missing_import(tmp_path: Path):
    module_impl = _module_impl(
        "y[:] = x", import_line="import torch", with_jit=False,
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL07")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol08_fail_on_module_file_missing_wrapper(tmp_path: Path):
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(32, 128)\ny[:] = x",
        with_wrapper=False,
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL08")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol48_fail_on_cube_list_element_dynamic(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 1024], pypto.DT_FP32), y: pypto.Tensor([1024, 1024], pypto.DT_FP32), m_l0: int):
    pypto.set_cube_tile_shapes([m_l0, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(32, 32)
    y[:] = x

def demo_wrapper(x, y, m_l0):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL"
    assert "m_l0" in finding.message


def test_ol48_fail_on_module_file_with_dynamic_tile_arg(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 module impl tile 非静态。"""
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(x.shape[0], 64)\ny[:] = x",
        shape="[1024, 128]",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL48")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ── Module-file coverage: OL23 (loop structure detection) ─────────────────


def test_ol23_warn_on_module_file_without_loop(tmp_path: Path):
    """module impl 漏写 loop 应在 Stage 5 阶段就 WARN。"""
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(32, 128)\ny[:] = x",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL23")
    assert finding.status == "WARN"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL49: unroll_list 只能在最内层 pypto.loop ────────────────────────────────


def test_ol49_pass_on_single_loop_with_unroll(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="batch", unroll_list=[16, 8, 4, 2, 1]):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "PASS"


def _nested_multivalue_unroll_impl():
    return _two_arg_impl(
        "B = x.shape[0]\n"
        "for b in pypto.loop(B, name=\"outer\"):\n"
        "    for k in pypto.loop(4, name=\"inner\", unroll_list=[4, 2, 1]):\n"
        "        x_tile = pypto.view(x, [1, 32], [b, k * 32])\n"
        "        y[:] = x",
    )


def test_ol49_pass_on_nested_with_unroll_on_innermost(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _nested_multivalue_unroll_impl()
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "PASS"


def test_ol49_pass_on_sibling_loops_each_with_unroll(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="loop1", unroll_list=[4, 2, 1]):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x
    for c in pypto.loop(B, name="loop2", unroll_list=[8, 4, 1]):
        x_tile = pypto.view(x, [1, 128], [c, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "PASS"


def test_ol49_fail_on_outer_with_unroll(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="outer", unroll_list=[8, 4, 2, 1]):
        for k in pypto.loop(4, name="inner"):
            x_tile = pypto.view(x, [1, 32], [b, k * 32])
            y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "FAIL"
    assert "unroll_list" in finding.message


def test_ol49_fail_on_3level_middle_with_unroll(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for a in pypto.loop(B, name="L1"):
        for b in pypto.loop(8, name="L2", unroll_list=[4, 2, 1]):
            for c in pypto.loop(4, name="L3"):
                x_tile = pypto.view(x, [1, 16], [a, b * 16])
                y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "FAIL"
    # 中间层 L2 含 unroll_list，但其 body 内还有 L3
    assert "L2" in finding.message or "unroll_list" in finding.message


def test_ol49_pass_when_no_loops(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # baseline kernel 中无 pypto.loop / unroll_list
    finding = run_rule(mod, op_dir, "OL49")
    assert finding.status == "PASS"


def test_ol49_fail_in_module_file(tmp_path: Path):
    """OL49 应同时扫描 modules/<op>_module*_impl.py。"""
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(16, 64)\n"
        "B = x.shape[0]\n"
        "for b in pypto.loop(B, name=\"outer\", unroll_list=[8, 4, 2, 1]):\n"
        "    for k in pypto.loop(4, name=\"inner\"):\n"
        "        x_tile = pypto.view(x, [1, 32], [b, k * 32])\n"
        "        y[:] = x",
        shape="[1024, 128]",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL49")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL56: Stage 6 之前 unroll_list 只能含单一值 ──────────────────────────────


def _ol56_impl(unroll_literal: str) -> str:
    return f"""import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="batch", unroll_list={unroll_literal}):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""


def _ol56_design(unroll_literal: str) -> str:
    return f"""---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---
# DESIGN

## §4 Loop structure

```python
for b in pypto.loop(B, name="batch", unroll_list={unroll_literal}):
    x_tile = pypto.view(x, [1, 128], [b, 0])
```
"""


def test_ol56_pass_single_value_one(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _ol56_impl("[1]"))
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "PASS"


def test_ol56_pass_single_value_nonone(tmp_path: Path):
    """单一非 1 值（Designer 有依据时可选）也应 PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _ol56_impl("[16]"))
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "PASS"


def test_ol56_fail_multivalue_impl(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _ol56_impl("[16, 8, 4, 2, 1]"))
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "FAIL"
    assert "unroll_list" in finding.message


def test_ol56_fail_multivalue_sibling_loops(tmp_path: Path):
    """直列（sibling）loop 中任一含多值 unroll_list 都应 FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="loop1", unroll_list=[1]):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x
    for c in pypto.loop(B, name="loop2", unroll_list=[8, 4, 1]):
        x_tile = pypto.view(x, [1, 128], [c, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "FAIL"


def test_ol56_fail_multivalue_nested_innermost(tmp_path: Path):
    """嵌套最内层含多值 unroll_list 也应 FAIL（与 OL49 正交）。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _nested_multivalue_unroll_impl()
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "FAIL"


def test_ol56_pass_no_unroll_list(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    for b in pypto.loop(B, name="batch"):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "PASS"


def test_ol56_pass_non_literal_unroll_list(tmp_path: Path):
    """unroll_list 为变量（非字面量）时无法静态判断，按 PASS 处理，避免误报。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 128], pypto.DT_FP32),
                y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 64)
    B = x.shape[0]
    cfg = [8, 4, 1]
    for b in pypto.loop(B, name="batch", unroll_list=cfg):
        x_tile = pypto.view(x, [1, 128], [b, 0])
        y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL56")
    assert finding.status == "PASS"


def test_ol56_fail_in_design_md(tmp_path: Path):
    """DESIGN.md 的 python 代码块中出现多值 unroll_list 应 FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = _ol56_design("[16, 8, 4, 2, 1]")
    write_file(op_dir / "DESIGN.md", design)
    finding = run_rule(mod, op_dir, "OL56", stage=4)
    assert finding.status == "FAIL"
    assert "DESIGN.md" in finding.message


def test_ol56_pass_design_md_single_value(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = _ol56_design("[1]")
    write_file(op_dir / "DESIGN.md", design)
    # impl 也是单值，确保整体 PASS
    write_file(op_dir / "demo_impl.py", _ol56_impl("[1]"))
    finding = run_rule(mod, op_dir, "OL56", stage=4)
    assert finding.status == "PASS"


def test_ol56_fail_in_module_file(tmp_path: Path):
    """OL56 应同时扫描 modules/<op>_module*_impl.py。"""
    module_impl = _module_impl(
        "pypto.set_vec_tile_shapes(16, 64)\n"
        "B = x.shape[0]\n"
        "for b in pypto.loop(B, name=\"batch\", unroll_list=[8, 4, 2, 1]):\n"
        "    x_tile = pypto.view(x, [1, 128], [b, 0])\n"
        "    y[:] = x",
        shape="[1024, 128]",
    )
    finding = _run_module_rule(tmp_path, module_impl, "OL56")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


# ─────────────────────────────────────────────────────────────────────────────
# OL57 — JIT 图代码内允许 pypto.loop / pypto.loop_unroll / range 循环 (禁止 while 和非 range 的 for)
# ─────────────────────────────────────────────────────────────────────────────


def test_ol57_pass_python_for_range_in_kernel_impl(tmp_path: Path):
    """GroupNorm 式: _kernel_impl 内 `for g in range(8)` → PASS (range 已放行)。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def _demo_kernel_impl(x, y):
    B = x.shape[0]
    for b in pypto.loop(B, name="batch"):
        for g in range(8):
            xg = pypto.view(x, [1, 8, 256, 256], [b, g * 8, 0, 0])
            pypto.assemble(xg, [b, g * 8, 0, 0], y)

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([1024, 128], pypto.DT_FP32), y: pypto.Tensor([1024, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(1, 16, 16, 32)
    _demo_kernel_impl(x, y)

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "PASS", finding.message


def test_ol57_pass_inverse_style_for_with_concat(tmp_path: Path):
    """inverse_pto 式: helper 内 `for i in range(8)` + list + concat → PASS (range 已放行)。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def inverse_pto(attn):
    lst = []
    for i in range(8):
        lst.append(pypto.view(attn, [16, 16], [16 * i, 16 * i]))
    return pypto.concat(lst, dim=1)

@pypto.frontend.jit
def demo_kernel_npu(attn: pypto.Tensor([128, 128], pypto.DT_FP32), out: pypto.Tensor([128, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = inverse_pto(attn)

def demo_wrapper(attn, out):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "PASS", finding.message


def test_ol57_fail_while_loop(tmp_path: Path):
    """JIT helper 内 `while` → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def _demo_kernel_impl(x, out):
    i = 0
    while i < 8:
        pypto.assemble(pypto.view(x, [16], [i * 16]), [i * 16], out)
        i += 1

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    _demo_kernel_impl(x, out)

def demo_wrapper(x, out):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "FAIL", finding.message
    assert "while" in finding.message


def test_ol57_fail_comprehension_with_pypto(tmp_path: Path):
    """JIT 内含 pypto 算子的推导式 → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    tiles = [pypto.view(x, [16], [i * 16]) for i in range(8)]
    out[:] = pypto.concat(tiles, dim=0)

def demo_wrapper(x, out):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "FAIL", finding.message
    assert "comprehension" in finding.message


def test_ol57_pass_only_pypto_loop(tmp_path: Path):
    """干净实现: 仅 pypto.loop → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def _demo_kernel_impl(x, out):
    B = x.shape[0]
    for b in pypto.loop(B, name="batch"):
        for t in pypto.loop(4, name="block", unroll_list=[1], submit_before_loop=True):
            blk = pypto.view(x, [1, 1000], [b, t * 1000])
            pypto.assemble(blk, [b, t * 1000], out)

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([1024, 4000], pypto.DT_FP32), out: pypto.Tensor([1024, 4000], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(16, 16)
    _demo_kernel_impl(x, out)

def demo_wrapper(x, out):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "PASS", finding.message


def test_ol57_pass_host_wrapper_python_loop(tmp_path: Path):
    """host wrapper 内的纯 Python 循环 (不含 pypto 算子) 不在 OL57 范围 → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    for b in pypto.loop(x.shape[0], name="b"):
        pypto.assemble(pypto.view(x, [1], [b]), [b], out)

def demo_wrapper(x):
    acc = 0
    for i in range(3):
        acc += i
    out = torch.empty_like(x)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL57")
    assert finding.status == "PASS", finding.message


# ─────────────────────────────────────────────────────────────────────────────
# OL48 — set_cube_tile_shapes 的 m/k/n 须为 2 元素 [L0,L1] 且 0<L0<=L1、L1%L0==0
# ─────────────────────────────────────────────────────────────────────────────


def _cube_impl(cube_args: str) -> str:
    return f"""import pypto
@pypto.frontend.jit
def demo_kernel_npu(
    a: pypto.Tensor([128, 128], pypto.DT_FP32),
    b: pypto.Tensor([128, 128], pypto.DT_FP32),
    out: pypto.Tensor([128, 128], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes({cube_args})
    out[:] = pypto.matmul(a, b, pypto.DT_FP32)

def demo_wrapper(a, b, out):
    return None
"""


def test_ol48_fail_cube_tile_one_element(tmp_path: Path):
    """set_cube_tile_shapes 单元素 list [16] → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _cube_impl("[16], [32], [64]"))
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL", finding.message
    assert "2 元素" in finding.message


def test_ol48_fail_cube_tile_l0_gt_l1(tmp_path: Path):
    """set_cube_tile_shapes L0 > L1 ([128,64]) → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _cube_impl("[128, 64], [128, 128], [128, 128]"))
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL", finding.message
    assert "L0 <= " in finding.message


def test_ol48_fail_cube_tile_not_divisible(tmp_path: Path):
    """set_cube_tile_shapes L1 % L0 != 0 ([3,64]) → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _cube_impl("[3, 64], [128, 128], [128, 128]"))
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "FAIL", finding.message
    assert "% " in finding.message


def test_ol48_pass_cube_tile_valid(tmp_path: Path):
    """合法 cube tile [64,128]/[64,128]/[128,256] → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_impl.py", _cube_impl("[64, 128], [64, 128], [128, 256]"))
    finding = run_rule(mod, op_dir, "OL48")
    assert finding.status == "PASS", finding.message


# ─────────────────────────────────────────────────────────────────────────────
# OL58 — Layer K wrapper output buffer must be torch.* pre-allocated
# ─────────────────────────────────────────────────────────────────────────────


def test_ol58_fail_pypto_zeros_in_wrapper(tmp_path: Path):
    """Matmul_Mish_Mish bug: pypto.zeros in host wrapper → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    out = pypto.zeros((x.shape[0],), dtype=pypto.DT_FP32, device=x.device)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "FAIL", finding.message
    assert "pypto.zeros" in finding.message


def test_ol58_fail_pypto_empty_in_wrapper(tmp_path: Path):
    """pypto.empty in host wrapper → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    out = pypto.empty((x.shape[0],), dtype=pypto.DT_FP32)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "FAIL", finding.message
    assert "pypto.empty" in finding.message


def test_ol58_fail_pypto_ones_in_wrapper(tmp_path: Path):
    """pypto.ones in host wrapper → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    out = pypto.ones((x.shape[0],), dtype=pypto.DT_FP32)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "FAIL", finding.message
    assert "pypto.ones" in finding.message


def test_ol58_pass_torch_empty_pre_allocated(tmp_path: Path):
    """Pass pattern: pre-allocate with torch.empty -> JIT call -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "PASS", finding.message


def test_ol58_pass_torch_empty_like(tmp_path: Path):
    """Pre-allocate with torch.empty_like -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    out = torch.empty_like(x)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "PASS", finding.message


def test_ol58_pass_torch_zeros_with_dtype_device(tmp_path: Path):
    """torch.zeros (dtype, device) → PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128, 20], pypto.DT_FP32), out: pypto.Tensor([128, 20], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128, 20)
    out[:] = x

def demo_wrapper(x):
    out = torch.zeros(x.shape[0], 20, dtype=torch.float32, device=x.device)
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "PASS", finding.message


def test_ol58_pass_reshaped_input_to_jit(tmp_path: Path):
    """Pass a .reshape view of a wrapper argument (not an output, intermediate view) to JIT -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(
    x2d: pypto.Tensor([128, 64], pypto.DT_FP32),
    out: pypto.Tensor([128, 64], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 64)
    out[:] = x2d

def demo_wrapper(x):
    x2d = x.reshape(-1, x.shape[-1])
    out = torch.empty(x2d.shape[0], x2d.shape[1], dtype=torch.float32, device=x.device)
    demo_kernel_npu(x2d, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "PASS", finding.message


def test_ol58_pass_alias_chain_to_torch_empty(tmp_path: Path):
    """Check B 应沿别名链识别 torch 预分配的输出 buffer。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    tmp = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    out = tmp
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "PASS", finding.message


def test_ol58_fail_output_not_allocated(tmp_path: Path):
    """Pass an identifier named 'out' to JIT while undefined in wrapper -> FAIL (Check B unknown path)."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x

def demo_wrapper(x):
    demo_kernel_npu(x, out)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "FAIL", finding.message
    assert "未在 wrapper 内分配" in finding.message


def test_ol58_skip_no_wrapper(tmp_path: Path):
    """wrapper 不在 → SKIP."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
import torch

@pypto.frontend.jit
def demo_kernel_npu(x: pypto.Tensor([128], pypto.DT_FP32), out: pypto.Tensor([128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(128)
    out[:] = x
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL58")
    assert finding.status == "SKIP", finding.message

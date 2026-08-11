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

"""D3 三文件分离规则（OL16-OL18）核心测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def test_ol16_fail_when_impl_imports_golden(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
from demo_golden import demo_golden

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL16")
    assert finding.status == "FAIL"


def test_ol16_pass_when_impl_clean(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL16")
    assert finding.status == "PASS"


def test_ol16_fail_on_module_file_importing_golden(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 module impl 误导入 golden。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
from demo_golden import demo_golden

@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                        y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL16")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol17_fail_when_test_has_jit_kernel(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import pypto
@pypto.frontend.jit
def demo_kernel(x, y):
    y[:] = x

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL17")
    assert finding.status == "FAIL"


def test_ol17_pass_when_test_clean(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL17")
    assert finding.status == "PASS"


def test_ol18_fail_when_test_missing_golden_import(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """from demo_impl import demo_wrapper

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL18")
    assert finding.status == "FAIL"


def test_ol18_pass_when_test_imports_both(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """from demo_impl import demo_wrapper
from demo_golden import demo_golden

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL18")
    assert finding.status == "PASS"

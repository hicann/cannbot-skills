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

import importlib.util
import json
import os
from pathlib import Path


def load_lint_module():
    path = Path(__file__).resolve().parents[1] / "pypto_op_lint.py"
    spec = importlib.util.spec_from_file_location("pypto_op_lint", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


SPEC_TEMPLATE = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
p0_shapes: [[1024,128], [1024,256], [1024,512]]
tolerance: {'atol': 0.001, 'rtol': 0.001}
---
# SPEC
"""

DESIGN_TEMPLATE = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---
# DESIGN
"""

API_TEMPLATE = """---
schema_version: 1
op_name: demo
---
# API REPORT
"""

IMPL_TEMPLATE = """import pypto
_N = pypto.DYNAMIC
_M = pypto.STATIC
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([_N, _M], pypto.DT_BF16), y: pypto.Tensor([_N, _M], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    tmp = pypto.cast(x, pypto.DataType.DT_FP32)
    y[:] = pypto.cast(tmp, pypto.DataType.DT_BF16)

def demo_wrapper(x, y):
    return None
"""

GOLDEN_TEMPLATE = """def demo_golden(x, y):
    return x
"""

TEST_TEMPLATE = """import torch
from numpy.testing import assert_allclose

def _run_and_check(name, n, m, scale):
    x = torch.randn(n, m, dtype=torch.bfloat16)
    y = x.float().numpy()
    assert_allclose(y, y, rtol=1e-3, atol=1e-3)

def test_level0_basic():
    _run_and_check('a', 1024, 128, 0.5)

def test_level1_basic():
    _run_and_check('b', 1024, 256, 0.5)
    _run_and_check('c', 1024, 512, 0.5)
"""


def build_stateless_op_dir(base: Path, op_name: str = "demo") -> Path:
    op_dir = base / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    write_file(op_dir / "SPEC.md", SPEC_TEMPLATE.replace("demo", op_name))
    write_file(op_dir / "DESIGN.md", DESIGN_TEMPLATE.replace("demo", op_name))
    write_file(op_dir / "API_REPORT.md", API_TEMPLATE.replace("demo", op_name))
    write_file(op_dir / f"{op_name}_impl.py", IMPL_TEMPLATE.replace("demo", op_name))
    write_file(op_dir / f"{op_name}_golden.py", GOLDEN_TEMPLATE.replace("demo", op_name))
    write_file(op_dir / f"test_{op_name}.py", TEST_TEMPLATE)
    write_file(op_dir / "README.md", "# readme\n")
    return op_dir


def run_rule(mod, op_dir: Path, rule_id: str, stage: int = 5):
    build_ctx = getattr(mod, "_build_context")
    run_checks = getattr(mod, "_run_checks")
    ctx = build_ctx(str(op_dir), stage)
    findings, _ = run_checks(ctx, [rule_id])
    return findings[0]


def run_stop(mod, cwd: str):
    os.environ.pop("PYPTO_OP_LINT_HOOK_INPUT", None)
    payload = {"cwd": cwd}
    os.environ["PYPTO_OP_LINT_HOOK_INPUT"] = json.dumps(payload, ensure_ascii=False)
    return mod.hook_stop()

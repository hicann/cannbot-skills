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

from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule


def test_ol39_and_ol40_pass_when_structured_docs_valid(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    f39 = run_rule(mod, op_dir, "OL39")
    f40 = run_rule(mod, op_dir, "OL40")
    assert f39.status == "PASS"
    assert f40.status == "PASS"


def test_ol39_fails_when_missing_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec_path = op_dir / "SPEC.md"
    spec_path.write_text("# SPEC only\n", encoding="utf-8")
    finding = run_rule(mod, op_dir, "OL39")
    assert finding.status == "FAIL"

#!/usr/bin/env python3
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

"""D2 工件完整性与流程合规规则（OL09-OL14, OL24）核心测试。"""
import json
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# ── OL09: SPEC.md 结构化章节校验 ──


def test_ol09_pass_when_spec_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
p0_shapes: [[1024, 128]]
tolerance: {'atol': 0.001, 'rtol': 0.001}
---
# SPEC
## 数学公式
$$ y = x $$
## 输入输出规格
shape: [N, M], dtype: bfloat16
## 精度要求
atol: 0.001, rtol: 0.001
"""
    write_file(op_dir / "SPEC.md", spec)
    finding = run_rule(mod, op_dir, "OL09", stage=1)
    assert finding.status == "PASS"


def test_ol09_fail_when_spec_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "SPEC.md").unlink()
    finding = run_rule(mod, op_dir, "OL09", stage=1)
    assert finding.status == "FAIL"


def test_ol09_aggregates_multiple_issues(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
p0_shapes: [[1024, 128]]
tolerance: {'atol': 0.001, 'rtol': 0.001}
---
# demo
some text without required sections
"""
    write_file(op_dir / "SPEC.md", spec)
    finding = run_rule(mod, op_dir, "OL09", stage=1)
    assert finding.status == "FAIL"
    assert "个问题" in finding.message
    assert "数学" in finding.message
    assert "输入输出" in finding.message
    assert "精度" in finding.message


# ── OL10: API_REPORT.md 结构化章节校验 ──

def test_ol10_pass_when_api_report_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    api = """---
schema_version: 1
op_name: demo
---
# API REPORT
## API 映射
mapping here
## 约束
constraints here
## Tiling
tiling here
"""
    write_file(op_dir / "API_REPORT.md", api)
    finding = run_rule(mod, op_dir, "OL10", stage=1)
    assert finding.status == "PASS"


def test_ol10_fail_when_missing_sections(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "API_REPORT.md", "# API REPORT\nsome text\n")
    finding = run_rule(mod, op_dir, "OL10", stage=1)
    assert finding.status == "FAIL"


# ── OL11: golden 文件可导入 ──

def test_ol11_pass_when_golden_importable(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL11", stage=2)
    assert finding.status == "PASS"


def test_ol11_fail_when_golden_has_syntax_error(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_golden.py", "def broken(\n")
    finding = run_rule(mod, op_dir, "OL11", stage=2)
    assert finding.status == "FAIL"


# ── OL12: DESIGN.md 结构化章节校验 ──

def test_ol12_pass_when_design_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---
# DESIGN
## API 映射
mapping here
## 数据切分
tiling here
## 验证方案
validation here
"""
    write_file(op_dir / "DESIGN.md", design)
    finding = run_rule(mod, op_dir, "OL12", stage=4)
    assert finding.status == "PASS"


def test_ol12_fail_when_design_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "DESIGN.md").unlink()
    finding = run_rule(mod, op_dir, "OL12", stage=4)
    assert finding.status == "FAIL"


# ── OL13: 三件套完整 ──

def test_ol13_pass_when_all_three_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL13", stage=5)
    assert finding.status == "PASS"


def test_ol13_fail_when_impl_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "demo_impl.py").unlink()
    finding = run_rule(mod, op_dir, "OL13", stage=5)
    assert finding.status == "FAIL"


# ── OL14: 精度通过 ──

def test_ol14_pass_when_stage5_completed(tmp_path: Path):
    """Stage 6 (Verification) entry needs Stage 5 completed in the new model."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "max_stage": 7,
        "current_stage": 6,
        "stage_status": {"5": "completed", "6": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL14", stage=6)
    assert finding.status == "PASS"


def test_ol14_fail_when_no_state_file(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL14", stage=6)
    assert finding.status == "FAIL"


# ── OL44: Stage 5 活跃 Phase 三件套（L0 跳过，L1 强制） ──


def _write_stage5_state(op_dir: Path, active_phase: str = "M1"):
    state = {
        "operator_name": "demo",
        "max_stage": 7,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
        "stage5_phases": {
            "active_phase": active_phase,
            "phase_status": {active_phase: {"status": "in_progress", "cycles": 0}},
        },
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))


def test_ol44_skip_for_l0_single_module(tmp_path: Path):
    """L0 op (MEMORY.md module_count:1, active M1, no modules/) → SKIP, not FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_stage5_state(op_dir)
    write_file(op_dir / "MEMORY.md", "decomposition_level: L0\nmodule_count: 1\n")
    finding = run_rule(mod, op_dir, "OL44", stage=5)
    assert finding.status == "SKIP"


def test_ol44_fail_for_l1_without_modules_dir(tmp_path: Path):
    """L1 op (module_count:2, active M1, no modules/) keeps failing."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_stage5_state(op_dir)
    write_file(op_dir / "MEMORY.md", "decomposition_level: L1\nmodule_count: 2\n")
    finding = run_rule(mod, op_dir, "OL44", stage=5)
    assert finding.status == "FAIL"


def test_ol44_fail_for_l1_when_module_count_unknown(tmp_path: Path):
    """No MEMORY/DESIGN module_count → keep default L1 behavior (FAIL, no modules/)."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_stage5_state(op_dir)
    finding = run_rule(mod, op_dir, "OL44", stage=5)
    assert finding.status == "FAIL"


# ── OL24: 状态文件结构合法 ──

def test_ol24_pass_when_state_valid(tmp_path: Path):
    """Schema v2.0 with max_stage, optional stage5_phases, etc. — PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "schema_version": "2.0",
        "max_stage": 8,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
        "stage_retry_count": {},
        "stage5_phases": {
            "active_phase": "M1",
            "phase_status": {"M1": {"status": "in_progress", "cycles": 0}},
            "max_cycles_per_phase": 10,
        },
        "artifact_hashes": {"spec_md": "abc"},
        "rollback_history": [],
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "PASS"


def test_ol24_warn_on_legacy_schema_without_max_stage(tmp_path: Path):
    """Legacy state.json (v1, no max_stage) is accepted with a WARN."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "WARN"


def test_ol24_fail_on_malformed_stage5_phases(tmp_path: Path):
    """stage5_phases must be a dict with phase_status."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "max_stage": 8,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
        "stage5_phases": "not a dict",
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"


def test_ol24_fail_on_malformed_rollback_history(tmp_path: Path):
    """rollback_history must be a list."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "max_stage": 8,
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
        "rollback_history": "not a list",
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"


def test_ol24_fail_when_state_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"


def test_ol24_fail_when_state_invalid_json(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / ".orchestrator_state.json", "not json")
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"

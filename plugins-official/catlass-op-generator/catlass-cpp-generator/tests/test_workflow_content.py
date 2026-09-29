# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "knowledge"
WORKFLOW = BUNDLE / "workflow"
FAMILY = BUNDLE / "operator/linear-attention"


def test_workflow_and_linear_attention_concepts_are_separated() -> None:
    workflow = {path.name for path in WORKFLOW.glob("*.md") if path.name != "index.md"}
    assert workflow == {
        "interface-and-golden-contract.md",
        "solution-design-reference.md",
        "stage-design-rules.md",
        "kernel-stage-sync-patterns.md",
        "precision-policy.md",
        "development-and-validation.md",
    }
    operator = {path.name for path in FAMILY.glob("*.md") if path.name != "index.md"}
    assert operator == {"matrix-inverse-patterns.md"}


def test_stage_rules_preserve_r01_to_r21() -> None:
    text = (WORKFLOW / "stage-design-rules.md").read_text(encoding="utf-8")
    rules = {int(value) for value in re.findall(r"\bR(\d{2})\b", text)}
    assert rules == set(range(1, 22))


def test_full_design_reasoning_projects_to_two_chapter_delivery() -> None:
    reference = (WORKFLOW / "solution-design-reference.md").read_text(encoding="utf-8")
    skill = (ROOT / "skills/catlass-cpp-design/SKILL.md").read_text(encoding="utf-8")
    template = (ROOT / "skills/catlass-cpp-design/templates/design.md.template").read_text(
        encoding="utf-8"
    )
    assert "# PR1069 原始正文" in reference
    assert "完整设计推演" in reference
    assert "docs/design.full.md" in reference
    assert "docs/design.full.md" in skill
    assert "不替代最终交付件" in skill
    titles = re.findall(r"^## \d+\. (.+)$", template, re.MULTILINE)
    assert titles == ["目标与数学语义", "Stage 总览与完整详设"]
    subsections = re.findall(r"^### \d+\.\d+ (.+)$", template, re.MULTILINE)
    assert subsections[-1] == "Workspace 总量"


def test_key_pr1069_contracts_remain_present() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (WORKFLOW, FAMILY)
        for path in directory.glob("*.md")
        if path.name != "index.md"
    )
    for phrase in (
        "reference/reference.py",
        "catlass-linear-attention-v1",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "Task Duration(us)",
        "max_abs_limit",
    ):
        assert phrase in combined


def test_kernel_timeout_rule_matches_pr1069() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    develop = (ROOT / "skills/catlass-cpp-develop/SKILL.md").read_text(encoding="utf-8")
    test = (ROOT / "skills/catlass-cpp-test/SKILL.md").read_text(encoding="utf-8")
    for text in (agents, develop, test):
        assert "60 秒无返回视为 kernel 超时" in text
        assert "清理进程和设备资源" in text
        assert "TilingKey" in text
        assert "blockDim" in text


def test_workflow_only_state_stays_outside_knowledge() -> None:
    test_skill = (ROOT / "skills/catlass-cpp-test/SKILL.md").read_text(encoding="utf-8")
    develop_skill = (ROOT / "skills/catlass-cpp-develop/SKILL.md").read_text(encoding="utf-8")
    assert "performance_optimize" in test_skill
    assert "precision_targeted" in develop_skill
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in BUNDLE.rglob("*.md")
        if path.name != "index.md"
    )
    assert "performance_optimize" not in combined
    assert "precision_targeted" not in combined


def test_architecture_gate_reaches_state_template_and_operator_knowledge() -> None:
    workflow_template = (
        ROOT / "skills/catlass-cpp-interface/templates/workflow.json"
    ).read_text(encoding="utf-8")
    design_template = (
        ROOT / "skills/catlass-cpp-design/templates/design.md.template"
    ).read_text(encoding="utf-8")
    matrix = (FAMILY / "matrix-inverse-patterns.md").read_text(encoding="utf-8")
    assert '"target_architecture": "pending"' in workflow_template
    for phrase in (
        "`target_architecture`",
        "CATLASS 架构标识",
        "Vector 执行模型",
        "Cube→Vector 数据路径",
        "Vector→Cube 数据路径",
    ):
        assert phrase in design_template
    assert 'architectures: ["atlas-a2-a3", "ascend950"]' in matrix
    assert "FixpipeParamsV220" in matrix
    assert "Ascend950 的 FP32 C0 为 8" in matrix

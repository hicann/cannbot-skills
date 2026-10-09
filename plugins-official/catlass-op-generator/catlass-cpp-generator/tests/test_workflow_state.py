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

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-cpp-interface/scripts/validate_workflow.py"
SPEC = importlib.util.spec_from_file_location("workflow_validator", SCRIPT)
workflow = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(workflow)
TEMPLATE = ROOT / "skills/catlass-cpp-interface/templates/workflow.json"


def initial() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def test_initial_state_is_valid() -> None:
    assert workflow.validate_workflow(initial()) == []


def test_sparse_initial_state_and_architecture_gate() -> None:
    sparse = json.loads(
        (ROOT / "knowledge/operator/sparse-flash-mla/workflow.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow.validate_workflow(sparse) == []
    assert (
        workflow.validate_workflow({**sparse, "target_architecture": "atlas_a2_a3"})
        == []
    )
    assert any(
        "atlas_a2_a3 only" in error
        for error in workflow.validate_workflow(
            {**sparse, "target_architecture": "ascend950"}
        )
    )
    assert workflow.validate_workflow(
        {**sparse, "workflow_id": "catlass-linear-attention-v1"}
    )


def test_bsa_identity_architecture_and_recovery() -> None:
    state = {
        **initial(),
        "algorithm_family": "block_sparse_attention",
        "workflow_id": "catlass-block-sparse-attention-arch22-v1",
    }
    assert workflow.validate_workflow(state) == []
    assert workflow.validate_workflow(
        {**state, "workflow_id": "catlass-sparse-flash-mla-v1"}
    )
    assert any(
        "atlas_a2_a3 only" in error
        for error in workflow.validate_workflow(
            {**state, "target_architecture": "ascend950"}
        )
    )
    state.update(
        stage="validation",
        target_architecture="atlas_a2_a3",
        operator_contract="frozen",
        golden_contract="frozen",
        issue_type="performance_optimize",
        resume_from="validation",
        validation_scope="full",
    )
    assert workflow.validate_workflow(state) == []


def test_all_recovery_routes_validate() -> None:
    base = initial()
    base.update(
        operator_contract="frozen",
        golden_contract="frozen",
        target_architecture="atlas_a2_a3",
    )
    scopes = {
        "precision_debug": "precision_targeted",
        "design_issue": "full",
        "reference": "full",
        "interface": "full",
        "performance_optimize": "full",
    }
    contract_overrides = {
        "reference": {"golden_contract": "provisional"},
        "interface": {
            "operator_contract": "provisional",
            "golden_contract": "provisional",
        },
    }
    for issue, stage in workflow.ISSUE_TO_RESUME.items():
        state = {
            **base,
            **contract_overrides.get(issue, {}),
            "stage": stage,
            "issue_type": issue,
            "resume_from": stage,
            "validation_scope": scopes[issue],
        }
        assert workflow.validate_workflow(state) == [], (
            issue,
            workflow.validate_workflow(state),
        )


def test_complete_requires_full_scope_and_no_issue() -> None:
    state = initial()
    state.update(
        stage="complete",
        operator_contract="frozen",
        golden_contract="frozen",
        target_architecture="ascend950",
        validation_scope="full",
    )
    assert workflow.validate_workflow(state) == []
    assert workflow.validate_workflow(
        {**state, "validation_scope": "precision_targeted"}
    )
    assert (
        workflow.validate_workflow(
            {
                **state,
                "stage": "validation",
                "issue_type": "performance_optimize",
                "resume_from": "validation",
            }
        )
        == []
    )


def test_architecture_must_be_frozen_after_interface() -> None:
    state = initial()
    state.update(
        stage="reference",
        operator_contract="frozen",
        target_architecture="pending",
    )
    errors = workflow.validate_workflow(state)
    assert any("frozen target_architecture" in error for error in errors)
    state["target_architecture"] = "atlas_a2_a3"
    assert workflow.validate_workflow(state) == []


def test_unknown_architecture_is_rejected() -> None:
    state = initial()
    state["target_architecture"] = "a5"
    assert any(
        "target_architecture must be one of" in error
        for error in workflow.validate_workflow(state)
    )

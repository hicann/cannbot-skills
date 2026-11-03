# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_SCRIPT = ROOT / "catlass-cpp-interface/scripts/validate_workflow.py"
TRANSITION_SCRIPT = ROOT / "catlass-cpp-interface/scripts/transition_workflow.py"
TEMPLATE = ROOT / "catlass-cpp-interface/templates/workflow.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load("catlass_workflow_validator", WORKFLOW_SCRIPT)
transition = _load("catlass_workflow_transition", TRANSITION_SCRIPT)


def initial() -> dict[str, object]:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def state(
    stage: str,
    issue_type: str | None = None,
    architecture: str = "atlas_a2_a3",
) -> dict[str, object]:
    return {
        "algorithm_family": "linear_attention",
        "target_architecture": architecture,
        "stage": stage,
        "issue_type": issue_type,
    }


def test_initial_template_contains_only_routing_state() -> None:
    data = initial()
    assert set(data) == workflow.WORKFLOW_FIELDS
    assert workflow.validate_workflow(data) == []


def test_registered_families_support_both_architectures() -> None:
    for family in workflow.ALGORITHM_FAMILIES:
        for architecture in workflow.TARGET_ARCHITECTURES:
            data = {
                **state("reference", architecture=architecture),
                "algorithm_family": family,
            }
            assert workflow.validate_workflow(data) == []


def test_legal_states_are_declared_explicitly() -> None:
    for stage, issue_type in workflow.VALID_STATES:
        assert workflow.validate_workflow(state(stage, issue_type)) == []

    assert workflow.validate_workflow(state("validation", "precision_debug"))
    assert workflow.validate_workflow(state("implementation", "performance_optimize"))
    assert workflow.validate_workflow(state("reference", "reference_issue"))


def test_pending_architecture_is_limited_to_interface() -> None:
    assert workflow.validate_workflow(state("interface", architecture="pending")) == []
    errors = workflow.validate_workflow(state("reference", architecture="pending"))
    assert any("requires a selected target_architecture" in error for error in errors)


def test_removed_and_unknown_fields_are_rejected() -> None:
    data = state("design")
    data["validation_scope"] = "full"
    errors = workflow.validate_workflow(data)
    assert any("unsupported fields: validation_scope" in error for error in errors)


def test_normal_chain_reaches_complete() -> None:
    data = initial()
    data = transition.transition_workflow(
        data,
        "interface_accepted",
        "atlas_a2_a3",
        "linear_attention",
    )
    assert (data["stage"], data["issue_type"]) == ("reference", None)
    for event, expected in (
        ("reference_accepted", "design"),
        ("design_accepted", "implementation"),
        ("implementation_accepted", "validation"),
        ("validation_accepted", "complete"),
    ):
        data = transition.transition_workflow(data, event)
        assert (data["stage"], data["issue_type"]) == (expected, None)


@pytest.mark.parametrize(
    ("source", "event", "target"),
    (
        (("reference", None), "interface_problem", ("interface", None)),
        (("design", None), "reference_problem", ("reference", None)),
        (("implementation", None), "design_problem", ("design", None)),
        (
            ("implementation", "precision_debug"),
            "reference_problem",
            ("reference", None),
        ),
        (("validation", None), "implementation_problem", ("implementation", None)),
        (
            ("validation", None),
            "precision_problem",
            ("implementation", "precision_debug"),
        ),
        (
            ("validation", None),
            "performance_candidate",
            ("design", "performance_optimize"),
        ),
        (("complete", None), "complete_evidence_invalid", ("validation", None)),
    ),
)
def test_problem_routes_return_to_the_owning_stage(
    source: tuple[str, str | None],
    event: str,
    target: tuple[str, str | None],
) -> None:
    data = transition.transition_workflow(state(*source), event)
    assert (data["stage"], data["issue_type"]) == target


def test_special_modes_return_to_the_normal_chain_after_acceptance() -> None:
    precision = transition.transition_workflow(
        state("implementation", "precision_debug"), "implementation_accepted"
    )
    assert (precision["stage"], precision["issue_type"]) == ("validation", None)

    performance = transition.transition_workflow(
        state("design", "performance_optimize"), "design_accepted"
    )
    assert (performance["stage"], performance["issue_type"]) == (
        "implementation",
        None,
    )


def test_architecture_reset_returns_to_interface_and_clears_selection() -> None:
    data = transition.transition_workflow(
        state("validation"), "target_architecture_reset"
    )
    assert data == state("interface", architecture="pending")


def test_undefined_transition_is_rejected_without_mutating_source() -> None:
    data = state("reference")
    original = dict(data)
    with pytest.raises(transition.TransitionError):
        transition.transition_workflow(data, "validation_accepted")
    assert data == original


def test_interface_cannot_exit_before_architecture_is_selected() -> None:
    with pytest.raises(transition.TransitionError):
        transition.transition_workflow(
            initial(),
            "interface_accepted",
            algorithm_family="linear_attention",
        )


def test_interface_cannot_exit_before_algorithm_family_is_selected() -> None:
    with pytest.raises(transition.TransitionError):
        transition.transition_workflow(
            initial(),
            "interface_accepted",
            target_architecture="atlas_a2_a3",
        )


def test_interface_acceptance_updates_algorithm_family() -> None:
    data = initial()
    data["algorithm_family"] = "general"
    updated = transition.transition_workflow(
        data,
        "interface_accepted",
        "ascend950",
        "flash_attention",
    )
    assert updated == {
        **state("reference", architecture="ascend950"),
        "algorithm_family": "flash_attention",
    }


def test_transition_cli_writes_the_validated_target_state(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(json.dumps(initial()), encoding="utf-8")
    with patch.object(
        sys,
        "argv",
        [
            "transition_workflow.py",
            "--workflow",
            str(workflow_path),
            "--event",
            "interface_accepted",
            "--target-architecture",
            "ascend950",
            "--algorithm-family",
            "linear_attention",
        ],
    ):
        assert transition.main() == 0
    updated = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert updated == state("reference", architecture="ascend950")


def test_architecture_error_uses_supported_values_and_official_names() -> None:
    data = initial()
    data["target_architecture"] = "a5"
    errors = workflow.validate_workflow(data)
    assert any(
        "atlas_a2_a3 (Atlas A2系列产品、Atlas A3系列产品)" in error
        and "ascend950 (Ascend 950PR&950DT系列产品)" in error
        for error in errors
    )

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Apply one registered transition to a CATLASS C++ workflow state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_workflow import (
    ALGORITHM_FAMILIES,
    TARGET_ARCHITECTURES,
    VALID_STATES,
    validate_workflow,
)


State = tuple[str, str | None]
TransitionKey = tuple[State, str]


def _routes(
    sources: tuple[State, ...], event: str, target: State
) -> dict[TransitionKey, State]:
    return {(source, event): target for source in sources}


DOWNSTREAM_OF_INTERFACE: tuple[State, ...] = (
    ("reference", None),
    ("design", None),
    ("design", "performance_optimize"),
    ("implementation", None),
    ("implementation", "precision_debug"),
    ("validation", None),
)
DOWNSTREAM_OF_REFERENCE: tuple[State, ...] = (
    ("design", None),
    ("design", "performance_optimize"),
    ("implementation", None),
    ("implementation", "precision_debug"),
    ("validation", None),
)
DOWNSTREAM_OF_DESIGN: tuple[State, ...] = (
    ("implementation", None),
    ("implementation", "precision_debug"),
    ("validation", None),
)


TRANSITIONS: dict[TransitionKey, State] = {
    (("interface", None), "interface_accepted"): ("reference", None),
    (("reference", None), "reference_accepted"): ("design", None),
    (("design", None), "design_accepted"): ("implementation", None),
    (
        ("design", "performance_optimize"),
        "design_accepted",
    ): ("implementation", None),
    (("implementation", None), "implementation_accepted"): (
        "validation",
        None,
    ),
    (
        ("implementation", "precision_debug"),
        "implementation_accepted",
    ): ("validation", None),
    (("validation", None), "validation_accepted"): ("complete", None),
    (("implementation", None), "precision_problem"): (
        "implementation",
        "precision_debug",
    ),
    (("validation", None), "precision_problem"): (
        "implementation",
        "precision_debug",
    ),
    (("validation", None), "implementation_problem"): ("implementation", None),
    (("validation", None), "performance_candidate"): (
        "design",
        "performance_optimize",
    ),
    (("complete", None), "complete_evidence_invalid"): ("validation", None),
    **_routes(DOWNSTREAM_OF_INTERFACE, "interface_problem", ("interface", None)),
    **_routes(DOWNSTREAM_OF_REFERENCE, "reference_problem", ("reference", None)),
    **_routes(DOWNSTREAM_OF_DESIGN, "design_problem", ("design", None)),
    **_routes(tuple(VALID_STATES), "target_architecture_reset", ("interface", None)),
}


class TransitionError(ValueError):
    """The requested workflow transition is not legal."""


def transition_workflow(
    data: dict[str, Any],
    event: str,
    target_architecture: str | None = None,
    algorithm_family: str | None = None,
) -> dict[str, Any]:
    source_errors = validate_workflow(data)
    if source_errors:
        raise TransitionError("; ".join(source_errors))

    source = (str(data["stage"]), data["issue_type"])
    target = TRANSITIONS.get((source, event))
    if target is None:
        raise TransitionError(f"event {event} is not valid from state {source}")

    updated = dict(data)
    updated["stage"], updated["issue_type"] = target
    if algorithm_family is not None:
        if event != "interface_accepted":
            raise TransitionError(
                "algorithm family can only be selected by interface_accepted"
            )
        if algorithm_family not in ALGORITHM_FAMILIES:
            raise TransitionError("unsupported algorithm family")
        updated["algorithm_family"] = algorithm_family

    if event == "target_architecture_reset":
        if target_architecture is not None:
            raise TransitionError(
                "target_architecture_reset does not accept a target architecture"
            )
        updated["target_architecture"] = "pending"
    elif target_architecture is not None:
        if event != "interface_accepted":
            raise TransitionError(
                "target architecture can only be selected by interface_accepted"
            )
        if target_architecture not in TARGET_ARCHITECTURES:
            raise TransitionError("unsupported target architecture")
        updated["target_architecture"] = target_architecture

    if event == "interface_accepted" and updated["target_architecture"] == "pending":
        raise TransitionError(
            "interface_accepted requires a selected target architecture"
        )
    if event == "interface_accepted" and algorithm_family is None:
        raise TransitionError("interface_accepted requires a selected algorithm family")

    target_errors = validate_workflow(updated)
    if target_errors:
        raise TransitionError("; ".join(target_errors))
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--event", required=True)
    parser.add_argument("--target-architecture", choices=tuple(TARGET_ARCHITECTURES))
    parser.add_argument("--algorithm-family", choices=ALGORITHM_FAMILIES)
    args = parser.parse_args()

    if not args.workflow.is_file():
        print(f"FAIL\n- workflow file not found: {args.workflow}")
        return 1
    try:
        data = json.loads(args.workflow.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL\n- invalid workflow JSON: {exc}")
        return 1
    if not isinstance(data, dict):
        print("FAIL\n- workflow JSON must be an object")
        return 1
    try:
        updated = transition_workflow(
            data,
            args.event,
            args.target_architecture,
            args.algorithm_family,
        )
    except TransitionError as exc:
        print(f"FAIL\n- {exc}")
        return 1

    temporary = args.workflow.with_name(f".{args.workflow.name}.tmp")
    temporary.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.workflow)
    print(
        "PASS: workflow transitioned to "
        f"stage={updated['stage']}, issue_type={updated['issue_type']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

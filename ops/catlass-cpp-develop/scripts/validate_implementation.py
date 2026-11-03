# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Run the generic 04 implementation gate over workflow, design and quick evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def implementation_architecture_errors(design_text: str) -> list[str]:
    """Check that implementation inputs still describe the performance architecture."""
    errors: list[str] = []
    if not re.search(r"并行", design_text):
        errors.append("implementation design must retain cross-work parallelism")
    if not re.search(r"流水", design_text):
        errors.append("implementation design must retain Stage pipelining")
    formal_serial_patterns = (
        r"正式设备方案\s*逐\s*work\s*串行",
        r"正式(?:设备)?实现[^。\n]{0,12}(?:采用|使用|为|按)\s*(?:逐\s*work\s*)?串行",
        r"实现准出[^。\n]{0,12}串行",
        r"性能\s*baseline[^。\n]{0,12}串行",
    )
    if any(re.search(pattern, design_text) for pattern in formal_serial_patterns):
        errors.append("formal implementation cannot use per-work serial execution")
    return errors


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(
    design_path: Path,
    workflow_path: Path,
    validation_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read workflow JSON: {workflow_path}: {exc}"]
    if not isinstance(workflow, dict):
        return ["workflow JSON must be an object"]
    workflow_validator = _load(
        "catlass_workflow_validator",
        ROOT / "catlass-cpp-interface/scripts/validate_workflow.py",
    )
    errors.extend(workflow_validator.validate_workflow(workflow))
    if workflow.get("stage") != "implementation":
        errors.append(
            "implementation validation requires workflow stage=implementation"
        )

    try:
        design_text = design_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read design document: {design_path}: {exc}")
        design_text = ""
    design_validator = _load(
        "catlass_design_validator",
        ROOT / "catlass-cpp-design/scripts/validate_design.py",
    )
    errors.extend(design_validator.validate(design_text))
    errors.extend(design_validator.workflow_alignment_errors(design_text, workflow))
    errors.extend(implementation_architecture_errors(design_text))

    try:
        validation_text = validation_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read validation record: {validation_path}: {exc}")
        validation_text = ""
    validation_validator = _load(
        "catlass_validation_validator",
        ROOT / "catlass-cpp-test/scripts/validate_validation.py",
    )
    errors.extend(validation_validator.validate(validation_text, workflow))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.design, args.workflow, args.validation)
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: implementation inputs and quick evidence are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

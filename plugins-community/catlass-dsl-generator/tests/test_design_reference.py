# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-dsl-design/scripts/validate_reference.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_inputs(root, *, invalid_output=False, mismatched_source=False):
    reference = root / "reference.py"
    expression = "1.0" if invalid_output else "x * 2 + bias"
    source = "def run(x, bias=1.0):\n    return {}\n".format(expression)
    reference.write_text(source, encoding="utf-8")
    definition = root / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "name": "affine",
                "axes": {"N": {"type": "var"}},
                "inputs": {
                    "x": {"shape": ["N"], "dtype": "float32"},
                    "bias": {"shape": None, "dtype": "float32"},
                },
                "outputs": {
                    "output": {"shape": ["N"], "dtype": "float32"}
                },
                "reference": source + ("# drift\n" if mismatched_source else ""),
            }
        ),
        encoding="utf-8",
    )
    workload = root / "workload.jsonl"
    workload.write_text(
        json.dumps(
            {
                "uuid": "correctness-main",
                "axes": {"N": 2},
                "inputs": {
                    "x": {"type": "random"},
                    "bias": {"type": "scalar", "value": 1.0},
                },
                "tolerance": {
                    "max_atol": 0.0,
                    "max_rtol": 0.0,
                    "required_matched_ratio": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return reference, definition, workload


def test_reference_validator_runs_definition_workloads(tmp_path):
    module = load_module()
    reference, definition, workload = write_inputs(tmp_path)
    result = module.validate_reference(reference, definition, workload)

    assert result["status"] == "passed"
    assert result["reference_sha256"] == hashlib.sha256(
        reference.read_bytes()
    ).hexdigest()
    assert result["definition_sha256"] == hashlib.sha256(
        definition.read_bytes()
    ).hexdigest()
    assert result["workload_sha256"] == hashlib.sha256(
        workload.read_bytes()
    ).hexdigest()
    assert result["cases"][0]["case_id"] == "correctness-main"
    assert result["cases"][0]["outputs"] == [
        {"path": "$", "shape": [2], "dtype": "float32", "device": "cpu"}
    ]


def test_reference_validator_normalizes_result_into_state(tmp_path):
    module = load_module()
    reference, definition, workload = write_inputs(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "schema": "catlass.dsl.workflow.v3",
        "config": {"approval": {"config_digest": "old"}},
        "config_digest": "old",
    }))

    result = module.validate_reference(reference, definition, workload)
    module.update_state(state_path, result)

    state = json.loads(state_path.read_text())
    normalized = state["config"]["reference_validation"]
    assert normalized["status"] == "passed"
    assert normalized["cases"][0]["case_id"] == "correctness-main"
    assert state["config"]["approval"]["config_digest"] == "<computed>"
    assert "config_digest" not in state


def test_reference_validator_rejects_source_drift_and_invalid_output(tmp_path):
    module = load_module()
    reference, definition, workload = write_inputs(
        tmp_path, mismatched_source=True
    )
    with pytest.raises(module.ReferenceError, match="逐字一致"):
        module.validate_reference(reference, definition, workload)

    reference, definition, workload = write_inputs(tmp_path, invalid_output=True)
    with pytest.raises(module.ReferenceError, match="reference 输出"):
        module.validate_reference(reference, definition, workload)


def test_reference_validator_rejects_opaque_case_axis(tmp_path):
    module = load_module()
    reference, definition, workload = write_inputs(tmp_path)
    definition_data = json.loads(definition.read_text(encoding="utf-8"))
    definition_data["axes"]["CASE"] = {"type": "var"}
    definition.write_text(json.dumps(definition_data), encoding="utf-8")
    workload_data = json.loads(workload.read_text(encoding="utf-8"))
    workload_data["axes"]["CASE"] = 0
    workload.write_text(json.dumps(workload_data) + "\n", encoding="utf-8")

    with pytest.raises(module.ReferenceError, match="完整配置必须保存在 workload.jsonl"):
        module.validate_reference(reference, definition, workload)


def test_reference_validator_accepts_attention_bench_format(tmp_path):
    module = load_module()
    source = "import torch\n\ndef run(x, mask=None):\n    return x if mask is None else x * mask\n"
    reference = tmp_path / "reference.py"
    reference.write_text(source, encoding="utf-8")
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "name": "attention_style",
                "op_type": "attention_style",
                "axes": {},
                "inputs": {
                    "x": {"description": "input; fp16/bf16"},
                    "mask": {"description": "optional mask"},
                },
                "outputs": {"output": {"description": "same as x"}},
                "reference": source,
                "kernel_reference": "upstream kernel text",
            }
        ),
        encoding="utf-8",
    )
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        json.dumps(
            {
                "uuid": "attention-fp16",
                "axes": None,
                "inputs": {
                    "x": {"type": "ones", "shape": [2, 4], "dtype": "float16"},
                    "mask": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.validate_reference(reference, definition, workload)

    assert result["status"] == "passed"
    assert result["cases"][0]["outputs"][0]["shape"] == [2, 4]
    assert result["cases"][0]["outputs"][0]["dtype"] == "float16"

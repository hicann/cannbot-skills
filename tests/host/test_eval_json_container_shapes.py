# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def load_module(monkeypatch):
    def load(relative):
        spec = importlib.util.spec_from_file_location(
            "report_host_" + Path(relative).stem, ROOT / relative
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize(
    "data",
    [
        [],
        None,
        {"skill_name": "demo", "evals": None},
        {"skill_name": "demo", "evals": [1]},
        {"skill_name": "demo", "evals": [{"config": None}]},
        {"skill_name": "demo", "evals": [{"expectations": None}]},
        {"skill_name": "demo", "evals": [{"expectations": [1]}]},
    ],
    ids=[
        "top-list",
        "top-null",
        "evals-null",
        "item-number",
        "config-null",
        "expectations-null",
        "expectation-number",
    ],
)
def test_invalid_containers_return_none(load_module, tmp_path, data):
    module = load_module("tests/system/scripts/evals_json_parser.py")
    file = tmp_path / "evals.json"
    file.write_text(json.dumps(data))
    assert module.parse_evals_json(file) is None


@pytest.mark.parametrize("kind", ["skill", "team"])
def test_valid_evaluation_retains_its_target_and_expectations(
    load_module, tmp_path, kind
):
    module = load_module("tests/system/scripts/evals_json_parser.py")
    file = tmp_path / "evals.json"
    file.write_text(
        json.dumps(
            {
                kind + "_name": "demo",
                "evals": [
                    {
                        "id": 1,
                        "prompt": "hello",
                        "config": {"max_tokens": 100},
                        "expectations": [{"type": "contains", "pattern": "hi"}],
                    }
                ],
            }
        )
    )
    result = module.parse_evals_json(file)
    assert result["target_type"] == kind
    assert result["target_name"] == "demo"
    assert result["evals"][0]["max_tokens"] == 100
    assert result["evals"][0]["expectations"] == [{"type": "contains", "pattern": "hi"}]

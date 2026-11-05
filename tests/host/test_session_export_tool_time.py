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
    "nested", [True, False], ids=["tool-state", "legacy-top-level"]
)
def test_export_file_counts_completed_tool_time(load_module, tmp_path, nested):
    module = load_module("tests/system/scripts/session_stats.py")
    part = {
        "type": "tool",
        "tool": "bash",
        "state": {"status": "completed", "input": {"command": "pytest -q"}},
    }
    (part["state"] if nested else part)["time"] = {"start": 1000, "end": 1450}
    file = tmp_path / "session.json"
    file.write_text(
        json.dumps({"info": {"id": "ses_test"}, "messages": [{"parts": [part]}]})
    )

    stats = module.SessionStats.from_export_file(str(file))
    assert stats.tool_time_ms == 450
    assert stats.tool_calls == {"bash": 1}
    assert stats.test_runs == 1


def test_export_text_and_reasoning_keep_their_top_level_times(load_module, tmp_path):
    module = load_module("tests/system/scripts/session_stats.py")
    file = tmp_path / "session.json"
    file.write_text(
        json.dumps(
            {
                "info": {"id": "ses_test"},
                "messages": [
                    {
                        "parts": [
                            {"type": "text", "time": {"start": 1000, "end": 1500}},
                            {"type": "reasoning", "time": {"start": 1000, "end": 1700}},
                        ]
                    }
                ],
            }
        )
    )
    stats = module.SessionStats.from_export_file(str(file))
    assert (stats.output_time_ms, stats.reasoning_time_ms, stats.tool_time_ms) == (
        500,
        700,
        0,
    )

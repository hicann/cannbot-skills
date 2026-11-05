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
import subprocess
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


def test_json_cli_emits_parseable_result_to_stdout(tmp_path):
    file = tmp_path / "session.json"
    file.write_text(
        json.dumps({"info": {"id": "ses_stdout", "title": "example"}, "messages": []})
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/system/scripts/session_stats.py"),
            str(file),
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["session_id"] == "ses_stdout"
    assert result.stderr == ""


def test_human_cli_preserves_report_channel(tmp_path):
    file = tmp_path / "session.json"
    file.write_text(json.dumps({"info": {"id": "ses_human"}, "messages": []}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/system/scripts/session_stats.py"),
            str(file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert "Session Stats Report: ses_human" in result.stderr

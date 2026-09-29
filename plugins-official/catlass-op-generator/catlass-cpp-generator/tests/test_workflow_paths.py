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

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], cwd: Path, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def test_python_tools_work_from_unrelated_directory(tmp_path: Path) -> None:
    other = tmp_path / "unrelated path"
    other.mkdir()
    workflow = ROOT / "skills/catlass-cpp-interface/templates/workflow.json"
    run(
        [
            sys.executable,
            str(ROOT / "skills/catlass-cpp-interface/scripts/validate_workflow.py"),
            "--workflow",
            str(workflow),
        ],
        other,
    )
    source = tmp_path / "reference.py"
    template = tmp_path / "definition.template.json"
    output = tmp_path / "definition.json"
    source.write_text("def reference(x):\n    return x\n", encoding="utf-8")
    shutil.copy2(ROOT / "skills/catlass-cpp-reference/templates/definition.template.json", template)
    run(
        [
            sys.executable,
            str(ROOT / "skills/catlass-cpp-reference/scripts/generate_definition.py"),
            "--source",
            str(source),
            "--template",
            str(template),
            "--output",
            str(output),
        ],
        other,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["reference"] == source.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_installer_check_and_conservative_uninstall(tmp_path: Path) -> None:
    project = tmp_path / "install target"
    project.mkdir()
    foreign = project / "keep.txt"
    foreign.write_text("keep", encoding="utf-8")
    init = ROOT / "init.sh"
    run(["bash", str(init), "project", "codex", str(project)], tmp_path)
    run(["bash", str(init), "--check", "project", "codex", str(project)], tmp_path)
    run(["bash", str(init), "--uninstall", "project", "codex", str(project)], tmp_path)
    assert foreign.read_text(encoding="utf-8") == "keep"
    assert not (project / ".agents/skills/catlass-cpp-interface").exists()
    assert not (project / ".agents/catlass-cpp-knowledge").exists()

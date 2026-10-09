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


def run(
    args: list[str], cwd: Path, expected: int = 0
) -> subprocess.CompletedProcess[str]:
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
    shutil.copy2(
        ROOT / "skills/catlass-cpp-reference/templates/definition.template.json",
        template,
    )
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
    assert json.loads(output.read_text(encoding="utf-8"))[
        "reference"
    ] == source.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
@pytest.mark.parametrize(
    "family,workflow_id",
    [
        ("linear_attention", "catlass-linear-attention-v1"),
        ("block_sparse_attention", "catlass-block-sparse-attention-arch22-v1"),
        ("sparse_flash_mla", "catlass-sparse-flash-mla-v1"),
    ],
)
def test_project_initializer_uses_dedicated_workflow(
    tmp_path: Path, family: str, workflow_id: str
) -> None:
    script = ROOT / "skills/catlass-cpp-interface/scripts/init_operator_project.sh"
    name = "catlass_custom_attention"
    args = ["bash", str(script), name, "--algorithm-family", family]
    run(args, tmp_path)
    marker = tmp_path / "operators" / name / "docs/workflow.json"
    state = json.loads(marker.read_text(encoding="utf-8"))
    assert state["algorithm_family"] == family
    assert state["workflow_id"] == workflow_id
    run(args, tmp_path)
    assert json.loads(marker.read_text(encoding="utf-8")) == state
    run(["bash", str(script), name], tmp_path)
    assert json.loads(marker.read_text(encoding="utf-8")) == state
    before = marker.read_bytes()
    other_family = (
        "linear_attention" if family != "linear_attention" else "sparse_flash_mla"
    )
    result = run(
        ["bash", str(script), name, "--algorithm-family", other_family],
        tmp_path,
        expected=1,
    )
    assert "does not match existing workflow marker" in result.stderr
    assert marker.read_bytes() == before


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

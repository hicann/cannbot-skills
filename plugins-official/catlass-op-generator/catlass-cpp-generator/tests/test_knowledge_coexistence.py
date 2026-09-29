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

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
CPP_SCRIPT = ROOT / "skills/catlass-cpp-knowledge/scripts/record_knowledge.py"
DSL_ROOT = REPO_ROOT / "plugins-community/catlass-dsl-generator"
DSL_SCRIPT = DSL_ROOT / "skills/catlass-dsl-knowledge/scripts/record_knowledge.py"


def digest_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.skipif(not DSL_SCRIPT.is_file(), reason="DSL plugin is not available in this checkout")
def test_cpp_and_dsl_knowledge_coexist_without_shared_paths(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(DSL_SCRIPT), "initialize", "--project-root", str(tmp_path)],
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dsl_root = tmp_path / ".catlass-dsl/knowledge"
    before = digest_tree(dsl_root)

    subprocess.run(
        [sys.executable, str(CPP_SCRIPT), "initialize", "--project-root", str(tmp_path)],
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    cpp_root = tmp_path / ".catlass-cpp/knowledge"
    assert dsl_root.is_dir() and cpp_root.is_dir()
    assert digest_tree(dsl_root) == before
    assert not (cpp_root / "dsl").exists()
    assert not (cpp_root / "learned").exists()

    cpp_skills = {path.name for path in (ROOT / "skills").iterdir() if path.is_dir()}
    dsl_skills = {path.name for path in (DSL_ROOT / "skills").iterdir() if path.is_dir()}
    assert cpp_skills.isdisjoint(dsl_skills)

#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused checks for the plan schema and effective edit gating."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from op_autoresearch.utils.console import emit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "engine"))
from create_plan import _validate_items
from quick_check import effective_edit_issue

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _repo_with_kernel(content: str) -> Path:
    td = Path(tempfile.mkdtemp(prefix="ar_edit_gate_"))
    _run(["git", "init"], td)
    _run(["git", "config", "user.name", "test"], td)
    _run(["git", "config", "user.email", "test@example.invalid"], td)
    (td / "kernel.py").write_text(content, encoding="utf-8")
    _run(["git", "add", "kernel.py"], td)
    _run(["git", "commit", "-m", "seed"], td)
    return td


def _issue_after(new_content: str | None) -> dict | None:
    td = _repo_with_kernel("x = 1\n")
    try:
        if new_content is not None:
            (td / "kernel.py").write_text(new_content, encoding="utf-8")
        cfg = SimpleNamespace(editable_files=["kernel.py"])
        return effective_edit_issue(str(td), cfg)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_effective_edit_gate() -> None:
    clean = _issue_after(None)
    assert clean and "Zero-edit" in clean["report"]

    comments = _issue_after("x = 1\n# explain the intended optimization\n")
    assert comments and "Comment-only edit" in comments["report"]

    code = _issue_after("x = 2\n")
    assert code is None


def test_plan_accepts_workflow_fields() -> None:
    base = {
        "desc": "Fuse the vector pass into one kernel",
        "rationale": (
            "This removes redundant global-memory traffic by fusing "
            "the producer and consumer vector passes."
        ),
    }
    _validate_items([dict(base), dict(base), dict(base)])


def main() -> int:
    test_effective_edit_gate()
    test_plan_accepts_workflow_fields()
    emit("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

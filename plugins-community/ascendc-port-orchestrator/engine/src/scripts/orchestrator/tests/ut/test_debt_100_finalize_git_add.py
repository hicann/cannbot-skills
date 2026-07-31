# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-100: finalize_op auto-stages the archive directory in git.

Background: port_a3 finalize was leaving 30-60 archive files untracked
locally (e.g. foreach_sqrt 2026-05-20 needed manual `git add` in commit
36461086 covering 32 files). The orchestrator successfully promoted the
files to `output/<project>/src/kernels/<op>/` but never staged them — so
peer agents pulling fresh saw only verification.json + a handful of stray
files, and `detect_plugin()` on a fresh checkout returned None for port_a3
ops.

Fix: end of finalize_op runs `git -C _PROJECT_ROOT add -- <rel(archive)>`.
Non-blocking on failure (not-a-repo / readonly / archive-outside-project).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_GIT = shutil.which("git")
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402
import finalize_dispatch as fd  # noqa: E402

# DEBT-201 (2026-07-06): finalize_op moved to finalize_dispatch.py and reads
# _PROJECT_ROOT by BARE NAME there. Patch _PROJECT_ROOT on finalize_dispatch
# (the module that actually resolves the name), not the finalize_pipeline
# re-export — otherwise the git-add path uses the real repo root, not tmp.


@pytest.fixture(autouse=True)
def _isolate_kb_auto_promote(monkeypatch):
    """Same isolation as test_finalize_pipeline.py — prevent real KB codex spawns."""
    from src.scripts.orchestrator import kb_auto_promote

    def _noop_run(*args, **kwargs):
        rpt = kb_auto_promote.PromotionBatchReport(markers_processed=0)
        rpt.finished_ts = rpt.started_ts
        return rpt
    monkeypatch.setattr(kb_auto_promote, "run_auto_promote", _noop_run)


def _seed_workspace(ws: Path):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "verification.json").write_text(json.dumps({"precision": {"status": "PASS"}}))
    (ws / "PROGRESS.md").write_text("# progress")
    (ws / "model.py").write_text("import torch\n")
    (ws / "model_new_ascendc.py").write_text("# pybind\n")
    (ws / "manifest.json").write_text(json.dumps({"op": "x"}))
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.h").write_text("// kernel")


def _init_git_repo(repo: Path):
    """Init a minimal git repo at `repo` so `git -C add` works."""
    assert _GIT is not None, "git executable not found"
    subprocess.run([_GIT, "init", "-q"], cwd=repo, check=True)
    subprocess.run([_GIT, "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run([_GIT, "config", "user.name", "test"], cwd=repo, check=True)
    # initial commit so HEAD exists
    (repo / ".gitkeep").write_text("")
    subprocess.run([_GIT, "add", ".gitkeep"], cwd=repo, check=True)
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)


def test_git_staged_when_archive_inside_project_root(tmp_path, monkeypatch):
    """Archive under _PROJECT_ROOT — finalize should run `git add` and stage files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    monkeypatch.setattr(fd, "_PROJECT_ROOT", repo)

    ws = repo / "workspace" / "test_op"
    archive_root = repo / "output" / "test_proj" / "src" / "kernels"
    archive_root.mkdir(parents=True)
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    assert rep.git_staged is True, f"errors={rep.errors}"
    # Verify the files are actually in git's index
    out = subprocess.run(
        [_GIT, "-C", str(repo), "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout
    staged = set(out.strip().splitlines())
    assert "output/test_proj/src/kernels/test_op/verification.json" in staged
    assert "output/test_proj/src/kernels/test_op/model.py" in staged
    assert "output/test_proj/src/kernels/test_op/kernel/k.h" in staged


def test_git_staged_false_when_archive_outside_project(tmp_path, monkeypatch):
    """Archive outside _PROJECT_ROOT (tmp_path scenario) — skip git-add cleanly.

    No error appended; just rep.git_staged = False. Finalize must still succeed.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    monkeypatch.setattr(fd, "_PROJECT_ROOT", repo)

    # Workspace + archive both OUTSIDE the project root
    ws = tmp_path / "elsewhere" / "test_op"
    archive_root = tmp_path / "elsewhere_archive"
    archive_root.mkdir()
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    assert rep.git_staged is False
    # The outside-project skip should NOT add an error (it's an intended path)
    git_errors = [e for e in rep.errors if "git-add auto-stage" in e]
    assert git_errors == [], f"unexpected git-add errors: {git_errors}"


def test_git_add_failure_non_blocking(tmp_path, monkeypatch):
    """If git fails (e.g. _PROJECT_ROOT isn't a repo), finalize still completes
    with rep.git_staged=False + error appended — does NOT raise.
    """
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    # Deliberately do NOT init git
    monkeypatch.setattr(fd, "_PROJECT_ROOT", not_a_repo)

    ws = not_a_repo / "workspace" / "test_op"
    archive_root = not_a_repo / "output" / "test_proj" / "src" / "kernels"
    archive_root.mkdir(parents=True)
    _seed_workspace(ws)

    rep = fp.finalize_op("test_op", ws, archive_root=archive_root)

    # Finalize itself succeeded (precision PASS, files promoted)
    assert "verification.json" in rep.files_promoted
    # But git-add failed cleanly
    assert rep.git_staged is False
    git_errors = [e for e in rep.errors if "git-add auto-stage" in e]
    assert len(git_errors) == 1, f"expected 1 git-add error, got: {git_errors}"

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

"""Behavior checks for whole-worktree cleanliness detection."""

import subprocess

from op_autoresearch.op.autoresearch.framework.git_repo import GitRepo


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_is_clean_covers_tracked_and_untracked_changes(tmp_path):
    _git(tmp_path, "init")
    tracked = tmp_path / "kernel.py"
    tracked.write_text("seed\n", encoding="utf-8")
    _git(tmp_path, "add", "kernel.py")
    _git(
        tmp_path,
        "-c",
        "user.name=AutoResearch Test",
        "-c",
        "user.email=autoresearch@example.invalid",
        "commit",
        "-m",
        "seed",
    )

    repo = GitRepo(str(tmp_path))
    assert repo.is_clean() is True

    tracked.write_text("changed\n", encoding="utf-8")
    assert repo.is_clean() is False

    tracked.write_text("seed\n", encoding="utf-8")
    assert repo.is_clean() is True

    (tmp_path / "untracked.py").write_text("new\n", encoding="utf-8")
    assert repo.is_clean() is False

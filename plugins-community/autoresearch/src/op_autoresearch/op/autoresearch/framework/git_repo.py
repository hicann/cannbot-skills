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

"""
GitRepo — git operations bound to a single task directory.

Before this module the autoresearch layer talked to git via 8
module-level functions in ``runner.py`` (``git_commit``,
``git_rollback_files``, ``git_current_commit``, ``git_dirty_files``,
``git_diff``, ``git_current_branch``, ``git_ensure_branch``,
``git_cleanup_branch``). Each one re-resolved the repo root via
``git rev-parse --show-toplevel`` on every call, and consumers in
session.py / loop.py / turn.py / autoresearch_workflow.py imported
them à la carte.

GitRepo collects them into one class with a cached ``repo_root``
property and a single ``task_dir`` instance, so callers hold one
handle instead of importing N helpers.
"""


import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .config import CommitResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitOptions:
    """Optional staging, branch, and delivery controls for a commit."""

    files: Optional[list[str]] = None
    push: bool = False
    task_name: Optional[str] = None
    expected_branch: Optional[str] = None


def _git_repo_root(task_dir: str) -> str:
    """Resolve the git repo root for ``task_dir``. Raises on failure.

    Module-private helper because every GitRepo instance caches its
    own resolved root and never calls this in the hot path — only
    during construction or first access.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=task_dir, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {task_dir}\n{result.stderr}")
    return result.stdout.strip()


def _git_add(repo_root: str, rel_path: str) -> bool:
    """``git add`` a single path. Returns True on success.

    On WSL2 with /mnt/c/ the stat cache can be stale, causing ``git
    add`` to skip genuinely modified files. The pre-add ``git diff``
    forces an index refresh for that path.
    """
    subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo_root, capture_output=True, check=False
    )
    result = subprocess.run(
        ["git", "add", rel_path],
        cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.warning(
            "git add failed for %s: %s", rel_path, result.stderr.strip()
        )
        return False
    return True


def _file_is_dirty(task_dir: str, repo_root: str, file_name: str,
                   rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", rel_path],
        capture_output=True, text=True, cwd=repo_root, check=False
    )
    if result.returncode == 0 and result.stdout.strip():
        return True
    if not os.path.exists(os.path.join(task_dir, file_name)):
        return False
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", rel_path],
        capture_output=True, text=True, cwd=repo_root, check=False
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _rollback_file(task_dir: str, repo_root: str, file_name: str,
                   rel_path: str) -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=repo_root, capture_output=True, check=False
    )
    if result.returncode != 0:
        file_path = os.path.join(task_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Removed untracked file: %s", rel_path)
        return
    result = subprocess.run(
        ["git", "checkout", "HEAD", "--", rel_path],
        cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.warning("Rollback failed for %s: %s", rel_path, result.stderr.strip())


def _checkout_fallback_branch(repo_root: str) -> None:
    for candidate in ("main", "master"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            capture_output=True, cwd=repo_root, check=False
        )
        if result.returncode == 0:
            subprocess.run(
                ["git", "checkout", candidate],
                capture_output=True, cwd=repo_root, check=False
            )
            return


class GitRepo:
    """Git operations bound to a single task directory.

    All methods that take a path are interpreted relative to
    ``task_dir`` and resolved against the cached ``repo_root``.
    Construction is lazy — ``repo_root`` is resolved on first access
    and cached for the rest of the instance lifetime.
    """

    def __init__(self, task_dir: str):
        self.task_dir = os.path.abspath(task_dir)
        self._repo_root: Optional[str] = None

    @property
    def repo_root(self) -> str:
        """Cached git repo root for ``self.task_dir``. Resolves lazily."""
        if self._repo_root is None:
            self._repo_root = _git_repo_root(self.task_dir)
        return self._repo_root

    # -- Read operations ---------------------------------------------------

    def current_commit(self) -> Optional[str]:
        """Return HEAD short hash, or None on any failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def current_branch(self) -> Optional[str]:
        """Return current branch name, or None on any failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def diff(self, base_commit: str, head: str = "HEAD",
             paths: Optional[list[str]] = None) -> Optional[str]:
        """Return ``git diff base_commit..head`` output, optionally scoped."""
        try:
            cmd = ["git", "diff", f"{base_commit}..{head}"]
            if paths:
                cmd.append("--")
                for p in paths:
                    cmd.append(self._rel(p))
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def dirty_files(self, files: list[str]) -> Optional[list[str]]:
        """Return the subset of ``files`` that have uncommitted changes
        or are untracked. Returns None on any unexpected exception.
        """
        try:
            dirty = set()
            for file_name in files:
                if _file_is_dirty(
                    self.task_dir, self.repo_root, file_name, self._rel(file_name)
                ):
                    dirty.add(file_name)

            return list(dirty)
        except Exception:
            return None

    def is_clean(self) -> Optional[bool]:
        """Return whether the whole repository has no tracked or untracked changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                check=False,
            )
            return result.returncode == 0 and not result.stdout.strip()
        except Exception:
            return None

    # -- Write operations --------------------------------------------------

    def commit(
        self,
        message: str,
        options: Optional[CommitOptions] = None,
    ) -> CommitResult:
        """Stage files and create a commit. Returns a CommitResult.

        Three outcomes:
          - committed: hash non-empty, commit succeeded
          - nothing_to_commit: nothing was staged (e.g. baseline / no-op)
          - error: commit command failed

        If ``expected_branch`` is set, refuses to commit when the current
        branch differs (defends against committing on the wrong branch
        after a manual checkout).
        """
        options = options or CommitOptions()
        try:
            self._assert_expected_branch(options.expected_branch)
            add_failures = self._stage_files(options.files)
            if not self._has_staged_changes():
                return _empty_commit_result(add_failures)
            commit_hash = self._commit_staged(message, options.task_name)
            if options.push:
                self._push_commit(commit_hash)
            return CommitResult(hash=commit_hash)
        except Exception as e:
            logger.exception("GitRepo.commit failed: %s", e)
            return CommitResult(error=str(e))

    def rollback_files(self, files: list[str]) -> None:
        """Roll back ``files`` to HEAD; remove untracked files entirely.

        For each file:
          - tracked: ``git checkout HEAD -- <path>`` restores to last commit
          - untracked: file is deleted from disk
        """
        try:
            for file_name in files:
                _rollback_file(
                    self.task_dir, self.repo_root, file_name, self._rel(file_name)
                )
        except Exception as e:
            logger.exception("GitRepo.rollback_files failed: %s", e)

    def ensure_branch(self, branch_name: str) -> str:
        """Switch to (or create) the named experiment branch.

        Stale branches from previous runs are deleted and recreated
        from current HEAD, so each experiment starts from a clean
        state. Returns the branch name on success, raises RuntimeError
        on failure.
        """
        repo_root = self.repo_root

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        current_branch = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch_name],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        branch_exists = result.returncode == 0

        if branch_exists:
            if current_branch == branch_name:
                _checkout_fallback_branch(repo_root)
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                capture_output=True, text=True, cwd=repo_root, check=False
            )
            logger.info("Deleted stale branch '%s'", branch_name)

        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create branch '{branch_name}': {result.stderr.strip()}"
            )

        logger.info("Created and switched to branch '%s'", branch_name)
        return branch_name

    def cleanup_branch(self, exp_branch: str, original_branch: str,
                       session_dir: str = "agent_session",
                       heartbeat_file: str = "RUNNING") -> None:
        """Switch back to the original branch and clean experiment artifacts.

        The exp branch is preserved for inspection. Artifacts created
        during the run (logs, plan.md, session dir, heartbeat) are
        removed before checkout to avoid dirty-tree conflicts.
        """
        repo_root = self.repo_root
        rel_dir = os.path.relpath(self.task_dir, repo_root)
        self._remove_experiment_artifacts(session_dir, heartbeat_file)

        # 2. Discard remaining uncommitted changes so checkout won't fail
        subprocess.run(
            ["git", "checkout", "--", rel_dir],
            capture_output=True, text=True, cwd=repo_root, check=False
        )

        # 3. Switch back to original branch (exp branch preserved)
        current = self.current_branch()
        if current == exp_branch:
            result = subprocess.run(
                ["git", "checkout", original_branch],
                capture_output=True, text=True, cwd=repo_root, check=False
            )
            if result.returncode != 0:
                logger.warning(
                    "Checkout %s failed: %s",
                    original_branch,
                    result.stderr.strip(),
                )
                return
            logger.info(
                "Switched back to '%s'; preserved experiment branch '%s'",
                original_branch,
                exp_branch,
            )
        else:
            logger.warning(
                "Not on experiment branch '%s' (current '%s'); skipping checkout",
                exp_branch,
                current,
            )

    def _remove_experiment_artifacts(
        self,
        session_dir: str,
        heartbeat_file: str,
    ) -> None:
        artifact_names = [
            "agent.log",
            "log.jsonl",
            "perf_log.md",
            "report.md",
            "plan.md",
            heartbeat_file,
        ]
        for name in artifact_names:
            path = os.path.join(self.task_dir, name)
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError:
                logger.debug("Could not remove %s", path, exc_info=True)
        for name in (session_dir, "__pycache__"):
            path = os.path.join(self.task_dir, name)
            if not os.path.isdir(path):
                continue
            try:
                shutil.rmtree(path)
            except OSError:
                logger.debug("Could not remove %s", path, exc_info=True)

    def _rel(self, fname: str) -> str:
        """Convert a task-relative file path to a repo-root-relative path."""
        return os.path.relpath(os.path.join(self.task_dir, fname), self.repo_root)

    def _assert_expected_branch(self, expected_branch: Optional[str]) -> None:
        if not expected_branch:
            return
        current = self.current_branch()
        if current and current != expected_branch:
            raise RuntimeError(
                f"Branch mismatch: on '{current}' but expected '{expected_branch}'. "
                "Aborting to prevent commits on the wrong branch."
            )

    def _stage_files(self, files: Optional[list[str]]) -> list[str]:
        paths = [self._rel(name) for name in files] if files else [
            os.path.relpath(self.task_dir, self.repo_root)
        ]
        return [path for path in paths if not _git_add(self.repo_root, path)]

    def _has_staged_changes(self) -> bool:
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.repo_root,
            capture_output=True,
            check=False,
        )
        return result.returncode != 0

    def _commit_staged(self, message: str, task_name: Optional[str]) -> str:
        author_name = task_name or "agent"
        result = subprocess.run(
            [
                "git", "-c", f"user.name={author_name}",
                "-c", "user.email=agent@autoresearch", "commit", "-m", message,
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"commit failed: {result.stderr.strip()}")
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
            check=False,
        )
        return head.stdout.strip()

    def _push_commit(self, commit_hash: str) -> None:
        result = subprocess.run(
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("Push failed: %s", result.stderr.strip())
            return
        logger.info("Pushed %s to remote", commit_hash)


def _empty_commit_result(add_failures: list[str]) -> CommitResult:
    if not add_failures:
        return CommitResult(nothing_to_commit=True)
    error = f"git add failed for: {', '.join(add_failures)}"
    logger.error("%s", error)
    return CommitResult(error=error)

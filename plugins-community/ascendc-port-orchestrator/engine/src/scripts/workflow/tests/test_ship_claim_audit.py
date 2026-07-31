# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for the ship_claim_audit PreToolUse hook.

The hook lives at src/scripts/workflow/ship_claim_audit.py and is wired
in `.claude/settings.json` to block outbound Discord messages that
frame partial worker output as wins without a commit SHA reachable
from origin/main.

Tests use a real temporary git repo so the merge-base verification
runs end-to-end. No mocking of git — the binary must behave exactly
as it will in production.

Anchor cases:

1. Win-word, no SHA → BLOCK.
2. Win-word + SHA reachable from origin/main → ALLOW.
3. Win-word + SHA NOT reachable from origin/main → BLOCK.
4. Plain status update, no win-words → ALLOW.
5. Non-monitored tool → ALLOW.
6. Win-word + $CLAUDE_PROJECT_DIR unset → BLOCK (fail-closed).
7. Empty payload → ALLOW (fail-open on hook bugs).
8. `edit_message` is monitored same as `reply`.
9. v1-style cite (path + pass_a substring, no real SHA) → BLOCK
   (regression guard against the dropped grep-level check).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_HOOK = _HERE.parent.parent / "ship_claim_audit.py"
_GIT = shutil.which("git")


def _run_hook(payload: dict, *, project_dir: str | None) -> tuple[int, str]:
    env = os.environ.copy()
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = project_dir
    else:
        env.pop("CLAUDE_PROJECT_DIR", None)
        env.pop("SHIP_CLAIM_AUDIT_PROJECT_DIR", None)
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return proc.returncode, proc.stderr


@pytest.fixture(scope="module")
def fake_repo(tmp_path_factory) -> tuple[str, str, str]:
    """Build a tiny git repo with origin/main and one extra branch.

    Yields:
      (repo_dir, sha_on_main, sha_not_on_main)
    """
    tmpdir = tmp_path_factory.mktemp("ship_claim_audit_test")
    if _GIT is None:
        pytest.skip("git executable not found")
    # Init bare upstream + working clone so origin/main is a real remote ref.
    upstream = tmpdir / "upstream.git"
    work = tmpdir / "work"
    subprocess.run([_GIT, "init", "--bare", str(upstream)], check=True,
                   capture_output=True)
    subprocess.run([_GIT, "clone", str(upstream), str(work)], check=True,
                   capture_output=True)
    # Identity (required for commit on minimal CI envs)
    for k, v in (("user.email", "test@test"), ("user.name", "test")):
        subprocess.run([_GIT, "-C", str(work), "config", k, v], check=True,
                       capture_output=True)
    (work / "README").write_text("hi\n")
    subprocess.run([_GIT, "-C", str(work), "add", "README"], check=True,
                   capture_output=True)
    subprocess.run([_GIT, "-C", str(work), "commit", "-m", "init"],
                   check=True, capture_output=True)
    subprocess.run([_GIT, "-C", str(work), "branch", "-M", "main"],
                   check=True, capture_output=True)
    subprocess.run([_GIT, "-C", str(work), "push", "-u", "origin", "main"],
                   check=True, capture_output=True)
    sha_on_main = subprocess.run(
        [_GIT, "-C", str(work), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Branch with a commit that is NOT pushed to main
    subprocess.run([_GIT, "-C", str(work), "checkout", "-b", "side"],
                   check=True, capture_output=True)
    (work / "OTHER").write_text("side\n")
    subprocess.run([_GIT, "-C", str(work), "add", "OTHER"], check=True,
                   capture_output=True)
    subprocess.run([_GIT, "-C", str(work), "commit", "-m", "side"],
                   check=True, capture_output=True)
    sha_not_on_main = subprocess.run(
        [_GIT, "-C", str(work), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    yield str(work), sha_on_main, sha_not_on_main


def test_win_word_without_sha_blocks(fake_repo) -> None:
    repo, _, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"text": "✅ 5_Cumsum 51/51 PASS"},
        },
        project_dir=repo,
    )
    assert rc == 2, f"expected block, got {rc}; stderr={err}"
    assert "BLOCKED" in err
    assert "hex-like candidates" in err


def test_win_word_with_sha_on_main_but_no_artifact_blocks(fake_repo) -> None:
    """Per a5 PR review 23:07Z: real SHA alone is insufficient — text must
    also reference verification.json or `pass_a ... PASS` so unrelated
    main commits don't accidentally satisfy the gate.
    """
    repo, sha, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"text": f"✅ closed in commit {sha[:12]}"},
        },
        project_dir=repo,
    )
    assert rc == 2, f"expected block (no artifact ref), got {rc}; stderr={err}"
    assert "verification.json" in err


def test_win_word_with_sha_on_main_and_artifact_allows(fake_repo) -> None:
    repo, sha, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {
                "text": f"✅ closed in {sha[:12]}; verification.json pass_a=PASS"
            },
        },
        project_dir=repo,
    )
    assert rc == 0, f"expected allow, got {rc}; stderr={err}"


def test_win_word_with_sha_and_pass_a_only_allows(fake_repo) -> None:
    """Either `verification.json` OR `pass_a ... PASS` substring satisfies
    the artifact-ref half of the requirement.
    """
    repo, sha, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"text": f"✅ closed at {sha[:12]}; pass_a=PASS 50/50"},
        },
        project_dir=repo,
    )
    assert rc == 0, f"expected allow, got {rc}; stderr={err}"


def test_win_word_with_sha_not_on_main_blocks(fake_repo) -> None:
    repo, _, side_sha = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"text": f"✅ done at commit {side_sha[:12]}"},
        },
        project_dir=repo,
    )
    assert rc == 2, f"expected block (side-branch sha), got {rc}; stderr={err}"


def test_plain_status_allows(fake_repo) -> None:
    repo, _, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {
                "text": "5_Cumsum state=await_optimizer, partial — not E2E closed"
            },
        },
        project_dir=repo,
    )
    assert rc == 0, f"expected allow, got {rc}; stderr={err}"


def test_non_monitored_tool_allows(fake_repo) -> None:
    repo, _, _ = fake_repo
    rc, _ = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "echo PASS win"}},
        project_dir=repo,
    )
    assert rc == 0


def test_win_word_without_project_dir_blocks() -> None:
    """Fail-closed when hook cannot verify."""
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"text": "✅ done"},
        },
        project_dir=None,
    )
    assert rc == 2, f"expected block (no project_dir), got {rc}; stderr={err}"
    assert "CLAUDE_PROJECT_DIR" in err


def test_empty_payload_fails_open(fake_repo) -> None:
    repo, _, _ = fake_repo
    rc, _ = _run_hook({}, project_dir=repo)
    assert rc == 0


def test_edit_message_monitored(fake_repo) -> None:
    repo, _, _ = fake_repo
    rc, _ = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__edit_message",
            "tool_input": {"text": "✅ updated"},
        },
        project_dir=repo,
    )
    assert rc == 2


def test_v1_style_substring_only_still_blocks(fake_repo) -> None:
    """The old grep-level check accepted an `output/...` path plus
    pass_a ... PASS` substrings as backing. v2 must reject them: those
    are still just text, not verifiable artifacts.
    """
    repo, _, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {
                "text": (
                    "✅ output/a3_to_a5_port/src/kernels/foo/ "
                    "verification.json pass_a=PASS"
                )
            },
        },
        project_dir=repo,
    )
    # No SHA present → must block, regardless of path / pass_a substrings.
    assert rc == 2, f"v1-style cite without SHA must still block, got {rc}; stderr={err}"


def test_short_sha_on_main_with_artifact_allows(fake_repo) -> None:
    """7-char abbreviated SHAs (git's default short form) must resolve."""
    repo, sha, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {
                "text": f"win at {sha[:7]} verification.json pass_a=PASS"
            },
        },
        project_dir=repo,
    )
    assert rc == 0, f"7-char SHA should resolve; got {rc}; stderr={err}"


def test_artifact_substring_alone_blocks(fake_repo) -> None:
    """`verification.json` without a verified SHA must still block —
    substring alone is forgeable text.
    """
    repo, _, _ = fake_repo
    rc, err = _run_hook(
        {
            "tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {
                "text": "✅ closed; verification.json pass_a=PASS"
            },
        },
        project_dir=repo,
    )
    assert rc == 2, f"expected block (no SHA), got {rc}; stderr={err}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

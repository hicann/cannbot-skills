# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for `_try_parse_json_tail` trailing-noise tolerance.

Empirical anchor: 2026-05-18 22_Nonzero finalize pass_b verifier
returned valid JSON followed by BiSheng warnings:

    {"results": [..., {"as_tuple": true, "status": "FAIL"}]}
    [Warning]: tiling struct [ReduceOpTilingDataV2] is conflict with one in file lp_norm_reduce.cc, line 41
    [Warning]: tiling struct [ReduceOpTilingDataV2] is conflict with one in file lp_norm_reduce.cc, line 41

The pre-fix parser rejected the whole stdout as "no parseable JSON"
and the finalize gate routed back to await_worker. The fix:
truncate at the last `}` to strip trailing noise; if that still fails,
use json.JSONDecoder().raw_decode() to forward-parse from each `{`
position.

Tests:

1. Clean JSON only — parses as before.
2. JSON followed by trailing `[Warning]:` noise — parses correctly.
3. JSON followed by multiple lines of noise — parses correctly.
4. Logs BEFORE the JSON (existing supported case) — still parses.
5. No JSON anywhere — returns None.
6. Malformed/truncated JSON — returns None (no spurious recovery).
"""
from __future__ import annotations

import sys
import pathlib
from unittest.mock import MagicMock

import pytest

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from phase_o5_runner import _try_parse_json_tail  # noqa: E402


def test_clean_json_parses() -> None:
    out = _try_parse_json_tail('{"status": "PASS", "n_pass": 10, "n_total": 10}')
    assert out == {"status": "PASS", "n_pass": 10, "n_total": 10}


def test_json_with_trailing_bisheng_warning_lines_parses() -> None:
    """The empirical 22_Nonzero case: BiSheng tiling warnings after JSON."""
    stdout = (
        '{"results": [{"shape": [16,16], "as_tuple": true, '
        '"status": "FAIL"}], "status": "FAIL"}\n'
        '[Warning]: tiling struct [ReduceOpTilingDataV2] is conflict '
        'with one in file lp_norm_reduce.cc, line 41\n'
        '[Warning]: tiling struct [ReduceOpTilingDataV2] is conflict '
        'with one in file lp_norm_reduce.cc, line 41\n'
    )
    out = _try_parse_json_tail(stdout)
    assert out is not None, "trailing BiSheng warnings must not block parsing"
    assert out["status"] == "FAIL"
    assert len(out["results"]) == 1
    assert out["results"][0]["status"] == "FAIL"


def test_json_with_many_trailing_lines_parses() -> None:
    """Defensive: more than two trailing lines."""
    stdout = '{"x": 1}\n' + "\n".join(f"[Warning]: noise line {i}" for i in range(10)) + "\n"
    out = _try_parse_json_tail(stdout)
    assert out == {"x": 1}


def test_logs_before_json_still_parse() -> None:
    """Existing-behavior preservation: verifier scripts may emit progress
    logs before the final JSON summary.
    """
    stdout = (
        '[runner] starting verification\n'
        '[runner] case 1/10 PASS\n'
        '[runner] case 2/10 PASS\n'
        '{"status": "PASS", "n_pass": 10, "n_total": 10}\n'
    )
    out = _try_parse_json_tail(stdout)
    assert out == {"status": "PASS", "n_pass": 10, "n_total": 10}


def test_logs_before_and_warnings_after_json() -> None:
    """Combined: progress logs before + BiSheng warnings after."""
    stdout = (
        '[runner] starting\n'
        '{"status": "PASS", "n_pass": 50, "n_total": 50}\n'
        '[Warning]: trailing noise\n'
    )
    out = _try_parse_json_tail(stdout)
    assert out == {"status": "PASS", "n_pass": 50, "n_total": 50}


def test_no_json_returns_none() -> None:
    """Pure noise (e.g. SSH timeout banner only) → None."""
    out = _try_parse_json_tail("SSH timed out\nno verifier output\n")
    assert out is None


def test_empty_stdout_returns_none() -> None:
    assert _try_parse_json_tail("") is None
    assert _try_parse_json_tail(None) is None  # type: ignore[arg-type]


def test_malformed_json_returns_none() -> None:
    """Truncated mid-object (e.g. SIGKILL during emit) must NOT silently
    recover to a partial dict — return None so the caller surfaces a
    runner_error rather than a fake PASS.
    """
    out = _try_parse_json_tail('{"x": 1, "y":')
    assert out is None


def test_nested_json_picks_outer() -> None:
    """If verifier wraps a per-case dict inside an outer summary, parser
    must return the outermost dict (the summary), not a fragment.
    """
    stdout = (
        '{"results": [{"status": "PASS"}, {"status": "PASS"}], '
        '"status": "PASS"}\n'
        '[Warning]: trailing\n'
    )
    out = _try_parse_json_tail(stdout)
    assert out is not None
    assert "results" in out and "status" in out
    assert len(out["results"]) == 2


# ────────────────────── P0gh-Gap-B: on-disk result.json fallback ──────────────────────

from phase_o5_runner import _try_fetch_remote_result_json  # noqa: E402


def _fake_subprocess_run_factory(*, stdout: str = "", returncode: int = 0,
                                  raise_exc=None):
    """Build a fake subprocess.run that returns a CompletedProcess-like obj.

    Captures the cmd list so tests can assert the docker exec form.
    """
    captured = {"cmds": []}

    def _fake(cmd, capture_output=True, text=True, timeout=60, **kw):
        captured["cmds"].append(cmd)
        if raise_exc is not None:
            raise raise_exc
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = ""
        return m
    return _fake, captured


def test_fetch_remote_result_json_success_with_password() -> None:
    """LIG empirical case: pass_b_runner writes valid JSON to disk; verifier
    stdout has only prose. SSH+cat fetches the disk file, returns parsed dict.
    """
    json_text = '{"tier1_pass": 34, "total": 38, "status": "PASS_WITHIN_TOLERANCE"}'
    fake, cap = _fake_subprocess_run_factory(stdout=json_text, returncode=0)
    out = _try_fetch_remote_result_json(
        a5_host="198.51.100.70", a5_user="root", a5_password="pw",
        a5_container="npu-a5-test",
        benchmark_root="/home/npu_user/AscendOpGenAgent",
        label="pass_b",
        ssh_opts=["-o", "StrictHostKeyChecking=no"],
        _subprocess_run=fake,
    )
    assert out == {"tier1_pass": 34, "total": 38, "status": "PASS_WITHIN_TOLERANCE"}
    # Verify the SSH cmd shape: sshpass + ssh + docker exec + cat
    assert cap["cmds"], "subprocess.run must have been called"
    cmd = cap["cmds"][0]
    assert cmd[0] == "sshpass" and "-p" in cmd and "pw" in cmd
    assert any("docker exec" in arg and "cat pass_b_result.json" in arg for arg in cmd)
    assert any("cd /home/npu_user/AscendOpGenAgent/current_task" in arg for arg in cmd)


def test_fetch_remote_result_json_key_auth_no_sshpass() -> None:
    """Without a5_password, falls back to key-auth SSH (no sshpass prefix)."""
    json_text = '{"tier1_pass": 1, "total": 1}'
    fake, cap = _fake_subprocess_run_factory(stdout=json_text, returncode=0)
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="",
        a5_container="container", benchmark_root="/root", label="pass_a",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out == {"tier1_pass": 1, "total": 1}
    cmd = cap["cmds"][0]
    assert cmd[0] == "ssh", f"key-auth path should start with `ssh` not `sshpass`; got {cmd[0]}"


def test_fetch_remote_result_json_file_missing_returns_none() -> None:
    """cat fails (file not present on remote) → returncode != 0 → None."""
    fake, _ = _fake_subprocess_run_factory(
        stdout="", returncode=1,  # cat: <file>: No such file...
    )
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="pw",
        a5_container="container", benchmark_root="/root", label="pass_b",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out is None


def test_fetch_remote_result_json_empty_stdout_returns_none() -> None:
    """returncode=0 but empty stdout → None (no spurious manufactured verdict)."""
    fake, _ = _fake_subprocess_run_factory(stdout="   \n", returncode=0)
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="pw",
        a5_container="container", benchmark_root="/root", label="pass_b",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out is None


def test_fetch_remote_result_json_invalid_json_returns_none() -> None:
    """Disk file present but contents not parseable as JSON → None
    (strict per main 'no graceful degrade').
    """
    fake, _ = _fake_subprocess_run_factory(stdout="not json at all", returncode=0)
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="pw",
        a5_container="container", benchmark_root="/root", label="pass_b",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out is None


def test_fetch_remote_result_json_non_dict_top_level_returns_none() -> None:
    """JSON file with non-dict top-level (e.g. array) → None.
    `_normalize_verifier_output` expects a dict.
    """
    fake, _ = _fake_subprocess_run_factory(stdout='[1, 2, 3]', returncode=0)
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="pw",
        a5_container="container", benchmark_root="/root", label="pass_b",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out is None


def test_fetch_remote_result_json_subprocess_exc_returns_none() -> None:
    """Tool-missing / timeout etc → None, don't crash phase_o5."""
    import subprocess as _sp
    fake, _ = _fake_subprocess_run_factory(
        raise_exc=_sp.TimeoutExpired(cmd="ssh", timeout=60),
    )
    out = _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="pw",
        a5_container="container", benchmark_root="/root", label="pass_b",
        ssh_opts=[], _subprocess_run=fake,
    )
    assert out is None


def test_fetch_remote_result_json_label_used_in_filename() -> None:
    """Label `pass_a` → file `pass_a_result.json`; `pass_b` → `pass_b_result.json`."""
    fake, cap = _fake_subprocess_run_factory(stdout='{}', returncode=0)
    _try_fetch_remote_result_json(
        a5_host="host", a5_user="root", a5_password="",
        a5_container="container", benchmark_root="/root", label="determinism",
        ssh_opts=[], _subprocess_run=fake,
    )
    cmd = cap["cmds"][0]
    assert any("cat determinism_result.json" in arg for arg in cmd), (
        f"label `determinism` should map to `determinism_result.json`; got cmd: {cmd}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

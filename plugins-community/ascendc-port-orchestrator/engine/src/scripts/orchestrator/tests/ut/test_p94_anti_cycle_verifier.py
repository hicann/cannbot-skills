# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P94 attack-id WORKER-SELF-CITING-VERIFIER regression tests.

Verifies that Phase O5 runner rejects verifier scripts that read
verification.json (the file they're supposed to independently
verify) — the foreach_abs original fraud pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5_runner as o5  # noqa: E402


def _seed_verifier(workspace: Path, name: str, body: str) -> Path:
    p = workspace / name
    p.write_text(body)
    return p


def test_self_citing_verifier_rejected(tmp_path):
    """Verifier whose CODE reads verification.json → REJECT."""
    body = """
import json
with open('verification.json') as f:
    d = json.load(f)
print(d['precision']['pass_a'])
"""
    _seed_verifier(tmp_path, "pass_a_runner.py", body)
    err = getattr(o5, '_verify_runner_independence')(tmp_path, "pass_a_runner.py")
    assert err is not None
    assert "verification.json" in err or "CYCLE" in err.upper()


def test_independent_verifier_accepted(tmp_path):
    """Verifier that runs actual NPU/CPU verification → accept."""
    body = """
import torch
import json
# Run actual computation
x = torch.randn(8, 16)
y = x.abs()
print(json.dumps({"tier1_pass": 8, "total": 8}))
"""
    _seed_verifier(tmp_path, "pass_a_runner.py", body)
    err = getattr(o5, '_verify_runner_independence')(tmp_path, "pass_a_runner.py")
    assert err is None


def test_comment_mention_does_not_trigger(tmp_path):
    """If 'verification.json' appears only in comments/docstrings,
    not in actual code lines, gate should NOT fire.
    """
    body = '''
"""This script eventually writes to verification.json downstream."""
# Note: caller writes verification.json AFTER we return.
import torch
x = torch.randn(8)
print({"tier1_pass": 8, "total": 8})
'''
    _seed_verifier(tmp_path, "pass_a_runner.py", body)
    err = getattr(o5, '_verify_runner_independence')(tmp_path, "pass_a_runner.py")
    assert err is None


def test_foreach_abs_pattern_caught(tmp_path):
    """Reproduce foreach_abs pattern: docstring admits non-execution
    + 8 verification.json references in code body.
    """
    body = '''
"""Pass A verifier — OL-132 Mode A port (foreach_abs).

This script does NOT execute on NPU (Mode A — upstream binary already
verified). Instead, it reads the persisted pass_a block from
verification.json which was populated from the source-identity argument.
"""
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

def _read_persisted_pass_a():
    vjson = WORKSPACE / "verification.json"
    if not vjson.is_file():
        return {"error": "verification.json not found"}
    with vjson.open() as f:
        data = json.load(f)
    return data.get("precision", {}).get("pass_a", {})

if __name__ == "__main__":
    print(json.dumps(_read_persisted_pass_a()))
'''
    _seed_verifier(tmp_path, "pass_a_runner.py", body)
    err = getattr(o5, '_verify_runner_independence')(tmp_path, "pass_a_runner.py")
    assert err is not None
    assert "verification.json" in err
    assert "CYCLE" in err.upper() or "cycle" in err.lower()


def test_missing_script_returns_none(tmp_path):
    """Non-existent script: returns None (caller handles missing
    elsewhere). Don't block on file-not-found here.
    """
    err = getattr(o5, '_verify_runner_independence')(tmp_path, "nonexistent.py")
    assert err is None

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-213(b): finalize asserts the harness was pristine for the run.

Background: a kernel worker edited `phase_o5_runner.py` (the O5 push-list —
the mechanism deciding whether O5 can SEE a backend's artifacts), left it
uncommitted, and an op was then finalized on that modified harness with
verdict VERIFIED. The subject modified the instrument, and the instrument
then verified the subject. No tool noticed.

Two halves are under test:
  1. the boundary — the scope must be the code that MEASURES, and must NOT
     fire on the worker's own product (workspace/, output/) or on the
     orchestrator's own runtime markers. This is what keeps the check from
     firing on every run and being learned-to-ignore.
  2. the assertion — DIRTY harness downgrades O5 VERIFIED -> PROVISIONAL,
     naming the dirty paths.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import harness_pristine as hp  # noqa: E402
import phase_o5  # noqa: E402

# tests/ut/conftest.py's autouse `_hermetic_harness_pristine` pins
# hp.harness_state to CLEAN for every ut test (so the ut gate doesn't depend on
# the developer's working tree). Bind the REAL implementation at import time —
# before any fixture runs — so the boundary tests below exercise actual git
# behaviour against real throwaway repos rather than the hermetic stub.
_REAL_HARNESS_STATE = hp.harness_state


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo shaped like a5_ops_slim: harness code + worker product, all
    committed clean.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _init_git_repo(root)
    _write(root / "src/scripts/orchestrator/phase_o5_runner.py", "# push list\n")
    _write(root / "src/scripts/orchestrator/plugins/base.py", "# plugin\n")
    _write(root / "src/scripts/patches/some_patch.py", "# patch\n")
    _write(root / "workspace/4_Abs/kernel.cpp", "// worker product\n")
    _write(root / "output/generated_ops/src/kernels/4_Abs/verification.json", "{}\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=root, check=True)
    return root


# --------------------------------------------------------------------------
# 1. The boundary
# --------------------------------------------------------------------------


def test_clean_harness_is_clean(repo: Path) -> None:
    st = _REAL_HARNESS_STATE(repo)
    assert st.state == hp.CLEAN
    assert st.dirty_paths == []
    assert st.is_dirty is False


def test_dirty_harness_file_is_dirty_and_names_the_path(repo: Path) -> None:
    """The exact DEBT-213 breach: phase_o5_runner.py edited, uncommitted."""
    _write(repo / "src/scripts/orchestrator/phase_o5_runner.py", "# EDITED push list\n")
    st = _REAL_HARNESS_STATE(repo)
    assert st.state == hp.DIRTY
    assert st.is_dirty is True
    assert "src/scripts/orchestrator/phase_o5_runner.py" in st.dirty_paths


def test_dirty_patches_dir_is_dirty(repo: Path) -> None:
    _write(repo / "src/scripts/patches/some_patch.py", "# EDITED\n")
    st = _REAL_HARNESS_STATE(repo)
    assert st.state == hp.DIRTY
    assert "src/scripts/patches/some_patch.py" in st.dirty_paths


def test_untracked_harness_module_is_dirty(repo: Path) -> None:
    """A NEW uncommitted harness module is a modified instrument too."""
    _write(repo / "src/scripts/orchestrator/sneaky_override.py", "# new\n")
    st = _REAL_HARNESS_STATE(repo)
    assert st.state == hp.DIRTY
    assert "src/scripts/orchestrator/sneaky_override.py" in st.dirty_paths


# --------------------------------------------------------------------------
# 2. No false positives — the blast-radius guard.
#    A dirty workspace/ or output/ is the worker's own product and is the
#    normal state of EVERY run. If any of these fire, the check is noise.
# --------------------------------------------------------------------------


def test_dirty_workspace_does_not_trigger(repo: Path) -> None:
    _write(repo / "workspace/4_Abs/kernel.cpp", "// edited by the worker\n")
    _write(repo / "workspace/4_Abs/brand_new.cpp", "// new worker file\n")
    assert _REAL_HARNESS_STATE(repo).state == hp.CLEAN


def test_dirty_output_does_not_trigger(repo: Path) -> None:
    _write(repo / "output/generated_ops/src/kernels/4_Abs/verification.json",
           json.dumps({"precision": {"status": "PASS"}}))
    _write(repo / "output/generated_ops/src/kernels/4_Abs/PROGRESS.md", "# new\n")
    assert _REAL_HARNESS_STATE(repo).state == hp.CLEAN


def test_orchestrator_runtime_marker_does_not_trigger(repo: Path) -> None:
    """`.kernel_worker_active` is written INSIDE the harness dir by
    agent_dispatch for every worker spawn. Measured on the live tree, it is
    the ONLY harness-path entry during a real run — counting it would fire
    this check on ~every run that ever spawned a worker.
    """
    _write(repo / "src/scripts/orchestrator/.kernel_worker_active", "")
    _write(repo / "src/scripts/orchestrator/.optimizer_active", "")
    _write(repo / "src/scripts/orchestrator/.opgen_state.json", "{}")
    assert _REAL_HARNESS_STATE(repo).state == hp.CLEAN


def test_pycache_does_not_trigger(repo: Path) -> None:
    _write(repo / "src/scripts/orchestrator/__pycache__/phase_o5.cpython-311.pyc", "x")
    _write(repo / "src/scripts/orchestrator/phase_o5.pyc", "x")
    assert _REAL_HARNESS_STATE(repo).state == hp.CLEAN


def test_dirty_docs_does_not_trigger(repo: Path) -> None:
    """Prose is not the instrument."""
    _write(repo / "docs/design/NOTES.md", "# notes\n")
    assert _REAL_HARNESS_STATE(repo).state == hp.CLEAN


# --------------------------------------------------------------------------
# 3. Graceful degradation — never crash, never silently pass
# --------------------------------------------------------------------------


def test_non_git_tree_degrades_to_unknown(tmp_path: Path) -> None:
    """A customer's unpacked bundle is not a checkout. Must not crash, must
    not claim CLEAN.
    """
    bundle = tmp_path / "bundle"
    (bundle / "src/scripts/orchestrator").mkdir(parents=True)
    st = _REAL_HARNESS_STATE(bundle)
    assert st.state == hp.UNKNOWN
    assert st.is_dirty is False
    assert st.reason


def test_missing_git_binary_degrades_to_unknown(repo: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(hp.subprocess, "run", _boom)
    st = _REAL_HARNESS_STATE(repo)
    assert st.state == hp.UNKNOWN
    assert "git executable not found" in st.reason


def test_git_timeout_degrades_to_unknown(repo: Path, monkeypatch) -> None:
    def _slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=30)
    monkeypatch.setattr(hp.subprocess, "run", _slow)
    assert _REAL_HARNESS_STATE(repo).state == hp.UNKNOWN


# --------------------------------------------------------------------------
# 4. The assertion: O5 verdict downgrade
# --------------------------------------------------------------------------


def _seed_verification(ws: Path, tier1: int = 8, total: int = 8) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "demo",
        "opgen_mode": "backward",
    }))
    (ws / "verification.json").write_text(json.dumps({
        "op": "demo",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": tier1, "total": total},
        },
    }))


def _runner_matching(tier1: int = 8, total: int = 8):
    def _r(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": tier1, "total": total})
    return _r


def _force_state(monkeypatch, state: hp.HarnessState) -> None:
    monkeypatch.setattr(hp, "harness_state", lambda *a, **k: state)


def test_o5_verified_unaffected_when_harness_clean(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    _seed_verification(ws)
    _force_state(monkeypatch, hp.HarnessState(hp.CLEAN, reason="matches HEAD"))
    rep = phase_o5.post_verify_for_finalize(ws, "demo", runner=_runner_matching())
    assert rep.verdict == "VERIFIED"
    assert rep.harness_git_state == hp.CLEAN
    assert rep.harness_dirty == []


def test_o5_verified_downgraded_to_provisional_when_harness_dirty(
    tmp_path: Path, monkeypatch
) -> None:
    """THE regression this ticket exists for: counts match, but the
    instrument was modified. Must NOT be VERIFIED, and must name the path.
    """
    ws = tmp_path / "demo"
    _seed_verification(ws)
    dirty = "src/scripts/orchestrator/phase_o5_runner.py"
    _force_state(monkeypatch, hp.HarnessState(hp.DIRTY, dirty_paths=[dirty],
                                              reason="1 uncommitted harness file"))
    rep = phase_o5.post_verify_for_finalize(ws, "demo", runner=_runner_matching())
    assert rep.verdict == "PROVISIONAL", (
        "dirty harness must never yield VERIFIED (DEBT-213(b))"
    )
    assert rep.verdict != "VERIFIED"
    assert rep.harness_git_state == hp.DIRTY
    assert dirty in rep.harness_dirty
    assert dirty in rep.summary, "the reason must NAME the dirty path"


def test_o5_unknown_does_not_downgrade(tmp_path: Path, monkeypatch) -> None:
    """UNKNOWN is recorded, not punished — else a non-git bundle flags every
    run and the check becomes noise.
    """
    ws = tmp_path / "demo"
    _seed_verification(ws)
    _force_state(monkeypatch, hp.HarnessState(hp.UNKNOWN, reason="not a git checkout"))
    rep = phase_o5.post_verify_for_finalize(ws, "demo", runner=_runner_matching())
    assert rep.verdict == "VERIFIED"
    assert rep.harness_git_state == hp.UNKNOWN


def test_o5_mismatch_not_relabelled_by_dirty_harness(tmp_path: Path, monkeypatch) -> None:
    """MISMATCH already refuses finalize; a dirty harness must not soften it
    into PROVISIONAL.
    """
    ws = tmp_path / "demo"
    _seed_verification(ws, tier1=8, total=8)
    _force_state(monkeypatch, hp.HarnessState(hp.DIRTY, dirty_paths=["src/scripts/patches/p.py"]))
    rep = phase_o5.post_verify_for_finalize(
        ws, "demo", runner=_runner_matching(tier1=3, total=8)
    )
    assert rep.verdict == "MISMATCH"


def test_o5_backward_no_claims_fails_before_harness_stamp(
    tmp_path: Path, monkeypatch
) -> None:
    ws = tmp_path / "demo"
    ws.mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "demo",
        "opgen_mode": "backward",
    }))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "N/A"}}
    }))
    dirty = "src/scripts/orchestrator/phase_o5_runner.py"
    _force_state(monkeypatch, hp.HarnessState(hp.DIRTY, dirty_paths=[dirty]))
    rep = phase_o5.post_verify_for_finalize(ws, "demo", runner=_runner_matching())
    assert rep.verdict == "RUNNER_FAILED"


def test_harness_probe_failure_does_not_break_o5(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    _seed_verification(ws)

    def _boom(*a, **k):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(hp, "harness_state", _boom)
    rep = phase_o5.post_verify_for_finalize(ws, "demo", runner=_runner_matching())
    assert rep.verdict == "VERIFIED"
    assert rep.harness_git_state == hp.UNKNOWN


# --------------------------------------------------------------------------
# 5. The fact is recorded — auditable after the process exits
# --------------------------------------------------------------------------


def test_record_harness_state_writes_dirty_block(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    _seed_verification(ws)
    rep = phase_o5.O5Report(verdict="PROVISIONAL")
    rep.harness_git_state = hp.DIRTY
    rep.harness_dirty = ["src/scripts/orchestrator/phase_o5_runner.py"]
    assert phase_o5.record_harness_state(ws, rep) is True

    v = json.loads((ws / "verification.json").read_text())
    block = v["harness_pristine"]
    assert block["state"] == hp.DIRTY
    assert block["o5_verdict"] == "PROVISIONAL"
    assert block["harness_dirty"] == ["src/scripts/orchestrator/phase_o5_runner.py"]
    assert "src/scripts/orchestrator" in block["scope"]
    # pre-existing content preserved (additive stamp)
    assert v["precision"]["status"] == "PASS"


def test_record_harness_state_writes_clean_block(tmp_path: Path) -> None:
    """CLEAN is recorded too: absence of the block must mean "no check ran",
    not "the check passed".
    """
    ws = tmp_path / "demo"
    _seed_verification(ws)
    rep = phase_o5.O5Report(verdict="VERIFIED")
    rep.harness_git_state = hp.CLEAN
    assert phase_o5.record_harness_state(ws, rep) is True
    v = json.loads((ws / "verification.json").read_text())
    assert v["harness_pristine"]["state"] == hp.CLEAN
    assert v["harness_pristine"]["o5_verdict"] == "VERIFIED"


def test_record_harness_state_fails_open(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    ws.mkdir(parents=True)  # no verification.json
    rep = phase_o5.O5Report(verdict="VERIFIED")
    assert phase_o5.record_harness_state(ws, rep) is False

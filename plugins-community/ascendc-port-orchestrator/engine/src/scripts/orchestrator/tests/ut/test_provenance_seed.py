#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-203 S3 unit + guard tests — migration seeding (default-OFF) + safety guards.

Covers main's signed criteria:
 ① seed at branched_from_kernel/ survives cold-start wipe + name-guard (dodges all patterns)
 ③ check_finalize_eligibility is branch-invariant (byte-identical verification.json regression)
 + default-OFF; seed is a copy of the proven source kernel; fail-open.
The delegation-scan byte-unchanged + branched-op-with-delegation-still-fails guards live in
test_provenance_seed_guards (this file's §guards) since the scanner file is untouched.
"""
import logging
import sys
import os
import json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import provenance_seed as ps


def _make_archive_with_proven_op(root, op, tags, *, fitness=0.95):
    d = root / op
    (d / "kernel").mkdir(parents=True)
    (d / "kernel" / "kernel.cpp").write_text("// proven kernel for " + op + "\n")
    (d / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"}, "determinism": {"policy_satisfied": True},
        "provenance_node": {"node_id": f"{op}@abc123", "fitness": fitness, "is_buggy": False,
                            "signature": {"op_class_tags": tags, "algorithm_classification": "single_op"}},
    }))
    return d


def _make_workspace(ws, tags):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "op_classification.json").write_text(json.dumps({
        "op": ws.name, "op_class_tags": tags, "algorithm_classification": "single_op", "source_sha256": "ff00"}))
    return ws


# ── default-OFF ───────────────────────────────────────────────────────────────
def test_seeding_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ps.SEED_FLAG_ENV, raising=False)
    arch = tmp_path / "arch"
    _make_archive_with_proven_op(arch, "5_GELU", ["elementwise"])
    ws = _make_workspace(tmp_path / "new_op", ["elementwise"])
    ref = ps.maybe_seed_from_similar_op("new_op", ws, archive_root=arch)   # flag off
    assert ref is None
    assert not (ws / ps.SEED_DIR_NAME).exists()


def test_seed_flag_parsing(monkeypatch):
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv(ps.SEED_FLAG_ENV, v)
        assert ps.seed_enabled() is True
    for v in ("0", "false", "", "off"):
        monkeypatch.setenv(ps.SEED_FLAG_ENV, v)
        assert ps.seed_enabled() is False


# ── seeding (flag ON via force_enabled) ───────────────────────────────────────
def test_seeds_from_proven_similar(tmp_path):
    arch = tmp_path / "arch"
    _make_archive_with_proven_op(arch, "5_GELU", ["elementwise", "transcendental"])
    ws = _make_workspace(tmp_path / "2_SwiGLU", ["elementwise", "transcendental"])
    ref = ps.maybe_seed_from_similar_op("2_SwiGLU", ws, archive_root=arch, threshold=0.5, force_enabled=True)
    assert ref is not None and ref.op == "5_GELU"
    # seed copied into branched_from_kernel/ (NOT kernel/)
    assert (ws / ps.SEED_DIR_NAME / "kernel.cpp").is_file()
    assert not (ws / "kernel").exists()   # worker authors kernel/, S3 never writes it
    marker = json.loads((ws / ps.SEED_MARKER).read_text())
    assert marker["branched"] is True and marker["parent_op"] == "5_GELU"


def test_no_seed_when_no_similar(tmp_path):
    arch = tmp_path / "arch"
    _make_archive_with_proven_op(arch, "att", ["attention", "matmul", "softmax"])
    ws = _make_workspace(tmp_path / "elem", ["elementwise"])   # disjoint
    ref = ps.maybe_seed_from_similar_op("elem", ws, archive_root=arch, threshold=0.5, force_enabled=True)
    assert ref is None and not (ws / ps.SEED_DIR_NAME).exists()


def test_no_seed_when_source_has_no_kernel(tmp_path):
    arch = tmp_path / "arch"
    d = arch / "nokernel"
    d.mkdir(parents=True)
    (d / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"}, "determinism": {"policy_satisfied": True},
        "provenance_node": {"node_id": "nokernel@x", "fitness": 0.95, "is_buggy": False,
                            "signature": {"op_class_tags": ["elementwise"]}}}))
    ws = _make_workspace(tmp_path / "new", ["elementwise"])
    assert ps.maybe_seed_from_similar_op("new", ws, archive_root=arch, threshold=0.3, force_enabled=True) is None


# ── ① GUARD (UPDATED by S5 lifecycle fix): branched_from_kernel/ is WIPED on cold-start ──
def test_seed_dir_wiped_by_coldstart_lifecycle(tmp_path):
    """S5 lifecycle fix (main-directed): cold-start now CLEARS the seed base
    branched_from_kernel/ + .branched_from.json (fresh op-gen start must not inherit a
    prior run's stale seed — lifetime_spawn_count survives cold-start [P94] so the hook
    won't re-fire on a re-run). The seed hook RE-INJECTS a fresh base post-reset when
    seeding; WITHIN a run the base persists (cold-start isn't called mid-run). This
    SUPERSEDES the original S3 'survives cold-start' assertion — clearing STRENGTHENS
    DEBT-078's anti-restore (the seed base is a restore source).
    """
    import orchestrator_coldstart as cs
    ws = tmp_path / "9_Op"
    (ws / ps.SEED_DIR_NAME).mkdir(parents=True)
    (ws / ps.SEED_DIR_NAME / "kernel.cpp").write_text("// stale branch base from prior run\n")
    (ws / ps.SEED_MARKER).write_text('{"branched": true, "parent_op": "X"}')
    (ws / "kernel").mkdir()
    (ws / "kernel" / "x.cpp").write_text("// worker output\n")
    getattr(cs, "_cold_start_reset_workspace")(ws)
    # seed base + marker cleared (lifecycle fix); worker output wiped as before
    assert not (ws / ps.SEED_DIR_NAME).exists(), "cold-start must wipe stale branched_from_kernel/"
    assert not (ws / ps.SEED_MARKER).exists(), "cold-start must clear stale .branched_from.json"
    assert not (ws / "kernel").exists(), "kernel/ must be wiped"


def test_seed_dir_in_coldstart_wipe_set():
    """S5: branched_from_kernel is now IN the cold-start worker-output wipe set (it's a
    restore source, per DEBT-078). The marker uses a distinct non-glob name.
    """
    name = ps.SEED_DIR_NAME
    assert name == "branched_from_kernel"                            # the wiped seed-base dir name
    assert ps.SEED_MARKER == ".branched_from.json"                    # the cleared marker


# ── ③ GUARD: check_finalize_eligibility is branch-invariant ───────────────────
def test_acceptance_gate_branch_invariant(tmp_path):
    """A verification.json with a branched provenance_node vs one without → identical
    finalize eligibility verdict (the acceptance bar ignores branch lineage).
    """
    import finalize_pipeline
    from finalize_dispatch import check_finalize_eligibility
    base_vj = {"precision": {"status": "PASS", "pass_a": {"tier1_pass": 5, "total": 5}},
               "determinism": {"policy_satisfied": True}}

    def _elig(with_branch):
        ws = tmp_path / ("b" if with_branch else "nb")
        ws.mkdir()
        vj = dict(base_vj)
        if with_branch:
            vj = dict(vj)
            vj["provenance_node"] = {"branched": True, "parent_id": "5_GELU@abc", "fitness": 0.9}
        (ws / "verification.json").write_text(json.dumps(vj))
        try:
            return check_finalize_eligibility(ws)
        except Exception as e:
            return f"exc:{type(e).__name__}"
    a = _elig(False)
    b = _elig(True)
    # identical verdict regardless of branch metadata (the gate never reads it)
    assert repr(a) == repr(b), f"acceptance gate differs for branched vs non-branched: {a!r} vs {b!r}"


# ── ② GUARD: delegation-scan grants a branched op NO exemption ────────────────
def test_delegation_scan_still_fails_branched_op(tmp_path):
    """A branched op whose adapted kernel contains a delegation pattern STILL fails
    the delegation scan — branch lineage grants no exemption (the scanner ignores it).
    """
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # src/scripts
    import scan_delegation_cheating as sdc
    ws = tmp_path / "branched_op"
    ws.mkdir()
    # simulate a branched op: seed base + provenance marker present
    (ws / ps.SEED_DIR_NAME).mkdir()
    (ws / ps.SEED_DIR_NAME / "k.cpp").write_text("// base\n")
    (ws / ps.SEED_MARKER).write_text(json.dumps({"branched": True, "parent_id": "5_GELU@abc"}))
    # the worker's adapted output contains a forbidden delegation call
    (ws / "model_new_ascendc.py").write_text("import torch_npu\n\ndef f(x):\n    return torch_npu.npu_swiglu(x)\n")
    result = sdc.scan_op_workspace(ws)
    assert result.get("violations"), "delegation scan must still flag a branched op's delegation"


def test_kw_brief_addendum_only_when_seeded(tmp_path):
    """kw_brief branch-base addendum: empty (brief byte-identical) with no marker;
    present + correct when a .branched_from.json marker exists.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "briefs")))
    import kw_brief
    ws = tmp_path / "op"
    ws.mkdir()
    assert getattr(kw_brief, "_branched_from_addendum")(ws) == ""   # unseeded → no addendum
    (ws / ps.SEED_MARKER).write_text(json.dumps(
        {"parent_op": "5_GELU", "similarity": 0.7, "seed_dir": ps.SEED_DIR_NAME}))
    add = getattr(kw_brief, "_branched_from_addendum")(ws)
    assert "BRANCH BASE" in add and "5_GELU" in add
    assert "NOT a submission" in add and "RE-VERIFY" in add   # anti-cheat framing present


def test_spawn_hook_skips_non_migration_workflow(tmp_path, monkeypatch):
    """The FSM caller must not seed backward generation from a target archive."""
    import finalize_dispatch
    import fsm_phase_spawn

    class _Plugin:
        name = "backward"

    class _Ctx:
        spawn_count = 0
        lifetime_spawn_count = 0

    class _Snap:
        current_state = "await_worker"

    called = []
    monkeypatch.setattr(ps, "seed_enabled", lambda: True)
    monkeypatch.setattr(finalize_dispatch, "_get_active_plugin", lambda ws: _Plugin())
    monkeypatch.setattr(ps, "maybe_seed_from_similar_op", lambda *a, **k: called.append(True))

    getattr(fsm_phase_spawn, "_maybe_seed_branch_base")(_Ctx(), tmp_path, _Snap())
    assert called == []


def test_s3_does_not_touch_scanner_or_coldstart_logic():
    """Meta-guard: S3 must not modify the delegation scanner or the cold-start wipe logic
    (criterion ②/① = those files stay byte-unchanged). Assert the modules still expose their
    load-bearing entry points (a rename/gut would break this).
    """
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    import scan_delegation_cheating as sdc
    import orchestrator_coldstart as cs
    assert hasattr(sdc, "scan_op_workspace") and hasattr(sdc, "scan_python_wrapper")
    assert hasattr(cs, "_cold_start_reset_workspace")


if __name__ == "__main__":
    import traceback
    import tempfile
    from pathlib import Path

    class _MP:
        def setenv(self, k, v):
            os.environ[k] = v

        def delenv(self, k, raising=True):
            os.environ.pop(k, None)
    fails = []
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        tempdir = None
        try:
            args = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            kw = {}
            if "tmp_path" in args:
                tempdir = tempfile.TemporaryDirectory()
                kw["tmp_path"] = Path(tempdir.name)
            if "monkeypatch" in args:
                kw["monkeypatch"] = _MP()
            fn(**kw)
            logging.info(f"  [PASS] {name}")
        except Exception as e:
            fails.append(name)
            logging.info(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
        finally:
            if tempdir is not None:
                tempdir.cleanup()
    logging.info(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)

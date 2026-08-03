#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DEBT-203 S3 WIRED e2e — the fsm_phase_spawn cold-start seed hook triggers end-to-end.

Proves (with no NPU needed) that the ACTUAL wired hook `_maybe_seed_branch_base` —
the one handle_spawn calls at the first worker spawn — fires end-to-end: with the
seeding flag ON, a fresh cold-start op + a proven similar op in the (plugin-resolved)
archive → the hook seeds `branched_from_kernel/` + drops `.branched_from.json`.
flag-OFF + non-first-spawn controls assert no seed. This is the wired counterpart to
the module-level tests in test_provenance_seed.py (main's wired-e2e gate), exercising
the real archive_root resolution (_PROJECT_ROOT + _get_active_plugin) inside the hook.
"""
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))          # src/scripts/orchestrator
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))    # src/scripts

import finalize_pipeline  # noqa: E402  (import parent FIRST to resolve the re-export dance)
import finalize_dispatch  # noqa: E402
import fsm_phase_spawn as S  # noqa: E402
import provenance_seed as ps  # noqa: E402


_LOG = logging.getLogger(__name__)


class _Snap:
    def __init__(self, state="await_worker"):
        self.current_state = state
        self.iter_counts = {}


def _bind_seed_modules(monkeypatch):
    """Bind the bare imports used by the wired hook to this test's modules.

    The full orchestrator suite intentionally exercises both package-qualified
    and flat imports. Pinning these three dynamic imports prevents collection
    order from making the hook observe a second module instance.
    """
    monkeypatch.setitem(sys.modules, "finalize_dispatch", finalize_dispatch)
    monkeypatch.setitem(sys.modules, "finalize_pipeline", finalize_pipeline)
    monkeypatch.setitem(sys.modules, "provenance_seed", ps)


def _ctx(op="2_NewAct", archive_project=None):
    env = SimpleNamespace(archive_project=archive_project) if archive_project else None
    return SimpleNamespace(
        op=op,
        spawn_count=0,
        lifetime_spawn_count=0,
        _resolve_env=lambda: env,
    )


def _make_corpus_and_ws(tmp_path, monkeypatch):
    """Create a proven archive operator and a similar fresh workspace.

    Route the hook's archive_root at the corpus through the _PROJECT_ROOT and
    _get_active_plugin monkeypatches.
    """
    _bind_seed_modules(monkeypatch)
    proj = tmp_path / "proj"
    kernels = proj / "output" / "npukernelbench" / "src" / "kernels"
    gelu = kernels / "5_GELU"
    (gelu / "kernel").mkdir(parents=True)
    (gelu / "kernel" / "5_GELU_kernel.cpp").write_text("// proven GELU kernel\n")
    (gelu / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"}, "determinism": {"policy_satisfied": True},
        "provenance_node": {"node_id": "5_GELU@abc123def456", "fitness": 0.95, "is_buggy": False,
                            "signature": {"op_class_tags": ["elementwise", "transcendental"],
                                          "algorithm_classification": "single_op"}}}))
    ws = tmp_path / "2_NewAct"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text("{}")
    (ws / "op_classification.json").write_text(json.dumps({
        "op": "2_NewAct", "op_class_tags": ["elementwise", "transcendental"],
        "algorithm_classification": "single_op", "source_sha256": "ff11"}))
    monkeypatch.setattr(finalize_pipeline, "_PROJECT_ROOT", proj)
    plugin = type("PortPlugin", (), {
        "name": "port_a3_to_a5",
        "archive_project_subdir": lambda self: "npukernelbench",
    })()
    monkeypatch.setattr(finalize_dispatch, "_get_active_plugin", lambda w: plugin)
    return ws


def test_hook_fires_and_seeds_when_flag_on(tmp_path, monkeypatch):
    ws = _make_corpus_and_ws(tmp_path, monkeypatch)
    monkeypatch.setenv(ps.SEED_FLAG_ENV, "1")   # flag ON
    ctx = _ctx()  # spawn_count=lifetime=0 → first spawn
    S._maybe_seed_branch_base(ctx, ws, _Snap("await_worker"))
    # the hook fired end-to-end: branch base seeded from the proven op + marker dropped
    assert (ws / ps.SEED_DIR_NAME / "5_GELU_kernel.cpp").is_file(), "hook did not seed branched_from_kernel/"
    marker = json.loads((ws / ps.SEED_MARKER).read_text())
    assert marker["branched"] is True and marker["parent_op"] == "5_GELU"
    assert not (ws / "kernel").exists()   # hook never writes kernel/ (the worker authors it)


def test_hook_noop_when_flag_off(tmp_path, monkeypatch):
    ws = _make_corpus_and_ws(tmp_path, monkeypatch)
    monkeypatch.delenv(ps.SEED_FLAG_ENV, raising=False)   # flag OFF (default)
    ctx = _ctx()
    S._maybe_seed_branch_base(ctx, ws, _Snap("await_worker"))
    assert not (ws / ps.SEED_DIR_NAME).exists()   # control: no seeding when flag off
    assert not (ws / ps.SEED_MARKER).exists()


def test_hook_noop_on_non_first_spawn(tmp_path, monkeypatch):
    ws = _make_corpus_and_ws(tmp_path, monkeypatch)
    monkeypatch.setenv(ps.SEED_FLAG_ENV, "1")
    ctx = _ctx()
    ctx.lifetime_spawn_count = 3   # not the first spawn (e.g. a resume) → no seeding
    S._maybe_seed_branch_base(ctx, ws, _Snap("await_worker"))
    assert not (ws / ps.SEED_DIR_NAME).exists()


def test_hook_noop_when_state_not_await_worker(tmp_path, monkeypatch):
    ws = _make_corpus_and_ws(tmp_path, monkeypatch)
    monkeypatch.setenv(ps.SEED_FLAG_ENV, "1")
    ctx = _ctx()
    S._maybe_seed_branch_base(ctx, ws, _Snap("await_probe"))   # not the kw first-spawn
    assert not (ws / ps.SEED_DIR_NAME).exists()


def test_hook_uses_target_suffixed_archive_from_env(tmp_path, monkeypatch):
    """Resolve archive_root from the target-suffixed environment project.

    Regression guard: use env.archive_project (for example,
    npukernelbench-a3-ds), not the
    plugin's base subdir (npukernelbench), which points at the wrong archive for
    -ds/-a3/-a2 agents. Proven op lives in the -a3-ds archive; assert the hook
    seeds from there.
    """
    _bind_seed_modules(monkeypatch)
    proj = tmp_path / "proj"
    # proven op in the TARGET-SUFFIXED archive only
    kernels = proj / "output" / "npukernelbench-a3-ds" / "src" / "kernels"
    add = kernels / "3_Add"
    (add / "kernel").mkdir(parents=True)
    (add / "kernel" / "3_Add_kernel.cpp").write_text("// proven add\n")
    (add / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"}, "determinism": {"policy_satisfied": True},
        "provenance_node": {"node_id": "3_Add@x", "fitness": 0.95, "is_buggy": False,
                            "signature": {"op_class_tags": ["elementwise"], "algorithm_classification": "single_op"}}}))
    ws = tmp_path / "4_Abs"
    ws.mkdir()
    (ws / "op_classification.json").write_text(json.dumps({
        "op": "4_Abs", "op_class_tags": ["elementwise"], "algorithm_classification": "single_op"}))
    monkeypatch.setattr(finalize_pipeline, "_PROJECT_ROOT", proj)
    plugin = type("PortPlugin", (), {
        "name": "port_a3_to_a5",
        "archive_project_subdir": lambda self: "npukernelbench",
    })()
    monkeypatch.setattr(finalize_dispatch, "_get_active_plugin", lambda w: plugin)
    monkeypatch.setenv(ps.SEED_FLAG_ENV, "1")
    # a fake ctx whose _resolve_env yields the target-suffixed archive_project
    fake_ctx = _ctx(op="4_Abs", archive_project="npukernelbench-a3-ds")
    S._maybe_seed_branch_base(fake_ctx, ws, _Snap("await_worker"))
    assert (ws / ps.SEED_DIR_NAME / "3_Add_kernel.cpp").is_file(), "hook did not use env.archive_project (-a3-ds)"
    assert json.loads((ws / ps.SEED_MARKER).read_text())["parent_op"] == "3_Add"


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class _MP:
        def __init__(self):
            self._saved = []

        @staticmethod
        def setenv(k, v):
            os.environ[k] = v

        @staticmethod
        def delenv(k, raising=True):
            os.environ.pop(k, None)

        def setattr(self, o, n, v):
            self._saved.append((o, n, getattr(o, n)))
            setattr(o, n, v)

        def undo(self):
            for o, n, v in reversed(self._saved):
                setattr(o, n, v)

    fails = []
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        mp = _MP()
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td), mp)
            _LOG.info("  [PASS] %s", name)
        except Exception:
            fails.append(name)
            _LOG.exception("  [FAIL] %s", name)
        finally:
            mp.undo()
    _LOG.info(
        "%s",
        "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}",
    )
    sys.exit(1 if fails else 0)

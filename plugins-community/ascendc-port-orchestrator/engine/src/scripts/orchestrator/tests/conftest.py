# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test isolation conftest — DEBT-47.

Root cause of 27 failed + 14 errors in the pre-commit unit gate:
  1. A legacy route test had a collection-time side effect that evicted the
     orchestrator.py MODULE from
     sys.modules and replaced it with the orchestrator PACKAGE object.  Any
     test file collected after it and using `import orchestrator as orch` got
     the PACKAGE (which lacks phase_o0, extract_canonical_handoff, etc.).

  2. test_refuse_detached.py's three unit tests each call
     sys.modules.pop("orchestrator") + fresh import + importlib.reload inside
     the test body — leaving a different module object in sys.modules["orchestrator"]
     after the test.  Later tests that do lazy `from orchestrator import X`
     inside their test bodies get the wrong identity.

  3. Several files do module-level `sys.path.insert(0, ...)` (at collection
     time), permanently widening sys.path.  Accumulated path entries can cause
     module-identity splits (same .py loaded twice under different names).

This conftest provides one autouse function-scoped fixture that snapshots the
critical global state BEFORE each test and restores it AFTER, regardless of
whether the test passes, fails, or errors.

Specifically it restores:
  - sys.path (list identity; duplicate entries added during the test are
    removed; entries removed during the test are re-added at their original
    indices — implemented as a full list replace).
  - sys.modules (shallow copy of all keys/values before the test; after the
    test, remove any new keys and restore any replaced/deleted values).
  - os.environ (dict snapshot; restores additions, deletions, modifications).
  - os.getcwd() (restores cwd if a test calls os.chdir).

Note: module-level (import-time) side effects in the test files themselves
cannot be undone by a function-scoped fixture because they run BEFORE the
fixture. The collection-time eviction was fixed by avoiding changes to
sys.modules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Reorg path stabilization (tests split into ut/ it/ ct/ subdirs, 2026-06)
# ─────────────────────────────────────────────────────────────────────────────
# Most test files locate their target MODULES with paths computed from their
# own __file__, e.g. `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
# under the assumption they live directly in tests/ (so parent.parent == the
# orchestrator/ dir). When a file is moved one level deeper (tests/ -> tests/ut/),
# that depth shifts by 1 and the computed path points at tests/ instead of
# orchestrator/ — breaking `import orchestrator`, `import agent_dispatch`, etc.
#
# Fix it ONCE here: this conftest's own location is STABLE (always
# orchestrator/tests/conftest.py) regardless of which subdir a test lives in.
# Anchor the real module roots from it and put them on sys.path. The per-file
# inserts then become harmless — the correct dir is already importable, and
# their stale computed path (now pointing at tests/ or a non-existent dir) just
# never resolves anything. This conftest applies to tests/ AND every subdir
# (ut/it/ct/integration), because pytest loads a directory's conftest for all
# descendants.
_CONFTEST = Path(__file__).resolve()
_TESTS_DIR = _CONFTEST.parent               # .../src/scripts/orchestrator/tests
_ORCH_DIR = _TESTS_DIR.parent               # .../src/scripts/orchestrator
_SCRIPTS_DIR = _ORCH_DIR.parent             # .../src/scripts
_SRC_DIR = _SCRIPTS_DIR.parent              # .../src
_REPO_ROOT = _SRC_DIR.parent                # repo root

# ORDER MATTERS — there is a NAME COLLISION on the bare name `orchestrator`:
#   * src/scripts/orchestrator/orchestrator.py  → the MODULE (257KB, holds most fns)
#   * src/scripts/orchestrator/                 → a PACKAGE (__init__.py re-exports nothing)
# Many tests do `from orchestrator import <fn>` where <fn> lives in the MODULE, so bare
# `import orchestrator` MUST resolve to orchestrator.py, NOT the package. That requires
# _ORCH_DIR (exposes orchestrator.py) to sit AHEAD of _SCRIPTS_DIR (exposes the package dir)
# on sys.path. Each entry below is insert(0)'d, so the LAST tuple item ends up at sys.path[0]
# — _ORCH_DIR is therefore listed LAST on purpose. This restores the pre-reorg behavior
# (per-file inserts used to put orchestrator/ first). Verified pre-move that nothing imports
# the bare `orchestrator` PACKAGE's submodules; only `src.scripts.orchestrator.*` package-
# qualified imports exist, and those resolve via _REPO_ROOT, unaffected by this ordering.
for _stable_path in (
    str(_TESTS_DIR),                 # `import _reorg_paths` (shared stable data-path anchors) from ut/it/ct
    str(_REPO_ROOT),                 # `from src.scripts.orchestrator import ...`
    str(_SCRIPTS_DIR),               # scripts-level flat modules (precision_eval_two_tier, lint scripts, ...)
    str(_SCRIPTS_DIR / "workflow"),  # `import <workflow module>`
    str(_ORCH_DIR / "plugins"),      # `import <plugin module>`
    str(_ORCH_DIR / "briefs"),       # `import briefs._common` etc.
    str(_ORCH_DIR / "precision"),    # `import precision_cannbot_adapter` (flat sibling import)
    str(_ORCH_DIR),                  # LAST → sys.path[0]: bare `import orchestrator` = orchestrator.py MODULE
):
    if _stable_path not in sys.path:
        sys.path.insert(0, _stable_path)


# DEBT (UT-de-env, 2026-06-30): the orchestrator briefs call `load_env()` which reads
# `workspace/.ascendc_env` — a GITIGNORED file present only on a provisioned dev box, ABSENT in a
# clean checkout / hermetic CI / cannbot-dev. Tests that build a brief therefore FALSE-FAIL with
# `FileNotFoundError: .ascendc_env not found` off the dev box (they pass locally only because the
# gitignored file happens to exist). The autouse fixture below makes the suite HERMETIC: when no
# real `.ascendc_env` and no `ASCENDC_ENV_PATH` override are present, it points load_env's
# `DEFAULT_ASCENDC_ENV` (the LOW-precedence default — so per-test monkeypatches of DEFAULT_ASCENDC_ENV
# or ASCENDC_ENV_PATH still override it) at a session tmp file holding a minimal valid env. On a
# provisioned box (real .ascendc_env present) it is a NO-OP — the real env is used unchanged. This
# changes test SETUP only, not tested code → behavior-preserving + refactor-agnostic.
_HERMETIC_ASCENDC_ENV = (
    "TARGET=a5\nBACKEND=ascendc\n"
    "A5_HOST=hermetic.test.local\nA5_USER=root\nA5_PASSWORD=\n"
    "A5_CONTAINER=hermetic_test\nA5_DEFAULT_NPU_ID=0\n"
    "A5_SOC_VERSION=Ascend950PR_9579\nA5_CANN_PATH=/opt/hermetic/cann\n"
    "SOC_VERSION=Ascend950PR_9579\nCANN_PATH=/opt/hermetic/cann\n"
    "A3_HOST=hermetic.test.local\nA3_USER=root\nA3_PASSWORD=\n"
    "A3_CONTAINER=hermetic_test_a3\nA3_CANN_PATH=/opt/hermetic/cann\n"
    "A3_SOC_VERSION=Ascend910_9382\nA3_DEFAULT_NPU_ID=0\n"
    "A3_WORKSPACE=/opt/hermetic/workspace\nA3_BACKUP_ROOT=/opt/hermetic/backup\n"
)


@pytest.fixture(scope="session")
def _hermetic_ascendc_env_file(tmp_path_factory):
    """Write the hermetic env to ONE session tmp file; reused by the per-test fixture."""
    path = tmp_path_factory.mktemp("hermetic_env") / ".ascendc_env"
    path.write_text(_HERMETIC_ASCENDC_ENV)
    return path


@pytest.fixture(autouse=True)
def _hermetic_ascendc_env(_hermetic_ascendc_env_file):
    """Point `load_env`'s DEFAULT at the hermetic env iff no real env / override is available.

    We set `_common.DEFAULT_ASCENDC_ENV` (the LOW-precedence default), NOT `ASCENDC_ENV_PATH`
    (the HIGH-precedence override) — so a test that monkeypatches `DEFAULT_ASCENDC_ENV` (to
    exercise a specific env scenario, e.g. missing A3_HOST) or sets `ASCENDC_ENV_PATH` STILL
    overrides this hermetic default. NO-OP on a provisioned box (real workspace/.ascendc_env) or
    when ASCENDC_ENV_PATH is already set. Function-scoped + restored, so it never leaks between
    tests. Behavior-preserving setup change only."""
    project_root = Path(__file__).resolve().parents[3]  # …/a5_ops_slim
    real_env = project_root / "workspace" / ".ascendc_env"
    if os.environ.get("ASCENDC_ENV_PATH") or real_env.exists():
        yield
        return
    # `briefs` lives under …/orchestrator (this file's parent dir). Ensure it is importable even
    # when a test runs in isolation (no other test has inserted that path yet); guard the import so
    # a test that doesn't touch briefs at all is unaffected if the package is absent.
    orch_dir = str(Path(__file__).resolve().parents[1])  # …/orchestrator
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        from briefs import _common
    except ImportError:
        yield
        return
    prev = _common.DEFAULT_ASCENDC_ENV
    _common.DEFAULT_ASCENDC_ENV = _hermetic_ascendc_env_file
    try:
        yield
    finally:
        _common.DEFAULT_ASCENDC_ENV = prev


@pytest.fixture(autouse=True)
def _hermetic_hook_registration(monkeypatch):
    """Keep checkout tests hermetic while production O0 remains fail-closed.

    A clean CI checkout intentionally has not run ``init.sh``, so it has no
    live direct-settings registration to validate. Patch only the external
    registration probe; the hook files and the rest of Phase O0 are still
    exercised. Dedicated O0 tests replace this patch per test to cover both
    READY and BLOCKED registration verdicts, and scripts/tests exercises the
    real read-only checker.
    """
    import phase_o0

    monkeypatch.setattr(
        phase_o0,
        "_check_hook_registration",
        lambda: ("hermetic-test", Path("hooks/hooks.json"), []),
    )


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Snapshot and restore sys.path, sys.modules, os.environ, os.getcwd()
    around every test in this directory.

    Implementation notes:
    - sys.path: store a *copy* of the list; after the test, splice the live
      list back to the snapshot (in-place, so other references to sys.path
      continue to work).
    - sys.modules: shallow-copy the dict; after the test, iterate the diff and
      restore.  Module *objects* that were replaced or reloaded get their
      original objects put back.  Brand-new keys added during the test get
      removed.  Keys present before the test but deleted during the test get
      re-added.  This covers importlib.reload() (which replaces the value for
      an existing key) as well as pop() + fresh-import (which first deletes
      then adds a new value for the key).
    - os.environ: same snapshot/restore pattern as sys.modules.
    - os.getcwd(): just restore the directory.
    """
    # ── snapshot ────────────────────────────────────────────────────────────
    path_snap: list[str] = list(sys.path)
    mods_snap: dict[str, Any] = dict(sys.modules)
    env_snap: dict[str, str] = dict(os.environ)
    cwd_snap: str = os.getcwd()

    # DEBT-47b: strip GIT_* leaked by the pre-commit hook context. When the suite
    # runs inside `git commit`, git exports GIT_DIR/GIT_INDEX_FILE pointing at the
    # REAL repo; any test that runs a `git -C <tmp>` subprocess (finalize git-add
    # tests, etc.) then operates on the real repo's index instead of its tmp one
    # → silently STAGES phantom archive paths (output/.../ctc_loss_v3) into the
    # outer index, polluting the very commit that ran the suite. Stripping GIT_*
    # makes git resolve the repo from -C/cwd (the tmp repo), as in production.
    # env_snap (above) restores them after the test (no global leak).
    for _gk in [k for k in os.environ if k.startswith("GIT_")]:
        del os.environ[_gk]

    yield  # ← test runs here

    # ── restore sys.path ────────────────────────────────────────────────────
    sys.path[:] = path_snap

    # ── restore sys.modules ─────────────────────────────────────────────────
    # 1. Remove keys added during the test.
    for key in list(sys.modules.keys()):
        if key not in mods_snap:
            del sys.modules[key]
    # 2. Restore keys that existed before the test but were removed or
    #    replaced (reload changes the value for the same key).
    for key, val in mods_snap.items():
        if sys.modules.get(key) is not val:
            sys.modules[key] = val

    # ── restore os.environ ──────────────────────────────────────────────────
    # Remove additions.
    for key in list(os.environ.keys()):
        if key not in env_snap:
            del os.environ[key]
    # Restore modifications and re-add deletions.
    for key, val in env_snap.items():
        if os.environ.get(key) != val:
            os.environ[key] = val

    # ── restore cwd ─────────────────────────────────────────────────────────
    try:
        if os.getcwd() != cwd_snap:
            os.chdir(cwd_snap)
    except OSError:
        # cwd_snap may no longer exist (tmp_path cleanup); nothing to do.
        pass


# --- B4 test-hygiene (2026-07-03): skip a5_ops-env-specific tests when their fixture is ABSENT ---
# These tests reference a5_ops-only paths/fixtures the cannbot bundle renamed or omits BY DESIGN:
#   - ascendc-op-gen SKILL.md → cannbot renamed to ops/ascendc-cross-gen-port (thin NL shell).
# Transparent SKIP (not deletion, not blanket): a5_ops CI (fixtures present) still runs them; only the bundle skips.
_ENGINE_SRC_B4 = _CONFTEST.parents[3]        # engine/src
_A5_SKILL_B4 = _ENGINE_SRC_B4 / "skills" / "ascendc-op-gen" / "SKILL.md"
_A5OPS_ENV_SKIP_B4 = {
    "test_skill_md_exists": _A5_SKILL_B4,
    "test_skill_md_mode_table_mentions_port_a3": _A5_SKILL_B4,
    "test_skill_md_yaml_drift_sentinel": _A5_SKILL_B4,
    "test_every_documented_orch_flag_exists_in_cli": _A5_SKILL_B4,
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        fx = _A5OPS_ENV_SKIP_B4.get(item.name)
        if fx is not None and not fx.exists():
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "a5_ops-env-specific (B4): fixture absent in cannbot bundle "
                        f"[{fx.name}] — renamed/omitted by design"
                    )
                )
            )

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Candidate-scope digest scheme v4 regression tests (2026-08-30, PR13 WP-A / A.0).

The 2026-08-29 hotfix added ``.opgen_same_signature.json`` and
``.kernel_worker_active`` to ``_CANDIDATE_RUNTIME_TOP_LEVEL`` (both are
controller/worker runtime markers written and removed INSIDE the O5 freeze
window — including them drifted the current-scope digest between O5 freeze and
the finalize gates and failed all 7 provenance gates on a 53/53 PASS
2_FFN_evo candidate).  The exclusion-list contract requires a
``CANDIDATE_DIGEST_SCHEME`` bump on every exclusion-semantics change, and the
regression test must prove the SEMANTIC end to end: writing/removing those
markers after a freeze must leave ``candidate_tree_sha256`` unchanged — not
merely that the symbols sit in the exclusion set.

Run: cd src/scripts && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
     orchestrator/tests/ut/test_candidate_scope_digest_v4.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/

from npubench import npubench_core as core  # noqa: E402
from npubench.npubench_core import (  # noqa: E402
    _CANDIDATE_RUNTIME_TOP_LEVEL,
    candidate_tree_sha256,
)


def _candidate_workspace(root: Path) -> Path:
    """Minimal candidate-shaped workspace: entry point + one kernel source."""
    workspace = root / "workspace" / "op_x"
    (workspace / "kernel").mkdir(parents=True)
    (workspace / "model_new_ascendc.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (workspace / "kernel" / "op_x.cpp").write_text("// candidate\n", encoding="utf-8")
    return workspace


def test_digest_scheme_bumped_to_v4() -> None:
    assert core.CANDIDATE_DIGEST_SCHEME == "npubench-candidate-scope/v4"


def test_runtime_markers_in_exclusion_set() -> None:
    assert ".opgen_same_signature.json" in _CANDIDATE_RUNTIME_TOP_LEVEL
    assert ".kernel_worker_active" in _CANDIDATE_RUNTIME_TOP_LEVEL


def test_freeze_window_marker_churn_leaves_digest_unchanged(tmp_path: Path) -> None:
    """End-to-end: write/delete the runtime markers after the O5 freeze."""
    workspace = _candidate_workspace(tmp_path)
    # Worker spawn creates the active marker before the freeze.
    (workspace / ".kernel_worker_active").write_text("12345\n", encoding="utf-8")
    frozen = candidate_tree_sha256(workspace)
    # O5 post-verify: the P0-1 same-signature counter rewrites its ledger.
    (workspace / ".opgen_same_signature.json").write_text(
        '{"engine": {"count": 1}}', encoding="utf-8"
    )
    # Worker exit removes the active marker.
    (workspace / ".kernel_worker_active").unlink()
    assert candidate_tree_sha256(workspace) == frozen
    # A second rewrite of the ledger (next O5 round) must still be inert.
    (workspace / ".opgen_same_signature.json").write_text(
        '{"engine": {"count": 2}}', encoding="utf-8"
    )
    assert candidate_tree_sha256(workspace) == frozen


def test_genuine_candidate_edit_still_drifts_digest(tmp_path: Path) -> None:
    """Guard against a vacuous digest: real candidate edits must move it."""
    workspace = _candidate_workspace(tmp_path)
    frozen = candidate_tree_sha256(workspace)
    (workspace / "kernel" / "op_x.cpp").write_text(
        "// candidate v2\n", encoding="utf-8"
    )
    assert candidate_tree_sha256(workspace) != frozen

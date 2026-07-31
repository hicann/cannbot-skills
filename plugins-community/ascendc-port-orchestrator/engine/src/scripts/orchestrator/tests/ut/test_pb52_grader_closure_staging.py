# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PB-52 regression (2026-07-05, task#7): the flat grader scripts' import closures
(`cannbench_grader` + `reference_provider/verify.py`) MUST be staged into the lane-isolated
`current_task` at arcnames matching the scripts' `sys.path.insert`, else the container-side
canonical grader hits ModuleNotFoundError → grader crash ("no JSON in stdout") → fallback runner
crash → RUNNER_FAILED (no tier2_status) → phase_o5 two-tier engagement gate → PERMANENT
finalize→await_worker rollback on a PASS kernel.

The fix stages the closures in `phase_o5_runner._resync_workspace_to_container` at:
  cannbench_grader  -> arcname `orchestrator/precision/cannbench_grader`
  verify.py         -> arcname `reference_provider/verify.py`
which mirror `precision_eval_two_tier.py`'s `sys.path.insert(parent/"orchestrator"/"precision")`
and `sys.path.insert(parent/"reference_provider")` (port_a3_two_tier reuses that sys.path via
`import precision_eval_two_tier`).
"""
import io
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]  # a5_ops repo root
_SCRIPTS = _ROOT / "src" / "scripts"


def test_pb52_closures_exist_and_syspath_matches():
    """The staged closures exist AND the grader's sys.path expects exactly those relative dirs."""
    assert (_SCRIPTS / "orchestrator" / "precision" / "cannbench_grader" / "__init__.py").is_file()
    assert (_SCRIPTS / "reference_provider" / "verify.py").is_file()
    two_tier = (_SCRIPTS / "precision_eval_two_tier.py").read_text()
    # the container-side grader resolves its imports via these relative sys.path dirs →
    # the staging arcnames MUST mirror them (drift here reopens PB-52).
    assert 'parent / "reference_provider"' in two_tier
    assert 'parent / "orchestrator" / "precision"' in two_tier


def test_pb52_staging_present_in_phase_o5_runner():
    """Guard against silent removal of the closure-staging — the bug returns if it's dropped."""
    src = (_SCRIPTS / "orchestrator" / "phase_o5_runner.py").read_text()
    assert "orchestrator/precision/cannbench_grader" in src, "PB-52 cannbench_grader staging removed"
    assert "reference_provider/verify.py" in src, "PB-52 verify.py staging removed"


def test_pb52_tar_stages_closures_at_expected_arcnames():
    """Building the tar with the fix's (src, arcname) pairs must place members at the
    sys.path-expected paths, with no __pycache__/.pyc leakage.
    """
    def _no_pycache(ti):
        return None if ("__pycache__" in ti.name or ti.name.endswith(".pyc")) else ti

    pairs = [
        (_SCRIPTS / "orchestrator" / "precision" / "cannbench_grader",
         "orchestrator/precision/cannbench_grader"),
        (_SCRIPTS / "reference_provider" / "verify.py",
         "reference_provider/verify.py"),
    ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for s, a in pairs:
            assert s.exists(), s
            tar.add(str(s), arcname=a, filter=_no_pycache)
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        names = set(tar.getnames())

    assert "orchestrator/precision/cannbench_grader/__init__.py" in names
    assert "orchestrator/precision/cannbench_grader/compare.py" in names
    assert "reference_provider/verify.py" in names
    assert not any(n.endswith(".pyc") or "__pycache__" in n for n in names), \
        "PB-52 staging leaked __pycache__/.pyc into the container tar"


# --- DEBT-201 tier-split staging (2026-07-08, same failure class as PB-52) -----------------
# precision_eval_two_tier.py was split into precision_tier1/precision_tier2 (2026-07-05).
# The O5 runner stages precision_eval_two_tier.py FLAT but must ALSO stage precision_tier1.py
# and precision_tier2.py (imported flat: `from precision_tier1 import ...`; precision_tier2
# also imports precision_tier1) — else container-side ModuleNotFoundError → grader crash →
# false finalize rollback on a PASS kernel. independent review hit this downstream during the v3.14.0
# cannbot re-sync (@914303c); fixed upstream so the next customer never hits it.

def test_tier_split_files_exist_and_imported_flat():
    """precision_eval_two_tier imports precision_tier1/tier2 FLAT; precision_tier2 imports tier1."""
    assert (_SCRIPTS / "precision_tier1.py").is_file()
    assert (_SCRIPTS / "precision_tier2.py").is_file()
    two_tier = (_SCRIPTS / "precision_eval_two_tier.py").read_text()
    assert "from precision_tier1 import" in two_tier
    assert "from precision_tier2 import" in two_tier
    tier2 = (_SCRIPTS / "precision_tier2.py").read_text()
    assert "from precision_tier1 import" in tier2, "tier2→tier1 transitive dep — both must stage"


def test_tier_split_staged_in_phase_o5_runner():
    """Guard against silent removal of the tier-split staging — the grader crash returns if dropped."""
    src = (_SCRIPTS / "orchestrator" / "phase_o5_runner.py").read_text()
    assert "precision_tier1.py" in src, "tier-split precision_tier1.py staging removed (DEBT-201)"
    assert "precision_tier2.py" in src, "tier-split precision_tier2.py staging removed (DEBT-201)"

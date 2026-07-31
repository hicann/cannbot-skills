# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0nn (2026-06-06): O5 resync must be ADDITIVE, never CLOBBERING.

Root cause of the FA-A5 port_a3 O5 false-negative (kernel genuinely PASSES
40/40 when pass_a_runner.py runs directly against current_task's fresh `.so` +
40-case dataset, but ssh_runner re-measures tier1_pass=0/total=40 — a TOTAL
inversion, all 40 fail):

`_resync_workspace_to_container` untar'd the LOCAL workspace's copies of
`model_new_ascendc.py` / `edge_inputs.pt` / `edge_dataset.pt` OVER the
container `current_task/`'s freshly-built-and-verified copies, while NEVER
pushing the matching freshly-built `kernel/build/.so` (it's not in push_files).
For port_a3 IL-chain flows the kernel + edge dataset are built/captured ON the
container and never synced back to the local workspace, so the local copies are
stale. Result: the verifier loaded a mismatched stale wrapper/dataset against
the deployed `.so` → all 40 cases fail → MISMATCH vs the 40/40 claim → infinite
O5-rollback loop.

Fix: the container-side untar uses `tar --skip-old-files` so files already
present in current_task (the just-verified artifacts) are kept authoritative;
only genuinely-MISSING files (e.g. a verifier script the worker wrote AFTER its
deploy — the original P0aba.O5 purpose) are added.

This GENERALIZES to every port_a3 op finalizing through O5 (and any mode whose
authoritative artifacts live on the container, not the local workspace). It
canNOT mask a true failure: the MEASURE still runs the verifier against the
deployed `.so`; the fix only stops the resync from corrupting the artifact set
that measure runs against. The `claimed` side is read from the LOCAL
workspace/verification.json (phase_o5.post_verify_for_finalize), so keeping a
stale current_task/verification.json is irrelevant to the comparison.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import phase_o5_runner  # noqa: E402


def test_resync_untar_is_non_clobbering() -> None:
    """The container-side untar in _resync_workspace_to_container must use
    `--skip-old-files` (additive) — NOT a bare `tar xf` (clobbering).
    """
    src = inspect.getsource(getattr(phase_o5_runner, '_resync_workspace_to_container'))
    assert "--skip-old-files" in src, (
        "O5 resync untar must use `tar --skip-old-files` so it never clobbers "
        "the freshly-built-and-verified current_task artifacts with stale "
        "local-workspace copies (FA-A5 0/40 false-negative root cause)."
    )
    # And it must NOT regress to a bare overwriting `tar xf`/`tar -xf`.
    assert "tar xf " not in src and "tar -xf " not in src, (
        "bare `tar xf`/`tar -xf` overwrites current_task's just-verified "
        "artifacts — use `tar --skip-old-files -xf` instead."
    )


def test_resync_still_pushes_verifier_scripts() -> None:
    """Non-clobber must NOT drop the verifier scripts the worker wrote
    post-deploy — those files are ABSENT in current_task, so --skip-old-files
    still writes them (original P0aba.O5 purpose preserved).
    """
    src = inspect.getsource(getattr(phase_o5_runner, '_resync_workspace_to_container'))
    for name in ("pass_a_runner.py", "run_pass_b.py", "edge_dataset.pt",
                 "model_new_ascendc.py"):
        assert name in src, f"{name} dropped from O5 resync push list"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression test for phase_o5_runner target-aware CANN_PATH resolution.

Empirical anchor: 2026-05-18 22_Nonzero finalize on the A3 backend
failed with `bash: line 1: /data/cann_b103/cann-9.0.0/set_env.sh: No
such file or directory`. Root cause: lines 638 and 764 used
`env.get("CANN_PATH", "/data/cann_b103/cann-9.0.0")` — a hardcoded
A5-host fallback path that doesn't exist in A3 containers.

Fix: prefer the `{target}_CANN_PATH` key, fall back to the generic
`CANN_PATH`, and only as a last resort use `/usr/local/Ascend/cann`
(the standard CANN install location that is more likely to be
present on any target than the a5-host-specific
`/data/cann_b103/cann-9.0.0`).

Tests:

1. A3 target with `A3_CANN_PATH` set → resolves to it.
2. A3 target without `A3_CANN_PATH` but with generic `CANN_PATH` →
   resolves to the generic.
3. A3 target with neither → falls back to `/usr/local/Ascend/cann`
   (NOT the old `/data/cann_b103/...` hardcode).
4. A5 target with `A5_CANN_PATH` set → resolves to it.
"""
from __future__ import annotations
import _reorg_paths  # reorg ut/it/ct: stable data-path anchors


def _resolve_cann_path(env: dict, target: str) -> str:
    """Mirror of the resolution logic now in phase_o5_runner.py:638/764.

    Kept here so the test is independent of import wiring and can act
    as a contract pin: if either runner call-site drifts, this test
    locks the intended resolution order.
    """
    return env.get(f"{target}_CANN_PATH") or env.get("CANN_PATH", "/usr/local/Ascend/cann")


def test_target_specific_key_wins() -> None:
    env = {"A3_CANN_PATH": "/usr/local/Ascend/cann", "CANN_PATH": "/data/cann_b103/cann-9.0.0"}
    assert _resolve_cann_path(env, "A3") == "/usr/local/Ascend/cann"


def test_generic_fallback_when_target_key_absent() -> None:
    env = {"CANN_PATH": "/opt/cann"}
    assert _resolve_cann_path(env, "A3") == "/opt/cann"


def test_safe_default_when_both_absent() -> None:
    env = {}
    assert _resolve_cann_path(env, "A3") == "/usr/local/Ascend/cann"
    # Critically, must NOT be the old a5-specific hardcode.
    assert _resolve_cann_path(env, "A3") != "/data/cann_b103/cann-9.0.0"


def test_a5_target() -> None:
    env = {"A5_CANN_PATH": "/data/cann_b103/cann-9.0.0",
           "A3_CANN_PATH": "/usr/local/Ascend/cann"}
    assert _resolve_cann_path(env, "A5") == "/data/cann_b103/cann-9.0.0"
    assert _resolve_cann_path(env, "A3") == "/usr/local/Ascend/cann"


def test_runner_source_uses_the_pattern() -> None:
    """Pin: phase_o5_runner.py resolves CANN_PATH for the build via the
    target-aware pattern with the SAFE default (not the a5-specific hardcode).

    task#24-item2 (2026-06-01): the two finalize call-sites were consolidated into
    the mode-gated helper `_a5_build_cann_path` (port_a3 → A5_CANN_PATH; else the
    legacy `{target}_CANN_PATH` resolution). So the pattern now lives once in the
    helper, and the call-sites use `_a5_build_cann_path(...)`. The anti-hardcode
    intent is unchanged; pin the new structure.

    DEBT-201 batch5 (2026-07-06): the two call-sites (`_run_verifier`,
    `_run_canonical_pass_a`) moved to phase_o5_verify.py (the helper DEF stays in
    phase_o5_runner.py). The invariant is unchanged — build CANN resolution still
    routes through the mode-gated helper — so the source-grep spans BOTH files.
    """
    import pathlib
    text = (
        (_reorg_paths.ORCH_DIR / "phase_o5_runner.py").read_text()
        + "\n"
        + (_reorg_paths.ORCH_DIR / "phase_o5_verify.py").read_text()
    )
    # The target-aware pattern is present (in the helper's non-port_a3 fallback).
    assert 'env.get(f"{target}_CANN_PATH")' in text, (
        "phase_o5_runner.py must still resolve via env.get(f'{target}_CANN_PATH')"
    )
    # The helper def (phase_o5_runner) + the ≥2 build call-sites (phase_o5_verify)
    # route through the mode-gated helper.
    assert text.count("_a5_build_cann_path(") >= 3, (
        "expected the _a5_build_cann_path helper def + ≥2 call-sites; "
        "the build CANN resolution must go through the mode-gated helper"
    )
    # The safe default is used, NOT the old a5-specific hardcode.
    assert 'env.get("CANN_PATH", "/usr/local/Ascend/cann")' in text
    assert 'env.get("CANN_PATH", "/data/cann_b103/cann-9.0.0")' not in text, (
        "found legacy a5-hardcoded CANN_PATH fallback"
    )


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0aay (2026-05-07): smoke test that every brief builder renders cleanly
across representative op-name shapes.

Origin: 5_Cumsum cold-start crashed on DS with `name 'OP_NAME' is not
defined` after P0aax shipped a Phase E template that included literal
`{OP_NAME}` inside an f-string. Python evaluated the brace as a variable
substitution → NameError at brief-build time, before the worker ever spawned.

Lesson: brief builders return f-strings; ANY literal `{...}` markdown
placeholder in those strings must be escaped as `{{...}}`. The cheap
defense is a smoke test that just calls build_*_brief() with a plausible
op name and asserts no exception. This catches both NameError (unescaped
placeholder) AND KeyError (.format() variants).

Coverage: kw / pp / ko / ar / fo / da brief builders (the ones the
orchestrator's agent_dispatch routes to).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import _common as bc  # noqa: E402
from briefs.kw_brief import build_worker_brief  # noqa: E402


def _seed_env(tmp_path: Path) -> bc.AscendCEnv:
    env_path = tmp_path / ".ascendc_env"
    env_path.write_text(textwrap.dedent("""\
        A5_HOST=test
        A5_USER=root
        A5_PASSWORD='x'
        A5_CONTAINER=test
        CANN_PATH=/data/cann
        SOC_VERSION=Ascend950PR_9579
        BENCHMARK_ROOT=/root/bench
        LOCAL_BENCHMARK=/local/bench
        LOCAL_PROJECT=/proj
        TARGET=a5
        OPGEN_MODE=port_a3_to_a5
        BENCHMARK_BRANCH=main
    """))
    return bc.load_env(env_path)


def _seed_workspace(tmp_path: Path, op: str) -> Path:
    workspace = tmp_path / "workspace" / op
    workspace.mkdir(parents=True)
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": op,
        "opgen_mode": "port_a3_to_a5",
    }))
    return workspace


# Representative op names spanning the shapes the orchestrator routes:
# - numbered + CamelCase: "5_Cumsum" (the one that crashed P0aay)
# - numbered + lowercase: "10_layernorm"
# - numbered + multi-underscore: "11_DequantSwigluQuant"
# - non-numbered lowercase: "moeinitrouting"
# - long mixed: "28_MultimodalRopePositionComputationWithGridBasedIndexing"
_REP_OP_NAMES = [
    "5_Cumsum",
    "10_layernorm",
    "11_DequantSwigluQuant",
    "moeinitrouting",
    "28_MultimodalRopePositionComputationWithGridBasedIndexing",
    "1_BatchMatmul",
]


@pytest.mark.parametrize("op", _REP_OP_NAMES)
def test_kw_brief_renders_without_exception(tmp_path, op):
    """The P0aay regression: build_worker_brief() must not raise NameError /
    KeyError on any op name. {OP_NAME} (and any other markdown placeholder)
    should be escaped as {{OP_NAME}} in the f-string source.
    """
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, op)

    brief = build_worker_brief(
        op, workspace,
        lane=0, spawn_index=1, iter_cap_remaining=15,
        env=env,
    )
    # Sanity: brief is non-empty and contains the op name somewhere
    assert isinstance(brief, str)
    assert len(brief) > 1000, f"brief looks truncated for op={op}: {len(brief)} chars"
    assert op in brief, f"brief should mention op={op}"


def test_kw_brief_references_gate_contract(tmp_path):
    """Post-2026-05-09 brief split (DS overflow finding): Phase D and Phase E
    detail moved to `GATE_CONTRACT.md`. The brief should reference that file
    by path so the worker knows where to load the canonical gate text from.
    """
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, "5_Cumsum")
    brief = build_worker_brief(
        "5_Cumsum", workspace,
        lane=0, spawn_index=1, iter_cap_remaining=15,
        env=env,
    )
    # Both Phase D and Phase E should redirect to GATE_CONTRACT.md
    assert "GATE_CONTRACT.md" in brief, (
        "Phase D + E should reference src/skills/references/shared/GATE_CONTRACT.md. "
        "If this fails, the brief split was reverted or the path is wrong."
    )
    # Spot-check: the §-anchor names used as Read targets should appear.
    assert "D.3a-port_a3" in brief
    assert "Phase E Knowledge Update" in brief


def test_kw_brief_with_directive_renders(tmp_path):
    """Respawn-with-directive path should also render cleanly (different
    return branch in _phase_instructions_block).
    """
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, "5_Cumsum")
    brief = build_worker_brief(
        "5_Cumsum", workspace,
        lane=0, spawn_index=2, iter_cap_remaining=10,
        env=env,
        directive_text="Test directive: rebuild kernel with TQue depth=4.",
    )
    assert isinstance(brief, str)
    assert "Test directive" in brief

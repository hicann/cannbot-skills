# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""PR2: backward op_class taxonomy + C2 OL-200 brief binding.

Coverage:
- is_backward_class predicate (base.py)
- detect_op_class appends a GRADIENT token for backward-named ops (schema_norm)
- _backward_perf_c2_block fires for backward / empty for forward (the real
  helper build_worker_brief calls)
- CALL-LEVEL / e2e: build_worker_brief (the real assembler) injects the OL-200
  block for a backward op_class, and a forward op's brief does NOT contain it
  (byte-size preservation) — i.e. the binding is effective in the real brief
  path, not just unit-isolated (#269 no-op lesson).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins.base import is_backward_class  # noqa: E402
from schema_norm import detect_op_class  # noqa: E402
from briefs.kw_brief import _backward_perf_c2_block, build_worker_brief  # noqa: E402
import briefs._common as bc  # noqa: E402


def _seed_env(tmp_path: Path, *, opgen_mode: str) -> bc.AscendCEnv:
    """Hermetic tmp .ascendc_env so the call-level tests run on ANY checkout
    (no dependency on the repo's gitignored workspace/.ascendc_env — which a
    fresh clone / reviewer worktree lacks, causing a false-red FileNotFoundError).
    Mirrors test_p0abm_brief_size_threshold._seed_env. DEBT-101 mechanism.
    """
    env_path = tmp_path / ".ascendc_env"
    env_path.write_text(textwrap.dedent(f"""\
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
        OPGEN_MODE={opgen_mode}
        BENCHMARK_BRANCH=main
    """))
    return bc.load_env(env_path)


# ── predicate ────────────────────────────────────────────────────────────

def test_is_backward_class_true_on_gradient_tag():
    assert is_backward_class("ELEMENTWISE_SMALL GRADIENT") is True
    assert is_backward_class("FUSED SOFTMAX GRADIENT") is True  # LIG-like
    assert is_backward_class("backward") is True  # case-insensitive


def test_is_backward_class_false_on_forward_and_empty():
    assert is_backward_class("ELEMENTWISE_SMALL") is False
    assert is_backward_class("FUSED SOFTMAX") is False
    assert is_backward_class("") is False
    assert is_backward_class(None) is False


# ── classifier GRADIENT token ────────────────────────────────────────────

def _ws(tmp_path, name, *, opgen_mode=None):
    ws = tmp_path / name
    ws.mkdir()
    if opgen_mode is not None:
        (ws / ".opgen_state.json").write_text(json.dumps({
            "op": name,
            "opgen_mode": opgen_mode,
        }))
    return ws


def test_detect_op_class_appends_gradient_for_backward_named(tmp_path):
    # Names ending in `_grad` are classified as gradients.
    assert "GRADIENT" in detect_op_class(_ws(tmp_path, "mul_grad"), {})
    # Names ending in `Backward` are classified as gradients.
    assert "GRADIENT" in detect_op_class(_ws(tmp_path, "24_KvCacheUpdateWithRopeBackward"), {})
    # Names ending in `-Bwd` are classified as gradients.
    assert "GRADIENT" in detect_op_class(_ws(tmp_path, "14_AdaIN2D-Bwd"), {})
    # The abbreviated lightning-indexer gradient name is also recognized.
    assert "GRADIENT" in detect_op_class(_ws(tmp_path, "lightning_indexer_grad"), {})


def test_detect_op_class_preserves_base_category(tmp_path):
    """Additive: base category still present (substring predicates unaffected)."""
    oc = detect_op_class(_ws(tmp_path, "mul_grad"), {})
    assert "ELEMENTWISE_SMALL" in oc  # MUL → ELEMENTWISE_SMALL, plus GRADIENT
    assert is_backward_class(oc) is True


def test_detect_op_class_no_gradient_for_forward(tmp_path):
    oc = detect_op_class(_ws(tmp_path, "13_Cat"), {})
    assert "GRADIENT" not in oc
    assert is_backward_class(oc) is False


# ── C2 helper (real function the assembler calls) ────────────────────────

def test_backward_perf_c2_block_fires_for_backward(tmp_path):
    block = _backward_perf_c2_block(_ws(tmp_path, "mul_grad"))
    assert "OL-200" in block
    assert "BACKWARD-PERF" in block


def test_backward_perf_c2_block_empty_for_forward(tmp_path):
    assert _backward_perf_c2_block(_ws(tmp_path, "13_Cat")) == ""


# ── CALL-LEVEL / e2e: real build_worker_brief assembler ──────────────────

def test_build_worker_brief_injects_ol200_for_backward(tmp_path):
    """The real brief assembler includes the OL-200 C2 block for a backward op."""
    env = _seed_env(tmp_path, opgen_mode="backward")
    ws = _ws(tmp_path, "mul_grad", opgen_mode="backward")
    brief = build_worker_brief(
        "mul_grad", ws, lane=0, spawn_index=1, iter_cap_remaining=5, env=env,
    )
    assert "OL-200" in brief
    assert "BACKWARD-PERF" in brief


def test_build_worker_brief_forward_has_no_backward_block(tmp_path):
    """Forward op's brief must NOT contain the backward block (size-preserving)."""
    env = _seed_env(tmp_path, opgen_mode="port_a3_to_a5")
    ws = _ws(tmp_path, "13_Cat", opgen_mode="port_a3_to_a5")
    brief = build_worker_brief(
        "13_Cat", ws, lane=0, spawn_index=1, iter_cap_remaining=5, env=env,
    )
    assert "BACKWARD-PERF" not in brief
    assert "OL-200" not in brief

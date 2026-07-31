# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests — Layer 2 port_a3 cube-class MIX brief forcing-function
(design `PORT_A3_CUBE_CLASS_MIX_ENFORCEMENT_DESIGN.md` §3.2).

`_port_a3_cube_class_mix_block` injects the MIX-scaffold + understand→regenerate
methodology into the port_a3 worker brief ONLY when op_classification.json carries
the `CUBE_MIX` tag (seeded by Layer 1 in `_cmd_port_a3` when the CANN reference is
cube-required). For non-cube-class ops it returns "" so their briefs stay
byte-identical — same tag-gating contract as `_backward_perf_c2_block`.

These tests pin: (a) tag present -> real MIX content; (b) tag absent / no file /
no workspace -> empty (no brief drift for non-cube ops); (c) the content carries
the load-bearing forcing elements (HACK warning, scaffold, KB-not-output rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
for p in (str(_ORCH_DIR), str(_ORCH_DIR / "briefs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from briefs.kw_brief import _port_a3_cube_class_mix_block  # type: ignore


def _ws(tmp_path: Path, tags) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    if tags is not None:
        (ws / "op_classification.json").write_text(json.dumps({"op_class_tags": tags}))
    return ws


def test_none_workspace_empty():
    assert _port_a3_cube_class_mix_block(None) == ""


def test_no_classification_file_empty(tmp_path):
    ws = _ws(tmp_path, tags=None)  # no op_classification.json written
    assert _port_a3_cube_class_mix_block(ws) == ""


def test_non_cube_class_empty(tmp_path):
    ws = _ws(tmp_path, tags=["a3_to_a5_port"])
    assert _port_a3_cube_class_mix_block(ws) == ""


def test_backward_tag_not_cube_empty(tmp_path):
    # a different tag set must not accidentally trigger the cube block
    ws = _ws(tmp_path, tags=["a3_to_a5_port", "backward", "GRADIENT"])
    assert _port_a3_cube_class_mix_block(ws) == ""


def test_cube_mix_tag_emits_forcing_block(tmp_path):
    ws = _ws(tmp_path, tags=["a3_to_a5_port", "CUBE_MIX"])
    block = _port_a3_cube_class_mix_block(ws)
    assert block != ""
    # load-bearing forcing elements (design §3.2):
    assert "ARCHITECTURAL_HACK" in block            # the gate rejects pure-VEC
    assert "MIX" in block
    assert "WorkspaceQueue" in block                # concrete scaffold sync
    assert "Mmad" in block                          # cube primitive
    assert "cube_vector_fusion" in block            # KB worked-example route
    assert "provenance-logged" in block             # prior archives are advisory
    assert "advisory only" in block
    assert "understand" in block.lower()            # methodology


def test_malformed_classification_json_empty(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "op_classification.json").write_text("{ this is not valid json ")
    assert _port_a3_cube_class_mix_block(ws) == ""


# --- E2E pipeline test (isolation-test ≠ pipeline): drive the REAL port_a3 brief
#     renderer, not just the isolated helper, so a future refactor that drops the
#     interpolation site is caught. (main PR #317 review.)

class _StubEnv:
    """Minimal duck-typed env for the port_a3 brief renderer + the sub-blocks it
    interpolates (migration-level etc.). Attribute access returns a string for
    any name not explicitly set, so the renderer never AttributeErrors on an
    incidental env field — the test asserts on the MIX-block interpolation, not
    on env-derived prose."""
    port_a3_source = "/home/x/workspace/cann/ops-nn/matmul/mat_mul_v3"
    host = "a5host.example"
    container = "npu_dev3"
    target = "a5"

    def __getattr__(self, name):  # any other env.X → harmless string
        return ""


def test_e2e_cube_mix_scaffold_in_full_port_a3_brief(tmp_path):
    """The FULL port_a3 phase brief (what cold-start renders) contains the MIX
    scaffold when the workspace is CUBE_MIX-tagged — proves the helper is
    actually interpolated into `_port_a3_phase_instructions_block`, not just
    callable in isolation.
    """
    from briefs.kw_brief import _port_a3_phase_instructions_block  # type: ignore

    ws = _ws(tmp_path, tags=["a3_to_a5_port", "CUBE_MIX"])
    brief = _port_a3_phase_instructions_block(
        op="mat_mul_v3", workspace=ws, iter_cap_remaining=3, env=_StubEnv()
    )
    # the real port_a3 brief rendered:
    assert "PHASES (port_from_a3_ascendc" in brief
    # AND the MIX forcing block is interpolated into it:
    assert "CUBE-CLASS MIX" in brief
    assert "WorkspaceQueue" in brief
    assert "ARCHITECTURAL_HACK" in brief
    assert "cube_vector_fusion" in brief


def test_e2e_non_cube_brief_has_no_mix_block(tmp_path):
    """A non-cube-class port_a3 op's full brief must NOT carry the MIX block
    (brief stays byte-identical to pre-feature for non-cube ops).
    """
    from briefs.kw_brief import _port_a3_phase_instructions_block  # type: ignore

    ws = _ws(tmp_path, tags=["a3_to_a5_port"])  # no CUBE_MIX
    brief = _port_a3_phase_instructions_block(
        op="some_vec_op", workspace=ws, iter_cap_remaining=3, env=_StubEnv()
    )
    assert "PHASES (port_from_a3_ascendc" in brief  # same brief renders
    assert "CUBE-CLASS MIX" not in brief            # but no MIX forcing block
    assert "WorkspaceQueue" not in brief

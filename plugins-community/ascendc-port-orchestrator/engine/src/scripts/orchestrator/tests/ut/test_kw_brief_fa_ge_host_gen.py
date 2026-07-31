# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""flash_attention_score-pbh-1 (2026-06-11, owner mandate) — kw_brief
GE-host-gen wiring test.

The port_a3 FA-class brief must instruct the worker to GENERATE the GE
op_host (def/infershape/tiling.cpp) by FOLLOWING GE_HOST_TRANSFORM_RECIPE.md
(CARRY def/infershape from the A3 arch22 input, REPLACE-HOOK tiling.cpp onto
the KB `wp_fa_host_tiling.h` shared `wfh::` layer) — NOT byte-copy CANN arch35
source. This is the brief-side companion to the GE_OPHOST_RAW_CANN_COPY
finalize gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import kw_brief  # noqa: E402


def _make_fa_ws(tmp_path: Path, op: str = "flash_attention_score") -> Path:
    """Workspace that resolves to FA-class (op_classification.json present +
    attention-named op so the FA-class template-assembly block fires)."""
    ws = tmp_path / op
    ws.mkdir()
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": ["FUSED", "SOFTMAX"]})
    )
    return ws


def test_ge_host_gen_block_standalone_content():
    """The standalone block names the recipe + the three transform classes."""
    block = getattr(kw_brief, '_fa_ge_host_gen_block')()
    assert "GE_HOST_TRANSFORM_RECIPE.md" in block
    assert "wp_fa_host_tiling.h" in block
    assert "wfh::" in block
    # the three per-file transform rules
    assert "infershape.cpp" in block and "CARRY" in block
    assert "def.cpp" in block and "PATCH" in block
    assert "tiling.cpp" in block and "REPLACE-HOOK" in block
    # names the finalize gate so worker knows the consequence
    assert "GE_OPHOST_RAW_CANN_COPY" in block
    # the red line: no arch35 copy
    assert 'arch35' in block


def test_fa_class_brief_embeds_ge_host_gen(tmp_path):
    """The full FA-class template-assembly brief carries the GE-host-gen step."""
    ws = _make_fa_ws(tmp_path)
    brief = getattr(kw_brief, '_fa_class_template_assembly_block')("flash_attention_score", ws)
    assert brief is not None, "FA-class brief must fire for attention-named op"
    assert "GE OP_HOST GENERATION" in brief
    assert "GE_HOST_TRANSFORM_RECIPE.md" in brief
    assert "wp_fa_host_tiling.h" in brief
    # Provenance contract permits only logged advisory target context.
    assert "provenance-tracked" in brief
    assert "advisory only" in brief
    assert "arch35" in brief


def test_non_fa_op_brief_absent(tmp_path):
    """A non-attention op gets no FA-class brief (so no GE-host-gen text)."""
    ws = _make_fa_ws(tmp_path, op="layer_norm")
    # op_classification has FUSED+SOFTMAX which is_fa_class would match; use a
    # name + tag that is neither attention-named nor FA-tagged.
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": ["ELEMENTWISE"]})
    )
    brief = getattr(kw_brief, '_fa_class_template_assembly_block')("layer_norm", ws)
    assert brief is None

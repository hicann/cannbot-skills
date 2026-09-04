# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""F1 regression: a prior-agent DIRECTIVE must not strip the FA-class template block."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from briefs.kw_brief import _phase_instructions_block  # noqa: E402
from briefs._common import AscendCEnv  # noqa: E402


def _fa_named_workspace(tmp_path):
    ws = tmp_path / "flash_attention_score_oc"
    ws.mkdir()
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": ["ATTENTION", "FUSED"]})
    )
    return ws


def _env(ws):
    env = AscendCEnv(
        target="a5",
        host="host",
        user="u",
        password="p",
        container="c",
        cann_path="/usr/local/Ascend/cann-9.2.0",
        soc_version="Ascend950DT_9582",
        benchmark_root="/tmp/bench",
        local_benchmark="",
        local_project="",
        archive_project="",
        build_archive_enabled=False,
    )
    env.opgen_mode = "port_a3_to_a5"
    env.port_a3_source = str(ws / ".source_migration_stage")
    return env


def _brief(ws, directive):
    return _phase_instructions_block(
        "flash_attention_score_oc",
        ws,
        iter_cap_remaining=9,
        directive_text=directive,
        handoff_from_prior="",
        env=_env(ws),
        backend="ascendc",
    )


def test_directive_preserves_fa_template_block(tmp_path):
    ws = _fa_named_workspace(tmp_path)
    brief = _brief(ws, "fix the QuePosition qualification")
    assert "DIRECTIVE FROM PRIOR AGENT" in brief
    assert "fa_class" in brief
    assert "wholeport" in brief or "MatmulBase" in brief or "MIX_AIC" in brief
    assert "# PHASES" in brief


def test_no_directive_still_emits_fa_template_block(tmp_path):
    ws = _fa_named_workspace(tmp_path)
    brief = _brief(ws, None)
    assert "fa_class" in brief
    assert "wholeport" in brief or "MatmulBase" in brief or "MIX_AIC" in brief

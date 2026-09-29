# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent


def test_legacy_entry_has_one_step_zero_forwarder() -> None:
    agents = LEGACY_ROOT / "AGENTS.md"
    if not agents.is_file():
        return
    text = agents.read_text(encoding="utf-8")
    assert text.count("Step 0：工作流分类") == 1
    assert "catlass-cpp-generator/scripts/select_operator_workflow.py" in text
    assert "legacy" in text and "原 Step 1-7" in text
    assert "需求接收阶段根据数学信息确认" in text
    assert "不要求用户显式说出" in text
    assert "分类信息不足时使用 `pending`" in text
    assert "用户明确确认的 `algorithm_family`" not in text
    assert "Step 1: 环境检查" in text
    assert "Step 7: 完成汇报" in text


def test_new_plugin_does_not_define_legacy_agents() -> None:
    assert not (ROOT / "agents").exists()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Golden lock for the scoped arch22 to arch35 worker prompt.

These hashes pin the reviewed input-provenance prompt for representative input
variations. Intentional contract edits must update the fixture and re-run the
source/target truth-boundary assertions.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile

import pytest


class _StubEnv:
    port_a3_source = "/home/x/workspace/cann/ops-nn/matmul/mat_mul_v3"
    host = "a5host.example"
    container = "npu_dev3"
    target = "a5"

    def __getattr__(self, name):  # any other env.X → harmless string
        return ""


def _ws(tags):
    d = pathlib.Path(tempfile.mkdtemp()) / "ws"
    d.mkdir(parents=True, exist_ok=True)
    if tags is not None:
        (d / "op_classification.json").write_text(json.dumps({"op_class_tags": tags}))
    return d


# (op, op_class_tags, iter_cap_remaining) -> sha256(output)
# Re-pinned 2026-08-31 (OKF-only 迁移): composed brief text now points at
# `kb/okf/**` + plugin `templates/fa_class/` (the previous pins were already
# stale from the A-core OKF 切换; this re-pin covers both).
# Re-pinned 2026-09-02 (issue #559 OKF-only 落地到 engine): the 2026-08-31 pins
# went stale again — on the 1358ec68 baseline the builders still emitted the
# legacy `kb/target/ascendc/**` paths. The OKF-only refactor (a4e3c03a) moved
# the engine pointers to `kb/okf/reference/**` + plugin `templates/fa_class/`;
# baseline-vs-HEAD output diff verified to be exactly that path relocation
# (all new paths exist in the plugin tree). Content change is the point.
# 2026-09-05 重钉：kb/okf/reference/ 目录重组后 brief 里的 KB 路径变化。
# 重钉前逐 case 在改动前后两棵树上生成 brief 做 diff 验证：4 个 case 差异 12–14 行，
# **非路径行为 0** —— 即变化仅为 KB 路径改写，无语义内容变动。
_GOLDEN = [
    ("mat_mul_v3", ["a3_to_a5_port", "CUBE_MIX"], 3,
     "a7faf9b28eca9026f647bca3e24ae7e4050e0fe08ae48f28ed2b1d786af33575"),
    ("some_vec_op", ["a3_to_a5_port"], 3,
     "09eb85eebc87d29dc5c34b856691e91721de3e166d10038958b0a9655c45df9b"),
    ("flash_attention_score", ["a3_to_a5_port", "FA_CLASS"], 2,
     "acf35f5734b2577835bab9e78731e91f690614b29b85ac9382809bfe0a7d9be2"),
    ("abs", ["a3_to_a5_port"], 1,
     "4a17d3982f8d632a4acd18fb37a298b588e4f28cbaf9fdab6cf4596aa855b79d"),
]


@pytest.mark.parametrize("op,tags,iter_cap,expected_sha", _GOLDEN)
def test_port_a3_phase_brief_byte_identical(op, tags, iter_cap, expected_sha):
    from briefs.kw_brief import _port_a3_phase_instructions_block  # type: ignore

    out = _port_a3_phase_instructions_block(
        op=op, workspace=_ws(tags), iter_cap_remaining=iter_cap, env=_StubEnv()
    )
    got = hashlib.sha256(out.encode()).hexdigest()
    assert got == expected_sha, (
        f"port_a3 phase brief changed for op={op!r} tags={tags} iter_cap={iter_cap}: "
        f"len={len(out)} sha={got} != reviewed golden {expected_sha}."
    )

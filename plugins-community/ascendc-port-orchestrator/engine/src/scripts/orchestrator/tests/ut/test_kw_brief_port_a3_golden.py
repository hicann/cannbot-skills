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
_GOLDEN = [
    ("mat_mul_v3", ["a3_to_a5_port", "CUBE_MIX"], 3,
     "c002c2061ea1edc980a51c144f05b55562e3cab7d4c438616c4eb392b4f9ead2"),
    ("some_vec_op", ["a3_to_a5_port"], 3,
     "fb7ed33096c420e9dd49fab76c36f9fa3b388af9b27908898e7fc96f2eadfe3b"),
    ("flash_attention_score", ["a3_to_a5_port", "FA_CLASS"], 2,
     "c2cc0c0db9b380d2c7f733d1dcc9c109534cc559a74dc2b15fb1565f9bcf5b21"),
    ("abs", ["a3_to_a5_port"], 1,
     "39f70626a398f40662538b862cac8e56c92f293080e67b0077f50f4f95101cbc"),
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

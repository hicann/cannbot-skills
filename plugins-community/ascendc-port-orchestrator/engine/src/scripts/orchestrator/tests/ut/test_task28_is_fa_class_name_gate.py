# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""task#28 — FA-class detection must use the op NAME, not FUSED+SOFTMAX tags.

§6 / blue repro: `/ascendc-op-gen hc_split_sinkhorn` (DeepSeek-V4 Sinkhorn split —
pure-Vector fused, NO matmul/Cube) classified with tags
`[fused, transcendental, softmax, reduction, normalization]`. The tag-based
`is_fa_class(op_class)` = "FUSED" in u and "SOFTMAX" in u → True (false positive),
tag-IDENTICAL to real FA (3_FusionAttention, grouped_query_attention). The ONLY
reliable FA signal is the op NAME (`is_attention_named`).

Post-IL-removal (task#17): FA-class ops are built by kw template-assembly. The
op-NAME gate (`is_attention_named`) remains the canonical FA discriminator used by
the FA-class routing / brief-injection hooks; these tests pin its behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from plugins.base import is_attention_named, is_fa_class  # noqa: E402

# The FA-shaped tag string both real FA AND hc_split_sinkhorn produce — the
# tag-based predicate cannot distinguish them; only the NAME can.
FA_SHAPED_TAGS = "FUSED TRANSCENDENTAL SOFTMAX REDUCTION NORMALIZATION"


# ── is_attention_named unit (the canonical FA-name discriminator) ────────────
@pytest.mark.parametrize("name,expected", [
    ("3_FusionAttention", True),
    ("grouped_query_attention", True),
    ("flash_attention_score", True),
    ("hc_split_sinkhorn", False),   # the §6 false-positive op
    ("layer_norm_fused", False),
    ("fused_quant_mat_mul", False),
    ("", False),
    (None, False),
])
def test_is_attention_named(name, expected):
    assert is_attention_named(name) is expected


def test_fa_class_keys_on_attention_tag_not_fused_softmax():
    """task#31: is_fa_class keys on the ATTENTION structural tag, so the
    FA-shaped tag string (FUSED+SOFTMAX+... but NO attention) does NOT match —
    this is what prevents the Sinkhorn false-positive. The op NAME gate
    (is_attention_named) is the independent discriminator for real FA.
    """
    assert is_fa_class(FA_SHAPED_TAGS) is False          # no ATTENTION tag
    assert is_fa_class("FUSED SOFTMAX ATTENTION") is True  # ATTENTION tag present
    # Name gate cleanly separates real FA from the Sinkhorn-shaped op.
    assert is_attention_named("3_FusionAttention") is True
    assert is_attention_named("hc_split_sinkhorn") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Op-class / op-name taxonomy predicates (extracted from plugins/base.py,
DEBT-201 god-class decompose).

Pure leaf module — no dependency on the plugin protocol or any plugin
implementation. `plugins.base` re-exports these names so every historical
import path (`from plugins.base import is_fa_class`, etc.) keeps working
byte-for-byte.
"""
from __future__ import annotations


# ────────────────────────────────────────────────────────────────────
# Op-class taxonomy predicates (FA-class Phase 1 — PR #122 design §3.2)
# ────────────────────────────────────────────────────────────────────
# `op_class` arrives as uppercase space-joined taxonomy tag-string from
# `schema_norm._detect_op_class(workspace, vj=None)`, e.g.
# `"FUSED SOFTMAX TRANSCENDENTAL REDUCTION"`. NOT lowercase enum equality.
# Plugin overrides reference these helpers instead of inlining substring
# tuples — eliminates lowercase-enum leak at the design↔code boundary.
# See docs/design/FA_CLASS_PROBLEM_SOLUTION_DESIGN.md §3.2 + R1+B1 folds.

def is_fa_class(op_class: str) -> bool:
    """FA-class predicate (narrow): the `attention` structural taxonomy tag.

    task#31 (2026-06-04): narrowed from the over-broad `FUSED + SOFTMAX` to the
    distinguishing structural tag `ATTENTION`. Root cause (task#28 disk audit):
    `FUSED + SOFTMAX` fired for `hc_split_sinkhorn` (pure-Vector Sinkhorn
    normalization — softmax, NO matmul/Cube), mis-routing a non-FA op into the
    structural-rewrite chain + the FA-class perf threshold + FA finalize gates. The
    classifier (`aog-op-classify`) now emits a dedicated `attention` op-family
    tag ONLY when the op exhibits the QK^T·V compute structure (score matmul →
    row-wise softmax → value matmul), which Sinkhorn does NOT. So `ATTENTION` in
    the uppercase space-joined tag-string is the reliable structural discriminator
    that `FUSED + SOFTMAX` never was.

    The classifier is LLM-driven (probabilistic, §5 of the task#31 design), so
    every FA-class call-site pairs this tag predicate with the deterministic
    op-name backstop `is_attention_named(op_name)` (belt-and-suspenders): the tag
    catches a mis-named FA op, the name catches a classifier tag-miss. Independent
    failure modes.

    Used (tag OR op-name backstop) by `ko_escalation_threshold` /
    `_is_fa_workspace` (extra_finalize_checks) / the kw_brief FA block. The
    FA-class routing gate DELIBERATELY stays op-name-only (task#28) and does
    NOT OR this tag: adding a tag-OR would WIDEN FA-class eligibility on an
    LLM-probabilistic tag (highest blast-radius — a classifier false-tag on a
    non-FA L4 op would mis-route it into the heavy template-assembly path).
    Revisit only after the task#31 §4 re-classification validation confirms the
    `attention` tag is reliably emitted with no non-FA false-positives.

    >>> is_fa_class("FUSED SOFTMAX ATTENTION TRANSCENDENTAL REDUCTION")
    True
    >>> is_fa_class("fused matmul attention softmax")  # case-insensitive
    True
    >>> is_fa_class("FUSED SOFTMAX TRANSCENDENTAL REDUCTION NORMALIZATION")  # Sinkhorn-shaped, no attention
    False
    >>> is_fa_class("FUSED ELEMENTWISE")  # not attention
    False
    >>> is_fa_class("")
    False
    >>> is_fa_class(None)  # graceful on None
    False
    """
    if not op_class:
        return False
    return "ATTENTION" in op_class.upper()


def is_attention_named(op_name: str) -> bool:
    """True iff the op NAME marks it as flash-attention-class (`attention` substring).

    task#28 (2026-06-01): the tag-based `is_fa_class` (FUSED+SOFTMAX) CANNOT
    distinguish a true FA op (QK^T·V, e.g. `3_FusionAttention` /
    `grouped_query_attention` / `flash_attention_score`) from a pure-vector fused
    op that merely contains a softmax (e.g. DeepSeek-V4 `hc_split_sinkhorn` —
    Sinkhorn row/col normalization). Disk-verified: real FA + Sinkhorn classify
    tag-identically (both `algorithm_classification="fused"`, both FUSED+SOFTMAX);
    NO op emits an `attention`/`fused_attention` tag. The ONLY reliable signal is
    the op NAME — every FA-family op name contains `attention`, Sinkhorn does not.
    Precedent: `orchestrator.py` already name-matches `attention` for FA seeding.

    Used by the FA-class routing gate so a fused vector op is NOT mis-routed
    into the FA template-assembly path (it goes the normal kw path). Safe
    fail-mode: a non-`attention`-named FA op would miss the FA path and take the
    normal kw path — degraded, not wrong. Root cause (classifier can't distinguish
    FA from fused-vec-softmax) is deferred to task#31 (classifier emits a
    distinguishing structural tag → gate can return to tag-based).

    >>> is_attention_named("3_FusionAttention")
    True
    >>> is_attention_named("grouped_query_attention")
    True
    >>> is_attention_named("hc_split_sinkhorn")  # fused vector softmax, NOT FA
    False
    >>> is_attention_named("")
    False
    >>> is_attention_named(None)  # graceful
    False
    """
    if not op_name:
        return False
    return "attention" in op_name.lower()


def is_l4_fused(op_class: str) -> bool:
    """L4-fused-class predicate (broad): FUSED taxonomy tag.

    True for any FUSED op including FA-class. Covers
    fused_attention / fused_attention_grad / fused_moe /
    fused_norm_matmul. Used by §3.1 plugin scope predicate
    (BenchmarkPlugin opts-in FA-class behavior for any FUSED op).

    `is_fa_class(x)` implies `is_l4_fused(x)`; reverse does not hold.

    >>> is_l4_fused("FUSED SOFTMAX TRANSCENDENTAL REDUCTION")
    True
    >>> is_l4_fused("FUSED MOE GATING")
    True
    >>> is_l4_fused("ELEMENTWISE")  # no FUSED tag
    False
    >>> is_l4_fused("")
    False
    >>> is_l4_fused(None)
    False
    """
    if not op_class:
        return False
    return "FUSED" in op_class.upper()


def is_backward_class(op_class: str) -> bool:
    """Backward/gradient-class predicate: GRADIENT or BACKWARD taxonomy tag.

    True for reverse (gradient) operators — the ops `training` needs (compute
    dL/dx, dL/dw from the upstream grad). Drives the C2 backward-perf brief
    block (OL-200 MIX_AIC cube/vec software-pipelining guidance) cross-mode:
    ANY mode generating a gradient op (the `backward` mode, or a port_a3 /
    benchmark backward op like LIG) surfaces the same perf knowledge.

    This is NOT a pure label — see `briefs/kw_brief.py` where
    `is_backward_class(op_class)` gates injection of the OL-200 whitebox
    perf-check into the worker brief (mirrors the `is_fa_class` block).
    Companion to the #273 contract-completeness gate (which, in B2/B3, uses
    the same op_class signal for the backward-completeness verdict).

    `op_class` is the uppercase space-joined taxonomy tag-string from
    `schema_norm.detect_op_class` (which appends a `GRADIENT` token for
    backward-named ops, e.g. `*_grad` / `*_backward` / `*_bwd`).

    >>> is_backward_class("ELEMENTWISE_SMALL GRADIENT")
    True
    >>> is_backward_class("FUSED SOFTMAX GRADIENT")  # LIG-like
    True
    >>> is_backward_class("backward")  # case-insensitive
    True
    >>> is_backward_class("ELEMENTWISE_SMALL")  # forward op
    False
    >>> is_backward_class("")
    False
    >>> is_backward_class(None)  # graceful on None
    False
    """
    if not op_class:
        return False
    u = op_class.upper()
    return "GRADIENT" in u or "BACKWARD" in u

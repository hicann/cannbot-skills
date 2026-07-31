# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""port_a3 migration-level decision plugin.

Maps `op_meta` → `MigrationLevel ∈ {L1, L2, L3, L4}` per PR #103
`ascendc-operator-A5-migration` decision tree. Drives kw_brief KB-reference
injection: each level pulls a different set of guides from
`src/skills/references/target/ascendc/migration/`.

Per Zheng directive 2026-05-16 17:42Z: plugin form, NO if-else outside.
Heuristics inside this plugin, not scattered across briefs.

OCP: adding a new heuristic = drop a new `LevelHeuristic` class + register
in `HEURISTICS` list. NO edit to existing heuristics or callers.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol


class MigrationLevel(str, enum.Enum):
    """Closed set per PR #103 SKILL.md decision tree."""
    L1 = "L1"   # basic adaptation (all ops)
    L2 = "L2"   # RegBase API rewrite (perf-critical / quant Cast / FP8)
    L3 = "L3"   # SIMT optimization (Scatter/Gather + simple-index + high-parallel)
    L4 = "L4"   # out of scope (tiling needs IsRegbaseSocVersion / UB shortage)


@dataclass(frozen=True)
class LevelDecision:
    level: MigrationLevel
    rationale: str
    guides: tuple[str, ...]                     # relative paths under migration/
    extra_subdirs: tuple[str, ...] = ()         # e.g. ("simt/", "cube/") to inject wholesale
    needs_escalation: bool = False              # True iff L4 — caller surfaces


class LevelHeuristic(Protocol):
    """One heuristic rule. Returns LevelDecision if applies, None to pass."""
    name: str

    def apply(self, op_meta: dict) -> Optional[LevelDecision]:
        ...


# ---------------------------------------------------------------------------
# Heuristics — each is a small class. Add new ones at bottom of HEURISTICS;
# no edits to existing.
# ---------------------------------------------------------------------------

def _is_fa_forward_class(op_meta: dict) -> bool:
    """Per OL-185 decision rule (2026-05-22 codified post-flat_quant ship):
    L4 IFF op is forward softmax/attention with Q×K@V tile-scheduling, online
    softmax with running max/sum, causal/sliding-window mask handling.
    Op-name pattern: flash_attention*, incre_flash_attention, mha_* forward,
    attention_score_* (NOT *_grad).

    Backward gradients (*_grad) of attention ops are GEMM/scatter/reduce based,
    NOT softmax forward → L2/L3 like flat_quant, NOT L4. The flat_quant
    calibration anchor (2014 LOC + cube+vec MIX_AIC + 8 CrossCoreSetFlag,
    shipped 8/8 PASS via single cold-start commit `7b3c7bf3` on origin/main)
    is the right reference.
    """
    op_class = (op_meta.get("op_class") or "").lower()
    op_name = (op_meta.get("op_name") or "").lower()
    combined = f"{op_class} {op_name}"
    # *_grad / gradient ops are backward → never FA-forward
    if "_grad" in combined or "gradient" in combined or "backward" in combined:
        return False
    fa_forward_markers = (
        "flash_attention", "incre_flash_attention",
        "attention_score", "mha_fwd", "mha_forward",
        "online_softmax",  # online-softmax-tile-scheduling, FA-class
    )
    return any(m in combined for m in fa_forward_markers)


class L4TilingIsRegbase:
    """L4 signal: tiling code uses IsRegbaseSocVersion AND op is FA-forward class.

    Per OL-185 (2026-05-22, post-flat_quant ship): IsRegbaseSocVersion in
    tiling is necessary-not-sufficient for L4. flat_quant tiling also uses
    IsRegbaseSocVersion (V220→V351 RegBase MicroAPI dispatch) but is L2/L3
    because it has no softmax/attention. The load-bearing L4 axis is
    op-class semantics, not tiling surface signal.

    LIG_grad incident 2026-05-23: worker reflexively returned L4 on
    IsRegbaseSocVersion trigger + abdicated via OL-175, despite OL-185
    explicitly disqualifying non-FA-forward ops from L4. Fix landed
    2026-05-24 (owner ultimatum): IsRegbaseSocVersion trigger now combines
    with FA-forward class check; non-FA ops with IsRegbase tiling downgrade
    to L2 (RegBase rewrite) with flat_quant calibration anchor.
    """
    name = "L4_TilingIsRegbase"

    @staticmethod
    def apply(op_meta: dict) -> Optional[LevelDecision]:
        tiling_signals = op_meta.get("tiling_signals", []) or []
        if not any("IsRegbaseSocVersion" in s for s in tiling_signals):
            return None
        # Tiling has IsRegbaseSocVersion. Check OL-185 narrower criterion:
        # is the op FA-forward class? If not, downgrade to L2 (flat_quant anchor).
        if _is_fa_forward_class(op_meta):
            return LevelDecision(
                level=MigrationLevel.L4,
                rationale=(
                    "tiling references IsRegbaseSocVersion + op_class is FA-forward "
                    "(softmax/attention) → architectural tiling rework needed "
                    "(OL-159 L4 path)"
                ),
                guides=("l4-simt-optimization-guide.md",),
                needs_escalation=True,
            )
        # OL-185 escape-hatch: non-FA-forward op with IsRegbase tiling → L2 anchor=flat_quant
        return LevelDecision(
            level=MigrationLevel.L2,
            rationale=(
                "tiling references IsRegbaseSocVersion BUT op is non-FA-forward "
                "(gemm/gather/scatter/reduce class per OL-185) → L2 RegBase rewrite "
                "with flat_quant calibration anchor, NOT L4 escalation"
            ),
            guides=("l1-implementation-guide.md", "l2-register-based-guide.md",
                    "l1-l2-implementation-guide.md"),
            extra_subdirs=("api-overview/",),
        )


class L4UbShortage:
    """L4 signal: UB budget < 40KB SIMT DCache reserve → escalate."""
    name = "L4_UBShortage"

    @staticmethod
    def apply(op_meta: dict) -> Optional[LevelDecision]:
        ub_kb = op_meta.get("ub_budget_kb")
        if isinstance(ub_kb, (int, float)) and ub_kb < 40:
            return LevelDecision(
                level=MigrationLevel.L4,
                rationale=f"UB budget {ub_kb}KB < 40KB SIMT DCache reserve",
                guides=("l4-simt-optimization-guide.md",),
                needs_escalation=True,
            )
        return None


class L3ScatterGatherSimt:
    """L3 SIMT eligibility: scatter/gather + simple-index + high-parallel + GM-direct."""
    name = "L3_ScatterGatherSIMT"

    @staticmethod
    def apply(op_meta: dict) -> Optional[LevelDecision]:
        op_class = (op_meta.get("op_class") or "").lower()
        op_name = (op_meta.get("op_name") or "").lower()
        pattern_tags = set(op_meta.get("pattern_tags") or [])

        scatter_gather_signals = {"scatter", "gather", "index_select", "indexput"}
        if (any(s in op_class for s in scatter_gather_signals)
                or any(s in op_name for s in scatter_gather_signals)
                or scatter_gather_signals & pattern_tags):
            simple_index = op_meta.get("index_complexity", "simple") == "simple"
            high_parallel = op_meta.get("numel_typical", 0) > 1 << 20  # > 1M elements
            if simple_index and high_parallel:
                return LevelDecision(
                    level=MigrationLevel.L3,
                    rationale=(
                        "scatter/gather op with simple indexing + high parallelism "
                        "→ L3 SIMT kernel alongside L1 base"
                    ),
                    guides=("l1-implementation-guide.md", "l3-simt-optimization-guide.md"),
                    extra_subdirs=("simt/",),
                )
        return None


class L2RegBaseRewrite:
    """L2 RegBase eligibility: perf-critical / quant-cast / overflow / FP8/HiFloat8."""
    name = "L2_RegBaseRewrite"

    _L2_OP_CLASSES = {
        "rmsnorm", "layernorm", "rope", "softmax",
        "quant", "dequant", "cast",
    }
    _L2_DTYPES = {"fp8", "hifloat8", "hif8", "e4m3", "e5m2"}

    def apply(self, op_meta: dict) -> Optional[LevelDecision]:
        op_class = (op_meta.get("op_class") or "").lower()
        op_name = (op_meta.get("op_name") or "").lower()
        dtypes = {d.lower() for d in (op_meta.get("dtypes") or [])}
        perf_critical = bool(op_meta.get("perf_critical"))
        has_overflow_control = bool(op_meta.get("has_overflow_control"))

        in_l2_class = any(c in op_class or c in op_name for c in self._L2_OP_CLASSES)
        has_l2_dtype = bool(dtypes & self._L2_DTYPES)

        if in_l2_class or has_l2_dtype or perf_critical or has_overflow_control:
            reasons = []
            if in_l2_class:
                reasons.append("op_class in L2 set (rmsnorm/rope/softmax/quant/cast)")
            if has_l2_dtype:
                reasons.append(f"L2 dtype: {sorted(dtypes & self._L2_DTYPES)}")
            if perf_critical:
                reasons.append("flagged perf_critical")
            if has_overflow_control:
                reasons.append("needs overflow control")
            return LevelDecision(
                level=MigrationLevel.L2,
                rationale="; ".join(reasons),
                guides=("l1-implementation-guide.md", "l2-register-based-guide.md",
                        "l1-l2-implementation-guide.md"),
                extra_subdirs=("reg-base-vector/", "memory-base-vector/"),
            )
        return None


class L1Default:
    """L1 fallback: always applies if no higher level matched."""
    name = "L1_Default"

    @staticmethod
    def apply(op_meta: dict) -> Optional[LevelDecision]:
        return LevelDecision(
            level=MigrationLevel.L1,
            rationale="default — no L2/L3/L4 signals matched",
            guides=("l1-implementation-guide.md",),
            extra_subdirs=("api-overview/",),
        )


# Order matters: L4 escalation > L3 SIMT > L2 RegBase > L1 default.
HEURISTICS: tuple[LevelHeuristic, ...] = (
    L4TilingIsRegbase(),
    L4UbShortage(),
    L3ScatterGatherSimt(),
    L2RegBaseRewrite(),
    L1Default(),
)


def decide_migration_level(op_meta: dict) -> LevelDecision:
    """Run heuristics in order; first non-None decision wins."""
    for h in HEURISTICS:
        d = h.apply(op_meta)
        if d is not None:
            return d
    # Should never reach — L1Default always returns.
    raise RuntimeError("no heuristic matched (including L1 default) — registry corrupt")

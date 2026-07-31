# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P114 migration-level decision plugin tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_level import (
    MigrationLevel, decide_migration_level, HEURISTICS,
)


def test_l1_default_when_no_signals():
    d = decide_migration_level({"op_name": "elementwise_add", "op_class": "elementwise"})
    assert d.level == MigrationLevel.L1
    assert "l1-implementation-guide.md" in d.guides


def test_l2_rmsnorm():
    d = decide_migration_level({"op_name": "RmsNorm", "op_class": "rmsnorm"})
    assert d.level == MigrationLevel.L2
    assert "l2-register-based-guide.md" in d.guides


def test_l2_fp8_dtype():
    d = decide_migration_level({"op_name": "cast_f32_to_f8", "op_class": "cast",
                                "dtypes": ["fp8", "fp32"]})
    assert d.level == MigrationLevel.L2


def test_l3_scatter_gather_simt():
    d = decide_migration_level({
        "op_name": "gather", "op_class": "gather",
        "index_complexity": "simple", "numel_typical": 2_000_000,
    })
    assert d.level == MigrationLevel.L3
    assert "l3-simt-optimization-guide.md" in d.guides
    assert "simt/" in d.extra_subdirs


def test_l3_not_when_index_complex():
    """Scatter/gather with complex indexing falls to L2 (perf-critical) or L1."""
    d = decide_migration_level({
        "op_name": "gather", "op_class": "gather",
        "index_complexity": "complex", "numel_typical": 2_000_000,
    })
    assert d.level != MigrationLevel.L3


def test_l3_not_when_low_parallel():
    d = decide_migration_level({
        "op_name": "scatter", "op_class": "scatter",
        "index_complexity": "simple", "numel_typical": 1000,
    })
    assert d.level != MigrationLevel.L3


def test_l4_tiling_isregbase_signal_fa_forward():
    """Per OL-185 (2026-05-24): IsRegbaseSocVersion tiling trigger now
    combines with FA-forward class check. FA-forward op with IsRegbase → L4.
    """
    d = decide_migration_level({
        "op_name": "flash_attention_score",
        "op_class": "flash_attention_score",
        "tiling_signals": ["uses IsRegbaseSocVersion to pick mode"],
    })
    assert d.level == MigrationLevel.L4
    assert d.needs_escalation is True


def test_l4_tiling_isregbase_non_fa_downgrades_to_l2():
    """OL-185 escape-hatch (LIG_grad fix 2026-05-24): non-FA-forward op with
    IsRegbaseSocVersion tiling is L2 (RegBase rewrite, flat_quant calibration),
    NOT L4. Critical regression guard — worker reflexively L4-classifies on
    surface tiling signal without OL-185 narrower criterion.
    """
    d = decide_migration_level({
        "op_name": "lightning_indexer_grad",
        "op_class": "lightning_indexer_grad",
        "tiling_signals": ["uses IsRegbaseSocVersion to pick mode"],
    })
    assert d.level == MigrationLevel.L2
    assert d.needs_escalation is False
    assert "non-FA" in d.rationale or "OL-185" in d.rationale or "flat_quant" in d.rationale
    assert d.guides == (
        "l1-implementation-guide.md",
        "l2-register-based-guide.md",
        "l1-l2-implementation-guide.md",
    )


def test_l4_tiling_isregbase_grad_op_downgrades_to_l2():
    """Backward gradient ops are GEMM/scatter/reduce based — NEVER L4 even
    with IsRegbase tiling, per OL-185.
    """
    for name in ("foo_grad", "attention_backward", "softmax_grad"):
        d = decide_migration_level({
            "op_name": name, "op_class": name,
            "tiling_signals": ["IsRegbaseSocVersion check"],
        })
        assert d.level == MigrationLevel.L2, f"{name} should downgrade to L2 (got {d.level})"


def test_l4_ub_shortage():
    d = decide_migration_level({"op_name": "foo", "ub_budget_kb": 30})
    assert d.level == MigrationLevel.L4
    assert d.needs_escalation is True


def test_l4_priority_over_l3():
    """L4 signals win over L3 even when both could fire — but ONLY when op
    is FA-forward class per OL-185. Non-FA gather/scatter with IsRegbase
    tiling falls through to L2 (OL-185 escape-hatch).
    """
    # FA-forward case → L4
    d = decide_migration_level({
        "op_name": "flash_attention_score", "op_class": "flash_attention_score",
        "index_complexity": "simple", "numel_typical": 10_000_000,
        "tiling_signals": ["IsRegbaseSocVersion check"],
    })
    assert d.level == MigrationLevel.L4
    # Non-FA gather → L2 (OL-185 escape-hatch); L4 ladder no longer wins
    d = decide_migration_level({
        "op_name": "gather", "op_class": "gather",
        "index_complexity": "simple", "numel_typical": 10_000_000,
        "tiling_signals": ["IsRegbaseSocVersion check"],
    })
    assert d.level == MigrationLevel.L2


def test_heuristics_registry_open():
    """Registry is a tuple of LevelHeuristic instances; tests can introspect."""
    names = [h.name for h in HEURISTICS]
    assert "L4_TilingIsRegbase" in names
    assert "L3_ScatterGatherSIMT" in names
    assert "L1_Default" in names

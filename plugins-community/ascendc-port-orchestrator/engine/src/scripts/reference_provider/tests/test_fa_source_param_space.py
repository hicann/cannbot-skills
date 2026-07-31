# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""case_gen FA source-param-space enumeration (DEBT-150, owner 2026-06-08 16:11).

Guards that case_gen enumerates from the A3 SOURCE's declared param-space (NOT hand-picked
bands / test-data) so every declared config is exercised → any kw capability-gap (the
graybox 49/64 missed fp32/dropout/D>128/TND/sparse/pse) is caught. The equivalence-gate
is the machine-checkable A5≡A3 check.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # reference_provider/

import fa_source_param_space as fps  # noqa: E402


def test_enumeration_covers_every_declared_value():
    """The load-bearing property: every declared dim-value appears in ≥1 enumerated case
    → equivalence-gate reports full coverage (no source-declared config silently dropped).
    """
    cases = fps.enumerate_fa_source_cases("sign_off")
    assert cases, "expected source-derived cases"
    gate = fps.fa_equivalence_gate([c["config"] for c in cases])
    assert gate["_equivalent"], f"case_gen misses declared configs: {gate['_gaps']}"


def test_covers_the_graybox_missed_dims():
    """The exact dims the graybox 49/64 dropped MUST be covered (they're declared in source)."""
    cases = fps.enumerate_fa_source_cases("sign_off")
    configs = [c["config"] for c in cases]
    dtypes = {c["dtype"] for c in configs}
    assert "float32" in dtypes, "fp32 (arch22-implemented, kw v1 NOT-implemented) must be covered"
    # fp8×3 are ABI-declared-but-NOT-arch22-implemented (GAP-1, independent review dig) → NOT enumerable
    # from arch22 (faithful port can't produce them); recorded as a known GAP, not covered.
    assert not ({"hifloat8", "float8_e5m2", "float8_e4m3"} & dtypes), \
        "fp8 are ABI-only (not arch22-implemented) → must NOT be enumerated as source-faithful"
    assert set(fps.ABI_DECLARED_UNIMPLEMENTED["dtype"]) == {"hifloat8", "float8_e5m2", "float8_e4m3"}
    assert any(c["has_dropout"] == 1 for c in configs), "dropout=1 (kw ignored) must be covered"
    assert any(c["head_dim"] in (512, 640, 768) for c in configs), "D>256 must be covered"
    assert any(c["layout"] == "TND" for c in configs), "TND layout (kw missed) must be covered"
    assert {c["sparse"] for c in configs} == set(range(10)), "all sparse 0-9 declared"
    # high-D × dropout interaction (spec 640/kp0.9, 768/kp0.8) present
    assert any(c["head_dim"] >= 640 and c["has_dropout"] == 1 for c in configs)


def test_source_faithful_clamps_arch22_d():
    """source_faithful=True → D clamped to arch22 DTemplateType enum {80,96,128} (≤128);
    default (arch35-target) includes 768.
    """
    faithful = {c["config"]["head_dim"] for c in fps.enumerate_fa_source_cases(source_faithful=True)}
    assert max(faithful) <= 128, f"source_faithful should clamp D≤128, got {sorted(faithful)}"
    target = {c["config"]["head_dim"] for c in fps.enumerate_fa_source_cases(source_faithful=False)}
    assert 768 in target, "default (arch35-target) must include D768"


def test_sel_validity_rejects_known_invalid():
    """SEL projection rejects the real arch22-SEL-excluded combo (rope + pse: SEL HasRope rows
    have HasPse=0). Per independent review source-authority (template_tiling_key.h L178-186), fp8 is NOT
    in arch22 SEL (GAP-1) so arch22 SEL must NOT constrain fp8 layout — fp8 layout is defined by
    the wholeport KB templates, not arch22 SEL.
    """
    base = {k: v[0] for k, v in fps.FA_SOURCE_PARAM_SPACE.items()}
    assert not getattr(fps, '_is_sel_valid')({**base, "has_rope": 1, "has_pse": 1})
    assert getattr(fps, '_is_sel_valid')({**base, "dtype": "float16", "layout": "BNSD"})
    # fp8 + BSND must NOT be rejected by the arch22 SEL gate (KB-template-defined, not arch22)
    assert getattr(fps, '_is_sel_valid')({**base, "dtype": "float8_e5m2", "layout": "BSND"})


def test_equivalence_gate_detects_a_gap():
    """If a covered-set MISSES a declared value (simulating a kw capability-gap), the gate
    flags it (this is the A5≡A3 catch the graybox lacked).
    """
    # a 'kw-coverage' that (like the graybox) only did fp16/bf16 dense D≤128 no-dropout
    kw_like = [{"dtype": "float16", "layout": "BNSD", "has_dropout": 0, "head_dim": 128,
                "sparse": 0, "has_pse": 0, "has_rope": 0, "has_atten_mask": 0, "impl_mode": 0,
                "seqlen": 256}]
    gate = fps.fa_equivalence_gate(kw_like)
    assert not gate["_equivalent"], "gate must flag the graybox-like gap"
    gap_dims = {d for d, _ in gate["_gaps"]}
    assert "dtype" in gap_dims and "has_dropout" in gap_dims and "head_dim" in gap_dims


# ---------------------------------------------------------------------------
# param_space_delta — the NL→kw structured contract (@main decision, owner 17:30):
# user-NL parsed ONCE into a structured delta, three-way-shared (kw/case_gen/gate).
# ---------------------------------------------------------------------------
def test_param_space_delta_validation():
    assert fps.validate_param_space_delta(None) == []
    assert fps.validate_param_space_delta({"head_dim": [256, 512]}) == []
    assert fps.validate_param_space_delta({"dtype": ["float8_e4m3"]}) == []
    # unknown dim rejected
    assert fps.validate_param_space_delta({"not_a_dim": [1]})
    # wrong value type rejected (numeric dim given str; enum dim given int)
    assert fps.validate_param_space_delta({"head_dim": ["512"]})
    assert fps.validate_param_space_delta({"layout": [3]})
    # empty / non-list rejected
    assert fps.validate_param_space_delta({"head_dim": []})
    assert fps.validate_param_space_delta({"head_dim": 512})


def test_resolve_param_space_raises_on_malformed_delta():
    """fail-loud: a malformed NL-parse is caught BEFORE the three-way consumers."""
    import pytest
    with pytest.raises(ValueError, match="param_space_delta"):
        fps.resolve_param_space(user_extensions={"bogus_dim": [1]})


def test_user_extension_flows_three_way():
    """A user-directed extension (e.g. 'D up to 512' beyond arch22's ≤128, with
    source_faithful base) appears in BOTH the enumerated cases AND the gate's declared
    space — i.e. case_gen + gate consume the SAME extended param-space (three-way-shared).
    """
    ext = {"head_dim": [256, 512]}
    cases = fps.enumerate_fa_source_cases(source_faithful=True, user_extensions=ext)
    dims = {c["config"]["head_dim"] for c in cases}
    assert 256 in dims and 512 in dims, "user-extension head_dims must appear in enumerated cases"
    assert max(dims) <= 512, "source_faithful base (≤128) + ext {256,512} → no 768"
    # the gate's declared space (resolve_param_space) also includes the extension
    sp = fps.resolve_param_space(source_faithful=True, user_extensions=ext)
    assert 256 in sp["head_dim"] and 512 in sp["head_dim"]
    assert 768 not in sp["head_dim"], "source_faithful + ext should not pull in 768"


# ---------------------------------------------------------------------------
# build_param_space_resolved — owner 2026-06-08 17:44: output extracted combos
# (source + NL) with provenance + FAST-STOP on infeasible (no best-efforts).
# ---------------------------------------------------------------------------
def test_value_provenance_tags():
    assert fps.value_provenance("dtype", "float16") == "arch22_source"
    assert fps.value_provenance("dtype", "float32") == "arch22_source"
    assert fps.value_provenance("dtype", "float8_e5m2") == "abi_declared_needs_kb_template"
    assert fps.value_provenance("dtype", "mxfp8") == "kb_template_extension"
    assert fps.value_provenance("head_dim", 128) == "arch22_source"      # arch22 DTemplateType
    assert fps.value_provenance("head_dim", 768) == "arch35_target_needs_kb_template"  # beyond arch22, has template
    # owner 2026-06-08 18:24: D=1024 = valid-but-no-template → kw mode-2 ATTEMPTS (not fast-stop)
    assert fps.value_provenance("head_dim", 1024) == "kw_debug_extension"
    assert fps.value_provenance("head_dim", 1280) == "kw_debug_extension"
    # genuinely illogical/impossible → unknown (fast-stop): nonexistent dtype, absurd/≤0 dims
    assert fps.value_provenance("dtype", "bogus_dtype") == "unknown"
    assert fps.value_provenance("head_dim", 99999) == "unknown"   # > sanity cap → absurd
    assert fps.value_provenance("head_dim", -128) == "unknown"    # ≤0 → nonsensical


def test_build_param_space_resolved_owner_example():
    """The owner's NL example (msg 17:41) structured: head_dim 128/256/512/768 + fp8/mxfp8 →
    resolved report is feasible (every value has a known origin), provenance tagged, combos
    enumerated. This is the 'output extracted combos (source+NL)' artifact.
    """
    ext = {"head_dim": [128, 256, 512, 768], "dtype": ["float8_e5m2", "float8_e4m3", "mxfp8"]}
    rep = fps.build_param_space_resolved(source_faithful=True, user_extensions=ext)
    assert rep["_feasible"], f"owner example must be feasible, infeasible={rep['_infeasible']}"
    assert rep["provenance"]["dtype"]["float8_e5m2"] == "abi_declared_needs_kb_template"
    assert rep["provenance"]["dtype"]["mxfp8"] == "kb_template_extension"
    assert rep["provenance"]["head_dim"]["512"] == "arch35_target_needs_kb_template"
    assert rep["provenance"]["head_dim"]["128"] == "arch22_source"
    assert rep["enumerated_case_count"] > 0
    fps.assert_feasible(rep)  # must NOT raise


def test_infeasible_nl_fast_stops():
    """owner 17:44 'no best-efforts': an NL extension with a value of no known origin →
    report flags infeasible AND assert_feasible hard-stops (does not silently drop/guess).
    """
    import pytest
    rep = fps.build_param_space_resolved(user_extensions={"dtype": ["fp9_bogus"]})
    assert not rep["_feasible"]
    assert any(b["value"] == "fp9_bogus" for b in rep["_infeasible"])
    with pytest.raises(fps.ParamSpaceInfeasible):
        fps.assert_feasible(rep)
    # malformed (unknown dim) still raises at structural-validation layer, before provenance
    with pytest.raises(ValueError, match="param_space_delta"):
        fps.build_param_space_resolved(user_extensions={"not_a_dim": [1]})


def test_beyond_template_dim_routes_to_kw_debug_not_fast_stop():
    """owner 2026-06-08 18:24: a valid-but-no-template request (D=1024) must NOT fast-stop —
    it's routed to kw's general generate-and-debug mode (mode-2, re-tile nearest template).
    Only genuinely illogical values halt. kw assembly is a fast-path specialization, NOT a
    replacement of kw's debug capability.
    """
    import pytest
    rep = fps.build_param_space_resolved(user_extensions={"head_dim": [1024]})
    assert rep["_feasible"], "D=1024 is attemptable (kw mode-2), must NOT be infeasible/halt"
    assert not rep["_infeasible"], "D=1024 must not be in the fast-stop list"
    assert any(x["value"] == 1024 for x in rep["_needs_kw_debug"]), \
        "D=1024 must be flagged for kw mode-2 debug-extension (route, not stop)"
    fps.assert_feasible(rep)  # must NOT raise — beyond-template ≠ infeasible
    # but a genuinely illogical value in the SAME request still fast-stops
    rep2 = fps.build_param_space_resolved(user_extensions={"head_dim": [1024, -7]})
    assert not rep2["_feasible"] and any(x["value"] == -7 for x in rep2["_infeasible"])
    with pytest.raises(fps.ParamSpaceInfeasible):
        fps.assert_feasible(rep2)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

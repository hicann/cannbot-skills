# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FA source-param-space enumeration — case_gen derives coverage from the A3 SOURCE's
declared config-space, NOT from hand-picked bands or test-data (owner 2026-06-08 16:11).

ROOT (graybox 49/64, DEBT-150): kw assembled from its 40-case dense test-data, not from
the A3 source's full declared param-space → A5 NOT functionally-equivalent to A3 (missing
fp32, dropout, D>128, sparse, pse). The source DECLARES its full space in arch22
`flash_attention_score_template_tiling_key.h` (ASCENDC_TPL_*_DECL) + op proto. case_gen
(this module) and kw must BOTH derive from that declaration. The machine-checkable
equivalence-gate: case_gen-coverage == source-declared == kw-assembled → set-diff = gap.

Param-space values below are extracted from arch22 (the port SOURCE) read 2026-06-08:
- `template_tiling_key.h` ASCENDC_TPL_*_DECL (dims + valid values)
- `common.h` STemplateType/DTemplateType enums (enum→size)
- `flash_attention_score_def.cpp` proto DataType list (dtype surface)

DataType note: the tiling-key DataType field (0-3) is an INTERNAL index; the op proto
declares the user-facing dtype surface {fp16, bf16, fp32, hifloat8, fp8_e5m2, fp8_e4m3}.
We enumerate the user-facing dtypes (what a test must exercise), not the internal index.

D-range note (arch22-source vs arch35-target): arch22 `DTemplateType` enum tops at 128
(80/96/128); the canonical V2-64 + arch35 target go to 768. `D_BUCKETS` below covers the
arch35-TARGET range (16…768) because the deliverable bar = the whitebox-achieved 64/64
(D768). A `source_faithful=True` flag clamps to the arch22-declared D≤128 if the bar is
arch22-source-equivalence instead (owner bar-call — both supported, default = target/768).
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# The A3-SOURCE-declared FA param-space (user-facing config dims).
# Internal tiling-impl dims (UB0/UB1/Block/Bmm1Format/Bmm2Source/BigDoubleBuffer/
# EnableL1Reuse/MatmulPolicyType/Regbase) are kw's tiling choices, NOT test-input
# axes — excluded here (case_gen tests user-facing CONFIG, not internal tiling).
# ---------------------------------------------------------------------------
# ABI-declared-but-NOT-arch22-implemented (GAP-1): the op-def
# proto declares fp8×3 (hifloat8/float8_e5m2/float8_e4m3) in its DataType ABI surface, but
# the arch22 KERNEL does not implement them (`grep fp8 arch22` = 0). So a faithful port FROM
# arch22 CANNOT produce them — they are an upstream/arch35 gap, NOT a case_gen/kw extraction
# miss. Excluded from the enumerable source space; recorded here so the gate reports them as a
# known GAP (owner: upstream) rather than silently omitting or falsely claiming them.
ABI_DECLARED_UNIMPLEMENTED: dict[str, list[Any]] = {
    "dtype": ["hifloat8", "float8_e5m2", "float8_e4m3"],
}

FA_SOURCE_PARAM_SPACE: dict[str, list[Any]] = {
    # arch22 KERNEL-implemented dtypes (3) — fp8×3 are ABI-declared-not-implemented (GAP-1
    # above), so a faithful arch22 port enumerates only these 3.
    "dtype": ["float16", "bfloat16", "float32"],
    # tiling_key Layout DECL 0-4.
    "layout": ["BSND", "SBH", "BSH", "BNSD", "TND"],
    # tiling_key feature bits.
    "has_dropout": [0, 1],          # HasDropOut DECL 0,1 (keep_prob<1 path)
    "has_atten_mask": [0, 1],       # HasAttenMask DECL 0,1
    "has_pse": [0, 1],              # HasPse DECL 0,1
    "has_rope": [0, 1],             # HasRope DECL 0,1
    "sparse": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # Sparse DECL 0-9
    "impl_mode": [0, 1, 2],         # ImplMode DECL 0,1,2 (precision modes)
    # head-dim buckets — arch35-target range (canonical bar = D768); see D-range note.
    # arch22-source-faithful = clamp to {80,96,128} (DTemplateType enum) via source_faithful.
    "head_dim": [16, 32, 64, 80, 96, 128, 256, 512, 640, 768],
    # seqlen tiles — STemplateType arch22 {16..128} are the tile sizes; actual S can be
    # any multiple; representative seqlens spanning the tile bands + tails.
    "seqlen": [64, 128, 192, 256, 384, 512, 1024, 2048],
}

# arch22-source-faithful D clamp (DTemplateType enum 80/96/128).
_ARCH22_D = [80, 96, 128]

# A head_dim beyond the known template set but ≤ this is treated as a VALID request that kw's
# general generate-and-debug mode (mode-2) can ATTEMPT (re-tile from the nearest template),
# NOT an illogical/impossible value that must fast-stop (owner 2026-06-08 18:24). Above this =
# physically absurd for current HW (would OOM) → illogical. The cap only separates
# "hard-but-attemptable" (D=1024/1280) from "nonsensical"; it is NOT a support guarantee.
_HEAD_DIM_SANITY_MAX = 2048


def _is_sel_valid(cfg: dict[str, Any]) -> bool:
    """Coarse SEL validity (the ASCENDC_TPL_SEL constraints, projected to user-facing dims).
    The full SEL is a list of template-families; this captures the user-facing rules that
    matter for test enumeration. Conservative:
    reject only the combos the SEL clearly excludes; allow the rest (a too-permissive gate
    over-tests, which is safe; a too-strict one would silently drop = the bug we're fixing)."""
    # NOTE (source-authority correction, template_tiling_key.h L178-186):
    # do NOT hang an arch22-SEL layout-restriction on fp8. arch22 SEL has only 3 dtype `#if`
    # guards (FLOAT16/BF16/FLOAT) — fp8 is NOT in arch22 SEL at all (GAP-1, ABI-only). The
    # `DataType=3 → Layout∈{SBH,BSH,BNSD}` rule lives INSIDE the fp16 `#if` block (it is an
    # fp16-internal rule, not fp8). fp8/mxfp8 enter only as a user-extension and their layout
    # support is defined by the wholeport KB templates (vf_basic_block_fullquant_*/_mx.h),
    # NOT by arch22 SEL — so the arch22 SEL gate must not constrain them.
    # rope is its own template-family; pair only with no-pse (SEL HasRope rows have HasPse=0).
    if cfg["has_rope"] == 1 and cfg["has_pse"] == 1:
        return False
    return True


def validate_param_space_delta(delta: "dict[str, list] | None") -> list[str]:
    """Validate a structured param_space_delta. Returns a list of error strings ([] = valid).

    The orchestrator parses a natural-language range exactly once and expands it
    against the declared dimension domain. For example, a request to add head
    dimensions greater than 128 and no greater than 512 becomes a mapping from
    head_dim to the explicit values 256 and 512. Values are lists of concrete
    scalars, never constraints that downstream consumers must expand again.

    This single mapping is shared by kernel assembly, case enumeration, and the
    equivalence gate. Sharing it prevents those consumers from independently
    interpreting the same request as different target spaces.

    Enforces: keys are known FA dims; values are lists of explicit (non-range) scalars; for
    enum dims (dtype/layout) values are strings, for numeric dims (head_dim/seqlen/sparse/…)
    values are ints. Fail-loud here so a malformed NL-parse is caught BEFORE it reaches the
    three-way consumers (a silently-wrong delta would corrupt kw + case_gen + gate together)."""
    if delta is None:
        return []
    errs: list[str] = []
    if not isinstance(delta, dict):
        return [f"param_space_delta must be a dict, got {type(delta).__name__}"]
    _str_dims = {"dtype", "layout"}
    for dim, values in delta.items():
        if dim not in FA_SOURCE_PARAM_SPACE:
            errs.append(f"unknown dim {dim!r} (not in FA source param-space {list(FA_SOURCE_PARAM_SPACE)})")
            continue
        if not isinstance(values, (list, tuple)) or not values:
            errs.append(f"dim {dim!r} values must be a non-empty list, got {values!r}")
            continue
        for v in values:
            if dim in _str_dims and not isinstance(v, str):
                errs.append(f"dim {dim!r} expects str values (enum), got {v!r}")
            elif dim not in _str_dims and isinstance(v, bool):
                errs.append(f"dim {dim!r} got bool {v!r}; expected int")
            elif dim not in _str_dims and not isinstance(v, int):
                errs.append(f"dim {dim!r} expects int values, got {v!r}")
    return errs


def resolve_param_space(source_faithful: bool = False,
                        user_extensions: "dict[str, list] | None" = None) -> dict[str, list]:
    """The effective param-space = source-declared (base) ∪ user-directed extensions.

    owner 2026-06-08 17:22: the A5 generation is guided by BOTH the input AscendC source
    (arch22-declared, the base) AND the user's natural-language request to /ascendc-op-gen
    (e.g. "add D>128<=512 support"). The orchestrator/brief parses the NL into a STRUCTURED
    `user_extensions` dict {dim: [extra values]} (NL→structured is upstream LLM work, not
    this module); here we merge it onto the source base. User says nothing → pure source
    declaration; user directs an extension → it's added (so A5 can exceed the arch22 source
    where the user/target asks, e.g. arch22 D≤128 + user "D≤512" → enumerate to 512)."""
    _errs = validate_param_space_delta(user_extensions)
    if _errs:
        raise ValueError("invalid param_space_delta (NL→structured parse must be well-formed "
                         "before three-way consumption): " + "; ".join(_errs))
    sp = {k: list(v) for k, v in FA_SOURCE_PARAM_SPACE.items()}
    if source_faithful:
        sp["head_dim"] = list(_ARCH22_D)
    if user_extensions:
        for dim, extra in user_extensions.items():
            base = sp.get(dim, [])
            sp[dim] = base + [v for v in extra if v not in base]
    return sp


def enumerate_fa_source_cases(coverage_tier: str = "sign_off",
                              source_faithful: bool = False,
                              user_extensions: "dict[str, list] | None" = None) -> list[dict]:
    """Enumerate test configs COVERING each declared dim-value + key interactions —
    derived from FA_SOURCE_PARAM_SPACE (the source declaration), NOT hand-picked bands.

    Coverage strategy (NOT full cartesian — SEL-valid product is huge): a base config +
    one-dim sweeps (each declared value of each dim covered at least once) + the key
    cross-interactions the source's hard paths pivot on (high-D × dropout, fp32 × layout,
    sparse-modes, mask+pse). Every declared dim-value appears in ≥1 case → any kw
    capability-gap (missing fp32/dropout/D768/sparse/pse/TND) is caught by ≥1 failing case.

    source_faithful=True clamps head_dim to arch22-declared {80,96,128} (A5≡A3-source bar);
    default False = arch35-target range incl 768 (canonical-64 bar)."""
    # the effective space = source-declared (∪ source_faithful clamp) ∪ user-extensions
    # (single resolve so the enumeration covers exactly what the gate declares — three-way).
    sp = resolve_param_space(source_faithful=source_faithful, user_extensions=user_extensions)

    base = {"dtype": "float16", "layout": "BNSD", "has_dropout": 0, "has_atten_mask": 0,
            "has_pse": 0, "has_rope": 0, "sparse": 0, "impl_mode": 0,
            "head_dim": 128, "seqlen": 256}

    cases: list[dict] = []
    seen: set = set()

    def _add(cfg: dict, name: str):
        if not _is_sel_valid(cfg):
            return
        key = tuple(sorted(cfg.items()))
        if key in seen:
            return
        seen.add(key)
        cases.append({"name": name, "config": dict(cfg)})

    _add(dict(base), "fa_base")
    # one-dim sweeps: cover every declared value of every dim.
    for dim, values in sp.items():
        for v in values:
            cfg = dict(base)
            cfg[dim] = v
            _add(cfg, f"fa_{dim}_{v}")
    # key cross-interactions (the source's hard paths the dense subset missed).
    interactions = [
        {**base, "head_dim": 768, "has_dropout": 1, "layout": "SBH"},   # high-D × dropout (spec 768/kp0.8)
        {**base, "head_dim": 640, "has_dropout": 1, "layout": "BNSD"},  # high-D × dropout (spec 640/kp0.9)
        {**base, "dtype": "float32", "head_dim": 128},                  # fp32 (kw NOT-implemented)
        {**base, "has_dropout": 1, "has_atten_mask": 1},                # dropout × mask
        {**base, "has_pse": 1, "sparse": 2},                            # pse × sparse
        {**base, "layout": "TND", "head_dim": 128},                     # TND layout (kw missed)
        {**base, "dtype": "bfloat16", "head_dim": 768, "has_dropout": 1},
    ]
    for i, cfg in enumerate(interactions):
        # respect the (possibly source_faithful-clamped) declared head_dim range
        if cfg["head_dim"] not in sp["head_dim"]:
            continue
        _add(cfg, f"fa_interaction_{i}")
    return cases


def fa_equivalence_gate(covered_configs: list[dict]) -> dict:
    """Machine-checkable A5≡A3 gate: does the covered set exercise EVERY declared dim-value?
    Returns per-dim {declared, covered, missing}. Any non-empty `missing` = a source-declared
    config the test-set (and therefore the verified kw capability) does NOT exercise = gap.

    covered_configs: list of {dim: value} (e.g. from kw-assembled-coverage OR case_gen-enum).
    """
    report: dict[str, dict] = {}
    all_missing = []
    for dim, declared in FA_SOURCE_PARAM_SPACE.items():
        covered = {c.get(dim) for c in covered_configs if dim in c}
        missing = [v for v in declared if v not in covered]
        report[dim] = {"declared": declared, "covered": sorted(covered, key=str),
                       "missing": missing}
        if missing:
            all_missing.append((dim, missing))
    report["_equivalent"] = (len(all_missing) == 0)
    report["_gaps"] = all_missing
    return report


# ---------------------------------------------------------------------------
# owner 2026-06-08 17:44: the pipeline MUST (a) explicitly output the extracted
# param-combos (source + NL-understanding) to logs/files, and (b) FAST-STOP on an
# infeasible/illogical NL request — no "best efforts". `build_param_space_resolved`
# is the case_gen-side mechanism: it resolves the space, tags every value with its
# PROVENANCE (where it can come from), and flags any value with no known origin as
# infeasible so the caller hard-stops after writing the report.
#
# Provenance taxonomy (case_gen-side; kw confirms actual template availability at
# assembly — this module reports "where it COULD come from", not "it built"):
#   arch22_source                  — declared + implemented in the arch22 kernel (faithful port).
#   abi_declared_needs_kb_template — fp8×3: op-def ABI declares, arch22 has no kernel
#                                    (GAP-1); satisfiable via KB fullquant templates IF present.
#   arch35_target_needs_kb_template— D>128 with a template: beyond arch22 (DTemplateType≤128); via wholeport/arch35.
#   kb_template_extension          — e.g. mxfp8: not in arch22 ABI at all; via *_mx.h KB templates.
#   kw_debug_extension             — valid request with NO template (e.g. D=1024): kw's general
#                                    generate-and-debug mode (mode-2) ATTEMPTS it by re-tiling the
#                                    nearest template. NOT a fast-stop — routed to kw mode-2; GAP-
#                                    with-evidence only if a real attempt genuinely fails (owner 18:24:
#                                    "在模板拼接基础上继续 debug 失败用例"). Distinct from 'unknown'.
#   unknown                        — illogical/impossible (no such dtype, nonsensical/absurd value)
#                                    → INFEASIBLE → fast-stop (owner 17:44, no best-efforts).
# ---------------------------------------------------------------------------
_VALUE_PROVENANCE: dict[str, dict] = {
    "dtype": {
        "hifloat8": "abi_declared_needs_kb_template",
        "float8_e5m2": "abi_declared_needs_kb_template",
        "float8_e4m3": "abi_declared_needs_kb_template",
        "mxfp8": "kb_template_extension",
    },
    "head_dim": {
        256: "arch35_target_needs_kb_template",
        512: "arch35_target_needs_kb_template",
        640: "arch35_target_needs_kb_template",
        768: "arch35_target_needs_kb_template",
    },
}


class ParamSpaceInfeasible(ValueError):
    """Raised when a resolved param-space contains a value with no known origin
    (owner 2026-06-08 17:44: an impossible/illogical NL combo must fast-stop, not best-effort)."""


def value_provenance(dim: str, value) -> str:
    """Where a single (dim, value) can come from. Explicit extension-map first, then the
    arch22-declared source list, else 'unknown' (infeasible)."""
    ext = _VALUE_PROVENANCE.get(dim, {})
    if value in ext:
        return ext[value]
    if dim in FA_SOURCE_PARAM_SPACE and value in FA_SOURCE_PARAM_SPACE[dim]:
        return "arch22_source"
    # beyond the source-declared + known-template set, but a VALID request kw's general
    # generate-and-debug mode (mode-2) can ATTEMPT by re-tiling the nearest template (e.g.
    # D=1024 from the D=768 template). owner 2026-06-08 18:24: do NOT fast-stop these — only
    # genuinely illogical/impossible values halt. Whether kw mode-2 succeeds is empirical
    # (the graybox decides); GAP-with-evidence if a real attempt fails. NOT a support promise.
    if dim == "head_dim" and isinstance(value, int) and not isinstance(value, bool) \
            and 0 < value <= _HEAD_DIM_SANITY_MAX:
        return "kw_debug_extension"
    return "unknown"


def build_param_space_resolved(source_faithful: bool = False,
                               user_extensions: "dict[str, list] | None" = None) -> dict:
    """Produce the machine-readable resolved-param-space report (→ param_space_resolved.json).

    Serves owner 2026-06-08 17:44: explicitly enumerate the extracted combos (source-declared
    base + NL-directed extensions), tag each value's provenance, and mark infeasible (unknown-
    origin) values. The caller writes this to the workspace/log THEN calls `assert_feasible`
    to fast-stop — so even on a fast-stop the extracted-combos + the reason are persisted.

    Validates the structured delta first (raises ValueError on malformed — fail-loud before
    any consumer). Does NOT itself raise on infeasible values (so the report is still written);
    `_feasible`/`_infeasible` carry the verdict for `assert_feasible`."""
    # structural validation (reuse): malformed NL→structured parse raises here.
    resolved = resolve_param_space(source_faithful=source_faithful, user_extensions=user_extensions)
    provenance: dict[str, dict] = {}
    infeasible: list = []      # unknown-origin → fast-stop (owner 17:44)
    needs_kw_debug: list = []  # valid-but-no-template → kw mode-2 attempts (owner 18:24), NOT a stop
    for dim, values in resolved.items():
        provenance[dim] = {}
        for v in values:
            origin = value_provenance(dim, v)
            provenance[dim][str(v)] = origin
            if origin == "unknown":
                infeasible.append({"dim": dim, "value": v})
            elif origin == "kw_debug_extension":
                needs_kw_debug.append({"dim": dim, "value": v})
    cases = enumerate_fa_source_cases(source_faithful=source_faithful, user_extensions=user_extensions)
    gate = fa_equivalence_gate([c["config"] for c in cases])
    return {
        "source_faithful": source_faithful,
        "source_base": {k: list(v) for k, v in FA_SOURCE_PARAM_SPACE.items()},
        "abi_declared_unimplemented": ABI_DECLARED_UNIMPLEMENTED,
        "user_extensions": user_extensions or {},
        "resolved": resolved,
        "provenance": provenance,
        "enumerated_case_count": len(cases),
        "equivalence_gate": {"equivalent": gate["_equivalent"], "gaps": gate["_gaps"]},
        "_infeasible": infeasible,
        "_needs_kw_debug": needs_kw_debug,
        "_feasible": (len(infeasible) == 0),  # kw_debug_extension does NOT make it infeasible
    }


def assert_feasible(report: dict) -> None:
    """Fast-stop (owner 17:44 'no best-efforts'): raise ParamSpaceInfeasible if the resolved
    report contains any unknown-origin value. Call AFTER persisting the report so the extracted
    combos + the infeasible reason are on disk before the pipeline halts."""
    if not report.get("_feasible", False):
        bad = report.get("_infeasible", [])
        raise ParamSpaceInfeasible(
            "param-space contains value(s) with no known origin (arch22 source / ABI+KB "
            "template / arch35 template) — NL request is infeasible, fast-stopping (no "
            f"best-efforts): {bad}")


if __name__ == "__main__":
    cases = enumerate_fa_source_cases("sign_off")
    print(f"enumerated {len(cases)} source-derived FA cases")
    gate = fa_equivalence_gate([c["config"] for c in cases])
    print(f"equivalence (case_gen covers all declared): {gate['_equivalent']}")
    if gate["_gaps"]:
        print(f"gaps: {gate['_gaps']}")
    # show dtype/dropout/D coverage (the graybox's missing dims)
    for d in ("dtype", "has_dropout", "head_dim", "layout", "sparse"):
        print(f"  {d}: covered {gate[d]['covered']}")

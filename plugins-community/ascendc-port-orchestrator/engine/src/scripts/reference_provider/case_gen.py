# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Op-agnostic test case generator for Phase O2.5 Reference Provider.

Given an op schema + coverage tier, emit a list of cases that stress **four
orthogonal axes** (NOT a full cross-product — see §Axes below).

CONTRACT:
    cases = generate_cases(schema, coverage_tier="sign_off", dtype=torch.float32)
    The returned cases have ``idx``, ``name``, ``shape``, and ``inputs`` fields.

AXES (three independent bands, not cross-produced):
    Band A — SHAPE stressors: N distinct shapes × single uniform distribution × 1 seed
             (targets the shape-tail / alignment / partition / tile / prime / 2D axis)
    Band B — DISTRIBUTION stressors: N distinct distributions × fixed base_shape
             (per rank) × K seeds (1/2/3 per tier)
             (targets precision-axis diversity at a representative shape)
    Band C — SCALAR probe variants: per scalar input's probe_values list × base_shape
             (targets degenerate scalar paths)

Rationale: full cross-product (shapes × distributions × seeds) would produce
~600+ cases at sign_off tier. Two independent 1-D sweeps cover the "most likely
to catch a bug" space at ~40 cases. For ops where cross-product coverage matters
(e.g. shape-dependent denormal triggers), add explicit cases to the per-op
`input_gen.py` beyond what this engine emits.

SCHEMA FORMAT:
    schema = {
        "op_name":       str,                  # e.g. "babelstream_triad"
        "formula":       str,                  # pseudo-code, e.g. "a[i] = b[i] + scalar*c[i]"
        "tensor_inputs": [                     # one entry per input tensor
            {"name": "b", "role": "operand"},  # role tags are informational
            {"name": "c", "role": "operand"},
        ],
        "scalar_inputs": [                     # optional scalar params
            {"name": "scalar", "dtype": "float", "default": 0.4,
             "probe_values": [0.4, -0.4, 0.0, -1.0]},
        ],
        "tensor_output": "a",                  # name (for verify.py output-key match)
        "rank": 1,                             # 1 for 1D, 2 for [H,W], etc.
        "max_case_tensor_bytes": 100 * 1024 * 1024,  # optional pre-allocation safety cap

        OPTIONAL extensions support fused ops with interdependent shapes:
        ``base_shape_filter`` is a callable accepting ``base_shape`` that skips
        invalid bases. Per-tensor overrides may define ``dtype``,
        ``shape_derive``, ``value_gen``, or ``invariant``. A shape derivation may
        accept either ``base_shape`` or ``(base_shape, scalars)``. Supported
        invariants are ``permutation``, ``positive``, ``non_negative``, and
        ``index_range:<N>``. Tensors without ``shape_derive`` use ``base_shape``.

        The ``optional`` boolean lets an input be ``None``. The generator creates
        materialized and absent variants, so each optional tensor doubles the
        relevant cases. This covers reference paths such as
        ``Model.forward(weight=None)``.

        A tensor with ``kind="list_of_tensors"`` is a variable-length list. Its
        ``list_length_plan`` controls the generated lengths, while an optional
        ``per_item_shape_derive`` accepts ``(base_shape, item_index, list_length)``
        or ``(base_shape, item_index, list_length, scalars)``. The resulting input
        is a list of ``torch.Tensor`` objects for models such as ``torch.cat``.

        Scalar input extensions include ``derive``, a callable accepting either
        ``base_shape`` or ``(base_shape, scalars)``, and ``invariant``. Scalar
        invariants include ``positive``, ``non_negative``, and
        ``le:<other_scalar_name>``.
    }

COVERAGE TIERS (mutually inclusive — sign_off adds to pilot, production adds to sign_off):
    pilot      ≈  15 cases: minimum to prove pipeline; ~1 per precision axis; single seed
    sign_off   ≈  50 cases: ADD shape-edge coverage (tail, degenerate, core-boundary, prime,
                            2D equivalence), 2 seeds per distribution
    production ≈ 150 cases: ADD NaN/Inf propagation, 3 seeds, 3 tile-boundary variants per
                            distribution, exhaustive shape cross-product

NOT YET IN SCOPE (needs separate engine):
    - reduction ops (Dot — per-element pattern same, but per-case shape semantics differ;
      reduce order tests need different template)
    - indexed/scatter ops (shape conjoint with index tensor; needs its own schema)
    - multi-dtype (fp16/bf16 — precision semantics per-dtype; future expansion)

USAGE IN PER-OP SCRIPT:
    from case_gen import generate_cases
    SCHEMA = {...}  # fill per op
    cases = generate_cases(SCHEMA, coverage_tier="sign_off", dtype=torch.float32)
    torch.save({"dtype": "float32", "op": SCHEMA["op_name"],
                "schema": SCHEMA, "cases": cases},
               "edge_inputs.pt")
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

from case_schema import (  # re-export (2026-07-05)
    _dtype_eps, _dtype_smallest_normal, _shape_plan, _attention_shape_bands,
    _matmul_shape_bands, _op_class_shape_bands, _distribution_plan, _tensor_shape_for,
    make_layout_dispatch, _tensor_dtype_for, _base_shape_for_rank,
    _A5, _OP_CLASS_SHAPE_BANDS, _TIER_2D_EQUIV, _TIER_NAN_INF, _TIER_SHAPE_EDGES,
)
from case_synth import (  # re-export (2026-07-05)
    _DEFAULT_MAX_CASE_TENSOR_BYTES, _guard_case_tensor_budget,
    _resolve_case_tensor_plan,
    _mk_tensor, _build_inputs, _build_list_of_tensors_item, _mk_invariant_gen,
    _mk_int_uniform_gen, _mk_bool_uniform_gen, _scalar_only_cases,
    _expand_list_of_tensors_lengths, _expand_optional_tensor_presence,
)

# Coverage knobs
_TIER_SEEDS_PER_DIST = {
    "pilot": 1,
    "sign_off": 2,
    "production": 3,
}


def _normal_tensor_values(numel: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Generate deterministic normal values for floating-point case inputs."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(numel, generator=generator, dtype=dtype)


# Hardware constants used to generate alignment-stress shapes. Current target is A5
# (Ascend950PR): 56 AIV cores, 32B alignment (= 8 fp32 / 16 fp16). UB tile size
# varies per kernel design but 4096 fp32 is a common choice (produces 16 KB × queues).


# ---------------------------------------------------------------------------
# Op-class-aware SEMANTIC shape bands (2026-06-08, owner case_gen-coverage directive).
#
# `_shape_plan` above is systematic on HW edges (align/tile/core/prime) but NOT on
# DOMAIN-SEMANTIC bands: an attention op's correctness + tiling pivot on seqlen-tile
# crossings and head-dim splits; a matmul's on M/N/K bands. For ops that declare an
# `op_class`, the engine prepends op-class-specific semantic shape bands so a
# benchmark-LESS schema gets comprehensive domain coverage BY CONSTRUCTION — the
# domain knowledge lives in the ENGINE and any op of that class inherits it.
#
# Bar (owner 2026-06-08): case_gen self-generated coverage ≥ a curated benchmark
# (e.g. FA V2-64) WITHOUT needing the benchmark — because real a3-port/op-gen agents
# have no benchmark reference, only case_gen output. Reconcile + the V2-64 bands:
# docs/analysis/FA_VERIFY_SET_VS_SPEC64_RECONCILE_2026_06_08.md and
# docs/analysis/CASE_GEN_VS_V2_FA_COVERAGE_2026_06_05.md §5.
#
# Add a new op-class = register a band-emitter in `_OP_CLASS_SHAPE_BANDS`. Each emitter
# returns [{name, shape}] in the op-class's canonical base layout (attention base =
# rank-4 [B, S, N, D]; `make_layout_dispatch` then views it as BSH/SBH/BNSD/BSND).
# OPT-IN: fires only when `schema["op_class"]` matches a registered emitter; ops
# without `op_class` are byte-identical to before (no behavior change, all-op safe).
# ---------------------------------------------------------------------------


# Registry: op_class (lowercased) -> band-emitter(coverage_tier, dtype, rank) -> [{name, shape}].
# Generalizes — register conv, moe-routing, etc. the same way (attention + matmul below).


def generate_cases(schema: dict[str, Any], coverage_tier: str = "sign_off",
                   dtype: torch.dtype = torch.float32) -> list[dict]:
    """Main entry. Returns list of case dicts keyed (idx, name, shape, inputs, meta).

    Three orthogonal bands (see module docstring §AXES):
      Band A — SHAPE stressors × uniform distribution
      Band B — DISTRIBUTION stressors × base_shape (rank-matched) × n_seeds
      Band C — SCALAR probe variants at base_shape

    Multi-axis sweeps (added 2026-05-18, Gap #3 unblock for 5_Cumsum /
    8_Sort / 9_TopK / 25_NLLLoss class of ops that legitimately admit
    multiple ranks and/or multiple dtypes in their reference spec):

      schema["ranks"] = [1, 2, 3, 4]
          → Loop generate_cases per rank, merge cases. Each emitted
            case carries `meta.rank` so downstream filtering is easy.
            Mutually exclusive with `schema["rank"]` (V1 single-rank).

      schema["dtypes"] = [torch.float32, torch.float16, torch.bfloat16]
          → Loop generate_cases per dtype, merge. The single-dtype
            `dtype=` keyword arg remains the V1 path; if both are
            provided, `schema["dtypes"]` takes precedence and the
            kwarg becomes a fallback for the not-listed case.

    Per-scalar callable extensions:

      scalar_inputs[i]["derive"] may take EITHER
          (base_shape,)                 # V1 1-arg form, unchanged
          (base_shape, scalars)         # 2-arg form (added 2026-05-18)
              where `scalars` is the dict of already-resolved scalars
              from the same case. Scalars are resolved in declaration
              order, so a later derive can reference an earlier one.

      scalar_inputs[i]["probe_values"] may be EITHER
          [v1, v2, ...]                 # V1 static list, unchanged
          lambda rank: [...]            # rank-dependent (added 2026-05-18)
          lambda base_shape: [...]      # shape-dependent variant
    """
    if coverage_tier not in _TIER_SHAPE_EDGES:
        raise ValueError(f"unknown tier: {coverage_tier}")
    configured_case_limit = schema.get("max_case_tensor_bytes")
    max_case_tensor_bytes = (
        int(configured_case_limit) if configured_case_limit is not None else None
    )
    if max_case_tensor_bytes is not None and max_case_tensor_bytes <= 0:
        raise ValueError("schema['max_case_tensor_bytes'] must be a positive integer")

    # Multi-rank sweep: recurse per rank, merge.
    ranks_list = schema.get("ranks")
    if ranks_list is not None:
        if not isinstance(ranks_list, (list, tuple)) or not ranks_list:
            raise ValueError("schema['ranks'] must be a non-empty list of ints")
        if "rank" in schema:
            raise ValueError(
                "schema cannot declare BOTH 'rank' (single) AND 'ranks' (list); "
                "use one"
            )
        merged: list[dict] = []
        idx = 0
        for r in ranks_list:
            sub_schema = dict(schema)
            sub_schema.pop("ranks", None)
            sub_schema["rank"] = int(r)
            sub_cases = generate_cases(sub_schema, coverage_tier=coverage_tier,
                                       dtype=dtype)
            for c in sub_cases:
                c = dict(c)
                c["idx"] = idx
                # Stash rank in name + meta so downstream consumers can filter.
                c["name"] = f"r{r}_{c['name']}"
                meta = dict(c.get("meta", {}))
                meta["rank"] = int(r)
                c["meta"] = meta
                merged.append(c)
                idx += 1
        return merged

    # Multi-dtype sweep: recurse per dtype, merge.
    dtypes_list = schema.get("dtypes")
    if dtypes_list is not None:
        if not isinstance(dtypes_list, (list, tuple)) or not dtypes_list:
            raise ValueError("schema['dtypes'] must be a non-empty list of torch.dtype")
        sub_schema = dict(schema)
        sub_schema.pop("dtypes", None)
        merged_dt: list[dict] = []
        idx = 0
        for dt in dtypes_list:
            sub_cases = generate_cases(sub_schema, coverage_tier=coverage_tier,
                                       dtype=dt)
            dt_tag = str(dt).replace("torch.", "")
            for c in sub_cases:
                c = dict(c)
                c["idx"] = idx
                c["name"] = f"{dt_tag}_{c['name']}"
                meta = dict(c.get("meta", {}))
                meta["dtype"] = dt_tag
                c["meta"] = meta
                merged_dt.append(c)
                idx += 1
        return merged_dt

    tensor_inputs = schema.get("tensor_inputs", [])
    scalar_inputs = schema.get("scalar_inputs", [])
    rank = schema.get("rank", 1)

    tensor_names = [t["name"] for t in tensor_inputs]

    # SCALAR-ONLY MODE (added 2026-04-22 for op#22-class ops whose inputs are all scalars).
    # Triggered when tensor_inputs is empty AND scalar_inputs has at least one entry with
    # probe_values. Produces cases by enumerating scalar probe combinations: a baseline case
    # (all defaults) + one case per (scalar × probe_value) variant while others stay default.
    # edge_dataset.pt is thin (just the scalar tuples); Model.forward is the generator.
    if not tensor_names:
        if not any(s.get("probe_values") for s in scalar_inputs):
            raise ValueError(
                "scalar-only schema must declare probe_values for at least one scalar_input "
                "(otherwise no variation to generate). Add probe_values: [list] to at least "
                "one entry in scalar_inputs, or add a tensor_input."
            )
        return _scalar_only_cases(schema, coverage_tier)

    if not tensor_names:
        raise ValueError(
            "schema must declare ≥1 tensor_input (or use scalar-only mode)"
        )

    shape_plans = _shape_plan(coverage_tier, dtype, rank)
    # Op-class SEMANTIC bands (2026-06-08, owner case_gen-coverage directive): prepend
    # domain-semantic shapes (attention seqlen/head-dim incl D640/768, etc) for ops with a
    # registered `schema["op_class"]`. OPT-IN — ops without op_class get [] → byte-identical.
    op_class_bands = _op_class_shape_bands(schema.get("op_class"), coverage_tier, dtype, rank)
    if op_class_bands:
        shape_plans = list(op_class_bands) + list(shape_plans)
    # Apply op-specific base shape filter if present (e.g. op#11 needs last-dim even).
    #
    # IMPORTANT: base_shape_filter receives VARIABLE-RANK shapes — `_shape_plan` emits
    # 1D shapes (degenerate / tile-boundary / prime bands always use rank 1) alongside
    # the requested rank's typical+edge shapes. For ops that require a specific rank,
    # the filter MUST reject shapes with the wrong length before unpacking them.
    # For a rank-four op, reject every shape whose length is not four, then unpack
    # its B/H/new-sequence/D dimensions and apply the remaining invariants.
    # Without this guard the filter crashes with `ValueError: not enough values to unpack`
    # when it tries to destructure a 1D shape. See workspace/kvcacheupdatewithropebackward/
    # input_gen.py for a worked example (op#24 rank-4 fused scatter+reduce).
    # 2026-04-28 op#11: allow SCHEMA-level extra shape plans for ops whose
    # constraint excludes ALL default shape_plan entries (e.g. last-dim must
    # be div by 128). Added to fix the "rank-2 hardcoded last-dim trap" — see
    # session handover §"未 codify 的隐含知识" #3 + KB OL-91.
    extra_shape_plans = schema.get("extra_shape_plans", [])
    if extra_shape_plans:
        shape_plans = list(extra_shape_plans) + list(shape_plans)

    base_shape_filter = schema.get("base_shape_filter")
    if base_shape_filter is not None:
        shape_plans = [s for s in shape_plans if base_shape_filter(s["shape"])]
    dist_plans = _distribution_plan(coverage_tier)
    n_seeds = _TIER_SEEDS_PER_DIST[coverage_tier]
    base_shape = _base_shape_for_rank(rank)
    # If a filter exists, validate our default base_shape too (fall back to first valid shape_plan shape)
    if base_shape_filter is not None and not base_shape_filter(base_shape):
        if shape_plans:
            base_shape = shape_plans[0]["shape"]
        else:
            raise ValueError("base_shape_filter excludes default base_shape AND every shape_plan entry")

    cases = []
    idx = 0
    # Preserve scalar native type (int vs float) per SCHEMA so runners can pass to
    # strict-typed ops (e.g. torch_npu expects int for mode flags, not 0.0).

    def _coerce_scalar(val, dtype_str):
        if dtype_str in ("int", "int32", "int64"):
            return int(val)
        if dtype_str == "bool":
            return bool(val)
        if dtype_str in ("str", "string"):
            # Pass-through for enum/string scalars (e.g. rotary_mode: "half"|"interleave").
            # Added 2026-04-22 for op#1 RotaryMul where mode is a string enum.
            return str(val)
        if dtype_str == "tuple_of_int":
            # P0aaa task #105 (2026-05-06): tuple-of-int scalar (e.g. Pad's
            # `pad: tuple[int]` with rank-dependent length). Value is a
            # list/tuple of ints; we tuple-ify for hashability + Pythonic
            # match to torch.nn.functional.pad's expected type.
            if isinstance(val, (list, tuple)):
                return tuple(int(x) for x in val)
            raise ValueError(f"tuple_of_int scalar value must be list/tuple, got {type(val).__name__}")
        return float(val)
    default_scalars = {s["name"]: _coerce_scalar(s.get("default", 0), s.get("dtype", "float"))
                       for s in scalar_inputs}
    _scalar_dtype = {s["name"]: s.get("dtype", "float") for s in scalar_inputs}

    def _build_scalars_for_shape(shape):
        """Per-case scalar values. Uses `derive(base_shape)` if present, else default.

        P0aaa task #105 (2026-05-06): tuple_of_int scalars with `length_derive`
        — value is a tuple of ints whose LENGTH depends on base_shape rank
        (e.g. Pad: 2D → 4 ints, 3D → 6 ints). Implementation:
          - dtype="tuple_of_int" + length_derive=lambda base: 2*len(base)
          - value drawn uniformly from `value_range` (default (0, 5))
          - if `derive` is also present, it overrides length_derive for this case

        Scalars without `derive` and without `length_derive` fall back to default.

        Gap #3 unblock (2026-05-18): `derive` may now take EITHER 1 arg
        `derive(base_shape)` (V1) OR 2 args `derive(base_shape, scalars)`
        where `scalars` is the already-resolved dict so far. Scalars are
        resolved in declaration order; a later scalar's derive can
        reference an earlier scalar's value (e.g. TopK's `k` depends on
        `dim` + base_shape).
        """
        import inspect as _inspect
        import random as _random
        scalars = dict(default_scalars)
        for s_spec in scalar_inputs:
            derive = s_spec.get("derive")
            length_derive = s_spec.get("length_derive")
            dtype_str = s_spec.get("dtype", "float")
            if callable(derive):
                # Inspect derive signature: 1-arg (base_shape) or 2-arg
                # (base_shape, scalars). Pass earlier-resolved scalars
                # when the function declares 2 params.
                try:
                    sig = _inspect.signature(derive)
                    n_params = len(sig.parameters)
                except (TypeError, ValueError):
                    n_params = 1
                if n_params >= 2:
                    value = derive(list(shape), dict(scalars))
                else:
                    value = derive(list(shape))
                scalars[s_spec["name"]] = _coerce_scalar(value, dtype_str)
            elif callable(length_derive) and dtype_str == "tuple_of_int":
                # Build a tuple of length `length_derive(base_shape)` filled with
                # ints uniformly drawn from value_range.
                length = int(length_derive(list(shape)))
                if length < 0:
                    raise ValueError(
                        f"length_derive for {s_spec['name']!r} returned {length}; "
                        f"must be non-negative"
                    )
                rng_lo, rng_hi = s_spec.get("value_range", (0, 5))
                rng = _random.Random(hash((s_spec["name"], tuple(shape), length)) & 0xFFFFFFFF)
                value = tuple(rng.randint(int(rng_lo), int(rng_hi)) for _ in range(length))
                scalars[s_spec["name"]] = value
        return scalars

    # --- Band A: SHAPE stressors × uniform distribution × 1 seed ---
    # Exhausts the SHAPE axis at a single distribution. For rank>1 ops, shape_plan
    # emits both flat and multi-dim variants; _mk_tensor reshapes correctly.
    randn_fn = next(p["fn"] for p in dist_plans if p["tag"] == "uniform")
    for shape_spec in shape_plans:
        shape = shape_spec["shape"]
        scalars_here = _build_scalars_for_shape(shape)
        # Per-shape-plan scalar override (opt-in): a plan may pin specific scalar values
        # for THIS shape via "scalars" (e.g. op-class bands pairing a high head-dim with
        # dropout keep_prob<1 + a specific layout — the "shape × config" coverage a
        # curated benchmark exercises). Filtered to DECLARED scalar names so a plan's
        # suggestion is silently ignored by schemas that don't declare that scalar
        # (generic-safe: the shape still lands; only the config hint is dropped).
        _plan_scalars = shape_spec.get("scalars")
        if _plan_scalars:
            scalars_here = {**scalars_here,
                            **{k: _coerce_scalar(v, _scalar_dtype.get(k, "float"))
                               for k, v in _plan_scalars.items() if k in default_scalars}}
        inputs = _build_inputs(tensor_inputs, shape, randn_fn, dtype, seed_base=1000,
                               scalars=scalars_here,
                               max_case_tensor_bytes=max_case_tensor_bytes,
                               case_label=f"shape_{shape_spec['name']}")
        inputs.update(scalars_here)
        cases.append({
            "idx": idx, "name": f"shape_{shape_spec['name']}",
            "shape": shape, "inputs": inputs,
            "meta": {"band": "A_shape", "distribution": "uniform",
                     "shape_class": shape_spec["name"], "seed_base": 1000},
        })
        idx += 1

    # --- Band B: DISTRIBUTION stressors × base_shape (rank-matched) × n_seeds ---
    base_scalars = _build_scalars_for_shape(base_shape)
    for dist in dist_plans:
        tag = dist["tag"]
        kind = dist["kind"]

        for seed_offset in range(n_seeds):
            case_label = f"dist_{tag}_seed{seed_offset}"
            resolved_plan = _resolve_case_tensor_plan(
                tensor_inputs, base_shape, dtype, base_scalars
            )
            _guard_case_tensor_budget(
                tensor_inputs,
                base_shape,
                dtype,
                base_scalars,
                max_case_tensor_bytes=max_case_tensor_bytes,
                case_label=case_label,
                resolved_plan=resolved_plan,
            )
            if kind in ("const", "single", "randn"):
                fn = dist["fn"]
                inputs = _build_inputs(tensor_inputs, base_shape, fn, dtype,
                                         seed_base=2000 + seed_offset * 37,
                                         scalars=base_scalars,
                                         max_case_tensor_bytes=max_case_tensor_bytes,
                                         case_label=case_label,
                                         resolved_plan=resolved_plan)
                inputs.update(base_scalars)
                cases.append({
                    "idx": idx, "name": f"dist_{tag}_seed{seed_offset}",
                    "shape": base_shape, "inputs": inputs,
                    "meta": {"distribution": tag, "seed_offset": seed_offset},
                })
                idx += 1

            elif kind == "correlated_cancel":
                # Cancellation needs all-but-one tensor drawn from randn, and one tensor
                # chosen so the op's computation approximates zero for some scalar value.
                # We can't know the op's formula here — we fall back to "negate scalar"
                # variant: emit a case with scalar = -default-scalar, inputs correlated to
                # tensor[0] = scalar * tensor[1] + tiny noise IF at least 2 tensor inputs
                # + 1 numeric scalar input exist. Otherwise skip (op-dependent).
                # Skip when first scalar is non-numeric (str/enum) — can't multiply/negate.
                # Also skip when first scalar has invariant="positive"/"non_negative" —
                # flipping sign would violate the invariant (handover T12, op#5 active_num).
                first_scalar_spec = scalar_inputs[0] if scalar_inputs else None
                first_scalar_dtype = first_scalar_spec.get("dtype", "float") if first_scalar_spec else None
                first_scalar_invariant = first_scalar_spec.get("invariant") if first_scalar_spec else None
                scalar_is_numeric = first_scalar_dtype not in ("str", "string", "bool")
                scalar_flippable = (scalar_is_numeric and
                                    first_scalar_invariant not in ("positive", "non_negative"))
                # Also honor per-case scalar derivation: if the first scalar is derived
                # (not a fixed default), skip — flipping would break the derivation semantics.
                if first_scalar_spec and callable(first_scalar_spec.get("derive")):
                    scalar_flippable = False
                if len(tensor_names) >= 2 and scalar_inputs and scalar_flippable:
                    g1 = torch.Generator().manual_seed(3000 + seed_offset * 11)
                    g2 = torch.Generator().manual_seed(3001 + seed_offset * 11)
                    # First tensor (t0) takes the correlated form ONLY if it doesn't have an invariant.
                    # If tensor_inputs[0] has an invariant (permutation/index_range/positive), the
                    # tensor cannot be overwritten with the cancellation value; fall back to
                    # generating both tensors normally per-invariant and just flip scalar sign.
                    t0_spec = tensor_inputs[0]
                    t1_spec = tensor_inputs[1]
                    t0_invariant = t0_spec.get("invariant")
                    t1_invariant = t1_spec.get("invariant")

                    if t0_invariant or t1_invariant:
                        # Invariant-preserving path: no forced cancellation values; only flip scalar
                        inputs = _build_inputs(tensor_inputs, base_shape, randn_fn, dtype,
                                               seed_base=3000 + seed_offset * 11,
                                               scalars=base_scalars,
                                               max_case_tensor_bytes=max_case_tensor_bytes,
                                               case_label=case_label,
                                               resolved_plan=resolved_plan)
                        inputs.update(base_scalars)
                        # Flip scalar (safe because invariant check above)
                        s_val = base_scalars[first_scalar_spec["name"]]
                        inputs[first_scalar_spec["name"]] = -s_val
                        cases.append({
                            "idx": idx, "name": f"dist_cancellation_seed{seed_offset}",
                            "shape": base_shape, "inputs": inputs,
                            "meta": {
                                "distribution": "cancellation",
                                "seed_offset": seed_offset,
                                "note": (
                                    "invariant-preserving: scalar "
                                    f"{first_scalar_spec['name']} flipped {s_val}→{-s_val}"
                                ),
                            },
                        })
                        idx += 1
                    else:
                        # Classic path: t0 = scalar * t1 + tiny noise, then flip scalar sign
                        t0_shape = list(resolved_plan[0].shapes[0])
                        t1_shape = list(resolved_plan[1].shapes[0])
                        t0_dtype = resolved_plan[0].dtype
                        t1_dtype = resolved_plan[1].dtype
                        # Require matching shape for the correlation pattern to be meaningful
                        if (t0_shape != t1_shape or t0_dtype != t1_dtype
                                or not t0_dtype.is_floating_point):
                            continue  # skip — shapes differ, cancellation doesn't apply
                        second = torch.randn(
                            int(math.prod(t1_shape)), generator=g1, dtype=t1_dtype
                        ).reshape(*t1_shape)
                        noise = torch.randn(
                            int(math.prod(t0_shape)), generator=g2, dtype=t0_dtype
                        ).reshape(*t0_shape) * _dtype_eps(t0_dtype) * 10
                        s_val = base_scalars[first_scalar_spec["name"]]
                        first = (s_val * second + noise).to(t0_dtype)
                        inputs = {tensor_names[0]: first, tensor_names[1]: second}
                        # Build remaining tensors honoring invariants
                        for i in range(2, len(tensor_names)):
                            t_spec = tensor_inputs[i]
                            t_shape = list(resolved_plan[i].shapes[0])
                            t_dtype = resolved_plan[i].dtype
                            t_inv = t_spec.get("invariant")
                            if t_inv:
                                gen = _mk_invariant_gen(t_inv, t_spec, base_scalars)
                            elif callable(t_spec.get("value_gen")):
                                gen = t_spec["value_gen"]
                            elif t_dtype in (torch.int8, torch.int32, torch.int64):
                                gen = _mk_int_uniform_gen(t_spec.get("int_range", (0, 1000)))
                            elif t_dtype == torch.bool:
                                gen = _mk_bool_uniform_gen()
                            else:
                                gen = _normal_tensor_values
                            inputs[tensor_names[i]] = _mk_tensor(gen, t_shape, t_dtype, 3100 + i)
                        inputs.update(base_scalars)
                        # Flip the scalar sign for cancellation
                        inputs[first_scalar_spec["name"]] = -s_val
                        cases.append({
                            "idx": idx, "name": f"dist_cancellation_seed{seed_offset}",
                            "shape": base_shape, "inputs": inputs,
                            "meta": {"distribution": "cancellation", "seed_offset": seed_offset,
                                     "note": f"{tensor_names[0]} ≈ {-s_val}*{tensor_names[1]}"},
                        })
                        idx += 1

            elif kind in ("sprinkle_nan", "sprinkle_inf"):
                # Skip sprinkle on first tensor if it has an invariant (NaN/Inf would break it)
                t0_invariant = tensor_inputs[0].get("invariant")
                if t0_invariant:
                    continue
                g = torch.Generator().manual_seed(4000 + seed_offset * 13)
                t0_shape = list(resolved_plan[0].shapes[0])
                t0_dtype = resolved_plan[0].dtype
                if not t0_dtype.is_floating_point:
                    continue
                base = torch.randn(
                    int(math.prod(t0_shape)), generator=g, dtype=t0_dtype
                ).reshape(*t0_shape)
                n = int(math.prod(t0_shape))
                sprinkle_idx = torch.randperm(n, generator=torch.Generator().manual_seed(4500))[: max(1, n // 10)]
                bad = base.clone().flatten()
                bad[sprinkle_idx] = float("nan") if kind == "sprinkle_nan" else float("inf")
                inputs = {tensor_names[0]: bad.reshape(*t0_shape)}
                # Build remaining tensors honoring invariants
                for i in range(1, len(tensor_names)):
                    t_spec = tensor_inputs[i]
                    t_shape = list(resolved_plan[i].shapes[0])
                    t_dtype = resolved_plan[i].dtype
                    t_inv = t_spec.get("invariant")
                    if t_inv:
                        gen = _mk_invariant_gen(t_inv, t_spec, base_scalars)
                    elif callable(t_spec.get("value_gen")):
                        gen = t_spec["value_gen"]
                    elif t_dtype in (torch.int8, torch.int32, torch.int64):
                        gen = _mk_int_uniform_gen(t_spec.get("int_range", (0, 1000)))
                    elif t_dtype == torch.bool:
                        gen = _mk_bool_uniform_gen()
                    else:
                        gen = _normal_tensor_values
                    inputs[tensor_names[i]] = _mk_tensor(gen, t_shape, t_dtype, 4100 + i)
                inputs.update(base_scalars)
                cases.append({
                    "idx": idx, "name": f"dist_{tag}_seed{seed_offset}",
                    "shape": base_shape, "inputs": inputs,
                    "meta": {"distribution": tag, "seed_offset": seed_offset,
                             "note": f"10% NaN/Inf in {tensor_names[0]}"},
                })
                idx += 1

    # --- Band C: SCALAR probe variants at base_shape ---
    for s_spec in scalar_inputs:
        # Skip scalar probe variants for scalars that are derived per-case — their value
        # is determined by shape, not by probe_values. Band C is for fixed-default scalars.
        if callable(s_spec.get("derive")):
            continue
        for probe_v in s_spec.get("probe_values", []):
            if probe_v == s_spec.get("default"):
                continue
            # Skip probe value if invariant rules it out
            inv = s_spec.get("invariant")
            if inv in ("positive",) and (isinstance(probe_v, (int, float)) and probe_v <= 0):
                continue
            if inv in ("non_negative",) and (isinstance(probe_v, (int, float)) and probe_v < 0):
                continue
            # P0aba_layout_dispatch (2026-05-07): build the per-case scalars
            # dict with the PROBE value substituted BEFORE passing to
            # _build_inputs. Previous order built tensors with default
            # scalar then overrode after — shape_derive inside _build_inputs
            # saw stale scalar so layout_dispatch tensors got the DEFAULT
            # layout's shape regardless of the probe value. Symptom (caught
            # by test_generate_cases_with_layout_dispatch_integration):
            # `case 7 layout=BSND base=[2,2,16,16]: tensor query shape
            # [2,2,256] != expected [2,2,16,16]` — the [2,2,256] is the
            # BSH (default) layout shape under a BSND-probe case.
            probed_scalars = dict(base_scalars)
            probed_scalars[s_spec["name"]] = _coerce_scalar(
                probe_v, s_spec.get("dtype", "float"))
            inputs = _build_inputs(tensor_inputs, base_shape, randn_fn, dtype, seed_base=5000,
                                   scalars=probed_scalars,
                                   max_case_tensor_bytes=max_case_tensor_bytes,
                                   case_label=f"scalar_{s_spec['name']}_{probe_v}")
            inputs.update(probed_scalars)
            cases.append({
                "idx": idx, "name": f"scalar_{s_spec['name']}_{probe_v}",
                "shape": base_shape, "inputs": inputs,
                "meta": {"distribution": "scalar_probe", "scalar": s_spec["name"], "value": probe_v},
            })
            idx += 1

    # V1.6.B (2026-05-19 / DEBT-069 Gap A): optional-tensor presence forking.
    # For each tensor_inputs[i] with `optional=True`, fork every existing case
    # into present/absent variants. With k optional tensors, case count
    # multiplies by 2^k. None-tensor cases get the tensor name → None in inputs.
    optional_tensor_names = [
        t["name"] for t in tensor_inputs if t.get("optional", False)
    ]
    if optional_tensor_names:
        cases = _expand_optional_tensor_presence(cases, optional_tensor_names)

    # V1.7 (2026-05-21 / 13_Cat BLOCK class): list_of_tensors length sweep.
    # For each kind="list_of_tensors" input, fork every existing case once
    # per length in `list_length_plan`. Case count multiplies by Π_i len(plan_i).
    # The per-case length value is stashed in inputs[`__list_length__<name>`]
    # so the rebuilt per-case tensor list uses the correct N at materialization.
    list_specs = [
        t for t in tensor_inputs if t.get("kind") == "list_of_tensors"
    ]
    if list_specs:
        cases = _expand_list_of_tensors_lengths(
            cases, list_specs, tensor_inputs, schema,
            coverage_tier=coverage_tier, global_dtype=dtype,
            max_case_tensor_bytes=max_case_tensor_bytes,
        )

    return cases


# ---------------- data-identity hash ----------------

def dataset_data_sha256(cases: list[dict]) -> str:
    """Version-independent hash of case tensor data.

    torch.save file bytes differ across torch versions (observed 2.11 vs 2.3 on
    BabelStream Copy pilot 2026-04-20 — file SHA differed but tensor data was
    bit-exact identical). This helper hashes ONLY the tensor data, independent
    of torch.save serialization format.

    Covers: idx + name + shape + each input (key + dtype + contiguous tensor
    bytes for tensors, or dtype + float-bits for scalars). Output-side cases
    don't apply here — this is called before reference evaluation.

    Used by input_gen.py to stamp manifest + edge_inputs.pt with `data_sha256`, and
    by reference generators to re-verify before writing outputs. Consumers
    enforce a matching `data_sha256` across all staged artifacts.
    """
    h = hashlib.sha256()
    for case in sorted(cases, key=lambda c: c["idx"]):
        _update_hash_for_case(h, case)
    return h.hexdigest()


def _update_hash_for_case(h: "hashlib._Hash", case: dict) -> None:
    """Add one case's bytes to a running sha256 stream.

    Extracted from `dataset_data_sha256` (2026-05-20, edge-data design S2)
    so that `case_data_sha256(case)` can produce a per-case hash using the
    EXACT same byte protocol as the dataset hash. Per-case hash is the
    load-bearing audit field for the new `edge_manifest.json` schema; it
    matches against `dataset_data_sha256(cases) == sha256(concat(per-case))`
    is INTENTIONALLY NOT guaranteed because the dataset hash includes
    cross-case sequencing (sort + case-separator semantics). The per-case
    hash is a stable identity for one case's tensor content.
    """
    h.update(struct.pack("<q", int(case["idx"])))
    h.update(str(case.get("name", "")).encode("utf-8"))
    h.update(b"|")
    shape = case.get("shape", [])
    h.update(struct.pack(f"<{len(shape)}q", *[int(s) for s in shape]))
    h.update(b"|")
    inputs = case["inputs"]
    for key in sorted(inputs.keys()):
        v = inputs[key]
        h.update(key.encode("utf-8"))
        h.update(b"=")
        if isinstance(v, torch.Tensor):
            t = v.detach().contiguous().cpu()
            h.update(b"T")
            h.update(str(t.dtype).encode("utf-8"))
            h.update(b":")
            h.update(struct.pack(f"<{t.ndim}q", *[int(s) for s in t.shape]))
            h.update(b":")
            # numpy() doesn't support bf16/fp16 on all platforms; view as int16/int32 bytes
            # (same bit-pattern, numpy-compatible).
            if t.dtype == torch.bfloat16:
                h.update(t.view(torch.int16).numpy().tobytes())
            elif t.dtype == torch.float16:
                h.update(t.view(torch.int16).numpy().tobytes())
            else:
                h.update(t.numpy().tobytes())
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            # V1.7 (2026-05-21): kind=list_of_tensors hash protocol — emit
            # length sentinel + per-item tensor bytes (using the same tensor
            # protocol as the single-tensor path above so list-of-Tensors
            # cases hash consistently with single-tensor cases that happen
            # to contain the same data).
            h.update(b"L")
            h.update(struct.pack("<q", len(v)))
            h.update(b":")
            for tt in v:
                t = tt.detach().contiguous().cpu()
                h.update(b"T")
                h.update(str(t.dtype).encode("utf-8"))
                h.update(b":")
                h.update(struct.pack(f"<{t.ndim}q", *[int(s) for s in t.shape]))
                h.update(b":")
                if t.dtype == torch.bfloat16:
                    h.update(t.view(torch.int16).numpy().tobytes())
                elif t.dtype == torch.float16:
                    h.update(t.view(torch.int16).numpy().tobytes())
                else:
                    h.update(t.numpy().tobytes())
                h.update(b";")
        elif isinstance(v, (int, float, bool)):
            h.update(b"S")
            h.update(type(v).__name__.encode("utf-8"))
            h.update(b":")
            h.update(struct.pack("<d", float(v)))
        else:
            h.update(b"O")
            h.update(repr(v).encode("utf-8"))
        h.update(b"|")


def case_data_sha256(case: dict) -> str:
    """Per-case version of `dataset_data_sha256`.

    Added 2026-05-20 (edge-data design S2). The new `edge_manifest.json`
    archives one `input_sha256` per case so the S5 finalize gate can verify
    regenerated inputs against the archive without re-loading the .pt blob.

    Uses the same byte protocol as `dataset_data_sha256` (per-case section)
    so the hash is stable across torch versions and matches between archive
    creation and A3-side regen verification.
    """
    h = hashlib.sha256()
    _update_hash_for_case(h, case)
    return h.hexdigest()


# ---------------- smoke test helpers ----------------

def _summary(cases: list[dict]) -> dict:
    shapes = {}
    dists = {}
    for c in cases:
        shapes[str(c["shape"])] = shapes.get(str(c["shape"]), 0) + 1
        tag = c["meta"].get("distribution", "unknown")
        dists[tag] = dists.get(tag, 0) + 1
    return {"n_cases": len(cases), "shape_buckets": shapes, "distribution_buckets": dists}


if __name__ == "__main__":
    # Self-test with Triad schema
    import json
    triad_schema = {
        "op_name": "babelstream_triad",
        "formula": "a[i] = b[i] + scalar * c[i]",
        "tensor_inputs": [{"name": "b", "role": "operand"}, {"name": "c", "role": "operand"}],
        "scalar_inputs": [{"name": "scalar", "dtype": "float", "default": 0.4,
                           "probe_values": [0.4, -0.4, 0.0, -1.0]}],
        "tensor_output": "a",
        "rank": 1,
    }
    for tier in ["pilot", "sign_off", "production"]:
        cases = generate_cases(triad_schema, coverage_tier=tier, dtype=torch.float32)
        print(f"tier={tier:10s} → {json.dumps(_summary(cases), indent=2)}")

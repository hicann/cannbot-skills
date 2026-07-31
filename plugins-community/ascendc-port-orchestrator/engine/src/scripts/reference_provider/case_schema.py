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

"""case_schema — shape/dtype/distribution/layout planning from an op signature, extracted
from case_gen.py (behavior-neutral, 2026-07-05). Leaf. Items kept in original order so the
_OP_CLASS_SHAPE_BANDS dispatch dict sees its shape-band funcs already defined."""
from __future__ import annotations
import math
import struct
from typing import Any

import torch

_TIER_SHAPE_EDGES = {
    "pilot": False,
    "sign_off": True,
    "production": True,
}
_TIER_NAN_INF = {
    "pilot": False,
    "sign_off": False,
    "production": True,
}
_TIER_2D_EQUIV = {
    "pilot": False,
    "sign_off": True,
    "production": True,
}
_A5 = {
    "n_cores": 56,
    "align_elems_fp32": 8,   # 32B / 4B
    "align_elems_fp16": 16,
    "default_tile_elems": 4096,
}


def _dtype_eps(dt: torch.dtype) -> float:
    return {
        torch.float32: 1.19e-7,
        torch.float16: 9.77e-4,
        torch.bfloat16: 7.81e-3,
        torch.float64: 2.22e-16,
    }[dt]


def _dtype_smallest_normal(dt: torch.dtype) -> float:
    return {
        torch.float32: 1.175494e-38,
        torch.float16: 6.103e-5,
        torch.bfloat16: 1.175494e-38,  # same mantissa range as fp32
        torch.float64: 2.225074e-308,
    }[dt]


def _shape_plan(coverage_tier: str, dtype: torch.dtype, rank: int) -> list[dict]:
    """Return list of {name, shape} pairs for this coverage tier.

    1D shapes span: degenerate / align-boundary / core-boundary / tile-boundary /
    typical / prime-non-aligned. 2D shapes added at sign_off+ when rank allows.
    """
    align = _A5["align_elems_fp32"] if dtype != torch.float16 else _A5["align_elems_fp16"]
    cores = _A5["n_cores"]
    tile = _A5["default_tile_elems"]

    shapes = []

    # --- TIER 1: TYPICAL (always) ---
    shapes += [
        {"name": "typical_1k", "shape": [1024]},
        {"name": "typical_32k", "shape": [32 * 1024]},
        {"name": "typical_1m", "shape": [1024 * 1024]},
    ]

    # --- TIER 2 (sign_off+): shape edges ---
    if _TIER_SHAPE_EDGES[coverage_tier]:
        shapes += [
            # Degenerate
            {"name": "degen_n1", "shape": [1]},
            {"name": "degen_n7", "shape": [align - 1]},                  # below alignment
            # Align boundary
            {"name": "align_1u", "shape": [align]},                       # exactly 1 aligned unit
            {"name": "align_1u_plus1", "shape": [align + 1]},                   # 1 unit + 1 tail
            # Core-partition boundary
            {"name": "core_lt_cores", "shape": [cores - 1]},                   # fewer elements than cores
            {"name": "core_eq_cores", "shape": [cores * align]},               # exactly 1 unit/core
            {"name": "core_not_multiple", "shape": [cores * align + 3]},          # 3-tail, uneven
            # Tile boundary
            {"name": "tile_minus1", "shape": [tile - 1]},                    # 1-short of tile
            {"name": "tile_plus1", "shape": [tile + 1]},                    # 1-over tile
            {"name": "tile_2_minus1", "shape": [2 * tile - 1]},
            # Prime / non-aligned (truly off-grid)
            {"name": "prime_1009", "shape": [1009]},
            {"name": "prime_32771", "shape": [32771]},
            # 2D equivalent (rank check)
        ]
        if rank >= 2 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d2_32x32", "shape": [32, 32]},        # = 1024 flat
                {"name": "d2_64x16", "shape": [64, 16]},        # = 1024 flat, different layout
            ]
        # P0aaa-extension (2026-05-06): rank-3 base_shape plan for ops like
        # 6_QuantMatmul [m,k,n], 9_QuantMatmulReduceSum [batch,m,k]. Multiple
        # representatives covering small / typical / non-square / batch=1.
        # base_shape_filter still gates which apply per-op.
        if rank >= 3 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d3_small", "shape": [16, 64, 64]},     # small all-square
                {"name": "d3_typical", "shape": [64, 256, 256]},   # typical
                {"name": "d3_non_square", "shape": [32, 128, 384]},   # non-square m≠k≠n
                {"name": "d3_batch1", "shape": [1, 256, 256]},    # batch=1 edge
                {"name": "d3_align_k", "shape": [16, 16, 256]},    # k at align boundary
                {"name": "d3_prime_inner", "shape": [16, 113, 64]},    # prime inner dim
            ]
        # rank-4 base_shape plan for ops like 2_FFN [M,K1,N1,N2],
        # 8_WeightQuantBatchmatmul [B,M,K,N], 1_LSTM-like [B,T,D,H].
        if rank >= 4 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d4_small", "shape": [16, 64, 64, 64]},     # all-square small
                {"name": "d4_typical", "shape": [64, 256, 256, 256]},  # typical
                {"name": "d4_non_square", "shape": [32, 128, 256, 64]},   # K1≠N1≠N2
                {"name": "d4_m1", "shape": [1, 256, 256, 256]},   # M=1 edge
                {"name": "d4_align_inner", "shape": [16, 16, 32, 64]},     # align-boundary inner dims
            ]

    # --- TIER 3 (production): even more boundaries ---
    if coverage_tier == "production":
        shapes += [
            {"name": "tile_3_plus3", "shape": [3 * tile + 3]},
            {"name": "huge_8m", "shape": [8 * 1024 * 1024]},  # multi-tile stress
        ]

    return shapes


def _attention_shape_bands(coverage_tier: str, dtype: torch.dtype, rank: int) -> list[dict]:
    """FA / attention semantic shape bands. Base layout = rank-4 [B, S, N, D].

    Covers the domain stressors FA correctness + tiling pivot on (which the generic
    `_shape_plan` square-ish shapes miss):
      - D (head_dim): single-tile (64/128) / multi-tile boundary (256) / multi-tile
        (512) / non-pow2 multi-tile (640) / max (768) — incl 640/768 the generic plan
        NEVER emits (the exact V2-64 gap that produced the case-gen reconcile).
      - S (seqlen): tile multiples + tails (64/128/192/256/384/512).
      - B (batch): 1/2/4.   N (head_num): 8/16/32.
    Systematic one-axis-at-a-time sweep around a baseline + representative combinations
    — comparable-to-exceeds a hand-curated set (FA V2-64) without a full cross-product.
    dtype + layout sweeps multiply via schema['dtypes'] + make_layout_dispatch.
    """
    if rank != 4:
        # attention base is rank-4 [B,S,N,D]; nothing semantic to add at other ranks.
        return []
    base_batch, base_sequence, base_heads, base_head_dim = 1, 256, 8, 128
    if not _TIER_SHAPE_EDGES[coverage_tier]:
        # tier-1 typical-only: a couple representative attention bases.
        return [
            {"name": "attn_typ_d128", "shape": [base_batch, base_sequence, base_heads, 128]},
            {"name": "attn_typ_d64", "shape": [base_batch, 128, base_heads, 64]},
        ]
    head_dim_bands = [64, 128, 256, 512, 640, 768]   # head-dim: single/multi-tile + 640/768
    sequence_bands = [64, 128, 192, 256, 384, 512]   # seqlen: tile multiples + tails
    batch_bands = [1, 2, 4]
    head_count_bands = [8, 16, 32]
    bands: list[dict] = []
    for d in head_dim_bands:                   # D sweep (key gap — incl 640/768)
        bands.append({"name": f"attn_D{d}", "shape": [base_batch, base_sequence, base_heads, d]})
    for s in sequence_bands:                   # S sweep (seqlen-tile bands)
        bands.append({"name": f"attn_S{s}", "shape": [base_batch, s, base_heads, base_head_dim]})
    for b in batch_bands:                      # B sweep (batch)
        bands.append({"name": f"attn_B{b}", "shape": [b, base_sequence, base_heads, base_head_dim]})
    for n in head_count_bands:                 # N sweep (head_num)
        bands.append({"name": f"attn_N{n}", "shape": [base_batch, base_sequence, n, base_head_dim]})
    bands += [                                 # representative cross-band combinations
        {"name": "attn_b4_n32_s512_d64", "shape": [4, 512, 32, 64]},   # large multicore, small-D
        {"name": "attn_b2_n16_s256_d128", "shape": [2, 256, 16, 128]},  # mid
        # High-D bands CARRY their dropout + layout config (coverage = shape × dropout).
        # Spec V2-64's high-D cases ARE the dropout cases: 640 is BNSD kp0.9, 768 is SBH
        # kp0.8. The per-plan "scalars" override is applied ONLY if the schema declares
        # that scalar (generic-safe — a schema without keep_prob/input_layout ignores it
        # and still gets the SHAPE). See generate_cases Band-A per-plan override.
        {"name": "attn_b1_n4_s192_d640", "shape": [1, 192, 4, 640],    # high-D non-pow2 + tail-S
         "scalars": {"input_layout": "BNSD", "keep_prob": 0.9}},
        {"name": "attn_b1_n2_s384_d768", "shape": [1, 384, 2, 768],    # max-D + 3-tile-S
         "scalars": {"input_layout": "SBH", "keep_prob": 0.8}},
    ]
    # Dropout (keep_prob<1) bands spanning low + high D — spec V2-64 has 6 kp<1 cases
    # NOT confined to high-D. These add mid-D dropout so (shape × dropout) is exercised
    # across the D range, not just at 640/768.
    bands += [
        {"name": "attn_d128_drop", "shape": [base_batch, base_sequence, base_heads, 128],
         "scalars": {"keep_prob": 0.9}},
        {"name": "attn_d512_drop", "shape": [1, 128, 8, 512],
         "scalars": {"keep_prob": 0.8}},
    ]
    if coverage_tier == "production":
        bands.append({"name": "attn_b4_n8_s512_d768", "shape": [4, 512, 8, 768]})  # max stress
    return bands


def _matmul_shape_bands(coverage_tier: str, dtype: torch.dtype, rank: int) -> list[dict]:
    """Matmul / GEMM semantic shape bands. Base layout = rank-3 [M, N, K]; the schema's
    shape_derive views it as A=[M,K], B=[K,N] (+ optional transpose/quant variants).

    A matmul's correctness + tiling pivot on cube-tile crossings (the 16x16x16 fractal /
    256-element L1 block on A5), NOT on square align edges — so the generic `_shape_plan`
    misses them. Bands sweep each of M / N / K across single-tile / multi-tile / tail
    (tile + remainder, off-by-one) + the tall-skinny / short-wide / deep-K aspect ratios
    that stress the accumulate-vs-output-tile balance. This is the 2nd op-class emitter —
    it proves the op-class mechanism generalizes (same registry, same per-plan scalar
    override, different domain bands).
    """
    if rank != 3:
        # matmul base is rank-3 [M,N,K]; nothing semantic to add at other ranks.
        return []
    base_m, base_n, base_k = 256, 256, 256
    if not _TIER_SHAPE_EDGES[coverage_tier]:
        return [{"name": "mm_typ", "shape": [256, 256, 256]},
                {"name": "mm_small", "shape": [64, 64, 64]}]
    matmul_dim_bands = [16, 64, 256, 512, 1024]  # single-tile -> multi-tile
    tail_dims = [17, 255, 257]                  # tile-boundary tails: tile-1, tile, tile+1
    bands: list[dict] = []
    for m in matmul_dim_bands:
        bands.append({"name": f"mm_M{m}", "shape": [m, base_n, base_k]})
    for n in matmul_dim_bands:
        bands.append({"name": f"mm_N{n}", "shape": [base_m, n, base_k]})
    for k in matmul_dim_bands:
        bands.append({"name": f"mm_K{k}", "shape": [base_m, base_n, k]})
    for t in tail_dims:
        bands.append({"name": f"mm_Mtail{t}", "shape": [t, base_n, base_k]})
        bands.append({"name": f"mm_Ktail{t}", "shape": [base_m, base_n, t]})   # K tail = reduction-depth remainder
    bands += [                                  # aspect-ratio stressors
        {"name": "mm_tall_skinny", "shape": [2048, 64, 256]},   # M >> N
        {"name": "mm_short_wide", "shape": [64, 2048, 256]},   # N >> M
        {"name": "mm_deep_k", "shape": [256, 256, 2048]},  # large reduction depth
    ]
    if coverage_tier == "production":
        bands.append({"name": "mm_large", "shape": [4096, 4096, 4096]})  # max stress
    return bands


_OP_CLASS_SHAPE_BANDS = {
    "attention": _attention_shape_bands,
    "fa_class": _attention_shape_bands,   # alias for the FA-class tag
    "matmul": _matmul_shape_bands,
    "gemm": _matmul_shape_bands,      # alias
}


def _op_class_shape_bands(op_class: "str | None", coverage_tier: str,
                          dtype: torch.dtype, rank: int) -> list[dict]:
    """Dispatch to the op-class semantic band-emitter; [] if no op_class / none registered."""
    if not op_class:
        return []
    emitter = _OP_CLASS_SHAPE_BANDS.get(str(op_class).lower())
    if emitter is None:
        return []
    return emitter(coverage_tier, dtype, rank)


def _distribution_plan(coverage_tier: str) -> list[dict]:
    """Return list of {name, gen_fn} for this tier.

    gen_fn: called as fn(n_elem, dtype, seed) -> torch.Tensor of shape (n_elem,).
    """
    plans = []

    def mk_const(v):
        return lambda n, dt, _seed: torch.full((n,), v, dtype=dt)

    def mk_randn():
        return lambda n, dt, seed: torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=dt)

    def mk_uniform(lo, hi):
        def _f(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            return (torch.rand(n, generator=g, dtype=dt) * (hi - lo) + lo).to(dt)
        return _f

    def mk_uniform_dtype_large():
        # large-but-VALID magnitude stressor. Scaled to sqrt(finfo(dtype).max)/16
        # so that pairwise products AND moderate accumulation (reduction depth up
        # to ~256 — covers FA Q@K over head-dim and topK/seq reductions) stay
        # within the dtype range. This is:
        #   * large (fp16 ~8-16, bf16 ~1e18 — far above the ~N(0,1) `uniform`
        #     distribution, stressing the high-magnitude regime), AND
        #   * finite (never inf on cast — the old fixed [1e20,1e30] made every
        #     fp16 value inf since finfo(fp16).max=6.5e4 → degenerate all-inf
        #     INPUTS → fp64 oracle nan/inf → unscoreable), AND
        #   * SCOREABLE for accumulation ops (gradients don't overflow the dtype).
        # Generated in fp32 then cast for precision. (Ops with very deep (>256)
        # accumulation may still overflow; the scorer's degenerate-guard handles
        # non-finite references — but the INPUTS are always valid + finite.)
        def _f(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            m = float(torch.finfo(dt).max) if dt.is_floating_point else 1e30
            scale = math.sqrt(m) / 16.0
            lo, hi = 0.5 * scale, 1.0 * scale
            vals = torch.rand(n, generator=g, dtype=torch.float32) * (hi - lo) + lo
            return vals.to(dt)
        return _f

    def mk_uniform_dtype_small():
        # small-but-REPRESENTABLE stressor scaled to the dtype's smallest normal.
        # The old fixed [1e-30,1e-20] underflowed every fp16 value to 0 (fp16
        # smallest-normal ~6e-5), making small_mag a trivial all-zero case.
        def _f(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            tiny = float(torch.finfo(dt).tiny) if dt.is_floating_point else 1e-20
            lo, hi = tiny * 16.0, tiny * 1024.0
            vals = torch.rand(n, generator=g, dtype=torch.float32) * (hi - lo) + lo
            return vals.to(dt)
        return _f

    def mk_mixed_sign():
        def _f(n, dt, seed):
            g1 = torch.Generator().manual_seed(seed)
            g2 = torch.Generator().manual_seed(seed + 17)
            t = torch.randn(n, generator=g1, dtype=dt)
            mask = (torch.rand(n, generator=g2) < 0.5).to(dt) * 2 - 1
            return (t * mask).to(dt)
        return _f

    def mk_denormal():
        def _f(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            small = _dtype_smallest_normal(dt)
            return (torch.rand(n, generator=g, dtype=dt) * (1.5 * small) + 0.5 * small).to(dt)
        return _f

    def mk_cancellation_pair(scalar_val):
        """For cancellation, we need correlated tensors; caller handles."""
        return scalar_val  # signal

    # --- Core distributions (all tiers) ---
    plans += [
        {"tag": "const_sanity", "kind": "const", "fn": mk_const(1.0)},
        {"tag": "uniform", "kind": "randn", "fn": mk_randn()},
        {"tag": "large_mag", "kind": "single", "fn": mk_uniform_dtype_large()},
        {"tag": "small_mag", "kind": "single", "fn": mk_uniform_dtype_small()},
        {"tag": "denormal", "kind": "single", "fn": mk_denormal()},
        {"tag": "mixed_sign", "kind": "single", "fn": mk_mixed_sign()},
    ]

    if coverage_tier in ("sign_off", "production"):
        plans += [
            {"tag": "cancellation", "kind": "correlated_cancel"},  # custom handling below
            {"tag": "const_near_zero", "kind": "const", "fn": mk_const(_dtype_eps(torch.float32) * 5)},
        ]

    if _TIER_NAN_INF[coverage_tier]:
        # WARNING: NaN cases break `torch.equal` bit-exact comparison since NaN != NaN.
        # verify.py must use NaN-aware compare (equal_nan=True or same-mask + value check)
        # when production tier cases are present. Pilot/sign_off do NOT emit these.
        plans += [
            {"tag": "sparse_nan", "kind": "sprinkle_nan"},
            {"tag": "sparse_inf", "kind": "sprinkle_inf"},
        ]

    return plans


def _tensor_shape_for(tinput: dict, base_shape: list[int],
                       scalars: dict[str, Any] | None = None) -> list[int]:
    """Compute per-tensor shape. Uses shape_derive if present, else base_shape.

    shape_derive contract:
      - 1-arg:  callable(base_shape) -> list[int]  (current behavior, backward-compat)
      - 2-arg:  callable(base_shape, scalars) -> list[int]  (added 2026-04-24)
        lets tensor shapes depend on scalar inputs — e.g. op#3 sampled_token_ids
        shape [num_queries, 1] where num_queries is a scalar input derived per-case.

    Example for op#11 (base_shape [N, 2H], 1-arg derive):
        x:      lambda s: list(s)            → [N, 2H]
        ws:     lambda s: [1, s[-1]]         → [1, 2H]

    Example for op#3 (base_shape [num_seqs], 2-arg derive, num_queries is derived scalar):
        sampled_token_ids: lambda s, sc: [sc["num_queries"], 1]
        block_tables:      lambda s, sc: [s[0], sc["max_blocks_per_seq"]]
    """
    import inspect
    derive = tinput.get("shape_derive")
    if derive is None:
        return list(base_shape)
    try:
        sig = inspect.signature(derive)
        n_positional = 0
        positional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for parameter in sig.parameters.values():
            if parameter.kind in positional_kinds:
                n_positional += 1
    except (TypeError, ValueError):
        n_positional = 1  # fallback: assume 1-arg
    if n_positional >= 2:
        result = derive(list(base_shape), dict(scalars or {}))
    else:
        result = derive(list(base_shape))
    if not isinstance(result, (list, tuple)) or not all(isinstance(x, int) and x >= 0 for x in result):
        raise ValueError(
            f"shape_derive for {tinput.get('name')!r} returned {result!r}; "
            f"must be list[int] of non-negative ints"
        )
    return list(result)


def make_layout_dispatch(
    layouts: dict[str, "list[int] | callable"],
    *,
    scalar_name: str = "input_layout",
) -> "callable":
    """P0aba_layout_dispatch (2026-05-07): build a 2-arg shape_derive that
    dispatches per-tensor shape based on a string-typed scalar.

    Designed for L4 attention-class ops where the operator accepts multiple
    layout strings (BSH / SBH / BSND / BNSD / TND etc.) that collapse or
    permute the same logical (B, S, N, D) dimensions:

        BSH  → [B, S, N*D]    (rank-3, head-dim collapsed into H)
        SBH  → [S, B, N*D]    (rank-3)
        BSND → [B, S, N, D]   (rank-4, head-dim explicit)
        BNSD → [B, N, S, D]   (rank-4, num_heads ahead of seq)
        TND  → [T, N, D]      (rank-3, batch+seq collapsed into T)

    All cases come from the SAME rank-4 base_shape (typically [B, S, N, D])
    and the layout scalar selects which view of those dims this tensor uses.
    Without this primitive, the SCHEMA author has to write the dispatch
    lambda by hand; this factory makes it declarative.

    Usage in SCHEMA:

        from case_gen import make_layout_dispatch

        SCHEMA = {
            "tensor_inputs": [
                {"name": "query",
                 "shape_derive": make_layout_dispatch({
                     "BSH":  lambda s: [s[0], s[1], s[2] * s[3]],
                     "SBH":  lambda s: [s[1], s[0], s[2] * s[3]],
                     "BSND": lambda s: [s[0], s[1], s[2], s[3]],
                     "BNSD": lambda s: [s[0], s[2], s[1], s[3]],
                 })},
                # key/value follow same pattern...
            ],
            "scalar_inputs": [
                {"name": "input_layout", "dtype": "str",
                 "default": "BSH",
                 "probe_values": ["BSH", "SBH", "BSND", "BNSD"]},
            ],
            "rank": 4,  # base_shape is [B, S, N, D]
            ...
        }

    Args:
        layouts: dict mapping layout string → either a fixed shape list[int]
                 OR a callable(base_shape) → list[int]. Callable form lets
                 the layout consult the actual base dims. Fixed form is for
                 layouts that don't vary by base.
        scalar_name: name of the layout scalar in scalar_inputs (defaults to
                     "input_layout"; some ops use "layout_query" / "layout_kv").

    Returns:
        2-arg shape_derive callable suitable for a tensor_input's
        `shape_derive` field. case_gen's _tensor_shape_for detects 2-arg
        signature and passes scalars dict.

    Raises (when called):
        ValueError if the resolved layout value is not in the layouts dict
        (catches typos / missing layout entries instead of silently
        falling back to base_shape).
    """
    def _dispatch(base, scalars):
        layout = scalars.get(scalar_name)
        if layout is None:
            raise ValueError(
                f"layout_dispatch: scalar {scalar_name!r} missing from scalars; "
                f"declare it in SCHEMA's scalar_inputs with probe_values "
                f"covering: {sorted(layouts.keys())}"
            )
        if layout not in layouts:
            raise ValueError(
                f"layout_dispatch: scalar {scalar_name}={layout!r} not in "
                f"declared layouts {sorted(layouts.keys())}; either add a "
                f"mapping for this layout OR remove it from probe_values"
            )
        spec = layouts[layout]
        if callable(spec):
            result = spec(base)
        elif isinstance(spec, (list, tuple)):
            result = spec
        else:
            raise ValueError(
                f"layout_dispatch: layouts[{layout!r}] must be list[int] or "
                f"callable(base) -> list[int], got {type(spec).__name__}"
            )
        return list(result)  # always return list (not tuple) per case_gen contract

    return _dispatch


def _tensor_dtype_for(tinput: dict, global_dtype: torch.dtype) -> torch.dtype:
    """Per-tensor dtype override (for heterogeneous ops like op#11 int32 x + fp32 scales)."""
    dt = tinput.get("dtype")
    if dt is None:
        return global_dtype
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        mapping = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16, "float64": torch.float64,
                   "int32": torch.int32, "int64": torch.int64, "int8": torch.int8,
                   "bool": torch.bool}
        if dt in mapping:
            return mapping[dt]
        raise ValueError(f"unknown dtype string {dt!r} in tensor_input")
    raise ValueError(f"dtype must be torch.dtype or str, got {type(dt)}")


def _base_shape_for_rank(rank: int) -> list[int]:
    """Pick a representative shape matching the op's rank for distribution-band cases.
    1D → [1024]; 2D → [32, 32]; 3D → [8, 8, 16]; 4D → [2, 2, 16, 16] (=1024 flat, last dim even for rotary-style ops);
    5D → [1, 2, 4, 8, 16] (=1024 flat; for video / 3D-conv / pool3d / NCDHW ops).
    """
    if rank == 1:
        return [1024]
    if rank == 2:
        return [32, 32]
    if rank == 3:
        return [8, 8, 16]
    if rank == 4:
        return [2, 2, 16, 16]
    if rank == 5:
        return [1, 2, 4, 8, 16]
    raise ValueError(f"rank={rank} not supported; extend _base_shape_for_rank.")

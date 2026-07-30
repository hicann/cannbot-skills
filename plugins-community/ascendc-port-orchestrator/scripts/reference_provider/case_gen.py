# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Op-agnostic test case generator for Phase O2.5 Reference Provider.

Given an operation schema and coverage tier, emit cases across three independent
bands rather than a full cross-product:

* Band A stresses shapes, including tails, alignment, partitioning, tiles,
  primes, and multidimensional layouts.
* Band B stresses distributions at a representative shape with one, two, or
  three seeds depending on the tier.
* Band C sweeps each scalar input's `probe_values` at the base shape.

A full shapes-by-distributions-by-seeds product would produce hundreds of
sign-off cases. Independent sweeps retain the highest-value coverage at roughly
forty to fifty cases. An operation that needs a specific cross-product, such as
a shape-dependent denormal case, should append explicit cases in its per-op
`input_gen.py`.

The primary call and result contract is:

```python
cases = generate_cases(schema, coverage_tier="sign_off", dtype=torch.float32)
```

Each result has integer `idx`, string `name`, integer-list `shape`, and an
`inputs` mapping from input name to tensor or scalar.

Core schema example:

```python
schema = {
    "op_name": "babelstream_triad",
    "formula": "a[i] = b[i] + scalar*c[i]",
    "tensor_inputs": [
        {"name": "b", "role": "operand"},
        {"name": "c", "role": "operand"},
    ],
    "scalar_inputs": [
        {
            "name": "scalar",
            "dtype": "float",
            "default": 0.4,
            "probe_values": [0.4, -0.4, 0.0, -1.0],
        },
    ],
    "tensor_output": "a",
    "rank": 1,
}
```

For fused operations with interdependent shapes, `base_shape_filter` rejects
invalid bases. Each tensor entry may override the global `dtype`; provide a
`shape_derive` callable taking either `base_shape` or
`base_shape, scalars_dict`; and provide a `value_gen` callable taking element
count, dtype, and seed. A tensor without `shape_derive` uses the base shape.

The optional tensor `invariant` accepts `permutation`, `positive`,
`non_negative`, or `index_range:<N>`, where N is an integer literal or scalar
input name. Correlated-cancellation and sprinkle bands do not overwrite a
tensor carrying one of these invariants, which keeps generated cases
semantically valid.

Optional tensor primitive, introduced in V1.6.B:

```python
tensor_inputs = [
    {"name": "input"},
    {"name": "target", "dtype": torch.int64, "invariant": "index_range_dim:dim"},
    {"name": "weight", "optional": True},
]
```

Each optional tensor doubles the presence combinations. The absent variant
stores Python `None` in `inputs[name]`, so a reference such as
`Model.forward(weight=None)` and the kernel under test both exercise the
unweighted path. The empirical anchor is 25_NLLLoss: 13 of 20 JSON cases use
`weight=None`, and preserving both variants exposed a one-character weighted
sum defect that an always-absent input would have missed.

Variable-length list primitive, introduced in V1.7:

```python
tensor_inputs = [
    {
        "name": "tensors",
        "kind": "list_of_tensors",
        "list_length_plan": [2, 3, 4],
        "per_item_shape_derive": lambda base, index, length: base,
    },
]
```

For every planned length N, the generated input is a Python list of N tensors.
The optional per-item shape callable may take three arguments
`base_shape, item_index, list_length` or four arguments with `scalars` added;
when omitted, every item uses the base shape. Combine this primitive with an
axis scalar such as `dim` for operations like `torch.cat`. It closes the
variable-length-input limitation previously documented for 13_Cat, 14_Split,
and similar operations.

A scalar entry may provide `derive`, a callable from base shape to value that
overrides `default`. Its optional `invariant` accepts `positive`,
`non_negative`, or informational `le:<other_scalar_name>`. The first two stop
the correlated-cancellation band from flipping the scalar's sign.

Coverage tiers are inclusive. `pilot` uses about 15 cases and one seed;
`sign_off` adds tail, degenerate, core-boundary, prime, and 2D-equivalence
shapes plus a second seed for about 50 cases; `production` adds NaN and Inf,
three seeds, three tile-boundary variants per distribution, and exhaustive
shape cross-products for about 150 cases.

Reduction-order semantics, indexed/scatter conjoint-shape semantics, and
multi-dtype precision policies may require operation-specific extensions.

Per-operation usage:

```python
from case_gen import generate_cases

SCHEMA = {...}
cases = generate_cases(SCHEMA, coverage_tier="sign_off", dtype=torch.float32)
torch.save(
    {"dtype": "float32", "op": SCHEMA["op_name"], "schema": SCHEMA, "cases": cases},
    "inputs.pt",
)
```
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

# Coverage knobs
_TIER_SHAPE_EDGES = {
    "pilot": False,
    "sign_off": True,
    "production": True,
}
_TIER_SEEDS_PER_DIST = {
    "pilot": 1,
    "sign_off": 2,
    "production": 3,
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


# Hardware constants used to generate alignment-stress shapes. Current target is A5
# (Ascend950PR): 56 AIV cores, 32B alignment (= 8 fp32 / 16 fp16). UB tile size
# varies per kernel design but 4096 fp32 is a common choice (produces 16 KB × queues).
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
        {"name": "typical_1k",  "shape": [1024]},
        {"name": "typical_32k", "shape": [32 * 1024]},
        {"name": "typical_1m",  "shape": [1024 * 1024]},
    ]

    # --- TIER 2 (sign_off+): shape edges ---
    if _TIER_SHAPE_EDGES[coverage_tier]:
        shapes += [
            # Degenerate
            {"name": "degen_n1",        "shape": [1]},
            {"name": "degen_n7",        "shape": [align - 1]},                  # below alignment
            # Align boundary
            {"name": "align_1u",        "shape": [align]},                       # exactly 1 aligned unit
            {"name": "align_1u_plus1",  "shape": [align + 1]},                   # 1 unit + 1 tail
            # Core-partition boundary
            {"name": "core_lt_cores",   "shape": [cores - 1]},                   # fewer elements than cores
            {"name": "core_eq_cores",   "shape": [cores * align]},               # exactly 1 unit/core
            {"name": "core_not_multiple", "shape": [cores * align + 3]},         # 3-tail, uneven
            # Tile boundary
            {"name": "tile_minus1",     "shape": [tile - 1]},                    # 1-short of tile
            {"name": "tile_plus1",      "shape": [tile + 1]},                    # 1-over tile
            {"name": "tile_2_minus1",   "shape": [2 * tile - 1]},
            # Prime / non-aligned (truly off-grid)
            {"name": "prime_1009",      "shape": [1009]},
            {"name": "prime_32771",     "shape": [32771]},
            # 2D equivalent (rank check)
        ]
        if rank >= 2 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d2_32x32",    "shape": [32, 32]},        # = 1024 flat
                {"name": "d2_64x16",    "shape": [64, 16]},        # = 1024 flat, different layout
            ]
        # P0aaa-extension (2026-05-06): rank-3 base_shape plan for ops like
        # 6_QuantMatmul [m,k,n], 9_QuantMatmulReduceSum [batch,m,k]. Multiple
        # representatives covering small / typical / non-square / batch=1.
        # base_shape_filter still gates which apply per-op.
        if rank >= 3 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d3_small",        "shape": [16, 64, 64]},     # small all-square
                {"name": "d3_typical",      "shape": [64, 256, 256]},   # typical
                {"name": "d3_non_square",   "shape": [32, 128, 384]},   # non-square m≠k≠n
                {"name": "d3_batch1",       "shape": [1, 256, 256]},    # batch=1 edge
                {"name": "d3_align_k",      "shape": [16, 16, 256]},    # k at align boundary
                {"name": "d3_prime_inner",  "shape": [16, 113, 64]},    # prime inner dim
            ]
        # rank-4 base_shape plan for ops like 2_FFN [M,K1,N1,N2],
        # 8_WeightQuantBatchmatmul [B,M,K,N], 1_LSTM-like [B,T,D,H].
        if rank >= 4 and _TIER_2D_EQUIV[coverage_tier]:
            shapes += [
                {"name": "d4_small",        "shape": [16, 64, 64, 64]},     # all-square small
                {"name": "d4_typical",      "shape": [64, 256, 256, 256]},  # typical
                {"name": "d4_non_square",   "shape": [32, 128, 256, 64]},   # K1≠N1≠N2
                {"name": "d4_m1",           "shape": [1, 256, 256, 256]},   # M=1 edge
                {"name": "d4_align_inner",  "shape": [16, 16, 32, 64]},     # align-boundary inner dims
            ]

    # --- TIER 3 (production): even more boundaries ---
    if coverage_tier == "production":
        shapes += [
            {"name": "tile_3_plus3",    "shape": [3 * tile + 3]},
            {"name": "huge_8m",         "shape": [8 * 1024 * 1024]},  # multi-tile stress
        ]

    return shapes


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
    base_B, base_S, base_N, base_D = 1, 256, 8, 128
    if not _TIER_SHAPE_EDGES[coverage_tier]:
        # tier-1 typical-only: a couple representative attention bases.
        return [
            {"name": "attn_typ_d128", "shape": [base_B, base_S, base_N, 128]},
            {"name": "attn_typ_d64",  "shape": [base_B, 128, base_N, 64]},
        ]
    D_BANDS = [64, 128, 256, 512, 640, 768]   # head-dim: single/multi-tile + 640/768
    S_BANDS = [64, 128, 192, 256, 384, 512]   # seqlen: tile multiples + tails
    B_BANDS = [1, 2, 4]
    N_BANDS = [8, 16, 32]
    bands: list[dict] = []
    for d in D_BANDS:                          # D sweep (key gap — incl 640/768)
        bands.append({"name": f"attn_D{d}", "shape": [base_B, base_S, base_N, d]})
    for s in S_BANDS:                          # S sweep (seqlen-tile bands)
        bands.append({"name": f"attn_S{s}", "shape": [base_B, s, base_N, base_D]})
    for b in B_BANDS:                          # B sweep (batch)
        bands.append({"name": f"attn_B{b}", "shape": [b, base_S, base_N, base_D]})
    for n in N_BANDS:                          # N sweep (head_num)
        bands.append({"name": f"attn_N{n}", "shape": [base_B, base_S, n, base_D]})
    bands += [                                 # representative cross-band combinations
        {"name": "attn_b4_n32_s512_d64",  "shape": [4, 512, 32, 64]},   # large multicore, small-D
        {"name": "attn_b2_n16_s256_d128", "shape": [2, 256, 16, 128]},  # mid
        # High-D bands CARRY their dropout + layout config (coverage = shape × dropout).
        # Spec V2-64's high-D cases ARE the dropout cases: 640 is BNSD kp0.9, 768 is SBH
        # kp0.8. The per-plan "scalars" override is applied ONLY if the schema declares
        # that scalar (generic-safe — a schema without keep_prob/input_layout ignores it
        # and still gets the SHAPE). See generate_cases Band-A per-plan override.
        {"name": "attn_b1_n4_s192_d640",  "shape": [1, 192, 4, 640],    # high-D non-pow2 + tail-S
         "scalars": {"input_layout": "BNSD", "keep_prob": 0.9}},
        {"name": "attn_b1_n2_s384_d768",  "shape": [1, 384, 2, 768],    # max-D + 3-tile-S
         "scalars": {"input_layout": "SBH", "keep_prob": 0.8}},
    ]
    # Dropout (keep_prob<1) bands spanning low + high D — spec V2-64 has 6 kp<1 cases
    # NOT confined to high-D. These add mid-D dropout so (shape × dropout) is exercised
    # across the D range, not just at 640/768.
    bands += [
        {"name": "attn_d128_drop", "shape": [base_B, base_S, base_N, 128],
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
    bM, bN, bK = 256, 256, 256
    if not _TIER_SHAPE_EDGES[coverage_tier]:
        return [{"name": "mm_typ", "shape": [256, 256, 256]},
                {"name": "mm_small", "shape": [64, 64, 64]}]
    MNK_BANDS = [16, 64, 256, 512, 1024]       # single-tile -> multi-tile
    TAILS = [17, 255, 257]                      # tile-boundary tails: tile-1, tile, tile+1
    bands: list[dict] = []
    for m in MNK_BANDS:
        bands.append({"name": f"mm_M{m}", "shape": [m, bN, bK]})
    for n in MNK_BANDS:
        bands.append({"name": f"mm_N{n}", "shape": [bM, n, bK]})
    for k in MNK_BANDS:
        bands.append({"name": f"mm_K{k}", "shape": [bM, bN, k]})
    for t in TAILS:
        bands.append({"name": f"mm_Mtail{t}", "shape": [t, bN, bK]})
        bands.append({"name": f"mm_Ktail{t}", "shape": [bM, bN, t]})   # K tail = reduction-depth remainder
    bands += [                                  # aspect-ratio stressors
        {"name": "mm_tall_skinny", "shape": [2048, 64, 256]},   # M >> N
        {"name": "mm_short_wide",  "shape": [64, 2048, 256]},   # N >> M
        {"name": "mm_deep_k",      "shape": [256, 256, 2048]},  # large reduction depth
    ]
    if coverage_tier == "production":
        bands.append({"name": "mm_large", "shape": [4096, 4096, 4096]})  # max stress
    return bands


# Registry: op_class (lowercased) -> band-emitter(coverage_tier, dtype, rank) -> [{name, shape}].
# Generalizes — register conv, moe-routing, etc. the same way (attention + matmul below).
_OP_CLASS_SHAPE_BANDS = {
    "attention": _attention_shape_bands,
    "fa_class":  _attention_shape_bands,   # alias for the FA-class tag
    "matmul":    _matmul_shape_bands,
    "gemm":      _matmul_shape_bands,      # alias
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
        {"tag": "uniform",      "kind": "randn",  "fn": mk_randn()},
        {"tag": "large_mag",    "kind": "single", "fn": mk_uniform_dtype_large()},
        {"tag": "small_mag",    "kind": "single", "fn": mk_uniform_dtype_small()},
        {"tag": "denormal",     "kind": "single", "fn": mk_denormal()},
        {"tag": "mixed_sign",   "kind": "single", "fn": mk_mixed_sign()},
    ]

    if coverage_tier in ("sign_off", "production"):
        plans += [
            {"tag": "cancellation",    "kind": "correlated_cancel"},  # custom handling below
            {"tag": "const_near_zero", "kind": "const",  "fn": mk_const(_dtype_eps(torch.float32) * 5)},
        ]

    if _TIER_NAN_INF[coverage_tier]:
        # WARNING: NaN cases break `torch.equal` bit-exact comparison since NaN != NaN.
        # verify.py must use NaN-aware compare (equal_nan=True or same-mask + value check)
        # when production tier cases are present. Pilot/sign_off do NOT emit these.
        plans += [
            {"tag": "sparse_nan",      "kind": "sprinkle_nan"},
            {"tag": "sparse_inf",      "kind": "sprinkle_inf"},
        ]

    return plans


def _mk_tensor(gen_fn, shape: list[int], dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Generate a 1D tensor of correct product-of-shape size, then reshape."""
    n = int(math.prod(shape))
    flat = gen_fn(n, dtype, seed)
    return flat.reshape(*shape)


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
        n_positional = sum(1 for p in sig.parameters.values()
                           if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                         inspect.Parameter.POSITIONAL_OR_KEYWORD))
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


def _build_inputs(tensor_inputs: list[dict], base_shape: list[int],
                   default_gen_fn, global_dtype: torch.dtype, seed_base: int,
                   force_gen_fn_per_tensor: dict | None = None,
                   scalars: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Construct all tensor inputs for one case, honoring per-tensor shape_derive + dtype.

    force_gen_fn_per_tensor: optional dict {tensor_name: callable(n, dtype, seed) -> tensor}
        Used by special bands (NaN sprinkle, cancellation) that want a specific tensor built
        differently from the band's default. HOWEVER: tensors declaring `invariant` in their
        schema entry (permutation / positive / non_negative / index_range:<N>) are exempt
        from forced override — those invariants must hold across all bands to keep cases
        semantically valid (fixes handover T12 "correlated_cancel breaks permutation").

    scalars: current per-case scalar values (for 2-arg shape_derive / invariant="index_range:<scalar>").
    """
    inputs: dict[str, torch.Tensor] = {}
    for i, tinput in enumerate(tensor_inputs):
        name = tinput["name"]
        # V1.7 (2026-05-21): list_of_tensors primitive. Build a Python list
        # of N tensors per scalars["__list_length__<name>"] (set by the
        # post-expansion pass `_expand_list_of_tensors_lengths`). Each item
        # may use per_item_shape_derive(base_shape, item_idx, N, scalars).
        if tinput.get("kind") == "list_of_tensors":
            inputs[name] = _build_list_of_tensors_item(
                tinput, base_shape, default_gen_fn, global_dtype,
                seed_base + i * 17, scalars or {},
            )
            continue
        shape = _tensor_shape_for(tinput, base_shape, scalars)
        dtype = _tensor_dtype_for(tinput, global_dtype)
        seed = seed_base + i * 17
        invariant = tinput.get("invariant")

        # Resolve generator: (force EXCEPT when invariant prohibits) > per-tensor value_gen > invariant auto-gen > default
        forced = force_gen_fn_per_tensor and name in force_gen_fn_per_tensor
        if forced and not invariant:
            gen = force_gen_fn_per_tensor[name]
        elif "value_gen" in tinput and callable(tinput["value_gen"]):
            gen = tinput["value_gen"]
        elif invariant:
            gen = _mk_invariant_gen(invariant, tinput, scalars or {}, base_shape)
        else:
            gen = default_gen_fn

        # Non-float dtypes may not accept randn/rand from torch_generator; use int variant
        # (skip when invariant gen already handles dtype)
        if dtype in (torch.int8, torch.int32, torch.int64) and gen is default_gen_fn:
            gen = _mk_int_uniform_gen(tinput.get("int_range", (0, 1000)))
        if dtype == torch.bool and gen is default_gen_fn:
            gen = _mk_bool_uniform_gen()

        inputs[name] = _mk_tensor(gen, shape, dtype, seed)
    return inputs


def _build_list_of_tensors_item(
    tinput: dict, base_shape: list[int], default_gen_fn,
    global_dtype: torch.dtype, seed_base: int, scalars: dict[str, Any],
) -> list[torch.Tensor]:
    """V1.7 (2026-05-21): build the per-case list of tensors for a
    `kind="list_of_tensors"` input. Used by torch.cat / torch.stack / similar
    variadic-list ops.

    The current list length is read from scalars[`__list_length__<name>`],
    set by `_expand_list_of_tensors_lengths` during the post-expansion pass.
    If absent (single-length case or non-expanded usage), falls back to the
    first entry of `tinput["list_length_plan"]`.

    Each list item uses:
      - `per_item_shape_derive(base_shape, item_idx, list_length[, scalars])`
        if present, else `base_shape` (uniform list).
      - The tensor's global dtype + the default generator (same as a normal
        single-tensor input). Per-item dtype/invariant overrides are NOT
        supported in V1.7 (future work; YAGNI until a real op needs it).

    Returns: list[torch.Tensor] of length N.
    """
    import inspect

    name = tinput["name"]
    list_length_plan = tinput.get("list_length_plan")
    if not list_length_plan:
        raise ValueError(
            f"tensor_input {name!r} declared kind=list_of_tensors but "
            f"missing list_length_plan (need non-empty list[int])"
        )
    # Per-case length (post-expansion sets the per-case value; otherwise default
    # to first plan value so a single-length-test sweep still works).
    n = int(scalars.get(f"__list_length__{name}", list_length_plan[0]))
    if n <= 0:
        raise ValueError(
            f"tensor_input {name!r}: per-case list_length must be > 0, got {n}"
        )

    derive = tinput.get("per_item_shape_derive")
    dtype = _tensor_dtype_for(tinput, global_dtype)

    items: list[torch.Tensor] = []
    for item_idx in range(n):
        if derive is not None:
            try:
                n_params = len(inspect.signature(derive).parameters)
            except (TypeError, ValueError):
                n_params = 3
            if n_params >= 4:
                shape = derive(base_shape, item_idx, n, scalars)
            else:
                shape = derive(base_shape, item_idx, n)
            if not isinstance(shape, (list, tuple)) or not all(isinstance(d, int) for d in shape):
                raise ValueError(
                    f"per_item_shape_derive for {name!r} item {item_idx}/{n} "
                    f"returned {shape!r}; expected list[int]"
                )
            shape = list(shape)
        else:
            shape = list(base_shape)

        gen = default_gen_fn
        if dtype in (torch.int8, torch.int32, torch.int64) and gen is default_gen_fn:
            gen = _mk_int_uniform_gen(tinput.get("int_range", (0, 1000)))
        if dtype == torch.bool and gen is default_gen_fn:
            gen = _mk_bool_uniform_gen()
        # Per-item seed: derive deterministically from seed_base + item_idx.
        items.append(_mk_tensor(gen, shape, dtype, seed_base + item_idx * 23))
    return items


def _mk_invariant_gen(invariant: str, tinput: dict, scalars: dict[str, Any],
                       base_shape: list[int] | None = None):
    """Generator factory for tensors with declared invariants.

    invariant ∈ {
        "permutation", "positive", "non_negative",
        "index_range:<spec>",
        "index_range_dim:<scalar>",      # NEW V3.7.5 (2026-05-02): scatter/gather/index ops
    }
    where <spec> is a scalar input name (resolved via scalars dict) or integer literal.

    `index_range_dim:<scalar>` resolves the upper bound dynamically per-case as
    `base_shape[normalize_dim(scalars[<scalar>], rank)]`. This is the canonical
    invariant for ops where index values must be valid offsets along a runtime-
    selected axis (torch.scatter, torch.gather, torch.index_select, torch.scatter_add,
    torch.gather along configurable dim). Without this, scalar-shaped ops like
    21_Scatter (where dim attr varies per-case and index.shape[dim] != x.shape[dim])
    cannot express "indices must be in [0, x.shape[dim])" cleanly. base_shape is
    the per-case base_shape passed by _build_inputs.
    """
    if invariant == "permutation":
        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed + 31337)
            return torch.randperm(n, generator=g).to(dt)
        return _fn
    if invariant == "non_negative":
        lo, hi = tinput.get("int_range", (0, 1000))

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randint(max(0, lo), max(1, hi), (n,), generator=g, dtype=dt)
        return _fn
    if invariant == "positive":
        lo, hi = tinput.get("int_range", (1, 1000))
        lo = max(1, lo)

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randint(lo, max(lo + 1, hi), (n,), generator=g, dtype=dt)
        return _fn
    if invariant.startswith("index_range_dim:"):
        # V3.7.5 (2026-05-02): scalar-shaped ops where index.shape[dim] != x.shape[dim]
        # but index values must lie in [0, x.shape[dim]). Resolves upper bound from
        # per-case base_shape + scalar value.
        scalar_name = invariant.split(":", 1)[1]
        if base_shape is None or len(base_shape) == 0:
            raise ValueError(
                f"invariant={invariant!r} requires base_shape but caller passed None/empty. "
                f"_build_inputs must pass base_shape through to _mk_invariant_gen."
            )
        rank = len(base_shape)
        dim_raw = scalars.get(scalar_name)
        if dim_raw is None:
            raise ValueError(
                f"invariant={invariant!r} refers to scalar {scalar_name!r} which is "
                f"absent from per-case scalars {sorted(scalars.keys())!r}. "
                f"Make sure scalar {scalar_name!r} is declared in SCHEMA's scalar_inputs."
            )
        # Normalize negative dim to [0, rank-1]
        dim_norm = int(dim_raw) if int(dim_raw) >= 0 else rank + int(dim_raw)
        if dim_norm < 0 or dim_norm >= rank:
            raise ValueError(
                f"invariant={invariant!r}: scalar {scalar_name!r}={dim_raw!r} "
                f"normalizes to {dim_norm} which is out of bounds for rank={rank}"
            )
        upper = int(base_shape[dim_norm])
        if upper <= 0:
            raise ValueError(
                f"invariant={invariant!r}: base_shape[{dim_norm}]={upper} must be positive"
            )
        allow_dup = bool(tinput.get("allow_dup_indices", True))  # scatter typically allows dup

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            if not allow_dup and n <= upper:
                return torch.randperm(upper, generator=g)[:n].to(dt)
            return torch.randint(0, upper, (n,), generator=g, dtype=dt)
        return _fn
    if invariant.startswith("index_range:"):
        # V3.2 (2026-04-24): unique-by-default when feasible (n <= upper).
        # Avoids generating duplicate indices that would race the reference
        # CANN scatter (op#12 cold-start lesson). Opt-in to duplicates via
        # `tensor_inputs[i]["allow_dup_indices"] = True` for ops that
        # specifically test scatter race semantics.
        spec = invariant.split(":", 1)[1]
        # Resolve: integer literal or scalar input name
        try:
            upper = int(spec)
        except ValueError as exc:
            upper = int(scalars.get(spec, 0))
            if upper <= 0:
                raise ValueError(
                    f"invariant={invariant!r} refers to scalar {spec!r} which is "
                    f"{scalars.get(spec)!r} — must be positive integer. "
                    f"Make sure scalar {spec!r} has a `derive` or `default` that resolves "
                    f"before this tensor is built."
                ) from exc
        allow_dup = bool(tinput.get("allow_dup_indices", False))

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            if not allow_dup and n <= upper:
                # Unique sample: randperm + slice
                return torch.randperm(upper, generator=g)[:n].to(dt)
            # n > upper or duplicates explicitly allowed: fall back to randint
            return torch.randint(0, upper, (n,), generator=g, dtype=dt)
        return _fn
    raise ValueError(f"unknown tensor invariant {invariant!r}")


def _mk_int_uniform_gen(int_range: tuple[int, int]):
    lo, hi = int_range

    def _f(n, dt, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(lo, hi, (n,), generator=g, dtype=dt)
    return _f


def _mk_bool_uniform_gen():
    def _f(n, _dt, seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.rand(n, generator=g) > 0.5)
    return _f


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


def _scalar_only_cases(schema: dict[str, Any], coverage_tier: str) -> list[dict]:
    """Scalar-only schema case generation (tensor_inputs empty).

    Coverage = baseline (all defaults) + per-scalar probe variants + at
    sign_off/production tier, also emit 1-2 "cross-probe" combinations
    (two scalars varied simultaneously) for interaction coverage.

    Each case's `inputs` dict is purely scalars. `shape` is omitted
    (set to [] — consumers should treat as "shape derived from scalars
    per op semantics"; edge_runner computes it at Model.forward time).
    """
    scalar_inputs = schema.get("scalar_inputs", [])

    def _coerce_scalar(val, dtype_str):
        if dtype_str in ("int", "int32", "int64"):
            return int(val)
        if dtype_str == "bool":
            return bool(val)
        if dtype_str in ("str", "string"):
            # Pass-through for enum/string scalars (e.g. rotary_mode: "half"|"interleave").
            # Added 2026-04-22 for op#1 RotaryMul where mode is a string enum.
            return str(val)
        return float(val)

    defaults = {s["name"]: _coerce_scalar(s.get("default", 0), s.get("dtype", "float"))
                for s in scalar_inputs}

    cases = []
    idx = 0

    # Baseline: all defaults.
    cases.append({
        "idx": idx, "name": "scalar_only_baseline",
        "shape": [], "inputs": dict(defaults),
        "meta": {"band": "S_baseline", "note": "all defaults"},
    })
    idx += 1

    # Per-scalar probe variants (like Band C in tensor mode).
    # 2026-05-18 Gap #3 unblock: probe_values may be a CALLABLE
    # (lambda rank: [...] or lambda base_shape: [...]) so the probe
    # set can be rank/shape-dependent — needed for ops like 9_TopK
    # whose valid `dim` range is {-rank..rank-1}.
    rank_for_probe = schema.get("rank")

    def _resolve_probe_values(s_spec):
        pv = s_spec.get("probe_values", [])
        if not callable(pv):
            return list(pv)
        import inspect as _inspect
        try:
            sig = _inspect.signature(pv)
            param_names = [p.name for p in sig.parameters.values()]
        except (TypeError, ValueError):
            param_names = []
        # Pass rank by default; if caller wants base_shape, use that
        # parameter name explicitly.
        if param_names and param_names[0] in ("base_shape", "shape"):
            return list(pv(_base_shape_for_rank(rank_for_probe or 1)))
        return list(pv(rank_for_probe if rank_for_probe is not None else 1))

    for s_spec in scalar_inputs:
        name = s_spec["name"]
        dtype_str = s_spec.get("dtype", "float")
        for probe_v in _resolve_probe_values(s_spec):
            if probe_v == s_spec.get("default"):
                continue
            inputs = dict(defaults)
            inputs[name] = _coerce_scalar(probe_v, dtype_str)
            cases.append({
                "idx": idx, "name": f"scalar_{name}_{probe_v}",
                "shape": [], "inputs": inputs,
                "meta": {"band": "S_probe", "scalar": name, "value": probe_v},
            })
            idx += 1

    # Sign_off + production: add cross-probe pairs (two scalars varied simultaneously).
    # Targets interaction bugs (e.g. past_kv_length > 0 AND sliding_window < seq_length).
    if coverage_tier in ("sign_off", "production"):
        probed = [s for s in scalar_inputs if s.get("probe_values")]
        if len(probed) >= 2:
            # Take first probe_value from each of the first two probed scalars.
            s1, s2 = probed[0], probed[1]
            s1_values = _resolve_probe_values(s1)
            s2_values = _resolve_probe_values(s2)
            v1 = next((v for v in s1_values if v != s1.get("default")), None)
            v2 = next((v for v in s2_values if v != s2.get("default")), None)
            if v1 is not None and v2 is not None:
                inputs = dict(defaults)
                inputs[s1["name"]] = _coerce_scalar(v1, s1.get("dtype", "float"))
                inputs[s2["name"]] = _coerce_scalar(v2, s2.get("dtype", "float"))
                cases.append({
                    "idx": idx,
                    "name": f"scalar_cross_{s1['name']}_{v1}_{s2['name']}_{v2}",
                    "shape": [], "inputs": inputs,
                    "meta": {"band": "S_cross", "scalars": [s1["name"], s2["name"]],
                             "values": [v1, v2], "note": "interaction probe"},
                })
                idx += 1

    return cases


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
        raise AssertionError(f"unknown tier: {coverage_tier}")

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

    shape_plans = _shape_plan(coverage_tier, dtype, rank)
    # Op-class SEMANTIC bands (2026-06-08, owner case_gen-coverage directive): prepend
    # domain-semantic shapes (attention seqlen/head-dim incl D640/768, etc) for ops with a
    # registered `schema["op_class"]`. OPT-IN — ops without op_class get [] → byte-identical.
    op_class_bands = _op_class_shape_bands(schema.get("op_class"), coverage_tier, dtype, rank)
    if op_class_bands:
        shape_plans = list(op_class_bands) + list(shape_plans)
    # Apply op-specific base shape filter if present (e.g. op#11 needs last-dim even).
    #
    # IMPORTANT: base_shape_filter receives VARIABLE-RANK shapes. A fixed-rank
    # filter must compare the shape length before unpacking dimensions. For a
    # rank-4 operation, accept length four first, then unpack B, H, new_seq, D
    # and apply constraints such as 16-byte divisibility and positive sequence
    # length. Otherwise a 1D shape raises a dimension-unpacking ValueError. See
    # workspace/kvcacheupdatewithropebackward/input_gen.py for the rank-4 fused
    # scatter-and-reduce example.
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
    dist_plans  = _distribution_plan(coverage_tier)
    n_seeds     = _TIER_SEEDS_PER_DIST[coverage_tier]
    base_shape  = _base_shape_for_rank(rank)
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
                               scalars=scalars_here)
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
            if kind in ("const", "single", "randn"):
                fn = dist["fn"]
                inputs = _build_inputs(tensor_inputs, base_shape, fn, dtype,
                                         seed_base=2000 + seed_offset * 37,
                                         scalars=base_scalars)
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
                                               scalars=base_scalars)
                        inputs.update(base_scalars)
                        # Flip scalar (safe because invariant check above)
                        s_val = base_scalars[first_scalar_spec["name"]]
                        inputs[first_scalar_spec["name"]] = -s_val
                        cases.append({
                            "idx": idx, "name": f"dist_cancellation_seed{seed_offset}",
                            "shape": base_shape, "inputs": inputs,
                            "meta": {"distribution": "cancellation", "seed_offset": seed_offset,
                                     "note": f"invariant-preserving: scalar {first_scalar_spec['name']} flipped {s_val}→{-s_val}"},
                        })
                        idx += 1
                    else:
                        # Classic path: t0 = scalar * t1 + tiny noise, then flip scalar sign
                        t0_shape = _tensor_shape_for(t0_spec, base_shape, base_scalars)
                        t1_shape = _tensor_shape_for(t1_spec, base_shape, base_scalars)
                        # Require matching shape for the correlation pattern to be meaningful
                        if t0_shape != t1_shape:
                            continue  # skip — shapes differ, cancellation doesn't apply
                        second = torch.randn(int(math.prod(t1_shape)), generator=g1, dtype=dtype).reshape(*t1_shape)
                        noise = torch.randn(int(math.prod(t0_shape)), generator=g2, dtype=dtype).reshape(*t0_shape) * _dtype_eps(dtype) * 10
                        s_val = base_scalars[first_scalar_spec["name"]]
                        first = (s_val * second + noise).to(dtype)
                        inputs = {tensor_names[0]: first, tensor_names[1]: second}
                        # Build remaining tensors honoring invariants
                        for i in range(2, len(tensor_names)):
                            t_spec = tensor_inputs[i]
                            t_shape = _tensor_shape_for(t_spec, base_shape, base_scalars)
                            t_dtype = _tensor_dtype_for(t_spec, dtype)
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
                                def _cancellation_randn(n, dt, seed):
                                    generator = torch.Generator().manual_seed(seed)
                                    return torch.randn(n, generator=generator, dtype=dt)

                                gen = _cancellation_randn
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
                t0_shape = _tensor_shape_for(tensor_inputs[0], base_shape, base_scalars)
                base = torch.randn(int(math.prod(t0_shape)), generator=g, dtype=dtype).reshape(*t0_shape)
                n = int(math.prod(t0_shape))
                sprinkle_idx = torch.randperm(n, generator=torch.Generator().manual_seed(4500))[: max(1, n // 10)]
                bad = base.clone().flatten()
                bad[sprinkle_idx] = float("nan") if kind == "sprinkle_nan" else float("inf")
                inputs = {tensor_names[0]: bad.reshape(*t0_shape)}
                # Build remaining tensors honoring invariants
                for i in range(1, len(tensor_names)):
                    t_spec = tensor_inputs[i]
                    t_shape = _tensor_shape_for(t_spec, base_shape, base_scalars)
                    t_dtype = _tensor_dtype_for(t_spec, dtype)
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
                        def _sprinkle_randn(n_el, dt, seed):
                            generator = torch.Generator().manual_seed(seed)
                            return torch.randn(n_el, generator=generator, dtype=dt)

                        gen = _sprinkle_randn
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
                                   scalars=probed_scalars)
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
        )

    return cases


def _expand_list_of_tensors_lengths(
    cases: list[dict], list_specs: list[dict], tensor_inputs: list[dict],
    schema: dict[str, Any], *, coverage_tier: str, global_dtype: torch.dtype,
) -> list[dict]:
    """V1.7 (2026-05-21): fork each case over the cartesian product of
    `list_length_plan` values across all `kind=list_of_tensors` inputs.

    For each fork (per-case x per-list-length-combo), re-materialize the
    list-of-tensors inputs at the new N (using the per-case base_shape from
    `case["shape"]` + scalars + the spec's `per_item_shape_derive`). Non-list
    inputs and scalars are preserved verbatim from the original case.

    Case count multiplier = Π_i len(spec_i.list_length_plan). For a single
    list with plan=[2,3,4] over 50 base cases → 150 cases.

    Mutates: new case `name` gets `_N<value>` suffix per list spec; `meta`
    records the per-list length value at `list_length_<name>` key.
    """
    import itertools
    if not list_specs:
        return cases

    # Cartesian product of all plans (handles multi-list ops if any exist;
    # 13_Cat has only one but future ops may have several).
    plans = [spec["list_length_plan"] for spec in list_specs]
    spec_names = [spec["name"] for spec in list_specs]
    combos = list(itertools.product(*plans))

    expanded: list[dict] = []
    next_idx = 0
    for c in cases:
        base_shape = c["shape"]
        # Scalars present in the case (everything in inputs that's not a Tensor
        # and not a list-of-Tensors and not None).
        scalars: dict[str, Any] = {
            k: v for k, v in c["inputs"].items()
            if not isinstance(v, torch.Tensor) and not isinstance(v, list) and v is not None
        }
        for combo in combos:
            # Inject per-fork list lengths into scalars dict so the per-item
            # shape derive callable + builder helper can read them.
            fork_scalars = dict(scalars)
            for name, n in zip(spec_names, combo):
                fork_scalars[f"__list_length__{name}"] = int(n)
            new_inputs = dict(c["inputs"])
            for spec, n in zip(list_specs, combo):
                # Re-materialize THIS list spec at the new length.
                # Seed deterministically from case idx + spec name + N so
                # the same (case, spec, N) triple always produces the same
                # tensors across runs.
                seed_base = (c["idx"] + 9001) * 31 + hash(spec["name"]) % 997 + int(n) * 7
                # Default generator: standard randn — matches single-tensor path.

                def _randn(num, dt, seed):
                    g = torch.Generator().manual_seed(seed)
                    return torch.randn(num, generator=g, dtype=dt)
                new_inputs[spec["name"]] = _build_list_of_tensors_item(
                    spec, base_shape, _randn, global_dtype, seed_base,
                    fork_scalars,
                )
            new_meta = dict(c.get("meta", {}))
            for name, n in zip(spec_names, combo):
                new_meta[f"list_length_{name}"] = int(n)
            length_tags = "_".join(
                f"{name}N{n}" for name, n in zip(spec_names, combo)
            )
            expanded.append({
                "idx": next_idx,
                "name": c["name"] + "_" + length_tags,
                "shape": c["shape"],
                "inputs": new_inputs,
                "meta": new_meta,
            })
            next_idx += 1
    return expanded


def _expand_optional_tensor_presence(
    cases: list[dict], optional_names: list[str]
) -> list[dict]:
    """Fork each case into 2^k variants for k optional tensors.

    Each fork sets `inputs[name]` to either the original tensor (present=True)
    or `None` (present=False). The case `name` and `meta` get a `_<name>{T|F}`
    suffix per optional tensor so downstream filters can locate the absent-path
    cases.

    Args:
        cases: list of case dicts as produced by the main band loops
        optional_names: tensor names declared `optional=True` in SCHEMA

    Returns:
        new list of cases, length = len(cases) * (2 ** len(optional_names))
    """
    if not optional_names:
        return cases
    expanded: list[dict] = []
    next_idx = 0
    # All 2^k presence patterns (k=len(optional_names))
    n_patterns = 1 << len(optional_names)
    for c in cases:
        for pattern in range(n_patterns):
            new_inputs = dict(c["inputs"])
            new_meta = dict(c.get("meta", {}))
            presence_tags = []
            for j, name in enumerate(optional_names):
                present = bool(pattern & (1 << j))
                if not present:
                    new_inputs[name] = None
                presence_tags.append(f"{name}{'T' if present else 'F'}")
                new_meta[f"optional_{name}_present"] = present
            new_case = {
                "idx": next_idx,
                "name": c["name"] + "_" + "_".join(presence_tags),
                "shape": c["shape"],
                "inputs": new_inputs,
                "meta": new_meta,
            }
            expanded.append(new_case)
            next_idx += 1
    return expanded


# ---------------- data-identity hash ----------------

def dataset_data_sha256(cases: list[dict]) -> str:
    """Version-independent hash of case tensor data.

    torch.save file bytes differ across torch versions (observed 2.11 vs 2.3 on
    BabelStream Copy pilot 2026-04-20 — file SHA differed but tensor data was
    bit-exact identical). This helper hashes ONLY the tensor data, independent
    of torch.save serialization format.

    Covers: idx + name + shape + each input (key + dtype + contiguous tensor
    bytes for tensors, or dtype + float-bits for scalars). Output-side cases
    do not apply here because the hash identifies reference inputs.

    Used by input_gen.py to stamp manifest + inputs.pt with `data_sha256`.
    Reference providers verify the same value before accepting outputs.
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

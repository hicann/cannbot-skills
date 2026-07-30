# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TEMPLATE — customize per op. Generates {inputs.pt, manifest.json} using the
schema-driven case_gen engine.

THE ENGINE HANDLES: shape coverage (degenerate / alignment / core-partition /
tile-boundary / prime / typical / 2D), distribution coverage (uniform / constants /
large/small_mag / denormal / cancellation / mixed_sign / scalar_probe / NaN/Inf
at production tier), seed variation, and cross-product.

YOUR JOB: fill the SCHEMA dict describing YOUR OP's inputs and scalars. That's it.

STEPS:
  1. Fill SCHEMA (3 fields: op_name, formula, tensor_inputs, scalar_inputs, tensor_output, rank).
  2. Pick COVERAGE_TIER: pilot (~15) / sign_off (~40) / production (~60).
  3. Run: `python3 input_gen.py` — writes inputs.pt + manifest.json.
  4. Verify the "if __name__" block's `assert SCHEMA["op_name"] != "<<<REPLACE>>>"`
     guard — this prevents accidentally running with the template's placeholder.

WORKED EXAMPLES
===============

Example 1 — Simple elementwise (BabelStream Triad, rank 1, no extensions):

```python
SCHEMA = {
    "op_name": "babelstream_triad",
    "formula": "a[i] = b[i] + scalar * c[i]",
    "tensor_inputs": [
        {"name": "b", "role": "operand"},
        {"name": "c", "role": "operand"},
    ],
    "scalar_inputs": [
        {
            "name": "scalar",
            "dtype": "float",
            "default": 0.4,
            "probe_values": [0.4, -0.4, 0.0],
        },
    ],
    "tensor_output": "a",
    "rank": 1,
}
```

Example 2 — Scalar-only HybridAttentionMaskPreparation (rank 0). Rank zero is
valid exactly when `tensor_inputs` is empty:

```python
SCHEMA = {
    "op_name": "hybrid_attention_mask",
    "formula": "bool_mask[b,h,t,s] = compute(B,H,T,S, attn_type, swa_window)",
    "tensor_inputs": [],
    "scalar_inputs": [
        {"name": "B", "dtype": "int", "default": 1, "probe_values": [1, 2, 4, 8]},
        {"name": "H", "dtype": "int", "default": 8, "probe_values": [1, 8, 16]},
        {
            "name": "attn_type",
            "dtype": "str",
            "default": "full",
            "probe_values": ["full", "swa", "causal"],
        },
    ],
    "tensor_output": "attention_mask",
    "rank": 0,
}
```

Example 3 — Fused KvCacheUpdateWithRopeBackward (rank 4) using
`shape_derive`, `value_gen`, and `base_shape_filter`. The shape-edge band
sweeps several ranks through the same filter, so a fixed-rank operation must
reject a nonmatching length before unpacking the dimensions. `key_states`
omits `shape_derive` and therefore uses the base shape.

```python
import torch

def _grad_cache_shape(base):
    B, H, new_seq, D = base
    return [B, H, max(new_seq * 2, new_seq + 1), D]

def _cos_sin_shape(base):
    B, H, new_seq, D = base
    return [B, new_seq, D]

def _cache_position_shape(base):
    B, H, new_seq, D = base
    return [new_seq]

def _cache_position_gen(n, dtype, seed):
    g = torch.Generator().manual_seed(seed + 31337)
    return torch.randperm(max(2 * n, n + 1), generator=g)[:n].to(torch.int64)

def _base_valid(base):
    if len(base) != 4:
        return False
    B, H, new_seq, D = base
    return D % 16 == 0 and D >= 4 and D % 2 == 0

SCHEMA = {
    "op_name": "KvCacheUpdateWithRopeBackward",
    "formula": "...",
    "tensor_inputs": [
        {
            "name": "grad_key_cache",
            "role": "cache_grad",
            "shape_derive": _grad_cache_shape,
        },
        {"name": "key_states", "role": "operand"},
        {"name": "cos", "role": "operand", "shape_derive": _cos_sin_shape},
        {
            "name": "cache_position",
            "role": "index",
            "dtype": torch.int64,
            "shape_derive": _cache_position_shape,
            "value_gen": _cache_position_gen,
        },
    ],
    "scalar_inputs": [],
    "tensor_output": "grad_key_states",
    "rank": 4,
    "base_shape_filter": _base_valid,
}
```

Example 4 — Layout-string sweep plus multiple outputs. An operation whose
input rank or shape is selected by a string layout is representable with
`make_layout_dispatch`; do not file `.workflow_exception_O2_5` merely because
the layout varies. A tuple-returning operation declares `tensor_outputs`
instead of `tensor_output`, and the verifier compares each declared name.

```python
from case_gen import make_layout_dispatch

_qkv = make_layout_dispatch({
    "BSH": lambda s: [s[0], s[1], s[2] * s[3]],
    "SBH": lambda s: [s[1], s[0], s[2] * s[3]],
    "BSND": lambda s: [s[0], s[1], s[2], s[3]],
    "BNSD": lambda s: [s[0], s[2], s[1], s[3]],
})
SCHEMA = {
    "op_name": "fusion_attention",
    "formula": "softmax(Q@K^T*scale)@V",
    "tensor_inputs": [
        {"name": name, "role": "operand", "shape_derive": _qkv}
        for name in ("query", "key", "value")
    ],
    "scalar_inputs": [
        {
            "name": "input_layout",
            "dtype": "str",
            "default": "BSH",
            "probe_values": ["BSH", "SBH", "BSND", "BNSD"],
        },
        {
            "name": "head_num",
            "dtype": "int",
            "default": 1,
            "derive": lambda base: int(base[2]),
        },
    ],
    "tensor_outputs": ["attention_out", "softmax_max", "softmax_sum"],
    "rank": 4,
    "base_shape_filter": lambda base: len(base) == 4 and base[3] % 16 == 0,
}
```

The FusionAttention form above was verified with nine pilot cases spanning all
four layouts. Canonical live examples are
`workspace/embeddingwithinitiallayernormbackward/input_gen.py`, which shows a
rank-3 indexed scatter with `shape_derive` and `value_gen`, and
`workspace/kvcacheupdatewithropebackward/input_gen.py`, which shows a rank-4
fused scatter/reduce with an interdependent maximum sequence length.

Each `tensor_inputs` entry has required `name` and optional `role`. Fused
entries may also provide `shape_derive`, a per-tensor `dtype`, and `value_gen`.
Each `scalar_inputs` entry may provide `name`, `dtype`, `default`, and
`probe_values`. For multiple outputs replace `tensor_output` with a
`tensor_outputs` list. Rank may be zero through four; zero is reserved for an
empty tensor-input list. A `base_shape_filter` callable can reject invalid
candidate shapes and must check the rank before unpacking dimensions.

When a fresh port cannot be represented by the schema, for example because of
an unsupported indirect multi-tensor dependency, record the schema gap in
`workspace/<op>/.workflow_exception_O2_5` and provide or reuse a
provenance-recorded deterministic *input recipe*.  That exception waives only
the standard ``SCHEMA``/``case_gen`` recipe form: execute the selected recipe in
the current run, then execute the staged arch22 source operator on a source NPU
in the current run and emit the normal capture provenance.  Cached, archived,
or committed source-NPU outputs never satisfy the truth contract.
"""
from __future__ import annotations
import hashlib
import json
import pathlib

import torch

# case_gen engine (lives next to this template after `cp` per README)
from case_gen import case_data_sha256, dataset_data_sha256, generate_cases


# YOU MUST EDIT THIS SCHEMA — placeholder values will fail the runtime guard.
SCHEMA = {
    "op_name":       "<<<REPLACE-PER-OP-e.g.-babelstream_triad>>>",
    "formula":       "<<<pseudo-code-e.g.-a[i] = b[i] + scalar * c[i]>>>",
    "tensor_inputs": [],
    "scalar_inputs": [],
    "tensor_output": "<<<REPLACE-output-tensor-name-e.g.-a>>>",
    "rank": 1,
}

COVERAGE_TIER = "sign_off"   # one of: pilot / sign_off / production
DTYPE = torch.float32        # or torch.float16, torch.bfloat16, torch.float64

# COVERAGE_SEED (added 2026-05-20, edge-data design S2): records the seed
# value under which this op's edge data was generated. Currently case_gen
# uses internal seed constants per-distribution; this field is the
# provenance record for the manifest (S5 finalize-gate hash check uses it).
# Future S3+: case_gen.generate_cases() will accept seed_base=COVERAGE_SEED
# so different seed values produce different (but reproducible) datasets.
COVERAGE_SEED = 0


# -----------------------------------------------------------------------------
# Below this line is boilerplate — no per-op edits needed.

def sha256_of_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _stringify_tensor_stat(t):
    if isinstance(t, torch.Tensor):
        return {"min": float(t.min()), "max": float(t.max()),
                "mean": float(t.mean()) if t.dtype.is_floating_point else None,
                "std": float(t.std()) if t.dtype.is_floating_point and t.numel() > 1 else None,
                "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    return {"value": t}


def _output_naming_errors(schema) -> list:
    """Guard check for output naming: single `tensor_output` (str, back-compat) OR
    multi-output `tensor_outputs` (list of names, for tuple-returning ops — e.g.
    attention returns attention_out/softmax_max/softmax_sum). At least ONE must be
    validly set (non-empty, no REPLACE placeholder); the verifier compares per-name.
    Returns a list of guard-error strings (empty = OK). Extracted for unit-test."""
    tout = schema.get("tensor_output", "")
    touts = schema.get("tensor_outputs") or []
    single_ok = bool(tout) and "REPLACE" not in tout
    multi_ok = bool(touts) and all(isinstance(n, str) and n and "REPLACE" not in n for n in touts)
    if single_ok or multi_ok:
        return []
    return [f"tensor_output={schema.get('tensor_output')!r} / tensor_outputs={touts!r} "
            "(set ONE: single output name, or non-empty list of names for multi-output ops)"]


def main():
    # Runtime guard — prevents accidentally committing / running with template placeholder.
    # Check all user-editable fields, not just the obvious ones.
    guard_errors = []
    if "REPLACE" in SCHEMA.get("op_name", "") or not SCHEMA.get("op_name"):
        guard_errors.append(f"op_name={SCHEMA.get('op_name')!r}")
    if "REPLACE" in SCHEMA.get("formula", "") or not SCHEMA.get("formula"):
        guard_errors.append(f"formula={SCHEMA.get('formula')!r}")
    guard_errors += _output_naming_errors(SCHEMA)
    # Scalar-only ops (tensor_inputs empty) are allowed iff scalar_inputs has probe_values
    # — case_gen's scalar-only mode (added 2026-04-22) produces cases by enumerating
    # scalar probe combinations. See case_gen._scalar_only_cases.
    if not SCHEMA.get("tensor_inputs"):
        scalars = SCHEMA.get("scalar_inputs", [])
        if not any(s.get("probe_values") for s in scalars):
            guard_errors.append(
                "tensor_inputs=<empty list> AND no scalar_input has probe_values — "
                "scalar-only mode requires at least one scalar with probe_values to vary"
            )
    for t in SCHEMA.get("tensor_inputs", []):
        if not isinstance(t, dict) or not t.get("name"):
            guard_errors.append(f"tensor_input missing 'name': {t!r}")
    # Scalar-only ops may set rank=0 (no tensor rank). Allow 0 only when tensor_inputs is empty.
    # Ranks 1-4 supported by case_gen._base_shape_for_rank.
    allowed_ranks = (0, 1, 2, 3, 4) if not SCHEMA.get("tensor_inputs") else (1, 2, 3, 4)
    if SCHEMA.get("rank", 0) not in allowed_ranks:
        guard_errors.append(f"rank={SCHEMA.get('rank')} not in {allowed_ranks}")
    if guard_errors:
        raise RuntimeError(
            "input_gen.py was NOT customized per op. Fill the SCHEMA dict at top of file. "
            "Errors:\n  - " + "\n  - ".join(guard_errors)
        )
    if COVERAGE_TIER not in ("pilot", "sign_off", "production"):
        raise RuntimeError(f"COVERAGE_TIER={COVERAGE_TIER!r} must be one of pilot/sign_off/production")

    out_dir = pathlib.Path(__file__).parent
    cases = generate_cases(SCHEMA, coverage_tier=COVERAGE_TIER, dtype=DTYPE)

    data_sha = dataset_data_sha256(cases)

    inputs_pt = out_dir / "inputs.pt"
    torch.save({
        "dtype": str(DTYPE).replace("torch.", ""),
        "op": SCHEMA["op_name"],
        "schema": SCHEMA,
        "coverage_tier": COVERAGE_TIER,
        "data_sha256": data_sha,
        "cases": cases,
    }, inputs_pt)

    # Manifest schema v2 (2026-05-20, edge-data design S2):
    # - top-level: `seed` (provenance for S5 finalize gate)
    # - per-case: `input_sha256` (load-bearing — S5 finalize-gate hash check)
    # - top-level: `manifest_schema_version` (forward-compat;
    #   readers compare to assert structure)
    # - top-level: `case_gen_version` (provenance — pins which version of
    #   case_gen generated the dataset so S5 can detect upstream drift)
    # Existing readers ignore unknown fields; this is a backwards-compatible
    # additive change.
    manifest = {
        "manifest_schema_version": 2,
        "op": SCHEMA["op_name"],
        "formula": SCHEMA["formula"],
        "dtype": str(DTYPE).replace("torch.", ""),
        "coverage_tier": COVERAGE_TIER,
        "seed": COVERAGE_SEED,
        "case_gen_version": "V1.6.B",  # bump when case_gen changes generation logic
        "data_sha256": data_sha,
        "inputs_pt_sha256": sha256_of_file(inputs_pt),
        "n_cases": len(cases),
        "cases": [
            {"idx": c["idx"], "name": c["name"], "shape": c["shape"],
             "input_sha256": case_data_sha256(c),  # NEW (S2): per-case hash
             "input_stats": {k: _stringify_tensor_stat(v) for k, v in c["inputs"].items()},
             "meta": c.get("meta", {})}
            for c in cases
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {inputs_pt} ({inputs_pt.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote manifest.json (tier={COVERAGE_TIER}, {len(cases)} cases)")
    print(f"data_sha256 (canonical, cross-torch-version):  {data_sha}")
    print(f"inputs_pt_sha256 (file-byte, version-sensitive): {manifest['inputs_pt_sha256']}")


if __name__ == "__main__":
    main()

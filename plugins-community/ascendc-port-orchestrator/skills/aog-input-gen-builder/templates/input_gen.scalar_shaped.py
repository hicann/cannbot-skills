# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""input_gen.py — SCALAR-SHAPED template (tensor shapes depend on scalar values).

Emitted by /aog-input-gen-builder for ops where one or more tensor shapes depend
on the current per-case scalar values, AND those scalars are themselves
derived from base_shape. Examples:
  - op#3 AdvanceStepFlashattn: [num_seqs] + [num_queries, 1] + [num_seqs, max_blocks]
  - vLLM-style schedule ops with count-scalars
  - Variable-batch ops where B/Q/H are scalars not base dims

Reference computation for int-only ops (all tensor dtypes int*) is handled
directly in this file (CPU-bit-exact); no A5 edge_runner needed. For ops
with float dtypes, this template emits a stub `compute_reference.py` TODO
at the bottom — orchestrator writes the A5 edge_runner.py separately.

Template filling guide:

Keep the rank check first in base_shape_filter because case_gen explores
multiple ranks. Add operator constraints only after that check. For the
AdvanceStepFlashattn pattern, num_seqs comes from the first base dimension,
num_queries is half of that dimension with a minimum of one, and
max_blocks_per_seq is derived from sequence length, block size, and safety
padding.

Tensor shapes may depend on both the base shape and derived scalar values:

    "tensor_inputs": [
        {"name": "input_tokens", "dtype": torch.int64,
         "shape_derive": lambda shape: list(shape)},
        {"name": "sampled_token_ids", "dtype": torch.int64,
         "shape_derive": lambda shape, scalars: [scalars["num_queries"], 1]},
        {"name": "seq_lens", "dtype": torch.int64,
         "shape_derive": lambda shape: list(shape),
         "invariant": "positive", "int_range": (1, 100)},
        {"name": "block_tables", "dtype": torch.int64,
         "shape_derive": lambda shape, scalars:
             [shape[0], scalars["max_blocks_per_seq"]],
         "int_range": (0, 200)},
    ],

Use derive for scalars computed from the base shape; it is mutually exclusive
with probe_values:

    "scalar_inputs": [
        {"name": "num_seqs", "dtype": "int", "invariant": "positive",
         "derive": lambda shape: shape[0]},
        {"name": "num_queries", "dtype": "int", "invariant": "positive",
         "derive": lambda shape: max(1, shape[0] // 2)},
        {"name": "block_size", "dtype": "int", "default": 8,
         "invariant": "positive", "probe_values": [1, 4, 16, 64]},
        {"name": "max_blocks_per_seq", "dtype": "int",
         "invariant": "positive",
         "derive": lambda shape: max(2, shape[0] // 4 + 1)},
    ],
"""
from __future__ import annotations
import json
import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "scripts" / "reference_provider"))
from case_gen import dataset_data_sha256, generate_cases  # noqa: E402


def _base_shape_valid(base):
    """Accept positive shapes of the configured rank before op-specific checks."""
    if len(base) != int("<<<RANK>>>"):
        return False
    return all(d >= 1 for d in base)


SCHEMA = {
    "op_name": "<<<OP_NAME>>>",
    "formula": "<<<FORMULA>>>",
    "tensor_inputs": [],
    "scalar_inputs": [],
    "tensor_output": "<<<OUTPUT_NAME>>>",   # null / "None" for in-place ops
    "rank": int("<<<RANK>>>"),
    "base_shape_filter": _base_shape_valid,
}

COVERAGE_TIER = "<<<COVERAGE_TIER>>>"
DTYPE = getattr(torch, "<<<DTYPE>>>".rsplit(".", 1)[-1])  # global default; per-tensor override via `dtype`


def main():
    out_dir = pathlib.Path(__file__).parent

    if "<<<" in str(SCHEMA) or "<<<" in COVERAGE_TIER:
        raise RuntimeError(
            "input_gen.py has unfilled <<<>>> placeholders — /aog-input-gen-builder "
            "skill should have filled these. If running standalone, fill manually."
        )

    cases = generate_cases(SCHEMA, coverage_tier=COVERAGE_TIER, dtype=DTYPE)

    payload = {
        "op": SCHEMA["op_name"],
        "formula": SCHEMA["formula"],
        "dtype": str(DTYPE).replace("torch.", ""),
        "coverage_tier": COVERAGE_TIER,
        "schema": _schema_to_json(SCHEMA),
        "cases": cases,
        "data_sha256": dataset_data_sha256(cases),
    }
    torch.save(payload, out_dir / "edge_inputs.pt")

    manifest = {
        "op": SCHEMA["op_name"],
        "formula": SCHEMA["formula"],
        "dtype": str(DTYPE).replace("torch.", ""),
        "coverage_tier": COVERAGE_TIER,
        "n_cases": len(cases),
        "data_sha256": payload["data_sha256"],
        "cases": [c.get("name", f"case_{i}") for i, c in enumerate(cases)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote edge_inputs.pt ({len(cases)} cases) + manifest.json")


def _schema_to_json(schema):
    out = {k: v for k, v in schema.items() if not callable(v)}
    if "tensor_inputs" in out:
        out["tensor_inputs"] = [
            {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in t.items() if not callable(v)}
            for t in schema["tensor_inputs"]
        ]
    if "scalar_inputs" in out:
        out["scalar_inputs"] = [
            {k: v for k, v in s.items() if not callable(v)}
            for s in schema["scalar_inputs"]
        ]
    return out


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""input_gen.py — SIMPLE template (1-3 tensors, same shape, minimal scalars).

Emitted by /aog-input-gen-builder for ops like pointwise arithmetic, reductions
with single input, or element-wise transforms.

USAGE: /aog-input-gen-builder fills the SCHEMA + COVERAGE_TIER + DTYPE slots,
then this file is written to workspace/{op}/input_gen.py.

A typical filled schema declares an operand, an optional per-tensor dtype
override for index data, and a probed scalar:

    "tensor_inputs": [
        {"name": "x", "role": "operand"},
        {"name": "idx", "dtype": torch.int64, "role": "index"},
    ],
    "scalar_inputs": [
        {"name": "alpha", "dtype": "float", "default": 0.5,
         "probe_values": [0.0, 0.5, 1.0, -0.5]},
    ],

Remove entries that the operator does not accept, and set rank to a value from
one through four that matches its tensor contract.
"""
from __future__ import annotations
import json
import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "scripts" / "reference_provider"))
from case_gen import dataset_data_sha256, generate_cases, case_data_sha256  # noqa: E402


def _stringify_tensor_stat(t):
    """Bug B (blue 2026-06-01): manifest cases[].input_stats must match the
    canonical input_gen.template.py schema so the reference provider's
    kwargs_from_workspace_manifest_case0 can build real-shape probe inputs.
    Mirrors input_gen.template.py:_stringify_tensor_stat."""
    if isinstance(t, torch.Tensor):
        return {"min": float(t.min()), "max": float(t.max()),
                "mean": float(t.mean()) if t.dtype.is_floating_point else None,
                "std": float(t.std()) if t.dtype.is_floating_point and t.numel() > 1 else None,
                "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    return {"value": t}


SCHEMA = {
    "op_name": "<<<OP_NAME>>>",
    "formula": "<<<FORMULA>>>",
    "tensor_inputs": [],
    "scalar_inputs": [],
    "tensor_output": "<<<OUTPUT_NAME>>>",
    "rank": 1,  # adjust 1/2/3/4 to match op
}

COVERAGE_TIER = "<<<COVERAGE_TIER>>>"   # pilot / sign_off / production
DTYPE = torch.float32                    # global default; per-tensor override via `dtype`


def main():
    out_dir = pathlib.Path(__file__).parent

    # Guard: catch leftover placeholders before generating noise cases.
    if "<<<" in str(SCHEMA) or "<<<" in COVERAGE_TIER:
        raise RuntimeError(
            "input_gen.py has unfilled <<<>>> placeholders — /aog-input-gen-builder "
            "skill should have filled these. If running standalone, fill manually."
        )

    cases = generate_cases(SCHEMA, coverage_tier=COVERAGE_TIER, dtype=DTYPE)
    schema_serializable = _schema_to_json(SCHEMA)

    payload = {
        "op": SCHEMA["op_name"],
        "formula": SCHEMA["formula"],
        "dtype": str(DTYPE).replace("torch.", ""),
        "coverage_tier": COVERAGE_TIER,
        "schema": schema_serializable,
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
        # Bug B (blue 2026-06-01): full per-case dict with input_stats (NOT just
        # string names) so downstream reference runners can
        # build real-shape probe inputs. Matches input_gen.template.py:292-298.
        "cases": [
            {"idx": c.get("idx", i), "name": c.get("name", f"case_{i}"),
             "shape": c.get("shape"),
             "input_sha256": case_data_sha256(c),
             "input_stats": {k: _stringify_tensor_stat(v) for k, v in c["inputs"].items()},
             "meta": c.get("meta", {})}
            for i, c in enumerate(cases)
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote edge_inputs.pt ({len(cases)} cases) + manifest.json")
    print(f"data_sha256: {payload['data_sha256']}")


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

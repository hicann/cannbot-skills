#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read-only precision-policy preflight; no DESIGN, Golden or device required."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from generator_support import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
_module_spec = importlib.util.spec_from_file_location(
    "_dsl_tolerance_contract", ROOT / "assets/case_contract.py"
)
_contract = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_contract)


def check_spec_tolerance(spec):
    section = spec.get("numerical_tolerance")
    output_dtypes = set()
    for output in spec.get("outputs") or []:
        output_dtypes.update(output.get("dtype_set") or [])
    for combination in (spec.get("dtype_policy") or {}).get("supported_combinations") or []:
        output_dtypes.update((combination.get("outputs") or {}).values())
    if section is None or section == {}:
        if not output_dtypes:
            raise ValueError("output dtypes required to select the default numerical policy")
        thresholds = {"float16": 2**-10, "bfloat16": 2**-7,
                      "float32": 2**-13, "float64": 2**-13}
        exact = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
        tolerance = {}
        for dtype in sorted(output_dtypes):
            if dtype in thresholds:
                t = thresholds[dtype]
                tolerance[dtype] = {"metric": "staged_relative", "backend": "native",
                                    "rtol": t, "max_relative_rtol": 10 * t,
                                    "epsilon": 1e-7, "fallback": {"reference_mode": "native"}}
            elif dtype in exact:
                tolerance[dtype] = {"metric": "bitwise_equal", "backend": "native"}
            else:
                raise ValueError(f"no default numerical policy for {dtype}")
        section = {"source": "built-in default numerical policy", "per_dtype": tolerance}
        spec["numerical_tolerance"] = section
    if not isinstance(section, dict):
        raise ValueError("numerical_tolerance must be a mapping")
    tolerance = section.get("per_dtype")
    _contract.validate_tolerance(tolerance)
    if "standard_stages" in section:
        raise ValueError(
            "standard_stages is not executable metadata; express the complete policy in per_dtype "
            "using staged_relative (no supplemental sidecar rules)"
        )
    if "evaluation_preconditions" in section:
        raise ValueError(
            "evaluation_preconditions must be expressed by the supported output/assertion contract, "
            "not ignored supplemental precision rules"
        )
    missing = output_dtypes - tolerance.keys()
    if missing:
        raise ValueError(f"missing output dtype tolerances: {sorted(missing)}")
    policies = {dtype: {**limits, "backend": limits.get("backend", "native")}
                for dtype, limits in tolerance.items()}
    effective = {
        dtype: _contract.staged_policy(dtype, limits)
        for dtype, limits in policies.items()
        if limits["metric"] == "staged_relative"
    }
    return {"status": "pass", "source": section.get("source"), "per_dtype": policies, "effective_fallback": effective}


def default_spec_tolerance(spec):
    """Propose canonical defaults without treating generated values as confirmed."""
    proposed = dict(spec)
    proposed.pop("numerical_tolerance", None)
    report = check_spec_tolerance(proposed)
    report["mode"] = "default_proposal"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--defaults", action="store_true",
                        help="propose built-in defaults without validating/replacing the existing precision policy")
    args = parser.parse_args()
    try:
        spec = load_yaml_mapping(args.spec)
        report = default_spec_tolerance(spec) if args.defaults else check_spec_tolerance(spec)
        report.update(
            spec=str(args.spec.resolve()),
            spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

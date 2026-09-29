# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unified precision comparison for the CATLASS linear-attention workflow.

Runners call ``compare_case`` with the output requirements from their test
plan, the observed outputs, and the masks generated from case parameters.
``compare`` supplies the shared numerical rule for one output.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}


def _load_array(path: Path):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".pt":
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, torch.Tensor):
            return value
        raise ValueError("a .pt output must contain one Tensor")
    raise ValueError("case arrays must use .npy or tensor-only .pt files")


def _dtype_name(value) -> str:
    if isinstance(value, np.ndarray):
        return value.dtype.name
    try:
        import torch
    except ImportError:
        raise ValueError("outputs must be NumPy arrays or PyTorch Tensors") from None
    if isinstance(value, torch.Tensor):
        return str(value.dtype).removeprefix("torch.")
    raise ValueError("outputs must be NumPy arrays or PyTorch Tensors")


def _numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    value = value.detach().cpu()
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return value.numpy()


def _finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all())


def _ulp(values: np.ndarray) -> np.ndarray:
    """Return a conservative ULP estimate for diagnostics.

    NumPy has no bfloat16 dtype on all supported versions.  For bfloat16 the
    float32 spacing is therefore only a diagnostic; max_abs remains the hard
    outlier gate.
    """

    values = values.astype(np.float32, copy=False)
    spacing = np.abs(np.spacing(values))
    return np.where(spacing == 0, np.finfo(np.float32).tiny, spacing)


def _metrics(actual: np.ndarray, golden: np.ndarray, atol: float, rtol: float) -> dict:
    metric_dtype = np.float64 if "float64" in (actual.dtype.name, golden.dtype.name) else np.float32
    actual = actual.astype(metric_dtype, copy=False)
    golden = golden.astype(metric_dtype, copy=False)
    diff = np.abs(actual - golden)
    tolerance = atol + rtol * np.abs(golden)
    close = diff <= tolerance
    relative = diff / (np.abs(golden) + 1e-7)
    return {
        "elements": int(diff.size),
        "matched_ratio": float(close.mean()) if diff.size else None,
        "error_count": int((~close).sum()),
        "max_abs": float(diff.max()) if diff.size else None,
        "max_ulp": float((diff / _ulp(golden)).max()) if diff.size else None,
        "mere": float(relative.mean()) if diff.size else None,
        "mare": float(relative.max()) if diff.size else None,
    }


def compare(
    actual,
    golden,
    *,
    dtype: str,
    policy: Mapping,
    valid_mask: Optional[np.ndarray] = None,
    regions: Optional[Mapping[str, np.ndarray]] = None,
) -> dict:
    """Compare one output; complete-case acceptance uses compare_case()."""

    actual_dtype = _dtype_name(actual)
    golden_dtype = _dtype_name(golden)
    if actual_dtype != dtype:
        return {"pass": False, "failure_reason": "dtype_mismatch",
                "dtype": dtype, "actual_dtype": actual_dtype}
    if golden_dtype not in FLOAT_DTYPES:
        raise ValueError("golden must have a floating-point dtype")
    if actual.shape != golden.shape:
        return {"pass": False, "failure_reason": "shape_mismatch"}
    actual, golden = _numpy(actual), _numpy(golden)
    if valid_mask is None:
        valid_mask = np.ones(actual.shape, dtype=bool)
    if valid_mask.shape != actual.shape or valid_mask.dtype != bool:
        raise ValueError("valid_mask must be a boolean array with the output shape")

    dtype_config = policy["dtype"].get(dtype)
    if dtype_config is None:
        raise ValueError(f"unsupported dtype in precision policy: {dtype}")
    atol = float(dtype_config["atol"])
    rtol = float(dtype_config["rtol"])
    max_abs_limit = float(dtype_config["max_abs_limit"])
    max_ulp_limit = dtype_config.get("max_ulp_limit")
    strict_ulp = bool(policy.get("strict_ulp", False) and max_ulp_limit is not None)

    actual_valid = actual[valid_mask]
    golden_valid = golden[valid_mask]
    report = {
        "precision_policy_version": policy.get("precision_policy_version"),
        "dtype": dtype,
        "actual_dtype": actual_dtype,
        "golden_dtype": golden_dtype,
        "shape": list(actual.shape),
        "atol": atol,
        "rtol": rtol,
        "global_matched_ratio_threshold": float(policy["global_matched_ratio"]),
        "region_matched_ratio_threshold": float(policy["region_matched_ratio"]),
        "critical_region_matched_ratio_threshold": float(policy["critical_region_matched_ratio"]),
        "max_abs_limit": max_abs_limit,
        "max_ulp_limit": max_ulp_limit,
        "strict_ulp": strict_ulp,
        "all_finite": _finite(actual_valid) and _finite(golden_valid),
        "regions": {},
    }
    if not report["all_finite"]:
        report["pass"] = False
        report["failure_reason"] = "non_finite_value"
        return report

    global_metrics = _metrics(actual_valid, golden_valid, atol, rtol)
    report["global"] = global_metrics
    hard_global = (
        global_metrics["matched_ratio"] is not None
        and global_metrics["matched_ratio"] >= float(policy["global_matched_ratio"])
        and global_metrics["max_abs"] <= max_abs_limit
    )
    if strict_ulp:
        hard_global = hard_global and global_metrics["max_ulp"] <= float(max_ulp_limit)

    region_pass = True
    for name, mask in (regions or {}).items():
        if mask.shape != actual.shape or mask.dtype != bool:
            raise ValueError(f"region {name!r} must be a boolean array with the output shape")
        selected = valid_mask & mask
        if not selected.any():
            report["regions"][name] = {"status": "SKIP", "pass": False}
            region_pass = False
            continue
        metrics = _metrics(actual[selected], golden[selected], atol, rtol)
        critical = name.lower().startswith(("tail", "boundary", "critical"))
        threshold = float(
            policy["critical_region_matched_ratio"] if critical else policy["region_matched_ratio"]
        )
        metrics.update({"status": "CHECKED", "threshold": threshold, "pass": (
            metrics["matched_ratio"] >= threshold and metrics["max_abs"] <= max_abs_limit
        )})
        if strict_ulp:
            metrics["pass"] = metrics["pass"] and metrics["max_ulp"] <= float(max_ulp_limit)
        report["regions"][name] = metrics
        region_pass = region_pass and metrics["pass"]

    report["pass"] = bool(hard_global and region_pass)
    if not report["pass"]:
        reasons = []
        if not hard_global:
            reasons.append("global_gate")
        if not region_pass:
            reasons.append("region_gate")
        report["failure_reason"] = ",".join(reasons)
    return report


def _case_error(case, reason: str, *, status: str = "CONFIG_ERROR") -> dict:
    return {"case": case, "status": status, "pass": False,
            "failure_reason": reason, "outputs": {}}


def _check_mask(mask, shape: tuple, name: str) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape:
        raise ValueError(f"{name} must be a boolean NumPy array with shape {shape}")


def compare_case(
    actual: Mapping,
    golden: Mapping,
    *,
    case: str,
    expected_outputs: Mapping,
    policy: Mapping,
    valid_masks: Optional[Mapping] = None,
    regions: Optional[Mapping] = None,
) -> dict:
    """Validate a case's independent output requirements before comparison.

    expected_outputs maps each required output name to shape, dtype and
    required_regions. Masks are generated from input parameters and the test
    plan; an empty required_regions list explicitly means no separate regions.
    """

    try:
        if not isinstance(case, str) or not case.strip():
            raise ValueError("case must have a nonempty name")
        if not isinstance(expected_outputs, Mapping) or not expected_outputs:
            raise ValueError("expected_outputs must declare the case's outputs")
        names = set(expected_outputs)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("output names must be nonempty strings")
        for label, values in (("actual", actual), ("golden", golden),
                              ("valid_masks", valid_masks), ("regions", regions)):
            if not isinstance(values, Mapping):
                raise ValueError(f"{label} must provide entries by output name")
            if set(values) != names:
                missing = sorted(names - set(values))
                extra = sorted(str(name) for name in set(values) - names)
                reason = f"{label} output names mismatch: missing={missing}, extra={extra}"
                if label == "actual":
                    return _case_error(case, reason, status="FAIL")
                raise ValueError(reason)

        # Validate all required checks before any numerical result can pass.
        for name, spec in expected_outputs.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"{name}: expected output specification is missing")
            shape = spec.get("shape")
            if not isinstance(shape, (list, tuple)) or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
            ):
                raise ValueError(f"{name}: shape must list nonnegative dimensions")
            shape = tuple(shape)
            dtype = spec.get("dtype")
            if dtype not in FLOAT_DTYPES or dtype not in policy["dtype"]:
                raise ValueError(f"{name}: dtype must be supported by the precision policy")
            required = spec.get("required_regions")
            if not isinstance(required, (list, tuple)) or any(
                not isinstance(region, str) or not region.strip() for region in required
            ) or len(set(required)) != len(required):
                raise ValueError(f"{name}: required_regions must explicitly list unique names")
            masks = regions[name]
            if not isinstance(masks, Mapping):
                raise ValueError(f"{name}: regions must be a mapping, including when empty")
            missing = sorted(set(required) - set(masks))
            if missing:
                raise ValueError(f"{name}: missing required regions {missing}")
            _check_mask(valid_masks[name], shape, f"{name}.valid_mask")
            if not valid_masks[name].any():
                raise ValueError(f"{name}: valid_mask selects no elements")
            for region, mask in masks.items():
                if not isinstance(region, str) or not region.strip():
                    raise ValueError(f"{name}: region names must be nonempty strings")
                _check_mask(mask, shape, f"{name}.{region}")
                if not (mask & valid_masks[name]).any():
                    raise ValueError(f"{name}.{region}: region selects no valid elements")
            _dtype_name(actual[name])
            if _dtype_name(golden[name]) not in FLOAT_DTYPES:
                raise ValueError(f"{name}: golden must have a floating-point dtype")
            if tuple(golden[name].shape) != shape:
                raise ValueError(f"{name}: golden shape differs from the expected output")

        outputs = {}
        for name, spec in expected_outputs.items():
            if tuple(actual[name].shape) != tuple(spec["shape"]):
                outputs[name] = {
                    "pass": False, "failure_reason": "shape_mismatch",
                    "expected_shape": list(spec["shape"]),
                    "actual_shape": list(actual[name].shape),
                }
            else:
                outputs[name] = compare(
                    actual[name], golden[name], dtype=spec["dtype"], policy=policy,
                    valid_mask=valid_masks[name], regions=regions[name],
                )
    except (ValueError, TypeError, KeyError) as exc:
        return _case_error(case, str(exc))

    passed = all(output["pass"] for output in outputs.values())
    report = {"case": case, "status": "PASS" if passed else "FAIL", "pass": passed,
              "precision_policy_version": policy.get("precision_policy_version"),
              "outputs": outputs}
    if not passed:
        report["failure_reason"] = "output_check_failed"
    return report


def _load_case(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("case data must be a JSON object")

    def array(relative_path):
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("case array paths must be nonempty strings")
        return _load_array(path.parent / relative_path)

    for key in ("actual", "golden", "valid_masks", "regions"):
        values = payload.get(key)
        if not isinstance(values, dict):
            raise ValueError(f"case data must include {key} by output name")
        if key == "regions":
            if any(not isinstance(masks, dict) for masks in values.values()):
                raise ValueError("regions must map each output to named mask paths")
            payload[key] = {name: {region: array(mask) for region, mask in masks.items()}
                            for name, masks in values.items()}
        else:
            payload[key] = {name: array(value) for name, value in values.items()}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="case data generated by the test runner")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    case = args.case.stem
    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        payload = _load_case(args.case)
        case = payload.get("case", case)
        report = compare_case(
            payload["actual"], payload["golden"], case=payload.get("case"),
            expected_outputs=payload.get("expected_outputs"), policy=policy,
            valid_masks=payload["valid_masks"], regions=payload["regions"],
        )
    except (OSError, EOFError, ValueError, TypeError, KeyError, RuntimeError, ImportError,
            pickle.UnpicklingError) as exc:
        report = _case_error(case, f"cannot load case inputs: {exc}")
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if report["status"] == "CONFIG_ERROR" else (0 if report["pass"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())

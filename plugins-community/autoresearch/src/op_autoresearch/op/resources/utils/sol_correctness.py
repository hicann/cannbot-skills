# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import random
import typing

import torch

_DEFAULT_ABSOLUTE_TOLERANCE = 1e-2
_DEFAULT_RELATIVE_TOLERANCE = 1e-2
_DEFAULT_MATCH_RATIO = 0.99
_ZERO_ERROR = 0.0
Tensor = torch.Tensor


class ToleranceSpec:
    def __init__(
        self, max_atol=_DEFAULT_ABSOLUTE_TOLERANCE,
        max_rtol=_DEFAULT_RELATIVE_TOLERANCE,
        required_matched_ratio=_DEFAULT_MATCH_RATIO, max_error_cap=None,
        allow_negative_inf=False, required_match_ratio=None,
    ):
        # Some datasets use the shorter alias; the explicit alias wins.
        ratio = (
            required_matched_ratio
            if required_match_ratio is None
            else required_match_ratio
        )
        values = (max_atol, max_rtol, ratio, max_error_cap, allow_negative_inf)
        names = (
            "max_atol",
            "max_rtol",
            "required_matched_ratio",
            "max_error_cap",
            "allow_negative_inf",
        )
        vars(self).update(zip(names, values))


class Correctness:
    def __init__(
        self,
        max_absolute_error=_ZERO_ERROR,
        max_relative_error=_ZERO_ERROR,
        has_nan=False,
        has_inf=False,
    ):
        names = ("max_absolute_error", "max_relative_error", "has_nan", "has_inf")
        vars(self).update(
            zip(names, (max_absolute_error, max_relative_error, has_nan, has_inf))
        )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python and Ascend PyTorch."""
    for seed_fn in (random.seed, torch.manual_seed):
        seed_fn(seed)
    try:
        importlib.import_module("torch_npu")  # registers torch.npu
    except ImportError:
        return
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.manual_seed_all(seed)


def _nonfinite_masks(
    output: Tensor,
    reference: Tensor,
    allow_negative_inf: bool,
) -> tuple[Tensor, Tensor]:
    output_bad = torch.logical_not(torch.isfinite(output))
    reference_bad = torch.logical_not(torch.isfinite(reference))
    if allow_negative_inf:
        accepted = torch.logical_and(output == -torch.inf, reference == -torch.inf)
        output_bad = torch.logical_and(output_bad, torch.logical_not(accepted))
        reference_bad = torch.logical_and(reference_bad, torch.logical_not(accepted))
    return output_bad, reference_bad


def _contains_true(tensor: Tensor) -> bool:
    return bool(torch.any(tensor).item())


def check_tensor_sanity(
    sol_tensor: Tensor,
    ref_tensor: Tensor,
    allow_negative_inf: bool = False,
) -> typing.Optional[Correctness]:
    """Check for non-finite values and all-zeros output."""
    output_bad, reference_bad = _nonfinite_masks(
        sol_tensor, ref_tensor, allow_negative_inf
    )
    if _contains_true(output_bad) or _contains_true(reference_bad):
        has_nan = any(
            _contains_true(torch.isnan(tensor))
            for tensor in (sol_tensor, ref_tensor)
        )
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    output_norm, reference_norm = (
        float(torch.linalg.vector_norm(tensor.float()).item())
        for tensor in (sol_tensor, ref_tensor)
    )
    if reference_norm > 0 and output_norm == 0:
        return Correctness(reference_norm, reference_norm)

    return None


def _drop_matching_negative_infinity(
    output: Tensor, reference: Tensor
) -> tuple[Tensor, Tensor]:
    keep = torch.logical_not(
        torch.logical_and(output == -torch.inf, reference == -torch.inf)
    )
    return output[keep], reference[keep]


def compute_error_stats(
    output: Tensor, reference: Tensor, tolerance: ToleranceSpec
) -> tuple[Correctness, bool]:
    """Compute numerical error between *output* and *reference*."""
    candidate, expected = (tensor.float() for tensor in (output, reference))
    invalid = check_tensor_sanity(
        candidate,
        expected,
        allow_negative_inf=tolerance.allow_negative_inf,
    )
    if invalid is not None:
        return invalid, True

    if tolerance.allow_negative_inf:
        candidate, expected = _drop_matching_negative_infinity(candidate, expected)
    if candidate.numel() == 0:
        return Correctness(), False

    error = torch.abs(candidate - expected)
    max_absolute = float(torch.amax(error).item())
    matches = torch.isclose(
        candidate,
        expected,
        rtol=tolerance.max_rtol,
        atol=tolerance.max_atol,
        equal_nan=False,
    )
    match_fraction = float(torch.count_nonzero(matches).item()) / error.numel()
    outside_tolerance = match_fraction < tolerance.required_matched_ratio
    cap = tolerance.max_error_cap
    outside_tolerance = outside_tolerance or (
        cap is not None and max_absolute > cap
    )

    denominator = torch.clamp(torch.abs(expected), min=tolerance.max_atol)
    max_relative = float(torch.amax(error / denominator).item())
    return Correctness(max_absolute, max_relative), outside_tolerance

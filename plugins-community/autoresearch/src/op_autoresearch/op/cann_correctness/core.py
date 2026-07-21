# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
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

"""
CANN-Bench tensor accuracy comparison engine.

Transplanted from CANN-Bench native kernel_eval/utils/compare.py and
kernel_eval/utils/thresholds.py. Only keeps the comparison logic needed
for pass/fail + error detail; scoring/report fields are omitted.
"""

import importlib
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from op_autoresearch.utils.console import emit

# ---------------------------------------------------------------------------
# Threshold tables (from kernel_eval/utils/thresholds.py)
# ---------------------------------------------------------------------------

PRECISION_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float64": 2**-13,
    "hifloat32": 2**-11,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
    "int8": 0, "int16": 0, "int32": 0, "int64": 0,
    "uint8": 0, "uint16": 0, "uint32": 0, "uint64": 0,
}

SMALL_VALUE_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float32": 2**-14,
    "float64": 2**-14,
    "hifloat32": 2**-12,
    "float8_e4m3fn": 2**-4,
    "float8_e5m2": 2**-3,
}

SMALL_VALUE_ERROR_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-16,
    "bfloat16": 2**-16,
    "float32": 2**-30,
    "float64": 2**-30,
    "hifloat32": 2**-28,
    "float8_e4m3fn": 2**-6,
    "float8_e5m2": 2**-5,
}

CANCEL_BOUNDARY_THRESHOLDS: Dict[str, float] = {
    "float32": 2**-8,
    "float64": 2**-8,
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "hifloat32": 2**-8,
    "float8_e4m3fn": 2**-1,
    "float8_e5m2": 2**-0,
}

CANCEL_ZERO_THRESHOLDS: Dict[str, float] = {
    "float32": 2**-8,
    "float64": 2**-8,
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "hifloat32": 2**-8,
    "float8_e4m3fn": 2**-1,
    "float8_e5m2": 2**-0,
}

_DEFAULT_DTYPE = "float32"


def _get_threshold(dtype_str: str) -> float:
    return PRECISION_THRESHOLDS.get(dtype_str.lower(), PRECISION_THRESHOLDS[_DEFAULT_DTYPE])


def _get_small_value_threshold(dtype_str: str) -> float:
    return SMALL_VALUE_THRESHOLDS.get(dtype_str.lower(), SMALL_VALUE_THRESHOLDS[_DEFAULT_DTYPE])


def _get_small_value_error(dtype_str: str) -> float:
    return SMALL_VALUE_ERROR_THRESHOLDS.get(dtype_str.lower(), SMALL_VALUE_ERROR_THRESHOLDS[_DEFAULT_DTYPE])


def _get_cancel_boundary(dtype_str: str) -> float:
    return CANCEL_BOUNDARY_THRESHOLDS.get(dtype_str.lower(), CANCEL_BOUNDARY_THRESHOLDS[_DEFAULT_DTYPE])


def _get_cancel_zero_threshold(dtype_str: str) -> float:
    return CANCEL_ZERO_THRESHOLDS.get(dtype_str.lower(), CANCEL_ZERO_THRESHOLDS[_DEFAULT_DTYPE])


# ---------------------------------------------------------------------------
# Integer dtype set
# ---------------------------------------------------------------------------

_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
for _optional_dtype_name in ("uint16", "uint32", "uint64"):
    _optional_dtype = getattr(torch, _optional_dtype_name, None)
    if _optional_dtype is not None:
        _INTEGER_DTYPES += (_optional_dtype,)
del _optional_dtype_name, _optional_dtype


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorComparisonOptions:
    """Optional policy and reference data for :func:`compare_tensors`."""

    dtype: str = "float32"
    threshold: Optional[float] = None
    native_output: Optional[Union[torch.Tensor, Tuple, List]] = None
    ignore_output_indices: Optional[List[int]] = None
    custom_thresholds: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class OutputAssertionContext:
    """Reference metadata needed to validate one candidate output."""

    index: int
    dtype: Optional[str] = None
    native_out: Any = None
    op_name: Optional[str] = None
    impl_inputs: Any = None
    all_impl_outputs: Any = None


@dataclass(frozen=True)
class _FloatState:
    output: torch.Tensor
    golden: torch.Tensor
    source_golden: torch.Tensor
    target_dtype: torch.dtype
    diff: torch.Tensor
    golden_abs: torch.Tensor
    relative_error: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class _ErrorDomain:
    mask: torch.Tensor
    total_count: int
    error_count: int
    passed: bool


def _normalize_outputs(output: Any) -> List[Any]:
    """Normalize output to a list of tensors (with None placeholders)."""
    if isinstance(output, torch.Tensor):
        return [output]
    if not isinstance(output, (tuple, list)):
        return []
    result: List[Any] = []
    for item in output:
        result.extend(_normalize_output_item(item))
    return result


def _normalize_output_item(item: Any) -> List[Any]:
    if isinstance(item, torch.Tensor):
        return [item]
    if isinstance(item, (tuple, list)):
        return [sub_item if isinstance(sub_item, torch.Tensor) else None for sub_item in item]
    return [None]


def _comparison_result(passed: bool, threshold: float, *, mere: float = 0.0,
                       mare: float = 0.0, error_msg: Optional[str] = None) -> Dict[str, Any]:
    return {
        "passed": passed,
        "mere": mere,
        "mare": mare,
        "threshold": threshold,
        "error_msg": error_msg,
    }


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cpu() if tensor.device.type == "npu" else tensor


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
    native_output: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compare a single tensor pair (MERE/MARE + small-value + cancellation).

    Returns: {"passed", "mere", "mare", "threshold", "error_msg"}
    """
    output = _cpu_tensor(output)
    golden = _cpu_tensor(golden)
    if output.shape != golden.shape:
        message = f"Shape mismatch: output={output.shape}, golden={golden.shape}"
        return _comparison_result(False, threshold, error_msg=message)
    bit_exact = _compare_bit_exact(output, golden, threshold)
    if bit_exact is not None:
        return bit_exact
    if output.dtype in _INTEGER_DTYPES:
        return _compare_integer(output, golden, threshold)
    state, special_result = _prepare_float_state(output, golden, threshold)
    if special_result is not None:
        return special_result
    valid_relative_error = state.relative_error[state.valid_mask]
    if len(valid_relative_error) == 0:
        return _comparison_result(True, threshold)
    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())
    mare_threshold = 10 * threshold
    if mere < threshold and mare < mare_threshold:
        return _comparison_result(True, threshold, mere=mere, mare=mare)
    return _compare_float_failure(state, dtype, threshold, native_output)


def _compare_bit_exact(output: torch.Tensor, golden: torch.Tensor,
                       threshold: float) -> Optional[Dict[str, Any]]:
    if threshold != 0 or not output.is_floating_point():
        return None
    bit_view = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }
    int_dtype = bit_view.get(output.dtype)
    if int_dtype is None:
        return None
    out_bits = output.contiguous().view(int_dtype)
    gold_bits = golden.to(output.dtype).contiguous().view(int_dtype)
    if torch.equal(out_bits, gold_bits):
        return _comparison_result(True, threshold)
    mismatch_count = int((out_bits != gold_bits).sum())
    message = f"bit-exact: {mismatch_count}/{output.numel()} elements differ"
    return _comparison_result(False, threshold, error_msg=message)


def _compare_integer(output: torch.Tensor, golden: torch.Tensor,
                     threshold: float) -> Dict[str, Any]:
    diff = torch.abs(output.long() - golden.long())
    mismatch_mask = diff > max(threshold, 0)
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count == 0:
        return _comparison_result(True, threshold)
    first_idx = int(mismatch_mask.reshape(-1).nonzero()[0].item())
    message = (
        f"integer mismatch: {mismatch_count}/{output.numel()} elements exceed tolerance {threshold}; "
        f"first_mismatch_flat_index={first_idx}, output={output.reshape(-1)[first_idx].item()}, "
        f"golden={golden.reshape(-1)[first_idx].item()}, diff={diff.reshape(-1)[first_idx].item()}"
    )
    return _comparison_result(False, threshold, error_msg=message)


def _prepare_float_state(output: torch.Tensor, golden: torch.Tensor,
                         threshold: float) -> Tuple[_FloatState, Optional[Dict[str, Any]]]:
    target_dtype = output.dtype
    output_fp64 = output.double()
    golden_truncated = golden.to(target_dtype).double()
    nan_result = _nan_mismatch_result(output_fp64, golden_truncated, threshold)
    if nan_result is not None:
        return None, nan_result
    inf_match_mask, inf_result = _normalize_infinities(
        output_fp64, golden_truncated, target_dtype, threshold
    )
    if inf_result is not None:
        return None, inf_result
    diff = torch.abs(output_fp64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    relative_error = diff / (golden_abs + 1e-7)
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    state = _FloatState(
        output_fp64, golden_truncated, golden, target_dtype,
        diff, golden_abs, relative_error, valid_mask,
    )
    return state, None


def _nan_mismatch_result(output: torch.Tensor, golden: torch.Tensor,
                         threshold: float) -> Optional[Dict[str, Any]]:
    nan_out = torch.isnan(output)
    nan_gold = torch.isnan(golden)
    if not torch.any(nan_out) and not torch.any(nan_gold):
        return None
    nan_diff = nan_out != nan_gold
    if not torch.any(nan_diff):
        return None
    first_idx = int(nan_diff.reshape(-1).nonzero()[0].item())
    message = (
        "NaN position mismatch: "
        f"output_nan={int(nan_out.sum().item())}, golden_nan={int(nan_gold.sum().item())}, "
        f"first_mismatch_flat_index={first_idx}, "
        f"output_is_nan={bool(nan_out.reshape(-1)[first_idx].item())}, "
        f"golden_is_nan={bool(nan_gold.reshape(-1)[first_idx].item())}"
    )
    return _comparison_result(False, threshold, error_msg=message)


def _normalize_infinities(output: torch.Tensor, golden: torch.Tensor,
                          target_dtype: torch.dtype, threshold: float):
    inf_match_mask = torch.zeros_like(output, dtype=torch.bool)
    inf_out = torch.isinf(output)
    inf_gold = torch.isinf(golden)
    if not torch.any(inf_out) and not torch.any(inf_gold):
        return inf_match_mask, None
    max_finite = float(torch.finfo(target_dtype).max)
    output_only = inf_out & ~inf_gold
    golden_only = inf_gold & ~inf_out
    if torch.any(output_only):
        output[output_only] = torch.sign(output[output_only]) * max_finite
    if torch.any(golden_only):
        golden[golden_only] = torch.sign(golden[golden_only]) * max_finite
    both_inf = inf_out & inf_gold
    if torch.any(both_inf) and not torch.all(torch.sign(output[both_inf]) == torch.sign(golden[both_inf])):
        return inf_match_mask, _comparison_result(False, threshold, error_msg="Inf sign mismatch")
    inf_match_mask[both_inf] = True
    return inf_match_mask, None


def _native_difference(state: _FloatState,
                       native_output: Optional[torch.Tensor]) -> torch.Tensor:
    if native_output is not None:
        native_output = _cpu_tensor(native_output)
        return torch.abs(native_output.double() - state.golden)
    native_golden = state.source_golden.to(state.target_dtype).double()
    return torch.abs(native_golden - state.golden)


def _small_value_domain(state: _FloatState, dtype: str,
                        cpu_diff: torch.Tensor) -> _ErrorDomain:
    small_value_threshold = _get_small_value_threshold(dtype)
    small_value_error = _get_small_value_error(dtype)
    small_value_mask = state.golden_abs < small_value_threshold
    small_value_mask[~state.valid_mask] = False
    small_value_total_count = int(small_value_mask.sum())
    small_value_error_count = int((small_value_mask & (state.diff > small_value_error)).sum())
    small_value_cpu_error_count = int(
        (small_value_mask & (cpu_diff > small_value_error)).sum()
    )
    passed = (
        small_value_error_count / max(small_value_cpu_error_count, 1)
    ) <= 2 if small_value_total_count > 0 else True
    return _ErrorDomain(small_value_mask, small_value_total_count, small_value_error_count, passed)


def _cancellation_domain(state: _FloatState, dtype: str, threshold: float,
                         cpu_diff: torch.Tensor) -> _ErrorDomain:
    cancel_boundary = _get_cancel_boundary(dtype)
    cancel_zero_threshold = _get_cancel_zero_threshold(dtype)
    output_abs = torch.abs(state.output)
    output_near_zero = output_abs < cancel_zero_threshold
    golden_in_cancel_range = (
        (state.golden_abs < cancel_boundary)
        & (state.golden_abs >= _get_small_value_threshold(dtype))
    )
    cancel_mask = output_near_zero & golden_in_cancel_range & state.valid_mask
    cancel_total_count = int(cancel_mask.sum())
    mare_threshold = 10 * threshold
    cancel_error_count = int((cancel_mask & (state.relative_error > mare_threshold)).sum())
    cpu_relative_error = cpu_diff / (state.golden_abs + 1e-7)
    cpu_error_count = int((cancel_mask & (cpu_relative_error > mare_threshold)).sum())
    passed = (cancel_error_count / max(cpu_error_count, 1)) <= 2 if cancel_total_count > 0 else True
    return _ErrorDomain(cancel_mask, cancel_total_count, cancel_error_count, passed)


def _compare_float_failure(state: _FloatState, dtype: str, threshold: float,
                           native_output: Optional[torch.Tensor]) -> Dict[str, Any]:
    cpu_diff = _native_difference(state, native_output)
    small = _small_value_domain(state, dtype, cpu_diff)
    cancel = _cancellation_domain(state, dtype, threshold, cpu_diff)
    mismatch_mask = state.relative_error > 10 * threshold
    mismatch_mask[~state.valid_mask] = False
    normal_mask = mismatch_mask & ~small.mask & ~cancel.mask
    normal_mismatch_count = int(normal_mask.sum())
    if normal_mismatch_count > 0:
        passed = False
        normal_re = state.relative_error[~small.mask & ~cancel.mask & state.valid_mask]
        mere = float(normal_re.mean()) if len(normal_re) > 0 else 0.0
        mare = float(normal_re.max()) if len(normal_re) > 0 else 0.0
    else:
        passed = small.passed and cancel.passed
        valid_error = state.relative_error[state.valid_mask]
        mere = float(valid_error.mean())
        mare = float(valid_error.max())
    error_msg = None if passed else _floating_error_message(
        state, normal_mismatch_count, small, cancel
    )
    return _comparison_result(passed, threshold, mere=mere, mare=mare, error_msg=error_msg)


def _floating_error_message(state: _FloatState, normal_count: int,
                            small: _ErrorDomain, cancel: _ErrorDomain) -> str:
    debug_re = torch.where(
        state.valid_mask,
        state.relative_error,
        torch.full_like(state.relative_error, -1.0),
    )
    worst_idx = int(torch.argmax(debug_re.reshape(-1)).item())
    return (
        f"floating mismatch: normal={normal_count}, "
        f"small_value={small.error_count}/{small.total_count}, "
        f"cancel={cancel.error_count}/{cancel.total_count}; "
        f"worst_flat_index={worst_idx}, output={state.output.reshape(-1)[worst_idx].item()}, "
        f"golden={state.golden.reshape(-1)[worst_idx].item()}, "
        f"diff={state.diff.reshape(-1)[worst_idx].item()}, "
        f"relative_error={state.relative_error.reshape(-1)[worst_idx].item()}"
    )


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    options: Optional[TensorComparisonOptions] = None,
) -> Dict[str, Any]:
    """Compare output tensors against golden reference (MERE/MARE standard).

    Returns: {"passed", "mere", "mare", "threshold", "error_msg", "output_results"}
    """
    options = options or TensorComparisonOptions()
    threshold = options.threshold
    if threshold is None:
        threshold = _output_threshold(options, options.dtype)
    try:
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        native_outputs = (
            _normalize_outputs(options.native_output)
            if options.native_output is not None else None
        )
        if len(outputs) != len(goldens):
            message = f"Output count mismatch: output={len(outputs)}, golden={len(goldens)}"
            return _comparison_summary(False, threshold, error_msg=message)
        return _compare_output_lists(outputs, goldens, native_outputs, options, threshold)
    except Exception as e:
        message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return _comparison_summary(False, threshold, error_msg=message)


def _output_threshold(options: TensorComparisonOptions, dtype: str) -> float:
    custom_thresholds = options.custom_thresholds or {}
    return custom_thresholds.get(dtype.lower(), _get_threshold(dtype))


def _compare_output_lists(outputs: List[Any], goldens: List[Any],
                          native_outputs: Optional[List[Any]], options: TensorComparisonOptions,
                          threshold: float) -> Dict[str, Any]:
    output_results = []
    mere_sum = 0.0
    mare_max = 0.0
    total_count = 0
    for index, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
        native_tensor = native_outputs[index] if native_outputs is not None else None
        result, count = _compare_output_pair(
            index, (out_tensor, gold_tensor), native_tensor, options, threshold
        )
        output_results.append(result)
        mere_sum += result["mere"] * count
        mare_max = max(mare_max, result["mare"])
        total_count += count
    all_passed = all(result["passed"] for result in output_results)
    error_msg = next(
        (
            result["error_msg"] for result in output_results
            if not result["passed"] and not (result.get("error_msg") or "").startswith("(skip")
        ),
        None,
    )
    mere = mere_sum / total_count if total_count > 0 else 0.0
    return _comparison_summary(
        all_passed, threshold, {"mere": mere, "mare": mare_max},
        error_msg, output_results,
    )


def _compare_output_pair(index: int, pair: Tuple[Any, Any], native_output: Any,
                         options: TensorComparisonOptions, threshold: float):
    output, golden = pair
    if options.ignore_output_indices and index in options.ignore_output_indices:
        return _indexed_result(index, threshold, "(skipped)"), 0
    if output is None or golden is None:
        return _indexed_result(index, threshold, "(None placeholder)"), 0
    dtype = str(output.dtype).replace("torch.", "")
    output_threshold = _output_threshold(options, dtype)
    result = _compare_single_tensor(output, golden, output_threshold, dtype, native_output)
    return {
        "index": index,
        "dtype": dtype,
        "passed": result["passed"],
        "mere": result["mere"],
        "mare": result["mare"],
        "threshold": output_threshold,
        "error_msg": result.get("error_msg", ""),
    }, output.numel()


def _indexed_result(index: int, threshold: float, error_msg: str) -> Dict[str, Any]:
    return {
        "index": index, "passed": True, "mere": 0.0, "mare": 0.0,
        "threshold": threshold, "error_msg": error_msg,
    }


def _comparison_summary(passed: bool, threshold: float,
                        metrics: Optional[Dict[str, float]] = None, error_msg: Optional[str] = None,
                        output_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    metrics = metrics or {}
    return {
        "passed": passed,
        "mere": metrics.get("mere", 0.0),
        "mare": metrics.get("mare", 0.0),
        "threshold": threshold,
        "error_msg": error_msg,
        "output_results": output_results or [],
    }


def dual_reference(model, inputs):
    """Compute the two references CANN-Bench's precision gate needs from a single
    reference callable, so the (already-identical) compare engine gets identical
    inputs to CANN-Bench:

    - golden: the reference at **FP64 on CPU** (CANN-Bench's fp64-CPU golden).
      NPU has no fp64, so the golden must run on CPU; floating inputs are moved
      to CPU and upcast to double.
    - native: the reference at the **target precision on CPU** (the same-precision
      reference the small-value / cancellation excusal compares against). It must
      be an INDEPENDENT reference, NOT the candidate's own path — if it were run
      on NPU it could coincide with a candidate that delegates to the same NPU
      builtins and vacuously pass the excusal, defeating the gate.

    Returns ``(golden_list, native_list)`` (each normalized to a list). Non-float
    inputs (indices / scalars / shapes) are passed through unchanged. Runs the
    reference on CPU (the golden formula is device-agnostic torch), matching
    CANN-Bench's fp64-CPU golden + CPU same-precision native.
    """
    def _to_cpu(x, fp64):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.double() if (fp64 and x.is_floating_point()) else x
        return x

    cpu_model = getattr(model, "cpu", lambda: model)()

    # native: target-precision reference on CPU (compute before upcasting model).
    native_inputs = [_to_cpu(x, fp64=False) for x in inputs]
    native = cpu_model(*native_inputs)

    # golden: FP64 reference on CPU.
    fp64_model = getattr(cpu_model, "double", lambda: cpu_model)()
    fp64_inputs = [_to_cpu(x, fp64=True) for x in inputs]
    golden = fp64_model(*fp64_inputs)

    return _normalize_outputs(golden), _normalize_outputs(native)


def _assert_registered_index_output(
    impl_out: Any,
    context: OutputAssertionContext,
) -> bool:
    spec = _index_gather_spec(context.op_name, context.index)
    if (
        spec is None
        or context.impl_inputs is None
        or context.all_impl_outputs is None
    ):
        return False
    inputs = (
        list(context.impl_inputs)
        if isinstance(context.impl_inputs, (list, tuple))
        else [context.impl_inputs]
    )
    outputs = (
        list(context.all_impl_outputs)
        if isinstance(context.all_impl_outputs, (list, tuple))
        else [context.all_impl_outputs]
    )
    x = inputs[spec["input"]]
    dim = int(inputs[spec["dim_arg"]])
    values = outputs[spec["value_output"]]
    passed, message = validate_index_output(x, dim, impl_out, values)
    emit(
        f"[cannbench_index_gather] output={context.index} "
        f"value_output={spec['value_output']} passed={passed}"
        + ("" if passed else f" msg={message}")
    )
    if not passed:
        raise AssertionError(
            f"index_gather (output {context.index}): {message}"
        )
    return True


def assert_outputs(
    impl_out: Any,
    golden_out: Any,
    context: OutputAssertionContext,
) -> None:
    """One-output CANN-Bench accuracy gate for a generated verify script: print
    one ``[cannbench_precision]`` line and raise ``AssertionError`` (a numerical
    kernel_miss) on failure, so the template stays a one-line call.

    ``golden_out`` should be the FP64 golden and ``native_out`` the same-precision
    reference (see :func:`dual_reference`); passing ``native_out`` makes the
    small-value / cancellation excusal use the same NPU/CPU-error ratio
    CANN-Bench's scoring uses. ``dtype`` names the target output precision (taken
    from ``native_out`` when available, else ``golden_out``).

    If ``op_name``'s output ``index`` is a registered index output (see
    :data:`_INDEX_GATHER_REGISTRY`), the element-wise MERE/MARE compare is
    replaced by CANN-Bench's tie-order-independent index_gather check —
    ``x.gather(dim, idx) == value_output`` — using ``impl_inputs`` /
    ``all_impl_outputs`` for x, dim and the paired value output.

    If ``op_name`` carries a per-op ``precision_thresholds`` block in its
    CANN-Bench proto (see :data:`_PRECISION_THRESHOLD_REGISTRY`), those dtype
    thresholds override the strict defaults for the MERE/MARE compare (e.g.
    int8 ±1 for quantized outputs), matching CANN-Bench's own scoring band.

    If ``op_name``'s output ``index`` is registered as compare-skipped (see
    :data:`_IGNORE_OUTPUT_REGISTRY` — proto ``compare: false`` WITHOUT an
    ``index_gather`` block, e.g. pure position indices), it is not checked at
    all, matching CANN-Bench's ``ignore_output_indices``.
    """
    index = context.index
    if _is_ignored_output(context.op_name, index):
        emit(
            f"[cannbench_precision] output={index} skipped=True "
            "(proto compare:false)"
        )
        return
    if _assert_registered_index_output(impl_out, context):
        return

    dtype = context.dtype
    if dtype is None:
        ref = context.native_out if context.native_out is not None else golden_out
        d = getattr(ref, "dtype", None)
        dtype = str(d).replace("torch.", "") if d is not None else "float32"
    custom_thresholds = _precision_threshold_spec(context.op_name)
    options = TensorComparisonOptions(
        dtype=dtype,
        native_output=context.native_out,
        custom_thresholds=custom_thresholds,
    )
    r = compare_tensors(impl_out, golden_out, options)
    emit(
        f"[cannbench_precision] output={index} dtype={dtype} "
        f"mere={r['mere']:.6e} mare={r['mare']:.6e} "
        f"threshold={r['threshold']:.6e} passed={r['passed']}"
    )
    if not r["passed"]:
        raise AssertionError(
            r.get("error_msg")
            or f"CANN-Bench MERE/MARE miss (output {index}, dtype {dtype})")


# ---------------------------------------------------------------------------
# Index-output "pointed-value" check (from CANN-Bench eval/index_check.py,
# issue #40). TopK / ArgSort / Cummin etc. return an index output whose golden
# (torch.topk) tie order is non-deterministic, so an element-wise index compare
# fails on ties even for a correct kernel. CANN-Bench marks the index output
# ``compare: false`` and instead checks, tie-order-independently, that the
# candidate's indices point to its own value output. Combined with the value
# output's MERE/MARE check against golden data, this validates wrong values and
# invalid indices while allowing equivalent tie orderings.
# ---------------------------------------------------------------------------


def validate_index_output(
    x: torch.Tensor,
    dim: int,
    idx_candidate: torch.Tensor,
    values_candidate: torch.Tensor,
) -> Tuple[bool, str]:
    """Check that ``x.gather(dim, idx)`` equals ``values`` element-wise (NaN-aware),
    independent of tie order. Returns ``(ok, msg)``. Transplanted verbatim from
    CANN-Bench ``eval/index_check.py::validate_index_output``.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor) \
            or not isinstance(values_candidate, torch.Tensor):
        return False, "index_gather: x / idx / values must all be Tensors"

    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    val_c = values_candidate.detach().cpu()

    if idx_c.dim() != x_c.dim():
        return False, f"index_gather: idx ndim {idx_c.dim()} != x ndim {x_c.dim()}"
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not (0 <= dim_n < x_c.dim()):
        return False, f"index_gather: dim={dim} out of range (x is {x_c.dim()}D)"

    if idx_c.shape != val_c.shape:
        return False, (f"index_gather: idx shape {tuple(idx_c.shape)} != value shape "
                       f"{tuple(val_c.shape)}")

    for d in range(x_c.dim()):
        if d != dim_n and idx_c.size(d) > x_c.size(d):
            return False, (f"index_gather: dim {d} idx size {idx_c.size(d)} > x "
                           f"{x_c.size(d)}")

    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return False, f"index_gather: idx out of range [{lo},{hi}], valid [0,{dim_size})"

    gathered = torch.gather(x_c, dim_n, idx_c).to(val_c.dtype)
    # NaN-aware (an all-NaN case with correct indices must pass; torch.equal would
    # reject it since NaN != NaN).
    if gathered.is_floating_point() or val_c.is_floating_point():
        both_nan = torch.isnan(gathered) & torch.isnan(val_c)
        eq = (gathered == val_c) | both_nan
    else:
        eq = gathered == val_c
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return False, (f"index_gather: candidate indices point to elements that "
                       f"differ from its value output ({mism} mismatches)")
    return True, ""


# op_name -> {index_output_pos: {"input", "dim_arg", "value_output"}}, all
# positional against the op's own call args / outputs. This mirrors the
# ``compare: false`` + ``index_gather`` blocks in CANN-Bench's proto.yaml for
# ops whose index output is tie-non-deterministic. Add an entry per such op.
_INDEX_GATHER_REGISTRY: Dict[str, Dict[int, Dict[str, int]]] = {
    # top_k(x, k, dim, largest) -> (values, idx): idx is output 1; validate it
    # points into x (arg 0) along dim (arg 2) reproducing the value output (0).
    "top_k": {1: {"input": 0, "dim_arg": 2, "value_output": 0}},
    "TopK": {1: {"input": 0, "dim_arg": 2, "value_output": 0}},
}


def _index_gather_spec(op_name: Optional[str], index: int) -> Optional[Dict[str, int]]:
    if not op_name:
        return None
    return _INDEX_GATHER_REGISTRY.get(op_name, {}).get(index)


# op_name (normalized: lowercased, underscores removed) -> set of output indices
# that CANN-Bench's proto marks ``compare: false`` WITHOUT an ``index_gather``
# block, i.e. skipped entirely (no candidate-vs-golden check). Distinct from
# _INDEX_GATHER_REGISTRY: those index outputs still get tie-order-independent
# gather validation; these are pure position indices with no gather relation to
# any value output, so CANN-Bench ignores them outright.
_IGNORE_OUTPUT_REGISTRY: Dict[str, set] = {
    # moe_gating_top_k_softmax(x, finished, k) -> (y, expert_idx, row_idx):
    # y is softmax(x) top-k values, so x.gather(dim, expert_idx) != y (no gather
    # check applies); row_idx is a bare row position. Proto marks both
    # compare:false with no index_gather -> skip outputs 1 and 2.
    "moegatingtopksoftmax": {1, 2},
}


def _is_ignored_output(op_name: Optional[str], index: int) -> bool:
    if not op_name:
        return False
    return index in _IGNORE_OUTPUT_REGISTRY.get(op_name.lower().replace("_", ""), set())


# ---------------------------------------------------------------------------
# Per-op precision-threshold overrides (from CANN-Bench proto.yaml
# ``operator.precision_thresholds``). The autoresearch scaffold generates the
# verify script from reference.py/task.yaml, NOT proto.yaml, so these per-op
# relaxations are otherwise dropped and the gate uses the strict dtype defaults
# in PRECISION_THRESHOLDS — spuriously failing e.g. quantized (int8 ±1) or
# division/normalization ops that CANN-Bench itself scores with a wider band.
# Keys are normalized (lowercased, underscores removed) so one entry matches
# both the snake_case op name and proto's CamelCase operator.name. Values are
# {lowercase-dtype: threshold}, forwarded verbatim to compare_tensors'
# custom_thresholds (a threshold applies to every output of that dtype).
# Mirror proto.yaml exactly; add an entry per op that carries the block.
# ---------------------------------------------------------------------------
_PRECISION_THRESHOLD_REGISTRY: Dict[str, Dict[str, float]] = {
    "foreachaddcdivscalar": {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "applyadamw": {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "applyrotaryposemb": {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "dynamicquant": {"int8": 1.0},
    "groupnorm": {"float16": 0.005, "float32": 0.005, "bfloat16": 0.01},
    "unsortedsegmentsum": {"float32": 0.001},
    "addrmsnormdynamicquant": {"int8": 1.0},
    "dequantswigluquant": {"int8": 1.0},
    "roialign": {"float16": 0.1, "float32": 0.01},
    "unique": {"bfloat16": 0.0, "float16": 0.0, "float32": 0.0, "int8": 0.0},
    "groupedmatmulswigluquant": {"int8": 1.0, "float32": 0.001},
    "gru": {"float32": 0.05, "float16": 0.05, "bfloat16": 0.05},
    "lstm": {"float32": 0.05, "float16": 0.05, "bfloat16": 0.05},
}


def _precision_threshold_spec(op_name: Optional[str]) -> Optional[Dict[str, float]]:
    """Per-op ``custom_thresholds`` for :func:`compare_tensors`, or None. Matches
    ``op_name`` case- and underscore-insensitively (snake_case == CamelCase).
    """
    if not op_name:
        return None
    return _PRECISION_THRESHOLD_REGISTRY.get(op_name.lower().replace("_", ""))


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    npu_runtime_available = True
    try:
        importlib.import_module("torch_npu")  # registers torch.npu
    except ImportError:
        npu_runtime_available = False
    if npu_runtime_available and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

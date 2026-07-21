# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Unified evaluation script with multi-case testing and geometric mean computation.

Supports both unary and binary operators with test cases defined in test_cases.py.

Output modes:
  --verbose   Full output including msprof subprocess logs (for debugging)
  (default)   Quiet mode — only case results, errors, and summary table
"""

import os
import re
import sys
import json
import logging
import argparse
import contextlib
import importlib.util
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Quiet-mode helpers: suppress msprof subprocess noise at the OS fd level
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _suppress_fds():
    """Redirect OS-level stdout/stderr to /dev/null.

    This captures output from subprocesses (msprof) that write directly
    to file descriptors 1/2, which Python's logging redirection cannot catch.
    """
    # Flush Python buffers before swapping fds to avoid losing buffered output
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY, 0o600)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        # Flush again after restoring to avoid interleaved output
        sys.stdout.flush()
        sys.stderr.flush()
        os.close(devnull)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


@contextlib.contextmanager
def _no_suppress():
    """No-op context manager (verbose mode)."""
    yield


# CRITICAL: Setup ASCEND_CUSTOM_OPP_PATH before importing torch_npu
# This is needed for custom operators to load correctly
if 'ASCEND_CUSTOM_OPP_PATH' not in os.environ:
    # Try to infer from command line args if not already set
    # Note: This is a workaround. Ideally, users should set it before running the script.
    for i, arg in enumerate(sys.argv):
        if arg == '--work-dir' and i + 1 < len(sys.argv):
            work_dir_arg = Path(sys.argv[i + 1]).resolve()
            custom_opp_path = work_dir_arg / "vendors" / "customize"
            if custom_opp_path.exists():
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
                lib_path = custom_opp_path / "op_api" / "lib"
                if lib_path.exists():
                    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if ld_path:
                        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{ld_path}"
                    else:
                        os.environ["LD_LIBRARY_PATH"] = str(lib_path)
            break
        elif not arg.startswith('--') and 'work_dir' not in locals():
            # Positional argument after op_name might be work-dir
            pass

import torch
import torch_npu
import numpy as np

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Per-operator pybind_lib isolation for parallel evaluation


def _inject_pybind_lib(work_dir_path):
    """Prepend work_dir/pybind_lib/ to sys.path[0] and invalidate any cached
    custom_ops_lib module, so each operator variant loads its own .so.

    Always moves pybind_lib to front (not just on first call) so that
    sequential multi-operator evaluation in the same process stays isolated.
    """
    pybind_lib = os.path.join(str(work_dir_path), "pybind_lib")
    if not os.path.isdir(pybind_lib):
        return
    # Always move to front, even if already present from an earlier call
    if pybind_lib in sys.path:
        sys.path.remove(pybind_lib)
    sys.path.insert(0, pybind_lib)
    # Always invalidate cached module so the next import reloads from the
    # newly-fronted path (guards against sequential per-operator evaluation)
    if "custom_ops_lib" in sys.modules:
        del sys.modules["custom_ops_lib"]
    logging.info(f"Injected pybind_lib: {pybind_lib}")


from precision import dual_inspect
from performance import measure_performance_msprof


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def resolve_api_description_path(work_dir: Path, api_desc: Optional[str] = None) -> Optional[Path]:
    """Resolve api_description path.

    Priority:
    1. explicit --api-desc path
    2. work_dir/api_description.md
    3. work_dir/api_desc.md
    """
    candidates = []
    if api_desc:
        candidates.append(Path(api_desc).expanduser())
    candidates.extend([
        work_dir / "api_description.md",
        work_dir / "api_desc.md",
    ])

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


_DTYPE_PATTERNS: Dict[str, List[str]] = {
    "bfloat16": [r"\bbfloat16\b", r"\bbf16\b"],
    "float16": [r"\bfloat16\b", r"\bfp16\b", r"\bhalf\b"],
    "float32": [r"\bfloat32\b", r"\bfp32\b", r"\bfloat\b"],
    "float64": [r"\bfloat64\b", r"\bfp64\b", r"\bdouble\b"],
    "int8": [r"\bint8\b"],
    "int16": [r"\bint16\b"],
    "int32": [r"\bint32\b"],
    "int64": [r"\bint64\b"],
    "uint8": [r"\buint8\b"],
    "bool": [r"\bbool\b"],
}


def infer_allowed_dtypes_from_api_description(api_desc_path: Path) -> Optional[set[str]]:
    """Best-effort extraction of dtype mentions from api_description.

    Keep this intentionally conservative: only gate when the doc clearly mentions
    concrete dtype tokens. If nothing reliable is found, return None and skip.
    """
    text = api_desc_path.read_text(encoding="utf-8")
    found = set()
    lower_text = text.lower()
    for dtype_name, regexes in _DTYPE_PATTERNS.items():
        if any(re.search(regex, lower_text) for regex in regexes):
            found.add(dtype_name)
    return found or None


def validate_api_description_dtype_guard(
        work_dir: Path, test_cases: Dict[str, Any], api_desc: Optional[str] = None) -> None:
    """Validate dtype expectations against api_description when available.

    Historical behavior assumed api_description lived under work_dir only.
    We now also support explicit --api-desc to avoid path loss across pipeline
    stages while keeping work_dir-first fallback for existing workflows.
    """
    api_desc_path = resolve_api_description_path(work_dir, api_desc)
    if api_desc_path is None:
        logging.warning("missing api_description.md/api_desc.md; skip API dtype alignment")
        return

    allowed_dtypes = infer_allowed_dtypes_from_api_description(api_desc_path)
    if not allowed_dtypes:
        logging.info(f"No explicit dtype markers found in {api_desc_path}; skip API dtype alignment")
        return

    for case in test_cases.get("cases", []):
        case_dtype = str(case.get("dtype", "")).lower()
        if not case_dtype:
            continue
        if case_dtype in allowed_dtypes:
            continue
        # Only reject dtypes that the recogniser covers.  Unknown families
        # (e.g. complex64) are skipped to avoid false rejections.
        if case_dtype in _DTYPE_PATTERNS:
            raise ValueError(
                f"Test case dtype '{case_dtype}' is not mentioned by api_description: {api_desc_path}. "
                f"Allowed dtypes inferred from doc: {sorted(allowed_dtypes)}"
            )


def load_test_cases(work_dir: Path) -> Tuple[Dict, Any]:
    """
    Dynamically load test_cases.py from work directory.

    Returns:
        Tuple of (TEST_CASES dict, module)
    """
    test_cases_path = work_dir / "test_cases.py"
    if not test_cases_path.exists():
        raise FileNotFoundError(
            f"test_cases.py not found in {work_dir}. "
            f"Please create it based on examples in skills/ascendc-evaluation/examples/"
        )

    spec = importlib.util.spec_from_file_location("test_cases", test_cases_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "TEST_CASES"):
        raise AttributeError(f"test_cases.py must define TEST_CASES dict")

    return module.loaded_cases, module


def setup_environment(work_dir: Path):
    """Set up ASCEND environment variables."""
    customize_opp_path = work_dir / "vendors" / "customize"
    if not customize_opp_path.exists():
        raise FileNotFoundError(f"vendors/customize not found in {work_dir}")

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(customize_opp_path)

    # Add lib to LD_LIBRARY_PATH
    op_api_lib_path = customize_opp_path / "op_api" / "lib"
    if op_api_lib_path.exists():
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if existing_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{op_api_lib_path}:{existing_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = str(op_api_lib_path)


def evaluate_multi_case(
    op_name: str,
    work_dir: Path,
    device_id: int = 0,
    flush_l2: bool = True,
    json_output_path: Optional[str] = None,
    skip_performance: bool = False,
    api_desc: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[bool, str, Dict]:
    """
    Multi-case evaluation with geometric mean.

    Args:
        op_name: Operator name
        work_dir: Working directory containing test_cases.py and operator files
        device_id: NPU device ID
        flush_l2: Whether to flush L2 cache
        json_output_path: Path to write JSON output
        skip_performance: Whether to skip performance measurement
        api_desc: Explicit path to api_description(.md) for dtype guard
        verbose: If False (default), suppress msprof noise and only print
                 essential case results + summary.  If True, print everything.

    Returns:
        Tuple of (all_passed, summary_message, results_dict)
    """
    quiet = not verbose
    # Choose the fd-level suppression context for msprof subprocess output
    _perf_ctx = _suppress_fds if quiet else _no_suppress

    # Set device
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    # Inject per-operator pybind_lib path for parallel isolation.
    # If work_dir is the *Custom project dir (e.g. FastGeluCustom/), pybind_lib
    # lives in its parent (the variant root).  Otherwise use work_dir directly.
    _pybind_work_dir = work_dir.parent if work_dir.name.endswith("Custom") else work_dir
    _inject_pybind_lib(_pybind_work_dir)

    # Load test cases
    logging.info(f"Loading test cases from {work_dir / 'test_cases.py'}")
    loaded_cases, test_module = load_test_cases(work_dir)

    validate_api_description_dtype_guard(work_dir, loaded_cases, api_desc=api_desc)

    operator_type = loaded_cases.get("operator_type", "unary")
    quantized_output = loaded_cases.get("quantized_output", False)
    cases = loaded_cases["cases"]

    logging.info(f"Operator: {TEST_CASES['operator']}")
    logging.info(f"Type: {operator_type}")
    logging.info(f"Test cases: {len(cases)}")

    # Load reference and custom implementations
    ref_code_path = work_dir / f"{op_name}_reference.py"
    custom_code_path = work_dir / f"{op_name}_custom.py"

    if not ref_code_path.exists():
        raise FileNotFoundError(f"Reference code not found: {ref_code_path}")
    if not custom_code_path.exists():
        raise FileNotFoundError(f"Custom code not found: {custom_code_path}")

    # Execute code to get Model and ModelNew classes
    ref_context = {}
    custom_context = {}
    exec(ref_code_path.read_text(encoding='utf-8'), ref_context)
    exec(custom_code_path.read_text(encoding='utf-8'), custom_context)

    ref_model_cls = ref_context.get("Model")
    custom_model_cls = custom_context.get("ModelNew")

    if not ref_model_cls or not custom_model_cls:
        raise ValueError("Code must define Model (reference) and ModelNew (custom)")

    # Multi-case evaluation loop
    results = []
    speedups = []
    all_passed = True

    logging.info(f"{'='*70}")
    logging.info(f"Starting multi-case evaluation")
    logging.info(f"{'='*70}")

    for idx, case in enumerate(cases):
        case_id = case.get("id")
        case_name = case.get("name")

        logging.info(f"\n[{idx+1}/{len(cases)}] Case {case_id}: {case_name}")
        logging.info(f"  {'-'*50}")

        try:
            # Get initialization parameters for this test case
            init_inputs = []
            if hasattr(test_module, 'get_init_inputs'):
                init_result = test_module.get_init_inputs(case)
                if init_result:
                    init_inputs = init_result

            # Initialize models with test case specific parameters
            ref_model = ref_model_cls(*init_inputs).to(device)
            custom_model = custom_model_cls(*init_inputs).to(device)
            ref_model.eval()
            custom_model.eval()

            # Generate inputs
            if operator_type == "unary":
                if not hasattr(test_module, "generate_input"):
                    raise AttributeError("test_cases.py must define generate_input() for unary operators")
                inputs = [test_module.generate_input(case, device=device)]
            elif operator_type in ("binary", "multi_input"):
                if not hasattr(test_module, "generate_inputs"):
                    raise AttributeError(f"test_cases.py must define generate_inputs() for {operator_type} operators")
                inputs = list(test_module.generate_inputs(case, device=device))
            else:
                raise ValueError(f"Unknown operator_type: {operator_type}")

            # Three-way precision evaluation
            logging.info(f"  Running precision evaluation...")

            with torch.no_grad():
                # Ref: NPU working dtype
                ref_output = ref_model(*inputs)

                # Ans: NPU custom kernel
                custom_output = custom_model(*inputs)

                # Golden: Compute on CPU with fp64 for float inputs, keep integers as-is
                # Move model to CPU temporarily
                ref_model_cpu = ref_model.cpu()
                golden_inputs = []
                for inp in inputs:
                    inp_cpu = inp.cpu()
                    # Only convert to double for floating-point types
                    if inp_cpu.is_floating_point():
                        golden_inputs.append(inp_cpu.double())
                    else:
                        # Keep integers as original dtype
                        golden_inputs.append(inp_cpu)
                golden_output = ref_model_cpu(*golden_inputs)
                # Move model back to NPU
                ref_model.to(device)

            # Move all outputs to CPU for comparison
            # Normalize outputs to list for uniform handling (single tensor or tuple)
            if isinstance(custom_output, (tuple, list)):
                custom_outputs = list(custom_output)
                ref_outputs = list(ref_output)
                golden_outputs = list(golden_output)
            else:
                custom_outputs = [custom_output]
                ref_outputs = [ref_output]
                golden_outputs = [golden_output]

            precision_passed = True
            precision_diagnosis = []
            first_precision_result = None
            for out_idx, (c_out, r_out, g_out) in enumerate(zip(custom_outputs, ref_outputs, golden_outputs)):
                out_name = f"Output_{out_idx}" if len(custom_outputs) > 1 else "Output"
                out_dtype = c_out.dtype
                # Quantization operators: ±1 LSB tolerance for integer outputs
                out_tolerances = None
                if quantized_output and not out_dtype.is_floating_point:
                    out_tolerances = {"atol": 1, "rtol": 0.0}
                pr = dual_inspect(
                    y_ans=c_out.cpu(),
                    y_ref=r_out.cpu(),
                    y_golden=g_out.cpu() if g_out.is_floating_point() else g_out.cpu().to(r_out.dtype),
                    name=out_name,
                    dtype=out_dtype,
                    tolerances=out_tolerances,
                )
                if first_precision_result is None:
                    first_precision_result = pr
                if not pr.passed:
                    precision_passed = False
                    diag = str(pr.diagnosis) if pr.diagnosis else "Unknown"
                    precision_diagnosis.append(f"{out_name}: {diag}")

            precision_result = first_precision_result

            if not precision_passed:
                diagnosis_str = "; ".join(precision_diagnosis) if precision_diagnosis else "Unknown"
                logging.warning(f"  ❌ Precision FAILED: {diagnosis_str}")
                # Always print failures (even in quiet mode) — critical for debugging
                if quiet:
                    print(f"[{idx+1}/{len(cases)}] {case_name:20s} ❌ FAIL  {diagnosis_str}", flush=True)
                all_passed = False
                results.append({
                    "case_id": case_id,
                    "case_name": case_name,
                    "status": "FAIL",
                    "reason": diagnosis_str,
                    "speedup": 0.0,
                })
                continue

            logging.info(f"  ✅ Precision PASSED")

            # Performance measurement with msprof (optional)
            speedup = 0.0
            perf_result = {}

            if not skip_performance:
                logging.info(f"  Running performance measurement (msprof)...")
                try:
                    # Extract device_id from device string (e.g., "npu:2" -> 2)
                    device_id_int = int(device.split(":")[-1]) if ":" in device else device_id

                    # In quiet mode, suppress msprof subprocess stdout/stderr
                    # (the bulk of noise: ~10KB per case of path/analysis logs)
                    with _perf_ctx():
                        perf_result = measure_performance_msprof(
                            ref_model=ref_model,
                            custom_model=custom_model,
                            inputs=inputs,
                            work_dir=work_dir,
                            case_name=case_name,
                            device_id=device_id_int,
                            num_trials=10,
                        )

                    speedup = perf_result.get("speedup", 0.0)
                    speedups.append(speedup)

                    logging.info(f"  Reference:        {perf_result.get('ref_time_us', 0.0):.2f}us")
                    logging.info(f"  Custom:           {perf_result.get('custom_time_us', 0.0):.2f}us")
                    logging.info(f"  Speedup:          {speedup:.2f}x")
                    logging.info(f"  Bottleneck class: {perf_result.get('bottleneck_class', 'Unknown')}")
                except Exception as perf_err:
                    logging.warning(f"  ⚠️  Performance measurement failed: {perf_err}")
                    logging.warning(f"  Continuing with precision-only results...")
            else:
                logging.info(f"  ⏭️  Performance measurement skipped")

            # Convert precision_result to dict for JSON serialization.
            # ratios is None for integer-dtype operators (see precision.py), so guard
            # against NoneType before accessing attributes.
            precision_dict = {
                "passed": precision_result.passed,
                "ratios": {
                    "max_re": precision_result.ratios.max_re,
                    "mean_re": precision_result.ratios.mean_re,
                    "rmse": precision_result.ratios.rmse,
                    "svec": precision_result.ratios.svec,
                } if precision_result.ratios is not None else None,
                "diagnosis": str(precision_result.diagnosis) if precision_result.diagnosis else None,
            }

            results.append({
                "case_id": case_id,
                "case_name": case_name,
                "status": "PASS",
                "speedup": speedup,
                "ref_time_us": perf_result.get("ref_time_us", 0.0),
                "custom_time_us": perf_result.get("custom_time_us", 0.0),
                "bottleneck_class": perf_result.get("bottleneck_class", "Unknown"),
                "precision": precision_dict,
            })

            # Quiet-mode: compact one-line per-case result
            if quiet:
                ref_us = perf_result.get("ref_time_us", 0.0)
                cust_us = perf_result.get("custom_time_us", 0.0)
                spd_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
                print(
                    f"[{idx+1}/{len(cases)}] {case_name:20s} ✅ PASS  "
                    f"ref={ref_us:>8.1f}us  custom={cust_us:>8.1f}us  "
                    f"speedup={spd_str:>7s}",
                    flush=True,
                )

        except Exception as e:
            logging.error(f"  ❌ Case failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            results.append({
                "case_id": case_id,
                "case_name": case_name,
                "status": "ERROR",
                "error": str(e),
                "speedup": 0.0,
            })
            # Always print errors (even in quiet mode) — critical for debugging
            if quiet:
                print(f"[{idx+1}/{len(cases)}] {case_name:20s} ❌ ERROR {e}", flush=True)

    # Compute geometric mean
    if speedups:
        geo_mean = float(np.exp(np.mean(np.log(speedups))))
        logging.info(f"\n{'='*70}")
        logging.info(f"Geometric mean speedup: {geo_mean:.2f}x (over {len(speedups)} passed tests)")
        logging.info(f"{'='*70}")
    else:
        geo_mean = 0.0
        logging.warning("No tests passed - cannot compute geometric mean")

    # Summary
    passed_count = sum(1 for r in results if r.get("status") == "PASS")
    summary = (
        f"[{'ALL PASSED' if all_passed else 'SOME FAILED'}] "
        f"{passed_count}/{len(cases)} tests passed, "
        f"geometric mean speedup={geo_mean:.2f}x"
    )

    logging.info(f"\n{summary}")

    # Quiet-mode: always print the summary line (visible regardless of log level)
    if quiet:
        print(f"\n{summary}", flush=True)

    # Write JSON output
    if json_output_path:
        output_data = {
            "operator": loaded_cases["operator"],
            "operator_type": operator_type,
            "total_cases": len(cases),
            "passed_cases": passed_count,
            "geometric_mean_speedup": geo_mean,
            "results": results,
        }
        json_path = Path(json_output_path)
        json_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        logging.info(f"Results written to: {json_path}")

    return all_passed, summary, {"geometric_mean": geo_mean, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation with multi-case testing")
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Work directory (default: output/<op_name>)")
    parser.add_argument("--device-id", type=int,
                        default=int(os.environ.get("ASCEND_DEVICE_ID", "0")),
                        help="NPU device ID (default: from ASCEND_DEVICE_ID or 0)")
    parser.add_argument("--no-flush-l2", action="store_true",
                        help="Disable L2 cache flushing (default: enabled)")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip performance measurement, precision only")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Path to write JSON output")
    parser.add_argument("--api-desc", type=str, default=None,
                        help="Explicit path to api_description(.md). If omitted, fallback to work_dir/api_description.md or api_desc.md")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full output including msprof subprocess logs. "
                        "Default is quiet mode: only case results, errors, and summary.")

    args = parser.parse_args()

    # Quiet mode by default: suppress INFO, keep WARNING/ERROR for diagnostics
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        if args.work_dir:
            work_dir = Path(args.work_dir).resolve()
        else:
            work_dir = Path("output").joinpath(args.op_name).resolve()

        _pybind_wd = work_dir.parent if work_dir.name.endswith("Custom") else work_dir
        _inject_pybind_lib(_pybind_wd)

        # Setup environment
        setup_environment(work_dir)

        # Run evaluation
        all_passed, summary, results = evaluate_multi_case(
            op_name=args.op_name,
            work_dir=work_dir,
            device_id=args.device_id,
            flush_l2=not args.no_flush_l2,
            json_output_path=args.json_output,
            skip_performance=args.skip_performance,
            api_desc=args.api_desc,
            verbose=args.verbose,
        )

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

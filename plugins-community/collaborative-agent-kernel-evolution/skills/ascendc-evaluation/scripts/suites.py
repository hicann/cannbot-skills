# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""High-level test suite functions for precision comparison.

Provides try_op(), compare_precision(), and smoke_test() that encapsulate
the three-way comparison pattern (golden vs ref vs ans) used across all
operator test files.
"""

import inspect
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch import nn

from constants import DEFAULT_TOLERANCES
from precision import _compute_comparison, _count_element_stats


# ---------------------------------------------------------------------------
# Custom exceptions for structured error reporting
# ---------------------------------------------------------------------------

class TestError(RuntimeError):
    """Test configuration error: golden/ref model fails when it should work.

    Fix the test code, not the target kernel.
    """
    pass


class TargetError(RuntimeError):
    """Target code error: the custom kernel (ans model) behaves unexpectedly.

    Includes: NotImplementedError in three-way/two-way, backward crash,
    namespace conflict (op name shadows SDK built-in in baseline mode).
    """
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_grad_names(model_cls: Type[nn.Module], inputs: Tuple) -> List[str]:
    """Extract forward() parameter names for float inputs (used as grad names)."""
    sig = inspect.signature(model_cls.forward)
    param_names = [p.name for p in list(sig.parameters.values())[1:]]  # skip self
    return [name for name, t in zip(param_names, inputs)
            if isinstance(t, torch.Tensor) and t.is_floating_point()]


def try_op(
    model_cls: Type[nn.Module],
    inputs: Tuple,
    init_inputs: Tuple[Any, ...] = (),
    *,
    role: Literal['ans', 'ref', 'golden'],
    require_grads: bool = False,
    n_warmup: int = 0,
    n_repeat: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[dict]:
    """Create model, place inputs on correct device, run forward.

    role='golden' -> CPU fp64
    role='ref'/'ans' -> explicit device (or CPU if None)

    Catches NotImplementedError in __init__ -> returns None.

    Returns:
        {
            'output': Tensor,
            'tracked': List[Tensor],
            'model': nn.Module,
            'fwd_inputs': List[Tensor],
            'device': torch.device,
        }
        or None
    """
    try:
        model = model_cls(*init_inputs)
    except NotImplementedError:
        return None

    if role == 'golden':
        actual_device = torch.device('cpu')
        fwd_inputs = []
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                fwd_inputs.append(t)
            elif t.is_floating_point():
                fwd_inputs.append(t.double())
            else:
                fwd_inputs.append(t.clone())
    else:
        actual_device = device if device is not None else torch.device('cpu')
        fwd_inputs = [t.to(actual_device) if isinstance(t, torch.Tensor) else t
                      for t in inputs]

    model = model.to(actual_device)

    tracked = []
    if require_grads:
        for i, t in enumerate(fwd_inputs):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t = t.detach().clone().requires_grad_(True)
                fwd_inputs[i] = t
                tracked.append(t)

    # Warmup iterations run without gradient tracking to stabilize timing.
    for _ in range(n_warmup):
        with torch.no_grad():
            model(*fwd_inputs)

    # run
    for _ in range(n_repeat):
        output = model(*fwd_inputs)

    return {
        'output': output,
        'tracked': tracked,
        'model': model,
        'fwd_inputs': fwd_inputs,
        'device': actual_device,
    }


# ---------------------------------------------------------------------------
# Precision comparison
# ---------------------------------------------------------------------------

def compare_precision(
    op_name: str,
    ans_model_cls: Type[nn.Module],
    ref_model_cls: Type[nn.Module],
    golden_model_cls: Type[nn.Module],
    inputs: Tuple,
    init_inputs: Tuple[Any, ...] = (),
    grad_output: Optional[torch.Tensor] = None,
    *,
    mode: Literal["three-way", "baseline", "two-way"] = "three-way",
    tolerances: Optional[dict] = None,
    case_meta: Optional[dict] = None,
    device: Optional[torch.device] = None,
    collector=None,
    output_names: Optional[List[str]] = None,
) -> dict:
    """Precision comparison with explicit mode validation.

    The caller declares the expected ``mode``:

    - **three-way** (default): ans, ref, golden all required.
      golden/ref failure → ``TestError``.
      ans failure → ``TargetError``.
    - **baseline**: ans expected to raise ``NotImplementedError``.
      ref + golden required.  If ans unexpectedly succeeds →
      ``TargetError``.
    - **two-way**: ans + ref required, golden optional.
      golden is only used to determine the small-value region boundary;
      if absent, ref is used instead.

    ``passed`` is determined solely by ``mismatch_rate`` (from isclose).

    Args:
        op_name: Operator name for collector metadata.
        ans_model_cls: ModelNew class (custom NPU kernel).
        ref_model_cls: Model class (PyTorch reference).
        golden_model_cls: Golden model class (often same as ref).
        inputs: Tuple of CPU tensors in working dtype.
        init_inputs: Tuple of model constructor arguments.
        grad_output: If provided, enables backward comparison.
        mode: Expected comparison mode.
        tolerances: Per-component tolerance dicts.
        case_meta: Dict with id, params, seed, distr for collector.
        output_names: Optional list of output component names for multi-output
            operators. When provided, the model's forward output is expected to
            be a tuple/list; each element is recorded under its corresponding
            name. When None (default), a single "Output" component is used.

    Returns:
        {
            "passed": bool,
            "mode": str,
            "note": str | None,
            "forward": {"Output": {component_dict}} | {"y": ..., "z": ...},
            "backward": {"Grad-x": {component_dict}, ...} | None,
        }

    Raises:
        TestError: golden/ref fail when they should work (fix the test).
        TargetError: ans fails or namespace conflict (fix the kernel).
    """
    if tolerances is None:
        tolerances = {}

    # --- Run forward for each role ---
    r_golden = try_op(golden_model_cls, inputs, init_inputs, role='golden', device=device)
    r_ref = try_op(ref_model_cls, inputs, init_inputs, role='ref', device=device)
    r_ans = try_op(ans_model_cls, inputs, init_inputs, role='ans', device=device)

    # --- Validate mode vs actual results ---
    note = None

    if mode == "three-way":
        # golden and ref must succeed — if not, it's a test bug
        if r_golden is None:
            raise TestError(
                f"mode='three-way' but golden model "
                f"({golden_model_cls.__name__}) raised NotImplementedError. "
                f"Fix golden_model_cls or use mode='two-way'."
            )
        if r_ref is None:
            raise TestError(
                f"mode='three-way' but ref model "
                f"({ref_model_cls.__name__}) raised NotImplementedError. "
                f"Fix ref_model_cls or use mode='two-way'."
            )
        # ans failure = target code bug
        if r_ans is None:
            raise TargetError(
                f"mode='three-way' but ans model "
                f"({ans_model_cls.__name__}) raised NotImplementedError. "
                f"Custom kernel not implemented."
            )

    elif mode == "baseline":
        # ref and golden must succeed
        if r_golden is None:
            raise TestError(
                f"mode='baseline' but golden model "
                f"({golden_model_cls.__name__}) raised NotImplementedError."
            )
        if r_ref is None:
            raise TestError(
                f"mode='baseline' but ref model "
                f"({ref_model_cls.__name__}) raised NotImplementedError."
            )
        # ans should NOT succeed — namespace conflict
        if r_ans is not None:
            raise TargetError(
                f"mode='baseline' but ans model "
                f"({ans_model_cls.__name__}) ran successfully. The op name "
                f"likely shadows a built-in SDK operator. Rename with a "
                f"unique suffix."
            )
        # Normal baseline: use ref as ans
        note = "baseline: ans not implemented, using ref as ans"

    elif mode == "two-way":
        # ans and ref must succeed
        if r_ref is None:
            raise TestError(
                f"mode='two-way' but ref model "
                f"({ref_model_cls.__name__}) raised NotImplementedError."
            )
        if r_ans is None:
            raise TargetError(
                f"mode='two-way' but ans model "
                f"({ans_model_cls.__name__}) raised NotImplementedError. "
                f"Custom kernel not implemented."
            )
        # golden is optional; if absent, ref is used for small-value range
        if r_golden is None:
            note = "two-way: golden not available, using ref for small-value range"

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # --- Build forward component(s) ---
    if output_names is not None:
        # Multi-output path: model returns tuple/list, one component per name
        forward = {}
        all_passed = True
        for idx, name in enumerate(output_names):
            fwd_component = _build_forward_component(
                r_ans, r_ref, r_golden, mode, tolerances.get(name), index=idx)
            forward[name] = fwd_component
            if not fwd_component["passed"]:
                all_passed = False
        result = {
            "passed": all_passed,
            "mode": mode,
            "note": note,
            "forward": forward,
            "backward": None,
        }
        if case_meta is not None and collector is not None:
            for name, fwd_component in forward.items():
                collector.add_forward(
                    name, fwd_component,
                    case_meta["params"], case_meta["seed"],
                    case_meta["distr"], case_meta["id"],
                )
            if note:
                collector.set_note(note)
            collector.set_mode(mode)
    else:
        # Single-output path (backward compatible)
        fwd_component = _build_forward_component(
            r_ans, r_ref, r_golden, mode, tolerances.get("Output"))
        result = {
            "passed": fwd_component["passed"],
            "mode": mode,
            "note": note,
            "forward": {"Output": fwd_component},
            "backward": None,
        }
        if case_meta is not None and collector is not None:
            collector.add_forward(
                "Output", fwd_component,
                case_meta["params"], case_meta["seed"],
                case_meta["distr"], case_meta["id"],
            )
            if note:
                collector.set_note(note)
            collector.set_mode(mode)

    # --- Backward ---
    if grad_output is not None:
        grad_names = _get_grad_names(ref_model_cls, inputs)
        backward = {}

        # Run forward+backward for each available role
        r_golden_bw = try_op(golden_model_cls, inputs, init_inputs,
                             role='golden', require_grads=True, device=device) if r_golden else None
        r_ref_bw = try_op(ref_model_cls, inputs, init_inputs,
                          role='ref', require_grads=True, device=device) if r_ref else None
        r_ans_bw = try_op(ans_model_cls, inputs, init_inputs,
                          role='ans', require_grads=True, device=device) if r_ans else None

        def _get_grads(r):
            if r is None:
                return None
            try:
                g_out = grad_output.to(dtype=r['output'].dtype, device=r['device'])
                return torch.autograd.grad(r['output'], r['tracked'], grad_outputs=g_out)
            except Exception:
                return None

        grads_golden = _get_grads(r_golden_bw)
        grads_ref = _get_grads(r_ref_bw)
        grads_ans = _get_grads(r_ans_bw)

        # Backward: golden/ref grads must work if forward worked
        if mode in ("three-way", "two-way"):
            if grads_ref is None:
                raise TestError(
                    " ref model forward succeeded but "
                    "backward failed. Check ref_model_cls.backward()."
                )
        if mode == "three-way":
            if grads_golden is None:
                raise TestError(
                    " golden model forward succeeded but "
                    "backward failed. Check golden_model_cls.backward()."
                )

        # ans backward failure = target code bug
        bw_mode = mode
        if grads_ans is None and mode in ("three-way", "two-way"):
            raise TargetError(
                f" ans model ({ans_model_cls.__name__}) forward "
                f"succeeded but backward failed."
            )
        elif grads_ans is None and mode == "baseline":
            bw_mode = "baseline"

        for idx, name in enumerate(grad_names):
            comp_name = f"Grad-{name}"

            g_golden = grads_golden[idx].cpu() if grads_golden else None
            g_ref = grads_ref[idx].cpu() if grads_ref else None
            g_ans = grads_ans[idx].cpu() if grads_ans else None

            # In baseline mode, use ref as ans
            if bw_mode == "baseline" and g_ans is None:
                g_ans = g_ref

            # golden fallback to ref for small-value range
            if g_golden is None:
                g_golden = g_ref

            grad_component = _build_component(
                g_ans, g_ref, g_golden, bw_mode,
                tolerances.get(comp_name))

            backward[comp_name] = grad_component

            if not grad_component["passed"]:
                result["passed"] = False

            if case_meta is not None and collector is not None:
                collector.add_backward(
                    comp_name, grad_component,
                    case_meta["params"], case_meta["seed"],
                    case_meta["distr"], case_meta["id"],
                )

        result["backward"] = backward

    return result


def _build_forward_component(r_ans, r_ref, r_golden, mode, tol, index=None):
    """Build forward component dict from try_op results.

    Args:
        index: If not None, extract element at this index from the output
            tuple/list. Used for multi-output operators.
    """
    def _extract(r):
        if r is None:
            return None
        out = r['output']
        if index is not None:
            out = out[index]
        return out.cpu()

    y_ans = _extract(r_ans)
    y_ref = _extract(r_ref)
    y_golden = _extract(r_golden)

    # baseline: use ref as ans
    if mode == "baseline":
        y_ans = y_ref

    # golden is used only for small-value range; if absent, use ref
    if y_golden is None:
        y_golden = y_ref

    return _build_component(y_ans, y_ref, y_golden, mode, tol)


def _resolve_component_tol(y_ans, y_ref, y_golden, tol):
    """Resolve tolerance dict; infer from first available tensor when tol is None."""
    if tol is not None:
        return tol
    for t in (y_ans, y_ref, y_golden):
        if t is not None:
            return DEFAULT_TOLERANCES.get(t.dtype, DEFAULT_TOLERANCES[torch.bfloat16])
    return DEFAULT_TOLERANCES[torch.bfloat16]


def _infer_working_dtype(y_ans, y_ref):
    """Infer working dtype from first available non-golden tensor (default bfloat16)."""
    for t in (y_ans, y_ref):
        if t is not None:
            return t.dtype
    return torch.bfloat16


def _pairwise_comparisons(component, y_ans, y_ref, y_golden, tol, dtype):
    """Populate component with pairwise comparison dicts for available tensor pairs."""
    atol = tol["atol"]
    rtol = tol["rtol"]
    ulp_tol = tol.get("ulp_tol")
    sv_th = tol.get("sv_th")
    sv_err = tol.get("sv_err")
    ulp_kw = dict(ulp_method=tol.get("ulp_method", "bitwise"),
                  include_subnormal=tol.get("include_subnormal", True))
    if y_ans is not None and y_golden is not None:
        component["ans_vs_golden"] = _compute_comparison(
            y_ans, y_golden, y_golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
            **ulp_kw,
        ).to_dict()
    if y_ref is not None and y_golden is not None:
        component["ref_vs_golden"] = _compute_comparison(
            y_ref, y_golden, y_golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
            **ulp_kw,
        ).to_dict()
    if y_ans is not None and y_ref is not None:
        component["ans_vs_ref"] = _compute_comparison(
            y_ans, y_ref, y_golden if y_golden is not None else y_ref,
            sv_th, sv_err, atol, rtol, ulp_tol, dtype,
            **ulp_kw,
        ).to_dict()


def _component_ratios(component, mode, dtype):
    """Informational ratios for three-way float comparisons (None otherwise)."""
    has_both = "ans_vs_golden" in component and "ref_vs_golden" in component
    if not (mode == "three-way" and has_both and dtype.is_floating_point):
        return None
    ref_floor = 1e-7
    avg = component["ans_vs_golden"]
    rvg = component["ref_vs_golden"]
    return {
        "max_re": avg["re_max"] / max(rvg["re_max"], ref_floor),
        "mean_re": avg["re_mean"] / max(rvg["re_mean"], ref_floor),
        "rmse": avg["rmse"] / max(rvg["rmse"], ref_floor),
        "svec": avg["svec"] / max(rvg["svec"], 1),
    }


def _component_nan_inf_fail(component) -> bool:
    """True if golden+ref are clean but ans has a NaN/Inf anomaly."""
    ans_stats = component.get("ans", {})
    ref_stats = component.get("ref", {})
    golden_stats = component.get("golden", {})
    golden_ref_clean = (
        golden_stats.get("nan", 0) == 0 and golden_stats.get("inf", 0) == 0
        and ref_stats.get("nan", 0) == 0 and ref_stats.get("inf", 0) == 0
    )
    ans_has_anomaly = ans_stats.get("nan", 0) > 0 or ans_stats.get("inf", 0) > 0
    return golden_ref_clean and ans_has_anomaly


def _build_component(y_ans, y_ref, y_golden, mode, tol):
    """Build component dict with metrics and pass/fail.

    ``passed`` = no NaN/Inf anomaly AND mismatch_rate == 0.0.
    NaN/Inf anomaly: golden+ref are clean but ans has NaN or Inf.
    """
    tol = _resolve_component_tol(y_ans, y_ref, y_golden, tol)
    dtype = _infer_working_dtype(y_ans, y_ref)
    sv_th = tol.get("sv_th")

    component: Dict[str, Any] = {}

    # Element stats for available tensors
    if y_ans is not None:
        component["ans"] = _count_element_stats(y_ans, sv_th).to_dict()
    if y_ref is not None:
        component["ref"] = _count_element_stats(y_ref, sv_th).to_dict()
    if y_golden is not None:
        component["golden"] = _count_element_stats(y_golden, sv_th).to_dict()

    _pairwise_comparisons(component, y_ans, y_ref, y_golden, tol, dtype)

    ratios = _component_ratios(component, mode, dtype)
    if ratios is not None:
        component["ratios"] = ratios

    # Mismatch rate (prefer ans_vs_golden per spec)
    if "ans_vs_golden" in component:
        mismatch = component["ans_vs_golden"]["mismatch_rate"]
    elif "ans_vs_ref" in component:
        mismatch = component["ans_vs_ref"]["mismatch_rate"]
    else:
        mismatch = 0.0
    component["mismatch_rate"] = mismatch

    # NOTE: This pass/fail logic mirrors precision.dual_inspect().
    # Changes to the rule must be synchronized in both places.
    component["passed"] = (not _component_nan_inf_fail(component)
                           and math.isclose(mismatch, 0.0, abs_tol=1e-12))
    return component


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(
    op_name: str,
    model_cls: Type[nn.Module],
    inputs: Tuple,
    init_inputs: Tuple[Any, ...] = (),
    target: Optional[torch.Tensor] = None,
    loss_func=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Simple smoke test: verify forward (and optionally backward) runs without error.

    Returns:
        {"passed": bool, "output_shape": list, "output_dtype": str, "error": str|None}
    """
    try:
        r = try_op(model_cls, inputs, init_inputs, role='ref', device=device)
        if r is None:
            return {"passed": False, "error": "Model raised NotImplementedError"}

        output = r['output']
        # Normalize to list for multi-output support
        outputs = output if isinstance(output, (tuple, list)) else [output]
        for i, o in enumerate(outputs):
            if not isinstance(o, torch.Tensor):
                raise AssertionError(f"Output {i} is not a Tensor")
            if o.is_floating_point():
                if not torch.isfinite(o).all():
                    raise AssertionError(f"Output {i} contains NaN/Inf")
            if not o.numel() > 0:
                raise AssertionError(f"Output {i} is empty")

        first = outputs[0]
        result = {
            "passed": True,
            "output_shape": list(first.shape),
            "output_dtype": str(first.dtype),
            "n_outputs": len(outputs),
            "error": None,
        }

        # Optionally test backward
        if loss_func and target is not None:
            r_bw = try_op(model_cls, inputs, init_inputs,
                          role='ref', require_grads=True, device=device)
            if r_bw is not None:
                t_target = target.to(r_bw['device'])
                loss = loss_func(r_bw['output'], t_target)
                loss.backward()
                result["backward"] = True

        return result
    except Exception as e:
        return {"passed": False, "error": str(e)}

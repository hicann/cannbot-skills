#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Kernel regression check for ``model_new_ascendc.py`` degeneracy.

Drift types:
  Type 1: No kernel import (pure PyTorch fallback — kernel not wired in)
  Type 2: Kernel imported but forward() never calls it (orphan import)
  Type 3: forward() calls kernel BUT also uses torch compute ops (partial fallback)
  Type 4: forward() has Python scalar for-loop (degenerated to per-element scalar code)

Backend kernel import patterns recognized:
  AscendC (a5_ops native):
    - `import _<name>_ext as <alias>` (pybind11 C-extension)
    - `from kernel.<module> import <kernel_fn>`
    - `import kernel.<module> as <alias>`
    - `from .kernel import <kernel_fn>` (relative)
Usage:
  python validate_kernel_regression.py <model_new_*.py> [--json]

Exit code: 0 = PASS, 1 = drift detected (regression).
"""
import argparse
import ast
import json
import sys

# ----------------------------------------------------------------------------
# Allow-lists (same semantics as cv-agent — buffer alloc + shape ops are OK)
# ----------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like", "tensor", "arange", "linspace",
    "as_tensor",
    # RNG / utilities (not compute on tensors)
    "manual_seed", "set_default_dtype", "set_default_device",
    "get_default_dtype",
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like",
}

# Qualifiers whose method calls are NEVER tensor compute (Python scalar math etc.)
SCALAR_MATH_QUALIFIERS = {
    "math", "np", "numpy", "operator", "statistics", "fractions",
    "os", "sys", "json", "Path", "logging",
}

ALLOWED_TENSOR_METHODS = {
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "item", "tolist",
    "requires_grad_", "zero_",
    "index_select", "is_npu",
}

ALLOWED_BUILTIN_FUNCS = {
    "min", "max", "abs", "len", "range", "int", "float", "bool",
    "list", "tuple", "str", "type", "isinstance", "print",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "hasattr", "getattr", "setattr",
}

FORBIDDEN_TENSOR_METHODS = {
    "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
    "argmax", "argmin", "var", "std",
    "matmul", "mm", "bmm", "addmm",
    "add", "sub", "mul", "div", "fmod", "remainder",
    "add_", "sub_", "mul_", "div_",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "relu_", "sigmoid_", "tanh_",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "sin", "cos", "clamp", "clamp_", "ceil", "floor", "round",
    "reciprocal", "neg", "sign",
    "softmax", "log_softmax",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
    "eq", "ne", "lt", "gt", "le", "ge", "where",
}

# ----------------------------------------------------------------------------
# AST helpers
# ----------------------------------------------------------------------------


def _resolve_call_name(node):
    """Return (qualifier, attr) for a Call node, or None.
    torch.empty -> ('torch', 'empty'); _ext.run(...) -> ('_ext', 'run'); foo(x) -> (None, 'foo').
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _is_kernel_module(module_path):
    """Match AscendC kernel-bearing module paths.

    Patterns matched:
      - kernel.X                    (AscendC subdirectory import)
      - kernel.X.Y                  (AscendC nested)
      - .kernel                     (relative AscendC)
      - .kernel.X                   (relative AscendC nested)
    Does NOT match: torch.*, torch.nn.*, F.*, etc.
    """
    if not module_path:
        return False
    parts = module_path.split(".")
    # AscendC: kernel.* OR .kernel(.X)
    if parts[0] == "kernel":
        return True
    return False


def _is_ext_module_alias(name):
    """Detect pybind11 C-ext alias name pattern (a5_ops convention).

    a5_ops convention: `import _<op_name>_ext as _ext` — C-extension binary loaded
    via the build dir (sys.path injected). The alias `_ext` or the leading-`_`
    name with trailing `_ext` is the signature.
    """
    if not name:
        return False
    # leading underscore + trailing _ext (C extension binding)
    if name.startswith("_") and name.endswith("_ext"):
        return True
    # common alias `_ext`
    if name == "_ext":
        return True
    return False


def find_kernel_imports(tree):
    """Find all imported AscendC kernel bindings.

    Returns dict: {bound_name: {"actual": str, "module": str, "kind": str, "line": int}}
    where kind is ``ascendc_ext`` or ``ascendc_kernel``.
    """
    kernels = {}
    for node in ast.walk(tree):
        # `from <module> import <name>` form
        if isinstance(node, ast.ImportFrom) and node.module:
            if _is_kernel_module(node.module):
                for alias in node.names:
                    bound = alias.asname if alias.asname else alias.name
                    kernels[bound] = {
                        "actual": alias.name, "module": node.module,
                        "kind": "ascendc_kernel", "line": node.lineno,
                    }
        # `import <module> as <alias>` form
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name
                if _is_ext_module_alias(alias.name) or _is_ext_module_alias(bound):
                    kernels[bound] = {
                        "actual": alias.name, "module": alias.name,
                        "kind": "ascendc_ext", "line": node.lineno,
                    }
                elif _is_kernel_module(alias.name):
                    kernels[bound] = {
                        "actual": alias.name, "module": alias.name,
                        "kind": "ascendc_kernel", "line": node.lineno,
                    }
    return kernels


def find_model_forward(tree):
    """Find ModelNew.forward or Model.forward. Prefer ModelNew if both exist."""
    model_new_forward = model_forward = None
    model_new_class = model_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "ModelNew":
                model_new_class = node
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                        model_new_forward = item
            elif node.name == "Model":
                model_class = node
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                        model_forward = item
    if model_new_forward:
        return model_new_forward, "ModelNew", model_new_class
    return model_forward, "Model", model_class


def _find_class_methods(class_node):
    """Build {method_name: FunctionDef} for instance methods of class_node (excludes nested classes)."""
    if class_node is None:
        return {}
    methods = {}
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = item
    return methods


def _method_invokes_kernel(method_node, kernel_bound_names, all_methods, visited=None):
    """Return True iff method_node body (transitively via self.<method>) invokes a kernel binding.

    Recursion bound: `visited` tracks names already scanned (cycle + redundancy guard).
    Limits depth implicitly by not re-entering visited methods.
    """
    if method_node is None:
        return False
    if visited is None:
        visited = set()
    if method_node.name in visited:
        return False
    visited.add(method_node.name)

    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if not resolved:
            continue
        qual, attr = resolved
        # Direct kernel-bound call
        if qual is None and attr in kernel_bound_names:
            return True
        # Qualified kernel attribute call (e.g. _ext.run_cat)
        if qual in kernel_bound_names:
            return True
        # self.<other_method>() — recurse
        if qual == "self" and attr in all_methods and attr != method_node.name:
            if _method_invokes_kernel(all_methods[attr], kernel_bound_names, all_methods, visited):
                return True
    return False


def check_kernel_called(forward_node, kernel_bound_names, class_node=None):
    """Return list of kernel call events in forward(). Patterns:
       1. direct: <bound>(x, y, ...)              — AscendC binding call
       2. attribute: <bound>.<method>(x, y, ...)  — _ext.run_cat(...) AscendC convention
       3. assign-then-call: k = <bound>(...); k(x, y)
       4. self-indirection: forward() calls self.<helper>() whose body invokes kernel
          — cv-agent's `self._build_kernel(...)` pattern + caching helpers
    """
    if forward_node is None:
        return []
    called = []
    kernel_var_names = set()

    # Fix C: pre-compute set of `self.<method>` names whose bodies transitively call kernel
    all_methods = _find_class_methods(class_node)
    indirection_methods = set()
    for name, method in all_methods.items():
        if name == "forward":
            continue
        if _method_invokes_kernel(method, kernel_bound_names, all_methods):
            indirection_methods.add(name)

    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if not resolved:
            continue
        qual, attr = resolved
        # direct: <kernel_bound>(...)
        if qual is None and attr in kernel_bound_names:
            called.append({"call": attr, "line": node.lineno, "pattern": "direct_call"})
        # attribute: <bound_module>.<func>(...) — AscendC `_ext.run_cat(...)`
        if qual in kernel_bound_names:
            called.append({"call": f"{qual}.{attr}", "line": node.lineno, "pattern": "attribute_call"})
        # Fix C: self.<helper>() where helper transitively invokes kernel
        if qual == "self" and attr in indirection_methods:
            called.append({"call": f"self.{attr}", "line": node.lineno, "pattern": "self_indirection"})
        # assign-then-call: kernel_var = <bound>(...); kernel_var(...)
        # Also catches: kernel_var = self.<helper>(...); kernel_var(...) — Fix C tail
        if isinstance(node, ast.Call):
            for ancestor in ast.walk(forward_node):
                if isinstance(ancestor, ast.Assign) and ancestor.value is node:
                    is_kernel_source = (
                        (qual is None and attr in kernel_bound_names)
                        or (qual == "self" and attr in indirection_methods)
                    )
                    if is_kernel_source:
                        for target in ancestor.targets:
                            if isinstance(target, ast.Name):
                                kernel_var_names.add(target.id)
        if qual is None and attr in kernel_var_names:
            called.append({"call": f"{attr}(...)", "line": node.lineno, "pattern": "kernel_invoke"})
    return called


def check_forbidden_torch_ops(forward_node, kernel_bound_names):
    """Return list of forbidden torch compute calls in forward()."""
    if forward_node is None:
        return []
    violations = []
    for node in ast.walk(forward_node):
        # @ matmul operator
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({"line": node.lineno, "call": "@",
                               "reason": "matrix multiply must be in kernel, not forward()"})
            continue
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if not resolved:
            continue
        qual, attr = resolved
        # allow kernel calls
        if qual is None and attr in kernel_bound_names:
            continue
        if qual in kernel_bound_names:
            continue
        # torch.xxx
        if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
            violations.append({"line": node.lineno, "call": f"torch.{attr}",
                               "reason": f"torch.{attr} is a compute op — must be in kernel"})
            continue
        # F.xxx / functional.xxx
        if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
            violations.append({"line": node.lineno, "call": f"{qual}.{attr}",
                               "reason": f"{qual}.{attr} is PyTorch compute — must be in kernel"})
            continue
        # builtins ok
        if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
            continue
        # Scalar-math qualifiers (math.ceil, np.array, etc.) — NEVER tensor compute
        if qual in SCALAR_MATH_QUALIFIERS:
            continue
        # tensor.method() forbidden compute
        if attr in FORBIDDEN_TENSOR_METHODS and qual not in (
                "torch", "F", "functional", "torch.nn.functional", "nn.functional"):
            violations.append({"line": node.lineno,
                               "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                               "reason": f"{attr} is a compute method — must be in kernel"})
    return violations


def check_scalar_for_loops(forward_node):
    """Detect `for i in range(N): ...` with tensor[i] indexing + compute inside."""
    if forward_node is None:
        return []
    violations = []
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.For):
            continue
        if isinstance(node.iter, ast.Call):
            resolved = _resolve_call_name(node.iter)
            if resolved and resolved == (None, "range"):
                loop_var = node.target.id if isinstance(node.target, ast.Name) else ""
                if _loop_has_tensor_indexing(node, loop_var) and _loop_has_computation(node):
                    violations.append({"line": node.lineno, "loop_var": loop_var,
                                       "reason": "scalar for-loop over tensor indices — must vectorize in kernel"})
    return violations


def _loop_has_tensor_indexing(for_node, loop_var):
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if isinstance(child, ast.Subscript):
            for sub in ast.walk(child.slice):
                if isinstance(sub, ast.Name) and sub.id == loop_var:
                    return True
    return False


def _loop_has_computation(for_node):
    binop_count = 0
    arith = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    for child in ast.walk(for_node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True
        if isinstance(child, ast.BinOp) and isinstance(child.op, arith):
            binop_count += 1
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if not resolved:
                continue
            qual, attr = resolved
            if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue
            if attr in FORBIDDEN_TENSOR_METHODS and qual not in (
                    "torch", "F", "functional", "torch.nn.functional", "nn.functional"):
                return True
            if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
                return True
            if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
                return True
    return binop_count >= 5


# ----------------------------------------------------------------------------
# Main validate
# ----------------------------------------------------------------------------

def validate(code, filepath="<unknown>"):
    """Statically check one `model_new_*.py` source for kernel-regression drift.

    Runs the four ordered checks (short-circuiting on the first failure): kernel
    import present, forward() actually calls it, forward() has no forbidden torch
    compute ops, forward() has no scalar python for-loop over tensor indices.

    Args:
        code: full Python source of the model file (str). Parsed with `ast`.
        filepath: label echoed back into the result for diagnostics (not read).

    Returns:
        dict with keys:
          `valid` (bool) — True only if all four checks pass;
          `regression_type` (int|None) — 1=no import, 2=orphan import, 3=partial
            torch fallback, 4=scalar for-loop; None when valid;
          `checks` — per-check {passed, ...details, error} sub-dicts;
          `suggestion` (str) — human-readable fix for the first failing check.
        A SyntaxError in `code` is reported as regression_type 1 (not raised).
    """
    result = {"valid": False, "filepath": filepath, "regression_type": None,
              "checks": {"kernel_imported": {"passed": False, "kernels": [], "error": None},
                         "kernel_called": {"passed": False, "called": [], "error": None},
                         "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
                         "no_scalar_for_loops": {"passed": False, "violations": [], "error": None}},
              "suggestion": ""}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["regression_type"] = 1
        result["suggestion"] = f"SyntaxError: {e}"
        return result

    # Check 1: kernel import
    kernels = find_kernel_imports(tree)
    result["checks"]["kernel_imported"]["kernels"] = [
        {"bound": b, "actual": v["actual"], "module": v["module"], "kind": v["kind"], "line": v["line"]}
        for b, v in kernels.items()]
    if not kernels:
        result["regression_type"] = 1
        result["suggestion"] = (
            "No kernel import detected. Expected: AscendC `import _<name>_ext as _ext` / "
            "`from kernel.<module> import <fn>`. forward() cannot call a "
            "kernel that isn't imported.")
        result["checks"]["kernel_imported"]["error"] = "no kernel-module import found"
        return result
    result["checks"]["kernel_imported"]["passed"] = True

    # Check 2: kernel called from forward
    kernel_bound_names = set(kernels.keys())
    forward_node, class_name, class_node = find_model_forward(tree)
    if forward_node is None:
        result["regression_type"] = 2
        result["suggestion"] = "No ModelNew.forward or Model.forward method found."
        result["checks"]["kernel_called"]["error"] = "no forward method"
        return result
    called = check_kernel_called(forward_node, kernel_bound_names, class_node=class_node)
    result["checks"]["kernel_called"]["called"] = called
    if not called:
        result["regression_type"] = 2
        result["suggestion"] = (
            f"Imported kernel binding(s) {sorted(kernel_bound_names)} but {class_name}.forward() "
            "never calls them. Orphan import — kernel not wired into the model.")
        result["checks"]["kernel_called"]["error"] = "kernel imported but not called"
        return result
    result["checks"]["kernel_called"]["passed"] = True

    # Check 3: forbidden torch compute ops in forward
    violations = check_forbidden_torch_ops(forward_node, kernel_bound_names)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations
    if violations:
        result["regression_type"] = 3
        detail = "; ".join(f"L{v['line']} {v['call']}" for v in violations[:5])
        result["suggestion"] = (
            f"forward() still uses PyTorch compute ops: {detail}. All compute must be in the "
            "kernel; forward() may only do buffer alloc (torch.empty etc.) + shape ops "
            "(.view/.reshape).")
        result["checks"]["no_forbidden_torch_ops"]["error"] = f"{len(violations)} forbidden ops"
        return result
    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # Check 4: scalar for-loops
    loop_violations = check_scalar_for_loops(forward_node)
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations
    if loop_violations:
        result["regression_type"] = 4
        detail = "; ".join(f"L{v['line']} for {v['loop_var']}" for v in loop_violations[:5])
        result["suggestion"] = (
            f"forward() has scalar Python for-loops over tensor indices: {detail}. Vectorize "
            "this inside the kernel — Python-level iteration defeats the purpose of the kernel.")
        result["checks"]["no_scalar_for_loops"]["error"] = f"{len(loop_violations)} scalar loops"
        return result
    result["checks"]["no_scalar_for_loops"]["passed"] = True

    result["valid"] = True
    return result


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AST static check for kernel regression in model_new_ascendc.py")
    parser.add_argument("file", help="path to model_new_*.py")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        msg = {"valid": False, "error": f"file not found: {args.file}"}
        print(json.dumps(msg) if args.json else f"[ERROR] {msg['error']}")
        sys.exit(1)

    result = validate(code, filepath=args.file)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["valid"]:
            kerns = result["checks"]["kernel_imported"]["kernels"]
            called = result["checks"]["kernel_called"]["called"]
            print("[PASS] kernel regression check OK")
            print(f"  imported: {', '.join(k['bound'] + ' ({})'.format(k['kind']) for k in kerns)}")
            print(f"  called: {', '.join(c['call'] for c in called)}")
        else:
            rtype = result["regression_type"]
            type_desc = {1: "no kernel import (pure PyTorch)",
                         2: "kernel imported but forward() does not call it",
                         3: "forward() uses forbidden PyTorch compute ops",
                         4: "scalar Python for-loop in forward()"}
            print(f"[FAIL] regression Type {rtype}: {type_desc.get(rtype, 'unknown')}")
            for name, chk in result["checks"].items():
                status = "PASS" if chk["passed"] else "FAIL"
                print(f"  [{status}] {name}")
                if chk.get("error"):
                    print(f"         {chk['error']}")
            tov = result["checks"]["no_forbidden_torch_ops"]["violations"]
            if tov:
                print("  forbidden-op details:")
                for v in tov:
                    print(f"    L{v['line']}: {v['call']} — {v['reason']}")
            lov = result["checks"]["no_scalar_for_loops"]["violations"]
            if lov:
                print("  scalar for-loop details:")
                for v in lov:
                    print(f"    L{v['line']}: for {v['loop_var']} — {v['reason']}")
            print(f"\n  fix: {result['suggestion']}")
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""A3 author invocation — wraps aog-a3-author skill (Gap-A fix, 2026-05-13).

For port_a3_to_a5 mode, the workspace needs per-op `run_a3_reference.py`
+ `input_gen.py` to capture A3 baseline. Hand-authoring per op is 1-2 h /
op which doesn't fit the 0-interaction harness goal. The aog-a3-author
skill auto-generates these scripts from the upstream
`examples/test_aclnn_<op>.cpp` + `op_host/<op>_proto.cpp` files.

This module:
1. Validates trigger conditions (port_a3_to_a5 mode + scripts missing)
2. Spawns `claude --print` with the aog-a3-author skill prompt
3. Validates the produced scripts (AST parse + invariant checks)
4. Drops `.a3_authored` marker on success OR `.a3_author_FAILED-<reason>` on
   failure; logs to `.a3_author_log.jsonl`

Mirror of kb_invoke.merge_one structure. Orchestrator decides WHEN to
invoke; LLM inside the skill decides CONTENT.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import operator as _operator
import re
import subprocess
import sys as _sys
from pathlib import Path
from typing import Optional

# Harness-decoupling: the claude invocation is owned by the Backend, not hardcoded here.
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # orchestrator/ for `backends` package
from backends import get_backend
from source_op import resolve_logical_op_name

_backend = get_backend()


# Path A (torch_npu wrapper) — required tokens in runner script
_PATH_A_RUNNER_TOKENS = (
    "torch_npu",        # must import the NPU backend
    "torch.npu.set_device(",
    "edge_inputs.pt",   # must read fixture
    "edge_dataset.pt",  # must write outputs
    "a3_baseline_perf.json",  # must write per-case timings
    "def main(",
    'if __name__ == "__main__"',
)
# Path B (cpp-binary wrapper) — required artifacts + runner tokens
_PATH_B_RUNNER_TOKENS = (
    "subprocess",       # must invoke the compiled binary
    "edge_inputs.pt",
    "edge_dataset.pt",
    "a3_baseline_perf.json",
    "def main(",
    'if __name__ == "__main__"',
)
_PATH_B_REQUIRED_FILES = (
    # Worker emits these in addition to run_a3_reference.py + input_gen.py
    # when Path B is selected (op has no torch_npu wrapper).
    "CMakeLists.txt",
    "build_runner.sh",
)
_REQUIRED_INPUT_GEN_TOKENS = (
    "edge_inputs.pt",   # must write fixture
    "manifest.json",    # must write per-case meta
    "def main(",
    'if __name__ == "__main__"',
    "torch.save",
    "MAX_CASE_TENSOR_BYTES",
    "MAX_DATASET_TENSOR_BYTES",
    "generate_cases",
)

_MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024
_MAX_DATASET_TENSOR_BYTES = 1024 * 1024 * 1024
_MAX_STATIC_INT_VALUE = 1 << 63


def _strip_hash_comments(text: str) -> str:
    """Remove ``#`` comments while preserving hashes inside quoted strings."""
    result: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\" and quote is not None:
            result.append(char)
            escaped = True
        elif char in {'"', "'"}:
            result.append(char)
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            newline = text.find("\n", index)
            if newline == -1:
                break
            result.append("\n")
            index = newline
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _strip_cmake_comments(text: str) -> str:
    """Remove CMake bracket and line comments before inspecting link items."""
    without_bracket_comments = re.sub(
        r"#\[(=*)\[.*?\]\1\]",
        "",
        text,
        flags=re.DOTALL,
    )
    return _strip_hash_comments(without_bracket_comments)


def _has_cmake_link_item(cmake_text: str, library: str) -> bool:
    """Return whether *library* occurs in one of the supported link forms.

    The author contract permits a bare CMake item (``ascendcl``), a linker
    option (``-lascendcl``), or an exact shared-object name/path
    (``libascendcl.so`` and numeric SONAME suffixes). Identifier-aware
    boundaries deliberately reject lookalikes such as ``myascendcl`` and
    ``libascendcl_helper.so``.
    """
    escaped_library = re.escape(library)
    item_boundary = r"A-Za-z0-9_.+\-"
    bare_or_dash_l = (
        rf"(?<![{item_boundary}])(?:-l)?{escaped_library}"
        rf"(?![{item_boundary}])"
    )
    shared_object = (
        rf"(?<![{item_boundary}])lib{escaped_library}\.so"
        rf"(?:\.[0-9]+)*(?![{item_boundary}])"
    )
    return bool(re.search(rf"(?:{bare_or_dash_l}|{shared_object})", cmake_text))


def _has_clean_first_build_command(script_text: str) -> bool:
    """Check for an executable ``cmake --build ... --clean-first`` command."""
    uncommented = _strip_hash_comments(script_text).replace("\\\n", " ")
    return bool(
        re.search(
            r"(?:^|&&|\|\||;)\s*cmake\s+--build\b"
            r"[^\n;&|]*--clean-first(?:\s|$)",
            uncommented,
            flags=re.MULTILINE,
        )
    )


def _valid_static_int(value: int) -> int | None:
    """Bound a statically evaluated value to the supported integer range."""
    return value if abs(value) <= _MAX_STATIC_INT_VALUE else None


def _static_integer_constant(node: ast.AST) -> int | None:
    """Return an integer literal while rejecting boolean constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value if not isinstance(node.value, bool) else None
    return None


def _static_unary_int_value(node: ast.AST) -> int | None:
    """Evaluate a unary plus or minus expression used in a byte budget."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _static_int_value(node.operand)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    return None


def _static_binary_int_value(node: ast.AST) -> int | None:
    """Evaluate one supported integer binary expression without executing code."""
    if not isinstance(node, ast.BinOp):
        return None
    left = _static_int_value(node.left)
    right = _static_int_value(node.right)
    if left is None or right is None:
        return None
    operations = {
        ast.Add: _operator.add,
        ast.Sub: _operator.sub,
        ast.Mult: _operator.mul,
        ast.FloorDiv: _operator.floordiv,
    }
    operation = operations.get(type(node.op))
    if operation is None:
        return None
    try:
        return _valid_static_int(operation(left, right))
    except (ArithmeticError, ValueError):
        return None


def _static_int_value(node: ast.AST) -> int | None:
    """Evaluate the small integer-only expressions used for byte budgets."""
    constant = _static_integer_constant(node)
    if constant is not None:
        return constant
    unary_value = _static_unary_int_value(node)
    if unary_value is not None:
        return unary_value
    return _static_binary_int_value(node)


def _named_annotation_value(statement: ast.stmt, name: str) -> ast.AST | None:
    """Return a direct module annotation value when it assigns to *name*."""
    if not isinstance(statement, ast.AnnAssign):
        return None
    if not isinstance(statement.target, ast.Name):
        return None
    if statement.target.id != name:
        return None
    return statement.value


def _module_assignments(tree: ast.Module, name: str) -> list[ast.AST]:
    """Return direct module-scope values assigned to *name*."""
    values: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in statement.targets):
                values.append(statement.value)
            continue
        annotation_value = _named_annotation_value(statement, name)
        if annotation_value is not None:
            values.append(annotation_value)
    return values


def _main_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "main":
                return statement
    return None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _has_call(function: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _call_name(node) == name
        for node in ast.walk(function)
    )


def _is_schema_limit_lookup(node: ast.AST) -> bool:
    """Return whether *node* reads the canonical SCHEMA max-case limit."""
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Name):
        return False
    if node.func.value.id != "SCHEMA" or node.func.attr != "get":
        return False
    if len(node.args) < 2:
        return False
    key, default = node.args[:2]
    if not isinstance(key, ast.Constant) or key.value != "max_case_tensor_bytes":
        return False
    return isinstance(default, ast.Name) and default.id == "MAX_CASE_TENSOR_BYTES"


def _has_schema_limit_lookup(function: ast.AST) -> bool:
    """Find SCHEMA.get('max_case_tensor_bytes', MAX_CASE_TENSOR_BYTES)."""
    return any(_is_schema_limit_lookup(node) for node in ast.walk(function))


def _ast_name_matches(node: ast.AST, name: str) -> bool:
    """Return whether *node* is an AST name with the expected identifier."""
    return isinstance(node, ast.Name) and node.id == name


def _matches_limit_relation(
    left: ast.AST,
    relation: ast.cmpop,
    right: ast.AST,
    value_name: str,
    limit_name: str,
    *,
    allowed: bool,
) -> bool:
    """Match either the accepted or rejected direction of a limit comparison."""
    forward = _ast_name_matches(left, value_name) and _ast_name_matches(right, limit_name)
    reverse = _ast_name_matches(left, limit_name) and _ast_name_matches(right, value_name)
    if allowed:
        return (forward and isinstance(relation, (ast.Lt, ast.LtE))) or (
            reverse and isinstance(relation, (ast.Gt, ast.GtE))
        )
    return (forward and isinstance(relation, (ast.Gt, ast.GtE))) or (
        reverse and isinstance(relation, (ast.Lt, ast.LtE))
    )


def _has_limit_relation(
    test: ast.AST,
    value_name: str,
    limit_name: str,
    *,
    allowed: bool,
) -> bool:
    """Find a comparison between a named value and its configured limit."""
    for comparison in ast.walk(test):
        if not isinstance(comparison, ast.Compare):
            continue
        operands = [comparison.left, *comparison.comparators]
        if any(
            _matches_limit_relation(
                left,
                relation,
                right,
                value_name,
                limit_name,
                allowed=allowed,
            )
            for left, relation, right in zip(
                operands, comparison.ops, operands[1:]
            )
        ):
            return True
    return False


def _rejects_above_limit(
    test: ast.AST,
    value_name: str,
    limit_name: str,
) -> bool:
    """Return whether a condition rejects a value above its configured limit."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _has_limit_relation(
            test.operand,
            value_name,
            limit_name,
            allowed=True,
        )
    if isinstance(test, ast.BoolOp):
        return any(
            _rejects_above_limit(value, value_name, limit_name)
            for value in test.values
        )
    return _has_limit_relation(test, value_name, limit_name, allowed=False)


def _condition_raises(condition: ast.If) -> bool:
    """Return whether the immediate body of an ``if`` contains a raise."""
    return any(
        isinstance(node, ast.Raise)
        for statement in condition.body
        for node in ast.walk(statement)
    )


def _has_raising_limit_gate(
    function: ast.AST,
    value_name: str,
    limit_name: str,
) -> bool:
    """Find an if/raise whose condition rejects values above the limit."""
    for condition in ast.walk(function):
        if not isinstance(condition, ast.If):
            continue
        if not _condition_raises(condition):
            continue
        if _rejects_above_limit(condition.test, value_name, limit_name):
            return True
    return False


def _schema_dict_limit_values(value: ast.AST) -> list[ast.AST]:
    """Collect a max-case limit value from a literal SCHEMA dictionary."""
    if not isinstance(value, ast.Dict):
        return []
    return [
        item
        for key, item in zip(value.keys, value.values)
        if isinstance(key, ast.Constant) and key.value == "max_case_tensor_bytes"
    ]


def _schema_subscript_limit_value(target: ast.AST, value: ast.AST) -> list[ast.AST]:
    """Collect an explicit SCHEMA max-case assignment made through a subscript."""
    if not (
        isinstance(target, ast.Subscript)
        and _ast_name_matches(target.value, "SCHEMA")
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "max_case_tensor_bytes"
    ):
        return []
    return [value]


def _schema_assignment_limit_values(node: ast.Assign | ast.AnnAssign) -> list[ast.AST]:
    """Collect explicit max-case values contributed by one assignment node."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    value = node.value
    if value is None:
        return []
    values: list[ast.AST] = []
    for target in targets:
        if _ast_name_matches(target, "SCHEMA"):
            values.extend(_schema_dict_limit_values(value))
        else:
            values.extend(_schema_subscript_limit_value(target, value))
    return values


def _schema_case_limit_overrides(tree: ast.Module) -> list[ast.AST]:
    """Collect explicit SCHEMA max-case overrides without evaluating SCHEMA."""
    values: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            values.extend(_schema_assignment_limit_values(node))
    return values


def _validate_input_gen_constants(tree: ast.Module) -> list[str]:
    """Validate the two canonical byte-budget constant assignments."""
    errors: list[str] = []
    for name, expected in (
        ("MAX_CASE_TENSOR_BYTES", _MAX_CASE_TENSOR_BYTES),
        ("MAX_DATASET_TENSOR_BYTES", _MAX_DATASET_TENSOR_BYTES),
    ):
        assignments = _module_assignments(tree, name)
        values = [_static_int_value(value) for value in assignments]
        if len(values) != 1 or values[0] != expected:
            errors.append(
                f"input_gen.py must assign {name} exactly once to {expected} bytes"
            )
    return errors


def _validate_input_gen_main(main: ast.AST) -> list[str]:
    """Validate the executable generation and byte-limit gates in ``main``."""
    errors: list[str] = []
    if not _has_call(main, "generate_cases"):
        errors.append("input_gen.py main() must call generate_cases(...)")
    if not _has_schema_limit_lookup(main):
        errors.append(
            "input_gen.py main() must read SCHEMA max_case_tensor_bytes with "
            "MAX_CASE_TENSOR_BYTES as its default"
        )
    if not _has_raising_limit_gate(
        main,
        "configured_case_limit",
        "MAX_CASE_TENSOR_BYTES",
    ):
        errors.append(
            "input_gen.py main() must raise when configured_case_limit exceeds "
            "MAX_CASE_TENSOR_BYTES"
        )
    if not _has_call(main, "_tensor_payload_bytes"):
        errors.append(
            "input_gen.py main() must calculate generated tensor payload bytes"
        )
    if not _has_raising_limit_gate(
        main,
        "dataset_tensor_bytes",
        "MAX_DATASET_TENSOR_BYTES",
    ):
        errors.append(
            "input_gen.py main() must raise when dataset_tensor_bytes exceeds "
            "MAX_DATASET_TENSOR_BYTES"
        )
    return errors


def _validate_schema_case_limit_overrides(tree: ast.Module) -> list[str]:
    """Reject static per-operation case limits above the canonical ceiling."""
    errors: list[str] = []
    for override in _schema_case_limit_overrides(tree):
        value = _static_int_value(override)
        if value is not None and not 0 < value <= _MAX_CASE_TENSOR_BYTES:
            errors.append(
                "input_gen.py SCHEMA max_case_tensor_bytes override exceeds "
                "the 100 MiB workflow limit"
            )
    return errors


def _validate_input_gen_ast(tree: ast.Module) -> list[str]:
    """Validate executable fixture-budget structure, not comment tokens."""
    errors = _validate_input_gen_constants(tree)

    main = _main_function(tree)
    if main is None:
        return [*errors, "input_gen.py missing executable main() function"]
    errors.extend(_validate_input_gen_main(main))
    errors.extend(_validate_schema_case_limit_overrides(tree))
    return errors


def should_trigger(workspace: Path, opgen_mode: str) -> tuple[bool, str]:
    """Strict trigger gate per CLAUDE.md user direction 2026-05-13:
    fire ONLY in port_a3_to_a5 mode AND when run_a3_reference.py is absent.

    Returns (should_fire, reason). reason is the audit-trail string.
    """
    if opgen_mode != "port_a3_to_a5":
        return False, f"opgen_mode={opgen_mode!r} (not port_a3_to_a5)"
    runner = workspace / "run_a3_reference.py"
    if runner.exists():
        return False, "run_a3_reference.py already present"
    return True, "port_a3_to_a5 mode + run_a3_reference.py absent"


def _new_log_entry(workspace: Path, op_dir: Path, opgen_mode: str) -> dict:
    """Create the audit record shared by every authoring exit path."""
    return {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "workspace": str(workspace),
        "op_dir": str(op_dir),
        "opgen_mode": opgen_mode,
    }


def _trigger_gate_result(workspace: Path, log_entry: dict, reason: str) -> dict:
    """Log and return the idempotent or wrong-mode trigger-gate result."""
    already_authored = "already present" in reason
    log_entry["verdict"] = "ALREADY_AUTHORED" if already_authored else "WRONG_MODE"
    _append_log(workspace, log_entry)
    return {
        "success": already_authored,
        "verdict": log_entry["verdict"],
        "errors": [] if already_authored else [reason],
        "log_entry": log_entry,
    }


def _find_aclnn_entry(op_dir: Path, op_name: str) -> Path | None:
    """Find a supported aclnn entry, retaining the conservative fallback path."""
    try:
        from phase_o25_a3_ref import derive_aclnn_entry  # type: ignore
        return derive_aclnn_entry(op_dir)
    except Exception:
        fallback = op_dir / "examples" / f"test_aclnn_{op_name}.cpp"
        return fallback if fallback.exists() else None


def _missing_cpp_result(
    workspace: Path,
    op_dir: Path,
    op_name: str,
    log_entry: dict,
) -> dict:
    """Log and return the fail-closed result when no aclnn entry exists."""
    log_entry["verdict"] = "MISSING_CPP"
    log_entry["errors"] = [
        f"aclnn entry does not exist in {op_dir} (tried "
        f"examples/test_aclnn_{op_name}.cpp, variant glob "
        f"examples/test_aclnn_{op_name}_*.cpp, _vN-stripped, "
        f"tests/ut/op_host fallback)"
    ]
    _append_log(workspace, log_entry)
    return {
        "success": False,
        "verdict": "MISSING_CPP",
        "errors": log_entry["errors"],
        "log_entry": log_entry,
    }


def _timeout_result(workspace: Path, log_entry: dict, timeout_sec: int) -> dict:
    """Log and return the fail-closed backend-timeout result."""
    log_entry["verdict"] = "LLM_TIMEOUT"
    log_entry["errors"] = [f"claude --print timed out after {timeout_sec}s"]
    _append_log(workspace, log_entry)
    return {
        "success": False,
        "verdict": "LLM_TIMEOUT",
        "errors": log_entry["errors"],
        "log_entry": log_entry,
    }


def _proxy_error_result(workspace: Path, log_entry: dict) -> dict:
    """Persist the proxy failure marker and return its fail-closed result."""
    log_entry["verdict"] = "LLM_PROXY_ERROR"
    log_entry["errors"] = ["Huawei proxy returned 407"]
    fail_marker = workspace / ".a3_author_FAILED-llm_proxy_error"
    fail_marker.write_text(f"ts={log_entry['ts']}\nerrors=Huawei proxy returned 407\n")
    _append_log(workspace, log_entry)
    return {
        "success": False,
        "verdict": "LLM_PROXY_ERROR",
        "errors": log_entry["errors"],
        "log_entry": log_entry,
    }


def _finish_authoring(
    workspace: Path,
    op_dir: Path,
    log_entry: dict,
) -> dict:
    """Validate artifacts, persist the corresponding marker, and log the verdict."""
    verdict, errors = _validate_authored_scripts(workspace)
    log_entry["verdict"] = verdict
    if errors:
        log_entry["errors"] = errors

    if verdict == "AUTHORED":
        marker = workspace / ".a3_authored"
        marker.write_text(
            f"ts={log_entry['ts']}\n"
            f"op_dir={op_dir}\n"
            "verdict=AUTHORED\n"
        )
        log_entry["marker"] = str(marker)
    else:
        fail_marker = workspace / f".a3_author_FAILED-{verdict.lower()}"
        fail_marker.write_text(f"ts={log_entry['ts']}\nerrors={'; '.join(errors)}\n")

    _append_log(workspace, log_entry)
    return {
        "success": verdict == "AUTHORED",
        "verdict": verdict,
        "errors": errors,
        "log_entry": log_entry,
    }


def author_one(
    workspace: Path,
    op_dir: Path,
    *,
    timeout_sec: int = 1200,
    opgen_mode: str = "port_a3_to_a5",
) -> dict:
    """Invoke aog-a3-author skill for one op.

    Args:
        workspace: workspace/<op>/ target dir (writes scripts here)
        op_dir: ops-nn op directory (reads test_aclnn + proto from here)
        timeout_sec: subprocess timeout (default 1200s = 20 min). History:
            300s was too tight on complex fused ops (group_norm_silu_quant
            2026-05-13 hit 300s parsing 9-tensor + 8-scalar signature).
            Bumped to 600s. 600s then proved too tight for Path B 5-file
            emit (fatrelu_mul 2026-05-17: .py files written at 10:00, .cpp
            + CMakeLists + build_runner.sh still mid-write when timeout
            fired). 1200s gives Path B comfortable margin. LLM needs time
            to read cpp + proto + write 2-5 valid scripts.
        opgen_mode: durable-state opgen_mode for trigger validation

    Returns:
        dict with keys:
            success: bool
            verdict: "ALREADY_AUTHORED" | "AUTHORED" | "MISSING_CPP" |
                     "PARSE_FAILED" | "INVARIANT_FAILED" | "LLM_TIMEOUT" |
                     "LLM_PROXY_ERROR" | "WRONG_MODE"
            errors: list[str]
            log_entry: dict (also appended to .a3_author_log.jsonl)
    """
    log_entry = _new_log_entry(workspace, op_dir, opgen_mode)

    # Trigger gate
    fire, reason = should_trigger(workspace, opgen_mode)
    log_entry["trigger_reason"] = reason
    if not fire:
        return _trigger_gate_result(workspace, log_entry, reason)
    op_name = resolve_logical_op_name(op_dir)

    # Input validation: op_dir must have an aclnn cpp. 2026-05-21 fix:
    # delegate to phase_o25_a3_ref.derive_aclnn_entry which handles
    # variant-suffix (`_a4w4` etc.) + `_vN`-stripped + tests/ut/ fallbacks.
    # Previously hard-coded `test_aclnn_<op>.cpp` only — blocked
    # grouped_matmul_swiglu_quant_v2 (5 variants, no base name) +
    # masked_select_v3 (aclnn named `test_aclnn_masked_select.cpp`).
    cpp_candidate = _find_aclnn_entry(op_dir, op_name)
    if cpp_candidate is None or not cpp_candidate.exists():
        return _missing_cpp_result(workspace, op_dir, op_name, log_entry)

    # Build skill prompt
    prompt = _build_prompt(op_dir, workspace)
    # Invocation via Backend (harness-decoupling): was hardcoded `claude --print` → CCBackend.dispatch(kind="skill").
    env = _backend.dispatch("aog-a3-author", prompt, kind="skill", timeout=timeout_sec)
    if env.raw_envelope.get("timed_out"):
        return _timeout_result(workspace, log_entry, timeout_sec)

    _stdout = env.output_text or ""
    log_entry["exit_code"] = env.raw_envelope.get("returncode")
    log_entry["stdout_tail"] = _stdout[-2000:]
    log_entry["stderr_tail"] = (env.raw_envelope.get("stderr") or "")[-1000:]

    # Detect Huawei proxy error pattern from the subprocess JSON output
    if _stdout and "corporate proxy Notification" in _stdout:
        return _proxy_error_result(workspace, log_entry)

    return _finish_authoring(workspace, op_dir, log_entry)


def _authored_script_paths(workspace: Path) -> tuple[Path, Path]:
    """Return the generated runner and fixture-generator paths for a workspace."""
    return workspace / "run_a3_reference.py", workspace / "input_gen.py"


def _missing_script_errors(paths: tuple[Path, Path]) -> list[str]:
    """Return the missing-artifact errors in the historical validation order."""
    return [f"{path.name} missing after skill run" for path in paths if not path.exists()]


def _parse_authored_scripts(
    paths: tuple[Path, Path],
) -> tuple[dict[Path, ast.Module], list[str]]:
    """Parse generated Python artifacts while retaining their parse diagnostics."""
    parsed_trees: dict[Path, ast.Module] = {}
    errors: list[str] = []
    for path in paths:
        try:
            parsed_trees[path] = ast.parse(path.read_text())
        except SyntaxError as error:
            errors.append(f"{path.name} ast.parse failed: {error}")
    return parsed_trees, errors


def _runner_token_errors(runner_text: str, path_b_present: bool) -> list[str]:
    """Validate the runner token contract for the selected authoring path."""
    tokens = _PATH_B_RUNNER_TOKENS if path_b_present else _PATH_A_RUNNER_TOKENS
    label = "Path-B (cpp-binary)" if path_b_present else "Path-A (torch_npu)"
    return [
        f"run_a3_reference.py [{label}] missing required token: {token!r}"
        for token in tokens
        if token not in runner_text
    ]


def _path_b_cmake_errors(workspace: Path) -> list[str]:
    """Validate Path-B runtime-link requirements from the generated CMake file."""
    cmake_text = _strip_cmake_comments((workspace / "CMakeLists.txt").read_text())
    errors: list[str] = []
    if _has_cmake_link_item(cmake_text, "ascend_hal"):
        errors.append(
            "CMakeLists.txt [Path-B] must not link ascend_hal; use the runtime "
            "ascendcl + nnopbase + opapi libraries"
        )
    for runtime_lib in ("ascendcl", "nnopbase", "opapi"):
        if not _has_cmake_link_item(cmake_text, runtime_lib):
            errors.append(
                f"CMakeLists.txt [Path-B] missing required runtime library: "
                f"{runtime_lib}"
            )
    return errors


def _path_b_build_runner_errors(workspace: Path) -> list[str]:
    """Validate that the Path-B build script executes a clean build command."""
    build_runner_text = (workspace / "build_runner.sh").read_text()
    uncommented = _strip_hash_comments(build_runner_text).strip()
    errors: list[str] = []
    if re.fullmatch(r"exit\s+0\s*;?", uncommented):
        errors.append("build_runner.sh [Path-B] is an empty exit 0 stub")
    if not _has_clean_first_build_command(build_runner_text):
        errors.append(
            "build_runner.sh [Path-B] must execute "
            "cmake --build ... --clean-first"
        )
    return errors


def _path_b_runner_artifact_errors(workspace: Path) -> list[str]:
    """Validate generated C++ runner files and their macro namespace safety."""
    runner_cpp_paths = sorted(workspace.glob("*_runner.cpp"))
    if not runner_cpp_paths:
        return ["Path-B missing required <op>_runner.cpp artifact"]
    errors: list[str] = []
    for cpp_path in runner_cpp_paths:
        generic_macro = re.search(
            r"^\s*#\s*define\s+(CHECK_RET|LOG_PRINT)\b",
            cpp_path.read_text(),
            re.MULTILINE,
        )
        if generic_macro:
            errors.append(
                f"{cpp_path.name} [Path-B] defines generic "
                f"{generic_macro.group(1)}, which may conflict with CANN headers; "
                "use an ordinary function or op-specific helper"
            )
    return errors


def _path_b_validation_errors(workspace: Path) -> list[str]:
    """Combine the Path-B CMake, build-script, and C++ artifact validation."""
    return [
        *_path_b_cmake_errors(workspace),
        *_path_b_build_runner_errors(workspace),
        *_path_b_runner_artifact_errors(workspace),
    ]


def _input_gen_token_errors(input_gen_text: str) -> list[str]:
    """Validate that the fixture generator contains the required output tokens."""
    return [
        f"input_gen.py missing required token: {token!r}"
        for token in _REQUIRED_INPUT_GEN_TOKENS
        if token not in input_gen_text
    ]


def _validate_authored_scripts(workspace: Path) -> tuple[str, list[str]]:
    """Validate that workspace has well-formed run_a3_reference.py + input_gen.py.

    Returns (verdict, errors). verdict is "AUTHORED" on success or one of
    PARSE_FAILED / INVARIANT_FAILED on failure.
    """
    errors: list[str] = []
    runner, input_gen = _authored_script_paths(workspace)
    paths = (runner, input_gen)
    errors.extend(_missing_script_errors(paths))

    if errors:
        return "PARSE_FAILED", errors

    parsed_trees, parse_errors = _parse_authored_scripts(paths)
    errors.extend(parse_errors)

    if errors:
        return "PARSE_FAILED", errors

    runner_text = runner.read_text()
    # Detect path: A (torch_npu wrapper) vs B (cpp-binary) by presence
    # of CMakeLists.txt + build_runner.sh in workspace.
    path_b_present = all(
        (workspace / f).exists() for f in _PATH_B_REQUIRED_FILES
    )
    errors.extend(_runner_token_errors(runner_text, path_b_present))

    if path_b_present:
        errors.extend(_path_b_validation_errors(workspace))

    input_gen_text = input_gen.read_text()
    errors.extend(_input_gen_token_errors(input_gen_text))
    errors.extend(_validate_input_gen_ast(parsed_trees[input_gen]))

    if errors:
        return "INVARIANT_FAILED", errors
    return "AUTHORED", []


def _prompt_source_and_path_selection(op_name: str, op_dir: Path, workspace: Path) -> str:
    """Build the source-reading and Path-A/Path-B selection prompt section."""
    return (
        f"Run the /aog-a3-author skill for op `{op_name}`. "
        f"Read the upstream aclnn signature from "
        f"`{op_dir}/examples/test_aclnn_{op_name}.cpp` "
        f"and the output-shape inference from "
        f"`{op_dir}/op_host/{op_name}_proto.cpp` (if present).\n\n"
        f"STEP 1 — decide Path A or Path B:\n"
        f"  - Run `python3 -c \"import torch_npu; print(dir(torch_npu))\" | "
        f"tr ',' '\\n' | grep -i '{op_name}'` to check whether a "
        f"torch_npu Python wrapper exists for `{op_name}` "
        f"(any of: torch_npu.npu_{op_name}, torch_npu._C._npu_{op_name}, "
        f"torch.ops.npu.{op_name}).\n"
        f"  - If yes → **Path A**: emit `input_gen.py` + `run_a3_reference.py` "
        f"that calls the torch_npu wrapper directly. Reference template: "
        f"`workspace/gather_elements_v2/run_a3_reference.py`.\n"
        f"  - If no → **Path B**: emit FIVE artifacts: `input_gen.py` + "
        f"`run_a3_reference.py` (subprocess-style) + `{op_name}_runner.cpp` "
        f"(modified copy of upstream test_aclnn cpp that reads per-case .bin "
        f"files + writes per-case output .bin files + emits timing on stdout) "
        f"+ `CMakeLists.txt` (links the CANN runtime libraries ascendcl + "
        f"nnopbase + opapi from the configured runtime lib64; do not link "
        f"ascend_hal, which is a driver/devlib library) + "
        f"`build_runner.sh` (`cmake -S . -B build && cmake --build build "
        f"--clean-first`, so stale link commands cannot survive a corrected CMake file).\n\n"
    )


def _prompt_fixture_generation_steps(reference_provider: Path, workspace: Path) -> str:
    """Build the input-generator requirements in the authoring prompt."""
    return (
        f"STEP 2 — emit to `{workspace}/`:\n"
        f"  - `input_gen.py`: **MUST use case_gen template, NOT hand-authored cases**. "
        f"Owner directive 2026-05-24T22:10Z: 必须用 case_gen 系统化展开 SCHEMA × COVERAGE_TIER. "
        f"Audit 2026-05-24 caught all 20 prior port_a3 archives hand-authored 8 cases → "
        f"under-coverage; 已 pass 的 op 必须用修复的 harness 重新 cold start 验证. \n"
        f"    1. `cp {reference_provider}/input_gen.template.py {workspace}/input_gen.py`\n"
        f"    2. `cp {reference_provider}/case_gen.py {workspace}/case_gen.py`\n"
        f"    3. Fill SCHEMA dict per op signature (op_name, formula, tensor_inputs, "
        f"scalar_inputs, tensor_output, rank). Read V220 source `op_host/<op>_def.cpp` + "
        f"`op_kernel/<op>.cpp` for input/output shape signature.\n"
        f"    4. Pick COVERAGE_TIER='sign_off' (~40 cases) or 'production' (~60 cases). "
        f"NEVER 'pilot' (~15) for port_a3 unless workspace marker says otherwise.\n"
        f"       Use one source-supported representative `rank` by default. Use "
        f"SCHEMA['ranks'] only when admitted arch22 source proves a rank-polymorphic "
        f"interface; do not mechanically sweep [1, 2, 3, 4].\n"
        f"    5. For ops where SCHEMA can't cleanly express interdependencies (e.g. fused "
        f"ops with sparse_indices ≤ S2 constraint), add `base_shape_filter` callback "
        f"per case_gen contract — NEVER fall back to hand-rolling 8 cases.\n"
        f"    6. Run `python3 input_gen.py` → writes edge_inputs.pt + manifest.json with "
        f"`data_sha256` + `coverage_tier` fields populated by case_gen (workflow_critic "
        f"O2_5.B.inv1 + inv3 require these).\n"
        f"    7. Retain the canonical `MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024` "
        f"pre-allocation guard. Each case's combined tensor payload must be <= 100 MiB; "
        f"retain `MAX_DATASET_TENSOR_BYTES = 1024 * 1024 * 1024` and reject the "
        f"generated aggregate tensor payload above 1 GiB before hashing or saving; "
        f"reduce representative shapes/rank variants and regenerate if the guard rejects "
        f"a case. Do not silently drop cases or lower the coverage tier.\n"
    )


def _prompt_runner_requirements(op_name: str, workspace: Path) -> str:
    """Build runner, Path-B artifact, invariant, and golden-reference instructions."""
    return (
        f"  - `run_a3_reference.py`: for each case, captures a3_outputs to "
        f"edge_dataset.pt + median timing to a3_baseline_perf.json. At the end "
        f"it MUST also write `a3_capture_manifest.json` = "
        f"{{\"n_total\": <#cases>, \"n_captured\": <#cases that actually got "
        f"a3_outputs>}} (task#25 Tier-1 capture gate — O2.5 declares READY only "
        f"when n_captured == n_total AND a3_baseline_perf.json median_ms_per_case "
        f"is non-empty; a partial/empty capture → CAPTURE_INCOMPLETE fail-fast, "
        f"NOT a silent CPU-truth degrade)\n"
        f"  - (Path B only) `<op>_runner.cpp` + `CMakeLists.txt` + "
        f"`build_runner.sh` per upstream test_aclnn pattern. Do not define generic "
        f"macros such as CHECK_RET (CANN headers already define them); use ordinary "
        f"functions or an op-specific prefix.\n\n"
        f"Invariants (validation gates will check):\n"
        f"  - Both .py files must `def main()` + `if __name__ == \"__main__\": main()`\n"
        f"  - Path A `run_a3_reference.py` must import torch_npu and call "
        f"torch.npu.set_device(0)\n"
        f"  - Path B `run_a3_reference.py` must import `subprocess` (to invoke "
        f"the binary) and still produce the same edge_dataset.pt + "
        f"a3_baseline_perf.json outputs\n\n"
        f"Golden reference (Path A): `workspace/ctc_loss_v3/` and "
        f"`workspace/gather_elements_v2/` — validated end-to-end."
    )


def _reference_provider_path() -> Path:
    """Resolve reference templates relative to this shipped orchestrator module."""
    # DEBT-191: resolve templates relative to THIS module rather than a hardcoded
    # a5_ops path, so a cannbot-only installation remains self-contained.
    return Path(__file__).resolve().parent.parent / "reference_provider"


def _build_prompt(op_dir: Path, workspace: Path) -> str:
    """Build the claude --print prompt that drives the aog-a3-author skill.

    The skill itself is documented in src/skills/aog-a3-author/SKILL.md;
    this prompt names the skill + provides concrete paths + selection guide
    between Path A (torch_npu wrapper) and Path B (cpp-binary)."""
    op_name = resolve_logical_op_name(op_dir)
    return "".join((
        _prompt_source_and_path_selection(op_name, op_dir, workspace),
        _prompt_fixture_generation_steps(_reference_provider_path(), workspace),
        _prompt_runner_requirements(op_name, workspace),
    ))


def _append_log(workspace: Path, log_entry: dict) -> None:
    log_path = workspace / ".a3_author_log.jsonl"
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        # Logging failure is not load-bearing — don't mask the real verdict
        pass

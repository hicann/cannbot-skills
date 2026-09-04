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
"""
AscendC Static Checker — validates generated AscendC code for common anti-patterns.

Runs 13 checks on .h and .cpp files and outputs a JSON report to stdout.

Usage:
    python3 src/scripts/ascendc_static_check.py <directory>

Example:
    python3 src/scripts/ascendc_static_check.py workspace/pooling_skills_test/generated/
    python3 src/scripts/ascendc_static_check.py output/src/pooling/

Exit codes:
    0 — all checks passed
    1 — one or more violations found
    2 — usage error (bad args, directory not found)

Requires Python 3.8+, no external dependencies.
"""

import hashlib
import json
import os
import re
import sys
from typing import Dict, List

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Violation = Dict[str, object]  # {"file": str, "line": int, "detail": str}
CheckResult = Dict[str, object]  # {"passed": bool, "violations": [...]}


# ---------------------------------------------------------------------------
# File reading helper
# ---------------------------------------------------------------------------
def read_lines(filepath: str) -> List[str]:
    """Read file lines, tolerating encoding errors."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Check 1: missing_namespace
#   Any .h file with __simt_vf__ but no `namespace ascendc_ops`
# ---------------------------------------------------------------------------
def check_missing_namespace(filepath: str, lines: List[str]) -> List[Violation]:
    if not filepath.endswith(".h"):
        return []

    has_simt_vf = False
    has_namespace = False
    simt_vf_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        is_comment = (stripped.startswith("//") or stripped.startswith("/*")
                      or stripped.startswith("*"))
        if not is_comment and "__simt_vf__" in line:
            has_simt_vf = True
            if simt_vf_line == 0:
                simt_vf_line = i
        if not is_comment and re.search(r"\bnamespace\s+ascendc_ops\b", line):
            has_namespace = True

    if has_simt_vf and not has_namespace:
        return [{
            "file": filepath,
            "line": simt_vf_line,
            "detail": "Header has __simt_vf__ kernels but no 'namespace ascendc_ops' declaration"
        }]
    return []


# ---------------------------------------------------------------------------
# Check 2: missing_kernel_operator
#   Any .h/.cpp with __aicore__ but no #include <kernel_operator.h>
#   or #include "kernel_operator.h"
# ---------------------------------------------------------------------------
_RE_KERNEL_OP_INCLUDE = re.compile(
    r'#\s*include\s*[<"]kernel_operator\.h[>"]'
)


def check_missing_kernel_operator(filepath: str, lines: List[str]) -> List[Violation]:
    has_aicore = False
    has_include = False
    aicore_line = 0

    for i, line in enumerate(lines, 1):
        if "__aicore__" in line:
            has_aicore = True
            if aicore_line == 0:
                aicore_line = i
        if _RE_KERNEL_OP_INCLUDE.search(line):
            has_include = True

    # Transitive-include resolution (2026-06-10, FA bf16/fp16 TU false positives): a .cpp
    # that includes a local header which includes kernel_operator.h is legitimate. Resolve
    # local quote-includes recursively (depth-limited, same-tree only) before flagging.
    if has_aicore and not has_include:
        import os as _os
        base = _os.path.dirname(_os.path.abspath(filepath))
        seen, queue = set(), []
        for ln in lines:
            m = re.match(r'\s*#include\s+"([^"]+)"', ln)
            if m:
                queue.append(m.group(1))
        depth = 0
        while queue and depth < 200 and not has_include:
            depth += 1
            inc = queue.pop()
            p = _os.path.normpath(_os.path.join(base, inc))
            if p in seen or not _os.path.isfile(p):
                continue
            seen.add(p)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if _RE_KERNEL_OP_INCLUDE.search(txt):
                has_include = True
                break
            for m in re.finditer(r'#include\s+"([^"]+)"', txt):
                queue.append(_os.path.relpath(_os.path.join(_os.path.dirname(p), m.group(1)), base))
    if has_aicore and not has_include:
        return [{
            "file": filepath,
            "line": aicore_line,
            "detail": "File uses __aicore__ but does not #include <kernel_operator.h>"
        }]
    return []


# ---------------------------------------------------------------------------
# Check 3: unconditional_simt_compat
#   #include "simt_compat.h" NOT inside #if.*ASCENDC_CPU_DEBUG
# ---------------------------------------------------------------------------
_RE_SIMT_COMPAT_INCLUDE = re.compile(r'#\s*include\s*"simt_compat\.h"')
_RE_CPU_DEBUG_IF = re.compile(r'#\s*if.*ASCENDC_CPU_DEBUG')


def check_unconditional_simt_compat(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    for i, line in enumerate(lines, 1):
        if _RE_SIMT_COMPAT_INCLUDE.search(line):
            # Check if the preceding non-blank line is #if.*ASCENDC_CPU_DEBUG
            guarded = False
            for j in range(i - 2, max(i - 4, -1), -1):  # look up to 2 lines back
                if j < 0:
                    break
                prev = lines[j].strip()
                if not prev:
                    continue  # skip blank lines
                if _RE_CPU_DEBUG_IF.search(prev):
                    guarded = True
                break  # only check the first non-blank preceding line
            if not guarded:
                violations.append({
                    "file": filepath,
                    "line": i,
                    "detail": '#include "simt_compat.h" is not guarded by '
                              "#if defined(ASCENDC_CPU_DEBUG)"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 4: bf16_static_cast
#   static_cast<float>(...) involving bfloat16 / bf16 on same or adjacent line
# ---------------------------------------------------------------------------
_RE_STATIC_CAST_FLOAT = re.compile(r"static_cast\s*<\s*float\s*>\s*\(")
_RE_BF16_TOKEN = re.compile(r"\b(?:bfloat16_t|bfloat16|bf16_t|bf16)\b", re.IGNORECASE)


def _is_comment_line(line: str) -> bool:
    """Check if a line is a single-line comment (// or /* ... */)."""
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def check_bf16_static_cast(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    n = len(lines)
    for i in range(n):
        # Skip lines that are comments — they often discuss the pattern
        if _is_comment_line(lines[i]):
            continue
        if _RE_STATIC_CAST_FLOAT.search(lines[i]):
            # Check current line and adjacent lines (i-1, i, i+1),
            # but only non-comment lines contribute bf16 tokens
            has_bf16_nearby = False
            for j in range(max(0, i - 1), min(n, i + 2)):
                if _is_comment_line(lines[j]):
                    continue
                if _RE_BF16_TOKEN.search(lines[j]):
                    has_bf16_nearby = True
                    break
            if has_bf16_nearby:
                violations.append({
                    "file": filepath,
                    "line": i + 1,
                    "detail": "static_cast<float>() used near bfloat16/bf16 type — "
                              "bisheng does not support this; use bit-manipulation helpers"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 5: simt_namespace
#   using namespace AscendC::Simt  (should be just AscendC)
# ---------------------------------------------------------------------------
_RE_SIMT_NAMESPACE = re.compile(r"using\s+namespace\s+AscendC\s*::\s*Simt")


def check_simt_namespace(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    for i, line in enumerate(lines, 1):
        if _RE_SIMT_NAMESPACE.search(line):
            violations.append({
                "file": filepath,
                "line": i,
                "detail": "'using namespace AscendC::Simt' — should be 'using namespace AscendC' "
                          "(Simt is accessed via AscendC::Simt::VF_CALL, not imported)"
            })
    return violations


# ---------------------------------------------------------------------------
# Check 6: float_fp16_param
#   In extern "C" functions whose name contains fp16/bf16, detect `float`
#   parameters with names containing "num" or "val" (heuristic for mistyped
#   scalar passing — fp16/bf16 scalars should be passed as uint16_t bits).
# ---------------------------------------------------------------------------
_RE_EXTERN_C_FUNC = re.compile(
    r'extern\s+"C".*?void\s+(\w+)\s*\(([^)]*)\)',
    re.DOTALL
)
_RE_FLOAT_SUSPECT_PARAM = re.compile(
    r"\bfloat\s+(\w*(?:num|val)\w*)\b", re.IGNORECASE
)


def check_float_fp16_param(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    full_text = "".join(lines)

    for m in _RE_EXTERN_C_FUNC.finditer(full_text):
        func_name = m.group(1)
        params_text = m.group(2)

        # Only check fp16/bf16 kernels
        if not re.search(r"(?:fp16|bf16)", func_name, re.IGNORECASE):
            continue

        for pm in _RE_FLOAT_SUSPECT_PARAM.finditer(params_text):
            param_name = pm.group(1)
            # Compute line number of the match
            match_pos = m.start(2) + pm.start()
            line_num = full_text[:match_pos].count("\n") + 1
            violations.append({
                "file": filepath,
                "line": line_num,
                "detail": "float parameter '{}' in fp16/bf16 kernel '{}' — "
                          "fp16/bf16 scalars should be passed as uint16_t bits "
                          "to avoid implicit promotion".format(param_name, func_name)
            })
    return violations


# ---------------------------------------------------------------------------
# Check 7: sort_bounds_missing
#   histogram[ without a preceding if.*<.*max_key guard within 5 lines
# ---------------------------------------------------------------------------
_RE_HISTOGRAM_ACCESS = re.compile(r"\bhistogram\s*\[")
_RE_MAX_KEY_GUARD = re.compile(r"\bif\b.*<.*\bmax_key\b")


def check_sort_bounds_missing(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    n = len(lines)
    for i in range(n):
        line_stripped = lines[i].lstrip()
        # Skip comment-only lines (// ... or /* ... */)
        if line_stripped.startswith("//") or line_stripped.startswith("/*"):
            continue
        if _RE_HISTOGRAM_ACCESS.search(lines[i]):
            # Check if any of the preceding 5 lines has a max_key guard
            guarded = False
            for j in range(max(0, i - 5), i):
                if _RE_MAX_KEY_GUARD.search(lines[j]):
                    guarded = True
                    break
            if not guarded:
                violations.append({
                    "file": filepath,
                    "line": i + 1,
                    "detail": "histogram[] access without preceding "
                              "'if ... < max_key' bounds guard within 5 lines"
                })
    return violations


_RE_SIMT_VF_NONVOID = re.compile(
    r'__simt_vf__\s+__aicore__\s+(?:inline\s+)?(?!void\b)(\w+)')


def check_simt_vf_nonvoid(filepath: str, lines: List[str]) -> List[Violation]:
    """__simt_vf__ functions MUST return void. Helper functions should not have __simt_vf__."""
    violations = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        m = _RE_SIMT_VF_NONVOID.search(line)
        if m:
            violations.append({
                "file": filepath,
                "line": i + 1,
                "detail": f"__simt_vf__ function returns '{m.group(1)}' instead of void "
                          f"— helper functions should use __aicore__ inline without __simt_vf__"
            })
    return violations


# ---------------------------------------------------------------------------
# Check 9: cann_wrapper_call
#   Detect calls to CANN built-in operator APIs (aclnn*, aclop*, acl_op_*,
#   torch_npu.*, npu_*) that indicate wrapping instead of genuine kernel code.
#   This is a reward-hacking guardrail: generated kernels must implement actual
#   computation logic, not forward to CANN built-in implementations.
# ---------------------------------------------------------------------------
# Sanctioned host-launch mechanism for the tilelang2ascendc route: the
# authored host helper drives ACLRT_LAUNCH_KERNEL through
# at_npu::native::OpCommand::RunOpApi on the current NPU stream.  Including
# these two headers is launch plumbing, NOT compute delegation — flagging them
# as CANN wrappers was a false positive (MUSEAttention 2026-08-22: a fully
# precision-PASS candidate was blocked at the delivery static gate solely on
# these two include lines).
_RE_SANCTIONED_TORCH_NPU_INCLUDE = re.compile(
    r'#\s*include\s*[<"]torch_npu/csrc/[^>"]+\.h[>"]'
)

_RE_CANN_WRAPPER_PATTERNS = [
    (re.compile(r'\baclnn[A-Z]\w*\s*\('), "aclnn* API call"),
    (re.compile(r'\baclop[A-Z]\w*\s*\('), "aclop* API call"),
    (re.compile(r'\bacl_op_\w+\s*\('), "acl_op_* API call"),
    (re.compile(r'\baclrtLaunchKernel\s*\('), "aclrtLaunchKernel call (launching pre-built kernel)"),
    (re.compile(r'#\s*include\s*[<"]acl/acl_op_compiler\.h[>"]'), "ACL op compiler header"),
    (re.compile(r'#\s*include\s*[<"]aclnn/\w+\.h[>"]'), "aclnn API header"),
    (re.compile(r'\btorch_npu\b'), "torch_npu reference in kernel code"),
    (re.compile(r'\bnpu_bridge\b'), "npu_bridge reference in kernel code"),
]


def check_cann_wrapper_call(filepath: str, lines: List[str]) -> List[Violation]:
    """Detect CANN built-in operator API calls — kernels must implement logic, not wrap.
    Excludes pybind11 bridge files (they legitimately reference torch_npu for binding).
    """
    # Skip pybind11 bridge files — they're the Python binding layer, not kernel code
    basename = os.path.basename(filepath)
    if basename.startswith("pybind11") or basename == "torch_binding.cpp":
        return []

    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if _RE_SANCTIONED_TORCH_NPU_INCLUDE.search(line):
            continue
        for pattern, desc in _RE_CANN_WRAPPER_PATTERNS:
            if pattern.search(line):
                violations.append({
                    "file": filepath,
                    "line": i,
                    "detail": f"CANN wrapper detected: {desc} — "
                              f"kernel must implement actual computation, not call built-in ops"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 10: datacopy_byte_count
#   The simple-count DataCopy overload takes an element/operand count. Reject
#   clear byte-count expressions before they can compile into an oversized DMA.
# ---------------------------------------------------------------------------
_RE_SIMPLE_DATACOPY_CALL = re.compile(r"\bDataCopy\s*\((.*?)\)\s*;", re.DOTALL)
_RE_BYTE_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*bytes?\w*\b", re.IGNORECASE)
_RE_SIZEOF_DIVISION = re.compile(r"/\s*sizeof\s*\(")


def _mask_cpp_comments(text: str) -> str:
    """Replace C/C++ comments with spaces while preserving line numbers."""
    def _preserve_newlines(match: re.Match) -> str:
        """Mask comment text without moving subsequent diagnostics."""
        return re.sub(r"[^\n]", " ", match.group(0))

    text = re.sub(r"/\*.*?\*/", _preserve_newlines, text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", _preserve_newlines, text)


def _split_call_arguments(body: str) -> List[str]:
    """Split a call body on top-level commas."""
    args = []
    start = 0
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(body):
        if char in "([{":
            stack.append(char)
        elif char in pairs and stack and stack[-1] == pairs[char]:
            stack.pop()
        elif char == "," and not stack:
            args.append(body[start:index].strip())
            start = index + 1
    args.append(body[start:].strip())
    return args


def check_datacopy_byte_count(filepath: str, lines: List[str]) -> List[Violation]:
    """Reject clear byte counts passed to simple-count ``DataCopy``."""
    code = _mask_cpp_comments("".join(lines))
    violations = []
    for match in _RE_SIMPLE_DATACOPY_CALL.finditer(code):
        args = _split_call_arguments(match.group(1))
        if len(args) != 3:
            continue
        count = args[2]
        sizeof_as_divisor = _RE_SIZEOF_DIVISION.search(count) is not None
        has_byte_name = _RE_BYTE_IDENTIFIER.search(count) is not None
        has_sizeof = re.search(r"\bsizeof\s*\(", count) is not None
        if sizeof_as_divisor or (not has_byte_name and not has_sizeof):
            continue
        violations.append({
            "file": filepath,
            "line": code[:match.start()].count("\n") + 1,
            "detail": "DataCopy third argument is a byte count; the simple-count "
                      "overload requires an element/operand count. Pass the aligned "
                      "element count directly, or use DataCopyPad for an unaligned tail."
        })
    return violations


# ---------------------------------------------------------------------------
# Check 11: datacopy_params_unit_mismatch
#   A variable blockCount with blockLen=1 and no stride is an ambiguous,
#   redundant contiguous copy. It commonly means an element count was used as
#   a number of 32-byte DMA bursts.
# ---------------------------------------------------------------------------
_RE_DATACOPY_PARAMS_DECL = re.compile(
    r"\bDataCopyParams\s+([A-Za-z_]\w*)\s*(?:(?:=\s*)?\{\s*\})?\s*;"
)
_RE_COUNT_LIKE_NAME = re.compile(
    r"\b(?:(?:cur|element|elem|tensor)\w*(?:count|length|num)|num(?:elements?|elems?))\w*\b",
    re.IGNORECASE,
)
_RE_EXPLICIT_BLOCK_UNIT = re.compile(r"\b\w*(?:block|burst|row)\w*\b", re.IGNORECASE)
_RE_ZERO_LITERAL = re.compile(
    r"^(?:static_cast\s*<[^>]+>\s*\(\s*)?0[uUlL]*(?:\s*\))?$"
)
_RE_ONE_LITERAL = re.compile(
    r"^(?:static_cast\s*<[^>]+>\s*\(\s*)?1[uUlL]*(?:\s*\))?$"
)


def check_datacopy_params_unit_mismatch(
    filepath: str, lines: List[str]
) -> List[Violation]:
    """Reject the known element-count-as-blockCount misuse."""
    code = _mask_cpp_comments("".join(lines))
    violations = []
    for declaration in _RE_DATACOPY_PARAMS_DECL.finditer(code):
        name = declaration.group(1)
        tail = code[declaration.end():declaration.end() + 4096]
        call_end = None
        for call in _RE_SIMPLE_DATACOPY_CALL.finditer(tail):
            args = _split_call_arguments(call.group(1))
            if len(args) == 3 and args[2].strip() == name:
                call_end = call.end()
                break
        if call_end is None:
            continue
        setup = tail[:call_end]
        block_lens = re.findall(rf"\b{re.escape(name)}\.blockLen\s*=\s*([^;]+);", setup)
        block_counts = re.findall(rf"\b{re.escape(name)}\.blockCount\s*=\s*([^;]+);", setup)
        if not block_lens or not block_counts:
            continue
        block_len = block_lens[-1].strip()
        block_count = block_counts[-1].strip()
        stride_values = re.findall(
            rf"\b{re.escape(name)}\.(?:srcStride|dstStride|srcGap|dstGap)\s*=\s*([^;]+);",
            setup,
        )
        has_nonzero_stride = any(
            _RE_ZERO_LITERAL.fullmatch(value.strip()) is None for value in stride_values
        )
        if (
            _RE_ONE_LITERAL.fullmatch(block_len) is None
            or has_nonzero_stride
            or _RE_EXPLICIT_BLOCK_UNIT.search(block_count)
            or not _RE_COUNT_LIKE_NAME.search(block_count)
        ):
            continue
        violations.append({
            "file": filepath,
            "line": code[:declaration.start()].count("\n") + 1,
            "detail": "DataCopyParams uses a count-like variable as blockCount with "
                      "blockLen=1 and no stride. blockCount counts DMA bursts and "
                      "blockLen counts 32-byte blocks; use simple-count DataCopy for "
                      "a contiguous element range or define genuine strided geometry."
        })
    return violations


# ---------------------------------------------------------------------------
# Check 12: cast_rint_same_dtype
#   CAST_RINT is a conversion rounding mode.  Applying it when source and
#   destination resolve to the same dtype can change values (the observed
#   fp32->fp32 failure rounded GELU gradients to 0/1).  Generic kernels that
#   also instantiate T=float must keep the Cast in an explicit non-float
#   constexpr branch and use a non-rounding path for float.
# ---------------------------------------------------------------------------
_RE_CAST_RINT_CALL = re.compile(
    r"\b(?:AscendC::)?Cast(?:\s*<\s*[^;()<>]+\s*>)?\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_RE_LOCAL_TENSOR_TYPE = re.compile(
    r"\b(?:AscendC::)?LocalTensor\s*<\s*(?P<decl_type>[^>]+?)\s*>\s*"
    r"(?P<decl_var>[A-Za-z_]\w*)|\bauto\s+(?P<auto_var>[A-Za-z_]\w*)\s*=\s*"
    r"[^;]*?(?:\.|\b)(?:Get|AllocTensor|DeQue)\s*<\s*"
    r"(?P<auto_type>[^>]+?)\s*>\s*\(", re.DOTALL,
)
_RE_TEMPLATE_CLASS = re.compile(
    r"\btemplate\s*<\s*(?:typename|class)\s+([A-Za-z_]\w*)\s*>\s*"
    r"(?:class|struct)\s+([A-Za-z_]\w*)", re.DOTALL,
)
_RE_LOCAL_TENSOR_OPERAND = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\[\]]+\]\s*)*$"
)


def _cast_inside_non_float_guard(prefix: str, template_param: str) -> bool:
    """Return true only while the canonical constexpr branch remains open."""
    # Braces inside ordinary C++ string/character literals are not scopes.
    prefix = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda match: " " * len(match.group(0)),
        prefix,
    )
    guard = re.compile(
        r"if\s+constexpr\s*\(\s*!\s*std::is_same_v\s*<\s*"
        + re.escape(template_param)
        + r"\s*,\s*float\s*>\s*\)\s*\{"
    )
    for match in guard.finditer(prefix):
        depth = 1
        for char in prefix[match.end():]:
            depth += (char == "{") - (char == "}")
            if depth == 0:
                break
        if depth > 0:
            return True
    return False


def check_cast_rint_same_dtype(filepath: str, lines: List[str]) -> List[Violation]:
    """Reject CAST_RINT calls that are or become same-dtype conversions."""
    code = _mask_cpp_comments("".join(lines))
    tensor_types = {}
    for match in _RE_LOCAL_TENSOR_TYPE.finditer(code):
        name = match.group("decl_var") or match.group("auto_var")
        dtype = match.group("decl_type") or match.group("auto_type")
        tensor_types[name] = re.sub(r"\s+", "", dtype)
    templates = list(_RE_TEMPLATE_CLASS.findall(code))
    sibling_code = ""
    if templates and os.path.isfile(filepath):
        directory = os.path.dirname(os.path.abspath(filepath))
        try:
            sibling_names = os.listdir(directory)
        except OSError:
            sibling_names = []
        for name in sibling_names:
            if not name.endswith((".h", ".cpp")):
                continue
            try:
                with open(os.path.join(directory, name), "r", encoding="utf-8",
                          errors="replace") as stream:
                    sibling_code += _mask_cpp_comments(stream.read())
            except OSError:
                continue

    violations = []
    for match in _RE_CAST_RINT_CALL.finditer(code):
        args = _split_call_arguments(match.group(1))
        if len(args) < 4 or "CAST_RINT" not in args[2]:
            continue
        dst_match = _RE_LOCAL_TENSOR_OPERAND.fullmatch(args[0])
        src_match = _RE_LOCAL_TENSOR_OPERAND.fullmatch(args[1])
        if dst_match is None or src_match is None:
            continue
        dst_type = tensor_types.get(dst_match.group(1))
        src_type = tensor_types.get(src_match.group(1))
        if not dst_type or not src_type:
            continue

        reason = ""
        if dst_type == src_type:
            reason = "both operands have element type '{}'".format(dst_type)
        else:
            pair = {dst_type, src_type}
            template = next(
                (
                    (param, entity) for param, entity in templates
                    if pair == {"float", param}
                ),
                None,
            )
            if template is None:
                continue
            generic, entity = template
            float_instance = r"\b{}\s*<\s*float\s*>".format(re.escape(entity))
            if re.search(float_instance, sibling_code) is None:
                continue
            if _cast_inside_non_float_guard(code[:match.start()], generic):
                continue
            reason = (
                "template parameter '{}' is instantiated as float in this kernel"
                .format(generic)
            )

        violations.append({
            "file": filepath,
            "line": code[:match.start()].count("\n") + 1,
            "detail": "CAST_RINT can perform a same-dtype conversion because {}; "
                      "fp32-to-fp32 CAST_RINT rounds values instead of preserving "
                      "them. Use Cast only in an explicit non-float constexpr "
                      "conversion branch and use a direct/non-rounding VEC path "
                      "when source and destination dtypes match."
                      .format(reason),
        })
    return violations


# ---------------------------------------------------------------------------
# Check 13: kernel_has_computation
#   Verify that kernel .h files contain actual computation logic (memory ops,
#   arithmetic, control flow) rather than being trivial stubs or wrappers.
#   A genuine AscendC kernel should have: TQue/TBuf declarations, DataCopy
#   or SetAtomicAdd/etc calls, and VEC or scalar computation.
# ---------------------------------------------------------------------------
# SIMD kernels use TQue/DataCopy/VEC; SIMT kernels use raw GM pointers + scalar loops.
# Both are legitimate — the check must detect WHICH style and validate accordingly.
_RE_SIMD_MARKERS = {
    "tque_or_tbuf": re.compile(r'\bT(?:Que|Buf)\s*<'),
    "data_copy": re.compile(r'\bDataCopy\b'),
    "vec_op": re.compile(r'\b(?:Add|Sub|Mul|Div|Abs|Exp|Reciprocal|Muls|Adds|Cast|'
                         r'ReduceSum|ReduceMax|ReduceMin|WholeReduceSum|Duplicate|'
                         r'Compare|Select|Gather|Scatter)\s*[<(]'),
    "pipe_or_enque": re.compile(r'\b(?:EnQue|DeQue|SetFlag|WaitFlag)\b'),
    "global_tensor": re.compile(r'\bGlobalTensor\s*<'),
    "local_tensor": re.compile(r'\bLocalTensor\s*<'),
}

_RE_SIMT_MARKERS = {
    "gm_addr": re.compile(r'\bGM_ADDR\b'),
    "simt_vf": re.compile(r'__simt_vf__'),
    "simt_thread": re.compile(r'\bSimt::(?:GetThreadIdx|GetThreadNum)\b'),
    "gm_pointer": re.compile(r'__gm__\s+(?:const\s+)?(?:float|half|int|uint)'),
    "scalar_loop": re.compile(r'\bfor\s*\(.*<.*\)'),
    "arithmetic": re.compile(r'[+\-*/]=|[+\-*/]\s'),
}

# Minimum markers for each style
_MIN_SIMD_MARKERS = 3
_MIN_SIMT_MARKERS = 3


def _kernel_tree_root(filepath: str) -> str:
    """Return the nearest enclosing ``kernel/`` directory, else the file's own dir."""
    cur = os.path.dirname(os.path.abspath(filepath))
    root = cur
    while True:
        if os.path.basename(cur) == "kernel":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return root
        cur = parent


def _kernel_tree_sources(root: str) -> List[str]:
    """List every ``.h``/``.cpp`` file under a kernel tree, in stable order."""
    sources: List[str] = []
    if not os.path.isdir(root):
        return sources
    for walk_root, _dirs, walk_files in os.walk(root):
        for sib in sorted(walk_files):
            if sib.endswith(".h") or sib.endswith(".cpp"):
                sources.append(os.path.join(walk_root, sib))
    return sources


def _sibling_carries_computation(filepath: str, markers: Dict[str, "re.Pattern"], min_required: int) -> bool:
    """Report whether a sibling kernel source carries the computation markers.

    Helper-header carve-out (2026-06-10, FA workspace_queue.h/kernel_common.h false
    positives): multi-file kernels legitimately split sync barriers / shared decls into
    compute-free headers. Only flag when NO sibling kernel file carries the
    computation — preserves stub-kernel detection, removes per-file FP.
    2026-08-24 extension: scan the whole kernel tree, not just the same
    directory — multi-dir projects (op_kernel/ + utils/) keep compute-free
    tiling/ABI headers in separate directories (FusionAttention
    utils/fa_tiling_data.h carries __aicore__ only in #ifdef host/device
    guards and was false-flagged while computation lives in
    ../op_kernel/fusion_attention.cpp).
    """
    for sib_path in _kernel_tree_sources(_kernel_tree_root(filepath)):
        if os.path.abspath(sib_path) == os.path.abspath(filepath):
            continue
        try:
            with open(sib_path, encoding="utf-8", errors="ignore") as sib_file:
                sib_text = sib_file.read()
        except OSError:
            continue
        if sum(1 for pat in markers.values() if pat.search(sib_text)) >= min_required:
            return True
    return False


def check_kernel_has_computation(filepath: str, lines: List[str]) -> List[Violation]:
    """Verify kernel files contain actual AscendC computation, not trivial stubs.
    Distinguishes SIMD (TQue/DataCopy/VEC) from SIMT (raw GM pointers/scalar loops).
    """
    if not filepath.endswith(".h"):
        return []

    full_text = "".join(lines)
    if "__aicore__" not in full_text:
        return []

    # Detect kernel style
    is_simt = bool(re.search(r'__simt_vf__', full_text))

    if is_simt:
        markers = _RE_SIMT_MARKERS
        min_required = _MIN_SIMT_MARKERS
        style = "SIMT"
    else:
        markers = _RE_SIMD_MARKERS
        min_required = _MIN_SIMD_MARKERS
        style = "SIMD"

    found = set()
    for name, pat in markers.items():
        if pat.search(full_text):
            found.add(name)

    if len(found) >= min_required:
        return []
    if _sibling_carries_computation(filepath, markers, min_required):
        return []  # computation lives in a sibling; this file is a helper header
    return [{
        "file": filepath,
        "line": 1,
        "detail": f"{style} kernel has only {len(found)}/{min_required} "
                  f"computation markers (found: {sorted(found)}). "
                  f"This may be a trivial stub or CANN wrapper."
    }]


# ---------------------------------------------------------------------------
# Registry of all checks
# ---------------------------------------------------------------------------
CHECKS = [
    ("missing_namespace", check_missing_namespace),
    ("missing_kernel_operator", check_missing_kernel_operator),
    ("unconditional_simt_compat", check_unconditional_simt_compat),
    ("bf16_static_cast", check_bf16_static_cast),
    ("simt_namespace", check_simt_namespace),
    ("float_fp16_param", check_float_fp16_param),
    ("sort_bounds_missing", check_sort_bounds_missing),
    ("simt_vf_nonvoid", check_simt_vf_nonvoid),
    ("cann_wrapper_call", check_cann_wrapper_call),
    ("datacopy_byte_count", check_datacopy_byte_count),
    ("datacopy_params_unit_mismatch", check_datacopy_params_unit_mismatch),
    ("cast_rint_same_dtype", check_cast_rint_same_dtype),
    ("kernel_has_computation", check_kernel_has_computation),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_prestaged_exempt_set(directory: str) -> set[str]:
    """Return intact files recorded by ``.upstream_prestaged.json``.

    Prestage manifests are provenance records, not blanket bypass markers.  A
    file is exempt only when its workspace-relative path stays inside the
    manifest root and its current SHA256 still matches the recorded digest.
    Missing, malformed, escaped, symlinked, or modified entries remain subject
    to every static check.
    """
    candidate_paths = [
        os.path.join(directory, ".upstream_prestaged.json"),
        os.path.join(
            os.path.dirname(os.path.abspath(directory)),
            ".upstream_prestaged.json",
        ),
    ]
    for manifest_path in candidate_paths:
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (json.JSONDecodeError, OSError):
            continue
        staged = manifest.get("staged_files", {})
        if not isinstance(staged, dict):
            continue
        workspace_root = os.path.realpath(os.path.dirname(manifest_path))
        exempt: set[str] = set()
        for rel, recorded_sha in staged.items():
            if not isinstance(rel, str) or not isinstance(recorded_sha, str):
                continue
            candidate = os.path.abspath(os.path.join(workspace_root, rel))
            try:
                if os.path.commonpath((workspace_root, candidate)) != workspace_root:
                    continue
            except ValueError:
                continue
            if os.path.islink(candidate) or not os.path.isfile(candidate):
                continue
            try:
                with open(candidate, "rb") as stream:
                    actual_sha = hashlib.sha256(stream.read()).hexdigest()
            except OSError:
                continue
            if actual_sha == recorded_sha.lower():
                exempt.add(candidate)
        return exempt
    return set()


def collect_files(directory: str) -> List[str]:
    """Collect C++ sources, excluding intact provenance-tracked prestage files."""
    exempt = _load_prestaged_exempt_set(directory)
    result = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith((".h", ".cpp")):
                full = os.path.join(root, fname)
                if os.path.abspath(full) in exempt:
                    continue
                result.append(full)
    return result


def run_checks(directory: str) -> dict:
    files = collect_files(directory)
    if not files:
        print("Warning: no .h or .cpp files found in '{}'".format(directory),
              file=sys.stderr)

    report = {
        "passed": True,
        "checks": {},
        "summary": "",
    }

    total_violations = 0
    checks_passed = 0

    for check_name, check_fn in CHECKS:
        all_violations = []  # type: List[Violation]
        for fpath in files:
            lines = read_lines(fpath)
            # Make paths relative to the scanned directory for cleaner output
            rel_path = os.path.relpath(fpath, directory)
            violations = check_fn(fpath, lines)
            # Rewrite file paths to relative
            for v in violations:
                v["file"] = rel_path
            all_violations.extend(violations)

        check_passed = len(all_violations) == 0
        if check_passed:
            checks_passed += 1
        else:
            report["passed"] = False

        total_violations += len(all_violations)
        report["checks"][check_name] = {
            "passed": check_passed,
            "violations": all_violations,
        }

    report["summary"] = "{}/{} checks passed, {} violations found".format(
        checks_passed, len(CHECKS), total_violations
    )

    return report


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: {} <directory>".format(sys.argv[0]), file=sys.stderr)
        return 2

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: '{}' is not a directory".format(directory), file=sys.stderr)
        return 2

    report = run_checks(directory)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

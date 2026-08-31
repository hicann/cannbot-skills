# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Authored-candidate delivery contract for the controlled CANN build.

This module owns everything that inspects the *authored* candidate before any
build tool runs: the delivery-file inventory, the comment/literal strippers used
by the anti-copy and framework-fallback gates, the AST helpers, the
TileLang2AscendC custom-op boundary validator, and the two content digests
(``kernel/CMakeLists.txt`` and the candidate source tree) that bind a build
receipt to the delivery it was produced from.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import platform
import re
import stat
import tokenize
from pathlib import Path
from typing import Any, Mapping, Sequence

from npubench.npubench_target_base import (
    CandidateContractError,
    TargetTransportError,
    _read_json,
    _sha,
)


_HOST_IS_AARCH64 = platform.machine() in ("aarch64", "arm64")


TILELANG2ASCENDC_SOURCE_KIND = "port-aclnn-tilelang2ascendc"


TILELANG2ASCENDC_CANDIDATE_KIND = "tilelang2ascendc_custom_op"


TILELANG2ASCENDC_STABLE_AUTHORED_FILES = frozenset(
    {
        "model_new_ascendc.py",
        "kernel/CMakeLists.txt",
        "kernel/register.cpp",
        # 2026-08-21: setup.py is standard setuptools packaging glue (the
        # same Extension/BuildExtension boilerplate as CMakeLists.txt and
        # register.cpp, both already whitelisted).  A byte-identical copy of
        # it carries no compute implementation, so rejecting it was an
        # unfair candidate kill — the anti-copy gates for op_host/op_kernel
        # compute files are unchanged.
        "kernel/setup.py",
    }
)


TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA = "cannbot.tilelang2ascendc_candidate_independence/v1"


def _candidate_delivery_files(workspace: Path) -> list[Path]:
    """Return the regular authored candidate files covered by the build digest."""
    root = Path(workspace).resolve()
    entry = root / "model_new_ascendc.py"
    kernel = root / "kernel"
    if entry.is_symlink() or not entry.is_file():
        raise TargetTransportError("candidate model_new_ascendc.py is missing or not a regular file")
    if kernel.is_symlink() or not kernel.is_dir():
        raise TargetTransportError("candidate kernel directory is missing or unsafe")
    paths: list[Path] = [entry]
    for path in sorted(kernel.rglob("*")):
        if path.is_symlink():
            raise TargetTransportError(f"candidate contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            relative = path.relative_to(root)
            # Build products are deliberately outside the authored candidate
            # contract and are recreated by the controlled builder.  Keeping
            # them out here aligns static scanning with _candidate_source_digest
            # and prevents an old binary containing a forbidden token from
            # poisoning a resumable source candidate.
            if "build" in relative.parts or path.suffix.lower() in {".so", ".o", ".a", ".pyc"}:
                continue
            paths.append(path)
    for path in paths:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise TargetTransportError(f"cannot inspect candidate {path.relative_to(root)}: {exc}") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TargetTransportError(
                f"candidate file must be a single regular file: {path.relative_to(root)}"
            )
    return paths


_TILELANG2ASCENDC_MODEL_FRAMEWORK_PATTERNS = (
    # Any torch.ops namespace other than the mandated ``npu`` one is framework dispatch:
    # the candidate contract below requires a registered ``torch.ops.npu`` call.
    (r"\btorch\s*\.\s*ops\s*\.\s*(?!npu\b)[A-Za-z_]\w*\s*\.",
     "framework torch.ops dispatch"),
    (r"\btorch\.nn\.functional\s*\.\s*[A-Za-z_]\w*\s*\(",
     "torch.nn.functional compute"),
    (r"\bF\s*\.\s*[A-Za-z_]\w*\s*\(", "F.* framework compute"),
    (r"\btorch\s*\.\s*(?:relu|gelu|silu|sigmoid|tanh|exp|log|sqrt|rsqrt|pow|"
     r"add|sub|mul|div|matmul|mm|bmm|sum|mean|max|min|softmax|where|cat|stack|"
     r"gather|scatter|nonzero|argmax|argmin|cumsum)\s*\(",
     "torch.* framework compute"),
    (r"\btorch_npu\b", "torch_npu delegation"),
    (r"\b(?:numpy|numpy\.|np\s*\.)", "NumPy/CPU fallback"),
    (r"\.(?:cpu|to_cpu)\s*\(\s*\)", "CPU tensor fallback"),
)


def _python_code_without_comments_or_literals(text: str) -> str:
    """Keep Python syntax tokens while hiding comments/docstrings from gates."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return tokenize.untokenize(
            (token.type, "" if token.type in {tokenize.COMMENT, tokenize.STRING} else token.string)
            for token in tokens
        )
    except (SyntaxError, tokenize.TokenError):
        # A syntax-invalid candidate is rejected by the target evaluator later;
        # retain a conservative lexical scan here instead of allowing comments
        # or a malformed string to hide a framework fallback.
        return re.sub(r"#[^\n]*", "", text)


def _python_code_without_comments(text: str) -> str:
    """Remove Python comments while retaining string literals for provenance."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return tokenize.untokenize(
            (token.type, "" if token.type == tokenize.COMMENT else token.string)
            for token in tokens
        )
    except (SyntaxError, tokenize.TokenError):
        return re.sub(r"#[^\n]*", "", text)


def _cxx_code_without_comments_or_literals(text: str) -> str:
    """Keep C/C++ tokens while excluding comments and string-literal decoys."""
    text = _cxx_code_without_comments(text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def _cxx_code_without_comments(text: str) -> str:
    """Remove C/C++ comments while retaining string literals and code."""
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _read_candidate_text(path: Path, label: str, candidate_kind: str = "ACLNN") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TargetTransportError(
            f"{candidate_kind} candidate {label} is not readable UTF-8"
        ) from exc


def _normalised_authored_text(path: Path, text: str) -> str | None:
    """Return code with comments/literals/formatting removed for copy checks."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        text = _python_code_without_comments(text)
    elif suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".asc"}:
        text = _cxx_code_without_comments(text)
    elif path.name == "CMakeLists.txt" or suffix == ".cmake":
        text = re.sub(r"#[^\n]*", "", text)
    else:
        return None
    return re.sub(r"\s+", " ", text).strip()


def _ast_attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _ast_contains_name(node: ast.AST | None, names: set[str]) -> bool:
    if node is None:
        return False
    return any(isinstance(item, ast.Name) and item.id in names for item in ast.walk(node))


def _ast_assignment_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    """Return every plain name bound by one assignment statement."""
    targets: list[ast.AST] = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in targets:
        for item in ast.walk(target):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                names.add(item.id)
    return names


_TILELANG2ASCENDC_CXX_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".asc"}
)
_TILELANG2ASCENDC_HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx"})
_TILELANG2ASCENDC_TRANSLATION_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
_TILELANG2ASCENDC_KERNEL_PREFIXES = ("kernel/op_host/", "kernel/op_kernel/")
_TILELANG2ASCENDC_KERNEL_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".asc"})
_TILELANG2ASCENDC_LAUNCH_RE = r"\bEXEC_KERNEL_CMD\b|\baclrtlaunch_[A-Za-z_]\w*\b|\bACLRT_LAUNCH_KERNEL\s*\("


def _read_tilelang2ascendc_text(path: Path, label: str) -> str:
    return _read_candidate_text(path, label, "TileLang2AscendC")


def _has_cxx_source(root: Path) -> bool:
    """Report whether a directory contains at least one C/C++ source file."""
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            return True
    return False


def _require_tilelang2ascendc_layout(workspace: Path) -> dict[str, Path]:
    """Return the mandatory authored files after checking the project layout."""
    required = {
        "model_new_ascendc.py": workspace / "model_new_ascendc.py",
        "kernel/CMakeLists.txt": workspace / "kernel" / "CMakeLists.txt",
        "kernel/register.cpp": workspace / "kernel" / "register.cpp",
    }
    for label, path in required.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise TargetTransportError(f"TileLang2AscendC candidate requires a regular {label}")
    for label in ("kernel/op_host", "kernel/op_kernel"):
        path = workspace / label
        if path.is_symlink() or not path.is_dir():
            raise TargetTransportError(f"TileLang2AscendC candidate requires {label}/")
        if not _has_cxx_source(path):
            raise TargetTransportError(f"TileLang2AscendC candidate {label}/ has no source files")
    return required


def _subclasses_module(model_class: ast.ClassDef) -> bool:
    for base in model_class.bases:
        chain = _ast_attribute_chain(base)
        if chain and chain[-1] == "Module":
            return True
    return False


def _class_function(model_class: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in model_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _tilelang2ascendc_model_forward(model_raw: str) -> ast.FunctionDef:
    """Return ModelNew.forward once the Python entry passes its gates."""
    model_code = _python_code_without_comments_or_literals(model_raw)
    for pattern, description in _TILELANG2ASCENDC_MODEL_FRAMEWORK_PATTERNS:
        match = re.search(pattern, model_code, flags=re.IGNORECASE)
        if match:
            raise TargetTransportError(
                f"TileLang2AscendC candidate uses forbidden {description}: {match.group(0)!r}"
            )
    try:
        model_tree = ast.parse(model_raw)
    except SyntaxError as exc:
        raise TargetTransportError(f"TileLang2AscendC model_new_ascendc.py is not valid Python: {exc}") from exc
    classes = {node.name: node for node in model_tree.body if isinstance(node, ast.ClassDef)}
    model_class = classes.get("ModelNew") or classes.get("Model")
    if model_class is None:
        raise TargetTransportError("TileLang2AscendC candidate must define ModelNew (or compatibility Model)")
    if not _subclasses_module(model_class):
        raise TargetTransportError(
            "TileLang2AscendC ModelNew must subclass torch.nn.Module (or imported Module)"
        )
    forward = _class_function(model_class, "forward")
    if forward is None:
        raise TargetTransportError("TileLang2AscendC candidate ModelNew must define synchronous forward")
    return forward


def _tilelang2ascendc_cxx_paths(workspace: Path, files: Sequence[Path]) -> list[Path]:
    """Select the authored C/C++ files that live under the kernel project."""
    kernel_root = workspace / "kernel"
    paths: list[Path] = []
    for path in files:
        if path.suffix.lower() not in _TILELANG2ASCENDC_CXX_SUFFIXES:
            continue
        if path.is_relative_to(kernel_root):
            paths.append(path)
    return paths


def _tilelang2ascendc_registered_ops(register_code: str) -> set[str]:
    """Return the op names register.cpp binds into the ``npu`` namespace."""
    if not re.search(
        r"\bTORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*npu\s*,", register_code
    ) or not re.search(
        r"\bTORCH_LIBRARY_IMPL\s*\(\s*npu\s*,", register_code
    ):
        raise TargetTransportError(
            "TileLang2AscendC register.cpp lacks TORCH_LIBRARY/TORCH_LIBRARY_IMPL for namespace npu"
        )
    if not re.search(r"\bPYBIND11_MODULE\s*\(", register_code):
        raise TargetTransportError(
            "TileLang2AscendC register.cpp lacks PYBIND11_MODULE; the built extension "
            "must export PyInit_<module> or the evaluator import fails"
        )
    registered = set(re.findall(r'\bm\.def\s*\(\s*["\']([A-Za-z0-9_]+)', register_code))
    registered.update(re.findall(r'\bm\.impl\s*\(\s*["\']([A-Za-z0-9_]+)', register_code))
    return registered


def _tilelang2ascendc_custom_op_calls(forward: ast.FunctionDef) -> list[ast.Call]:
    """Collect the ``torch.ops.npu.*`` calls made by ModelNew.forward."""
    calls: list[ast.Call] = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call):
            continue
        if _ast_attribute_chain(node.func)[:3] == ("torch", "ops", "npu"):
            calls.append(node)
    if not calls:
        raise TargetTransportError("TileLang2AscendC ModelNew.forward has no torch.ops.npu call")
    return calls


def _tilelang2ascendc_matched_ops(forward_calls: Sequence[ast.Call], registered: set[str]) -> list[str]:
    """Intersect the called custom ops with the ops register.cpp declares."""
    called_ops: set[str] = set()
    for node in forward_calls:
        chain = _ast_attribute_chain(node.func)
        if len(chain) == 4:
            called_ops.add(chain[3])
    matched = sorted(called_ops.intersection(registered))
    if not matched:
        raise TargetTransportError(
            "TileLang2AscendC ModelNew.forward must call a registered torch.ops.npu custom op"
        )
    return matched


def _derives_from_custom_op(value: ast.AST, forward_calls: Sequence[ast.Call]) -> bool:
    for node in ast.walk(value):
        if isinstance(node, ast.Call) and node in forward_calls:
            return True
    return False


def _returns_custom_op_result(
    node: ast.Return, result_names: set[str], forward_calls: Sequence[ast.Call]
) -> bool:
    if node.value is None:
        return False
    if _ast_contains_name(node.value, result_names):
        return True
    return _derives_from_custom_op(node.value, forward_calls)


def _require_custom_op_derived_return(
    forward: ast.FunctionDef, forward_calls: Sequence[ast.Call]
) -> None:
    """Require the forward return to flow from the custom-op call itself."""
    result_names: set[str] = set()
    for node in ast.walk(forward):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and _derives_from_custom_op(node.value, forward_calls)
        ):
            result_names.update(_ast_assignment_names(node))
    returns = [node for node in ast.walk(forward) if isinstance(node, ast.Return)]
    derived = any(
        _returns_custom_op_result(node, result_names, forward_calls) for node in returns
    )
    if not returns or not derived:
        raise TargetTransportError("TileLang2AscendC ModelNew.forward return is not derived from its custom op")


def _tilelang2ascendc_device_code(workspace: Path, cxx_code: Mapping[Path, str]) -> str:
    """Join the device-side sources: op_kernel plus kernel-root headers."""
    kernel_root = workspace / "kernel"
    op_kernel = kernel_root / "op_kernel"
    parts: list[str] = []
    for path, code in cxx_code.items():
        if path.is_relative_to(op_kernel):
            parts.append(code)
            continue
        if path.parent == kernel_root and path.suffix.lower() in _TILELANG2ASCENDC_HEADER_SUFFIXES:
            parts.append(code)
    return "\n".join(parts)


def _validate_tilelang2ascendc_device_code(
    device_code: str, translation_units: Sequence[Path]
) -> None:
    """Require real AscendC device evidence and reject host-only constructs."""
    if not translation_units:
        raise TargetTransportError("TileLang2AscendC candidate has no authored C/C++ translation unit")
    if not re.search(r"\b__global__\b", device_code) or not re.search(r"\b__aicore__\b", device_code):
        raise TargetTransportError("TileLang2AscendC op_kernel has no __global__ + __aicore__ device entry")
    if not re.search(r"\bAscendC\s*::\s*[A-Za-z_]\w*", device_code):
        raise TargetTransportError("TileLang2AscendC op_kernel has no AscendC device implementation evidence")
    if re.search(
        r"\bstd\s*::\s*(?:min|max)\s*(?:<[^;{}()]*>\s*)?\(|#\s*include\s*<algorithm>",
        device_code,
    ):
        raise TargetTransportError(
            "TileLang2AscendC op_kernel uses host STL min/max or <algorithm>; "
            "use device-safe scalar logic instead"
        )


def _validate_tilelang2ascendc_cmake(cmake_raw: str, register_code: str) -> None:
    """Check the authored CMake target, host arch paths and module naming."""
    if not re.search(r"\b(?:ascendc_library|add_library)\s*\(", cmake_raw):
        raise TargetTransportError("TileLang2AscendC kernel/CMakeLists.txt has no CANN build target")
    # 2026-08-21 (batch campaign): candidates repeatedly inherited hardcoded
    # x86_64-linux CANN paths from sources generated on x86 hosts, failing the
    # aarch64 build with a raw compiler error.  Fail fast with an actionable
    # repair hint instead.
    if _HOST_IS_AARCH64 and "x86_64-linux" in cmake_raw:
        raise TargetTransportError(
            "TileLang2AscendC kernel/CMakeLists.txt hardcodes x86_64-linux CANN "
            "paths on an aarch64 target — replace every x86_64-linux with "
            "aarch64-linux (candidate repair required)"
        )
    # 2026-08-21 (batch campaign): the evaluator imports the extension under the
    # name model_new_ascendc.py imports, which must equal BOTH the
    # PYBIND11_MODULE name (PyInit symbol) and the CMake OUTPUT_NAME (built
    # filename).  A mismatch builds fine and then dies late in the isolated
    # fixture stage with a bare ModuleNotFoundError.  Fail fast instead.
    register_pybind = re.findall(r"\bPYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", register_code)
    cmake_outputs = re.findall(r"\bOUTPUT_NAME\s+([A-Za-z_][A-Za-z0-9_]*)", cmake_raw)
    if register_pybind and cmake_outputs and not (
        set(register_pybind) & set(cmake_outputs)
    ):
        raise TargetTransportError(
            "TileLang2AscendC extension module name mismatch: PYBIND11_MODULE "
            f"{sorted(set(register_pybind))} != CMake OUTPUT_NAME "
            f"{sorted(set(cmake_outputs))} — align register.cpp, "
            "kernel/CMakeLists.txt OUTPUT_NAME and model_new_ascendc.py import "
            "(candidate repair required)"
        )


def _tilelang2ascendc_device_entries(device_code: str) -> list[str]:
    entries = sorted(set(re.findall(
        r"\b(?:extern\s+\"C\"\s+)?(?:__global__\s+__aicore__|__aicore__\s+__global__)\s+void\s+([A-Za-z_]\w*)\s*\(",
        device_code,
    )))
    if not entries:
        raise TargetTransportError(
            "TileLang2AscendC op_kernel has no recognizable __global__ + __aicore__ void entry"
        )
    return entries


def _tilelang2ascendc_cxx_code(workspace: Path, cxx_paths: Sequence[Path]) -> dict[Path, str]:
    code: dict[Path, str] = {}
    for path in cxx_paths:
        label = str(path.relative_to(workspace))
        code[path] = _cxx_code_without_comments_or_literals(_read_tilelang2ascendc_text(path, label))
    return code


def _validate_tilelang2ascendc_kernel_boundary(
    workspace: Path, files: Sequence[Path]
) -> Mapping[str, Any]:
    """Validate the authored TileLang2AscendC custom-op boundary.

    This is the sole authored-candidate validator.  TileLang2AscendC output
    uses ``register.cpp`` + ``torch.ops.npu`` and a nested ``kernel/`` project;
    it does not use the direct route's PyBind ABI and must not be rejected for
    lacking ``kernel/pybind11.cpp``.
    """
    workspace = Path(workspace).resolve()
    required = _require_tilelang2ascendc_layout(workspace)
    model_raw = _read_tilelang2ascendc_text(required["model_new_ascendc.py"], "model_new_ascendc.py")
    forward = _tilelang2ascendc_model_forward(model_raw)
    cxx_paths = _tilelang2ascendc_cxx_paths(workspace, files)
    cxx_code = _tilelang2ascendc_cxx_code(workspace, cxx_paths)
    register_raw = _read_tilelang2ascendc_text(required["kernel/register.cpp"], "kernel/register.cpp")
    register_code = _cxx_code_without_comments(register_raw)
    registered = _tilelang2ascendc_registered_ops(register_code)
    forward_calls = _tilelang2ascendc_custom_op_calls(forward)
    matched = _tilelang2ascendc_matched_ops(forward_calls, registered)
    _require_custom_op_derived_return(forward, forward_calls)

    translation_units: list[Path] = []
    for path in cxx_paths:
        if path.suffix.lower() in _TILELANG2ASCENDC_TRANSLATION_SUFFIXES:
            translation_units.append(path)
    device_code = _tilelang2ascendc_device_code(workspace, cxx_code)
    _validate_tilelang2ascendc_device_code(device_code, translation_units)
    # Launch evidence (2026-08-21 scope fix): the launch idiom may legally live in
    # a candidate-owned helper header (e.g. kernel/utils/*.h wrapping
    # ACLRT_LAUNCH_KERNEL) instead of appearing literally in op_host/*.cpp.
    # Searching only op_host rejected a real candidate whose
    # kernel/utils/torch_kernel_helper.h macro body contains a genuine
    # ACLRT_LAUNCH_KERNEL call.  Scan the WHOLE authored kernel tree
    # (comment/literal-stripped) so the evidence check follows the evidence.
    candidate_code = "\n".join(cxx_code.values())
    if not re.search(_TILELANG2ASCENDC_LAUNCH_RE, candidate_code):
        raise TargetTransportError("TileLang2AscendC candidate has no host kernel-launch evidence")
    cmake_raw = _read_tilelang2ascendc_text(required["kernel/CMakeLists.txt"], "kernel/CMakeLists.txt")
    _validate_tilelang2ascendc_cmake(cmake_raw, register_code)
    device_entries = _tilelang2ascendc_device_entries(device_code)
    launch_evidence = sorted(
        set(
            re.findall(
                r"\b(?:EXEC_KERNEL_CMD|aclrtlaunch_[A-Za-z_]\w*|ACLRT_LAUNCH_KERNEL)\b",
                candidate_code,
            )
        )
    )
    return {
        "schema": TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA,
        "format": "tilelang2ascendc",
        "python_entry": "model_new_ascendc.py:ModelNew.forward",
        "registered_ops": sorted(registered),
        "custom_op_calls": matched,
        "kernel_translation_units": sorted(str(path.relative_to(workspace)) for path in translation_units),
        "device_entries": device_entries,
        "host_launch_evidence": launch_evidence,
        "pybind_required": False,
    }


def _validate_candidate_for_controlled_build(
    workspace: Path,
    source_kind: str,
    source_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Validate the route-specific authored candidate boundary before CANN."""
    workspace = Path(workspace)
    if source_kind != TILELANG2ASCENDC_SOURCE_KIND:
        raise CandidateContractError(
            f"unsupported controlled candidate source kind: {source_kind!r}"
        )
    try:
        return _validate_tilelang2ascendc_candidate_for_build(workspace, source_manifest)
    except CandidateContractError:
        raise
    except TargetTransportError as exc:
        # Route authored-candidate contract violations to worker repair
        # (mrb CandidateContractError framework); the tilelang2ascendc validator
        # still signals them as TargetTransportError internally.
        raise CandidateContractError(str(exc)) from exc


def _require_candidate_entry_files(workspace: Path) -> None:
    entry = workspace / "model_new_ascendc.py"
    cmake = workspace / "kernel" / "CMakeLists.txt"
    for path, label in ((entry, "model_new_ascendc.py"), (cmake, "kernel/CMakeLists.txt")):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise TargetTransportError(f"TileLang2AscendC candidate requires a regular {label}")


def _load_tilelang2ascendc_source_manifest(workspace: Path) -> Mapping[str, Any]:
    """Re-resolve the authenticated source manifest through the transport.

    The binding is imported inside the function on purpose: ``npubench_target``
    stays the single owner of the verifier, the late import keeps the module
    graph acyclic, and the name is re-resolved on every call so a UT patch of
    that module attribute is still observed here.
    """
    from npubench.npubench_target import _verified_tilelang2ascendc_source_manifest

    state = _read_json(workspace / ".opgen_state.json", "durable state")
    return _verified_tilelang2ascendc_source_manifest(workspace, state)


def _staged_source_hashes(source_entries: Sequence[Any]) -> set[tuple[Any, Any]]:
    hashes: set[tuple[Any, Any]] = set()
    for entry in source_entries:
        if isinstance(entry, Mapping):
            hashes.add((entry.get("size"), entry.get("sha256")))
    return hashes


def _is_staged_kernel_path(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_TILELANG2ASCENDC_KERNEL_PREFIXES)


def _read_staged_normalised_text(staged_path: Path, relative: str) -> str | None:
    try:
        return _normalised_authored_text(staged_path, staged_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise TargetTransportError(
            f"cannot compare staged TileLang2AscendC provenance for {relative}"
        ) from exc


def _staged_kernel_normalised_text(
    source_root: Path, source_entries: Sequence[Any]
) -> dict[str, list[str]]:
    """Index the staged kernel sources by their comment/format-free text."""
    staged: dict[str, list[str]] = {}
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            continue
        relative = entry.get("path")
        if not _is_staged_kernel_path(relative):
            continue
        staged_path = source_root / str(relative)
        if staged_path.suffix.lower() not in _TILELANG2ASCENDC_KERNEL_SOURCE_SUFFIXES:
            continue
        if not staged_path.is_file():
            continue
        normalised = _read_staged_normalised_text(staged_path, str(relative))
        if normalised:
            staged.setdefault(normalised, []).append(str(relative))
    return staged


def _candidate_scan_text(path: Path, text: str) -> str:
    """Strip comments (and decoy literals) so the content gates see real code."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _python_code_without_comments_or_literals(text)
    if suffix in _TILELANG2ASCENDC_CXX_SUFFIXES:
        return _cxx_code_without_comments_or_literals(text)
    if path.name == "CMakeLists.txt" or suffix == ".cmake":
        return re.sub(r"#[^\n]*", "", text)
    return ""


def _reject_forbidden_candidate_text(
    relative: str, scan_text: str, content: bytes, source_root: Path
) -> None:
    if re.search(r"\baclnn[A-Za-z0-9_]*\b|\bacl_op(?!_compiler\b)[A-Za-z0-9_]*\b", scan_text, re.IGNORECASE):
        raise TargetTransportError(
            f"TileLang2AscendC candidate references forbidden ACLNN dispatcher/API text: {relative}"
        )
    if str(source_root).encode() in content or b".tilelang2ascendc_source" in content:
        raise TargetTransportError(
            f"TileLang2AscendC candidate references the read-only source stage: {relative}"
        )


def _reject_comment_only_kernel_change(
    relative: str, candidate_normalised: str | None, staged_kernel_normalised: Mapping[str, list[str]]
) -> None:
    """Reject a kernel file whose only delta from the stage is cosmetic."""
    if not candidate_normalised:
        return
    if not relative.startswith(_TILELANG2ASCENDC_KERNEL_PREFIXES):
        return
    if candidate_normalised not in staged_kernel_normalised:
        return
    matches = ", ".join(staged_kernel_normalised[candidate_normalised][:3])
    raise TargetTransportError(
        "TileLang2AscendC candidate only changes comments/formatting from "
        f"staged kernel source ({matches}): {relative}"
    )


def _classify_identical_candidate_file(
    relative: str, candidate_normalised: str | None, staged_kernel_normalised: Mapping[str, list[str]]
) -> str:
    if relative in TILELANG2ASCENDC_STABLE_AUTHORED_FILES:
        return "unchanged"
    _reject_comment_only_kernel_change(relative, candidate_normalised, staged_kernel_normalised)
    raise TargetTransportError(
        f"TileLang2AscendC candidate contains a byte-identical staged source file: {relative}"
    )


def _staged_text_matches(staged_path: Path, relative: str, candidate_normalised: str | None) -> bool:
    try:
        staged_text = staged_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TargetTransportError(
            f"cannot compare TileLang2AscendC candidate provenance for {relative}"
        ) from exc
    staged_normalised = _normalised_authored_text(staged_path, staged_text)
    return staged_normalised is not None and staged_normalised == candidate_normalised


def _read_candidate_bytes(path: Path, relative: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TargetTransportError(f"cannot read TileLang2AscendC candidate {relative}: {exc}") from exc


def _decode_candidate_bytes(content: bytes, relative: str) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TargetTransportError(
            f"TileLang2AscendC candidate {relative} is not readable UTF-8"
        ) from exc


def _classify_candidate_file(
    path: Path,
    relative: str,
    source_root: Path,
    source_hashes: set[tuple[Any, Any]],
    staged_kernel_normalised: Mapping[str, list[str]],
) -> str:
    """Return ``unchanged`` / ``changed_kernel`` / ``other`` for one file."""
    content = _read_candidate_bytes(path, relative)
    text = _decode_candidate_bytes(content, relative)
    scan_text = _candidate_scan_text(path, text)
    candidate_normalised = _normalised_authored_text(path, text)
    _reject_forbidden_candidate_text(relative, scan_text, content, source_root)
    if (len(content), hashlib.sha256(content).hexdigest()) in source_hashes:
        return _classify_identical_candidate_file(
            relative, candidate_normalised, staged_kernel_normalised
        )
    staged_path = source_root / relative
    if relative in TILELANG2ASCENDC_STABLE_AUTHORED_FILES and staged_path.is_file():
        if _staged_text_matches(staged_path, relative, candidate_normalised):
            return "unchanged"
    _reject_comment_only_kernel_change(relative, candidate_normalised, staged_kernel_normalised)
    if relative.startswith(_TILELANG2ASCENDC_KERNEL_PREFIXES):
        return "changed_kernel"
    return "other"


def _validate_tilelang2ascendc_candidate_for_build(
    workspace: Path,
    source_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate the authored TileLang2AscendC candidate boundary before CANN."""
    root = Path(workspace)
    _require_candidate_entry_files(root)
    if source_manifest is None:
        source_manifest = _load_tilelang2ascendc_source_manifest(root)
    source_entries = source_manifest.get("files")
    if not isinstance(source_entries, list):
        raise TargetTransportError("TileLang2AscendC source manifest has no file inventory")
    source_hashes = _staged_source_hashes(source_entries)
    resolved_root = root.resolve()
    source_root = resolved_root / ".tilelang2ascendc_source"
    candidate_files = _candidate_delivery_files(workspace)
    staged_kernel_normalised = _staged_kernel_normalised_text(source_root, source_entries)
    unchanged_files: list[str] = []
    changed_kernel_files: list[str] = []
    for path in candidate_files:
        relative = path.relative_to(resolved_root).as_posix()
        outcome = _classify_candidate_file(
            path, relative, source_root, source_hashes, staged_kernel_normalised
        )
        if outcome == "unchanged":
            unchanged_files.append(relative)
        elif outcome == "changed_kernel":
            changed_kernel_files.append(relative)
    if not changed_kernel_files:
        raise TargetTransportError(
            "TileLang2AscendC candidate must author at least one changed kernel/op_host or "
            "kernel/op_kernel source file"
        )
    proof = dict(_validate_tilelang2ascendc_kernel_boundary(workspace, candidate_files))
    proof["unchanged_stable_files"] = sorted(unchanged_files)
    proof["changed_kernel_files"] = sorted(changed_kernel_files)
    return proof


def _authored_cmake_sha256(workspace: Path) -> str:
    path = Path(workspace) / "kernel" / "CMakeLists.txt"
    if path.is_symlink() or not path.is_file():
        raise TargetTransportError(
            f"authored CMakeLists.txt is missing or not a regular file: {path}"
        )
    try:
        return _sha(path)
    except OSError as exc:
        raise TargetTransportError(f"cannot hash authored CMakeLists.txt: {exc}") from exc


def _candidate_source_digest(workspace: Path) -> str:
    """Hash only the authored candidate delivery, never state or source trees."""
    root = Path(workspace).resolve()
    entries: list[tuple[str, str]] = []
    entry = root / "model_new_ascendc.py"
    kernel = root / "kernel"
    if entry.is_symlink() or not entry.is_file():
        raise TargetTransportError("candidate model_new_ascendc.py is missing or not a regular file")
    if kernel.is_symlink() or not kernel.is_dir():
        raise TargetTransportError("candidate kernel directory is missing or unsafe")
    for path in (entry, *sorted(kernel.rglob("*"))):
        if path.is_symlink():
            raise TargetTransportError(f"candidate contains a symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "build" in relative.parts or path.suffix.lower() in {".so", ".o", ".a", ".pyc"}:
            continue
        entries.append((relative.as_posix(), _sha(path)))
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

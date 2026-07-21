# Copyright 2025 Huawei Technologies Co., Ltd
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
CodeChecker: 代码检查器

纯静态检查流程（不调用 LLM）：
1. ast.parse 语法检查
2. py_compile 编译检查
3. import 可用性检查
4. 中文文本混入检测
5. DSL/arch 合规性检测（反作弊：每个 DSL 各一个 _ComplianceCheck 类，
   各自独立 owns 自己的策略字段，``CodeChecker.__init__`` 不再感知任何
   单 DSL 的策略 schema）
"""

import ast
import importlib.resources
import importlib.util
import logging
import os
import py_compile
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy: single source of truth is op_autoresearch/op/config/code_checker.yaml.
# Loaded once at import; missing/malformed keys surface as KeyError / TypeError
# on first access (no redundant validation layer).
# ---------------------------------------------------------------------------

with importlib.resources.files("op_autoresearch.op.config").joinpath(
    "code_checker.yaml"
).open("r", encoding="utf-8") as _f:
    _POLICY = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Module-level constants derived from _POLICY. These are SHARED across
# multiple compliance checks (Triton and CATLASS use them when scanning
# forward() for forbidden torch ops).
# Storing them once at module scope avoids each Check class re-loading them
# and keeps the per-Check __init__ focused on Check-specific literals.
# ---------------------------------------------------------------------------

_STRAY_TEXT_RE = re.compile(
    "[" + "".join(
        f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in _POLICY["stray_text"]["unicode_ranges"]
    ) + "]{" + str(_POLICY["stray_text"]["min_run"]) + ",}"
)

_TRITON_MODULE_NAME: str = _POLICY["triton_module_name"]
_TRITON_DECORATORS: frozenset = frozenset(_POLICY["triton_decorators"])
_TORCH_COMPUTE_OPS_HARD: frozenset = frozenset(_POLICY["torch_compute_ops_hard"])
_TORCH_COMPUTE_OPS_SOFT: frozenset = frozenset(_POLICY["torch_compute_ops_soft"])
_TORCH_CALL_PREFIXES: frozenset = frozenset(_POLICY["torch_call_prefixes"])
_TORCH_CALL_PREFIXES_ORDERED: tuple = tuple(
    sorted(_TORCH_CALL_PREFIXES, key=len, reverse=True)
)
_DSL_COMPLIANCE_PREFIXES: tuple = tuple(_POLICY["dsl_compliance_prefixes"])
_ASCENDC_TEXT_SUFFIXES: frozenset = frozenset(
    _POLICY["ascendc_anti_cheat"]["text_suffixes"]
)
_ASCENDC_TEXT_FILENAMES: frozenset = frozenset(
    _POLICY["ascendc_anti_cheat"]["text_filenames"]
)


# ---------------------------------------------------------------------------
# Free helpers — AST navigation + shared decorator/prefix matchers.
# ---------------------------------------------------------------------------

def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    target = _POLICY["kernel_class_name"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == target:
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    target = _POLICY["kernel_forward_method"]
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target:
            return item
    return None


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def _collect_import_aliases(tree: ast.Module) -> Dict[str, str]:
    """Build a map of local-name → dotted-module-name from import statements.

    Recognizes bare-name decorators like ``@jit`` (from ``from triton import jit``)
    by resolving the alias back to its origin module. Only collects aliases that
    resolve to the Triton namespace — unrelated ``@jit`` from other
    libraries won't be misclassified.
    """
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        aliases.update(_triton_aliases_from_import(node))
    return aliases


def _triton_aliases_from_import(node: ast.AST) -> Dict[str, str]:
    if isinstance(node, ast.Import):
        return {
            alias.asname or alias.name: alias.name
            for alias in node.names
            if alias.name.split(".")[0] == _TRITON_MODULE_NAME
        }
    if not isinstance(node, ast.ImportFrom) or not node.module:
        return {}
    if node.module.split(".")[0] != _TRITON_MODULE_NAME:
        return {}
    return {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for alias in node.names
    }


def _is_triton_decorator(node: ast.expr,
                         import_aliases: Optional[Dict[str, str]] = None) -> bool:
    """True for ``@triton.jit`` / ``@triton.<dec>`` / ``@jit`` (when ``from
    triton import jit``). Handles bare name, dotted attribute, and called
    decorator forms.
    """
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func, import_aliases)
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == _TRITON_MODULE_NAME
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Name) and import_aliases:
        resolved = import_aliases.get(node.id, "")
        parts = resolved.rsplit(".", 1)
        if (len(parts) == 2 and parts[0] == _TRITON_MODULE_NAME
                and parts[1] in _TRITON_DECORATORS):
            return True
    return False


def _match_torch_call_prefix(call_name: str) -> Optional[str]:
    """Return the longest matching torch-namespace prefix for ``call_name``,
    or None. Longer prefixes win (``torch.nn.functional`` before ``torch``).
    """
    for prefix in _TORCH_CALL_PREFIXES_ORDERED:
        if call_name.startswith(f"{prefix}."):
            return prefix
    return None


def _fmt_calls(calls: List[tuple], limit: int = 5) -> str:
    """Render ``[(line, name), ...]`` as ``name(第line行), ... 等（共 N 处）``."""
    summary = ", ".join(f"{name}(第{line}行)" for line, name in calls[:limit])
    if len(calls) > limit:
        summary += f" 等（共 {len(calls)} 处）"
    return summary


@dataclass
class CheckError:
    """检查错误信息"""
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str
    fix_strategy: str = "fix"  # "fix" 或 "rewrite"


# ===========================================================================
# Compliance checks — each Check class owns its own policy state. Adding a
# new DSL anti-cheat = new ``_<dsl>ComplianceCheck`` subclass + 1 line in
# ``CodeChecker._CHECKS``. ``CodeChecker.__init__`` does NOT know any DSL.
# ===========================================================================


@dataclass(frozen=True)
class _ComputePolicy:
    hard_etype: str
    kernel_label: str
    soft_etype: Optional[str] = None
    skip_prefix: Optional[str] = None


def _forbidden_compute_in_forward(
    forward_node, policy: _ComputePolicy, kernel_present: bool = True
) -> List[Dict]:
    """Reject framework compute that replaces a selected DSL kernel."""
    hard_calls: List[tuple] = []
    soft_calls: List[tuple] = []
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_name = _dotted_name(node.func)
            if (
                policy.skip_prefix
                and call_name
                and call_name.startswith(policy.skip_prefix)
            ):
                continue
            if not call_name or not _match_torch_call_prefix(call_name):
                continue
            method = node.func.attr
            if method in _TORCH_COMPUTE_OPS_HARD:
                hard_calls.append((node.lineno, call_name))
            elif policy.soft_etype and method in _TORCH_COMPUTE_OPS_SOFT:
                soft_calls.append((node.lineno, call_name))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            hard_calls.append((node.lineno, "@ (matmul operator)"))

    if hard_calls:
        return [_hard_compute_error(hard_calls, policy)]
    if not policy.soft_etype or not soft_calls:
        return []
    if not kernel_present:
        return [_soft_compute_error(soft_calls, policy)]
    logger.warning(
        "CodeChecker compliance: forward() calls %s and also contains %s "
        "torch helper calls: %s",
        policy.kernel_label,
        len(soft_calls),
        _fmt_calls(soft_calls),
    )
    return []


def _hard_compute_error(calls: List[tuple], policy: _ComputePolicy) -> Dict:
    return _err(
        calls[0][0],
        policy.hard_etype,
        f"forward() contains {len(calls)} forbidden torch compute calls: "
        f"{_fmt_calls(calls)}. Core compute must be implemented in "
        f"{policy.kernel_label}.",
        f"Move core compute into {policy.kernel_label}; keep forward() for "
        "input setup, kernel launch, and output assembly.",
    )


def _soft_compute_error(calls: List[tuple], policy: _ComputePolicy) -> Dict:
    return _err(
        calls[0][0],
        policy.soft_etype or "torch_api_without_kernel",
        f"forward() contains {len(calls)} torch compute calls without launching "
        f"{policy.kernel_label}: {_fmt_calls(calls)}.",
        "Implement core compute in the kernel; retain helper operations only "
        "around a real kernel launch.",
    )


class _ComplianceCheck:
    """Base for per-DSL/per-arch static checks. Subclasses load their own
    policy literals in ``__init__`` (from ``_POLICY``), declare when to
    fire via ``applies(checker)``, and produce error dicts in ``run(code,
    checker)``. State is shared across all CodeChecker instances (the
    policy is module-immutable; checks have no per-instance mutable state).
    """

    name: str = ""

    @staticmethod
    def applies(_checker: "CodeChecker") -> bool:
        return True

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        raise NotImplementedError


def _err(line: int, error_type: str, detail: str, suggestion: str,
         *, snippet: str = "") -> Dict:
    """Build a CodeChecker error dict — the shape shared by every check."""
    return {"line": line, "error_type": error_type, "detail": detail,
            "suggestion": suggestion, "code_snippet": snippet,
            "fix_strategy": "rewrite"}


def _decorated_functions(tree: ast.Module, predicate, aliases) -> set:
    """Names of functions carrying a decorator matched by ``predicate``."""
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(predicate(dec, aliases) for dec in node.decorator_list):
                out.add(node.name)
    return out


class _KernelDSLComplianceCheck(_ComplianceCheck):
    """Shared skeleton for AST-based kernel anti-cheat checks: a kernel must
    be DEFINED and USED, and ModelNew.forward()
    must not delegate core compute to a torch high-level API.

    A subclass supplies only DATA + two small strategy hooks: which dsl it matches
    (``dsl_prefix`` startswith, or ``dsl_exact``), how to spot kernels
    (:meth:`_find_kernels`) and their use (:meth:`_find_used`), the error-type
    names + messages, and the forbidden-compute etypes. The run() flow — parse →
    'no kernel' → 'kernel not called' → the shared ``_forbidden_compute_in_forward``
    scan — lives here once, so a new AST DSL is ~15 lines.
    """

    dsl_prefix: Optional[str] = None      # startswith match (Triton)
    dsl_exact: Optional[str] = None       # exact match (CATLASS)
    no_kernel_etype: str = ""
    not_called_etype: str = ""            # "" → skip the 'not called' stage
    hard_etype: str = ""
    soft_etype: Optional[str] = None
    kernel_label: str = ""
    skip_prefix: Optional[str] = None
    # (detail_template, suggestion). detail may reference {dsl} / {kernels}.
    no_kernel_msg: Tuple[str, str] = ("", "")
    not_called_msg: Tuple[str, str] = ("", "")

    def applies(self, checker: "CodeChecker") -> bool:
        if self.dsl_prefix is not None:
            return checker.dsl.startswith(self.dsl_prefix)
        return checker.dsl == self.dsl_exact

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        aliases = _collect_import_aliases(tree)

        kernels = self._find_kernels(tree, aliases)
        if not kernels:
            detail, suggestion = self.no_kernel_msg
            return [_err(0, self.no_kernel_etype,
                         detail.format(dsl=checker.dsl), suggestion)]

        errors: List[Dict] = []
        used = self._find_used(tree, kernels) if self.not_called_etype else kernels
        if self.not_called_etype and not used:
            detail, suggestion = self.not_called_msg
            errors.append(_err(0, self.not_called_etype,
                               detail.format(kernels=sorted(kernels)), suggestion))

        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return errors
        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return errors
        policy = _ComputePolicy(
            hard_etype=self.hard_etype,
            soft_etype=self.soft_etype,
            kernel_label=self.kernel_label,
            skip_prefix=self.skip_prefix,
        )
        errors.extend(
            _forbidden_compute_in_forward(forward_node, policy, bool(used))
        )
        return errors

    def _find_kernels(self, tree: ast.Module, aliases: Dict[str, str]) -> set:
        """Return the set of kernel-bearing names (empty → 'no kernel')."""
        raise NotImplementedError

    def _find_used(self, tree: ast.Module, kernels: set) -> set:
        """Return the subset of ``kernels`` proven to be launched/called."""
        raise NotImplementedError


class _TritonComplianceCheck(_KernelDSLComplianceCheck):
    """triton_ascend: a ``@triton.jit`` kernel must be
    defined AND launched via ``kernel[grid](...)``; forward() no hard torch API.
    """

    name = "triton_compliance"
    dsl_prefix = "triton"
    no_kernel_etype = "no_triton_kernel"
    not_called_etype = "triton_kernel_not_called"
    hard_etype = "torch_api_instead_of_kernel"
    soft_etype = "torch_api_without_kernel"
    kernel_label = "triton kernel"
    no_kernel_msg = (
        "DSL 指定为 {dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
        "代码可能使用了 torch 高层 API 替代 triton kernel 实现。",
        "请确保代码中包含至少一个 @triton.jit 装饰的 kernel 函数，"
        "并在 ModelNew.forward() 中通过 kernel[grid](...) 语法调用它。")
    not_called_msg = (
        "定义了 triton kernel 函数 {kernels}，但代码中未找到任何 kernel[grid](...) "
        "形式的调用。kernel 函数可能只是装饰性的，实际计算未使用 triton。",
        "请在 ModelNew.forward() 或其辅助方法中，"
        "通过 kernel_name[grid_size](...) 语法启动 triton kernel。")

    def _find_kernels(self, tree, aliases):
        return _decorated_functions(tree, _is_triton_decorator, aliases)

    def _find_used(self, tree, kernels):
        # Count launches in helpers as well as forward().
        used: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in kernels:
                    used.add(value.id)
        return used


class _CatlassComplianceCheck(_ComplianceCheck):
    """ascendc_catlass 反作弊：ModelNew 中必须出现 torch.ops.catlass.xxx
    调用，forward() 不允许 torch 高层硬算子（除合法的 catlass 调用本身）。
    """

    name = "catlass_compliance"

    def __init__(self):
        _c = _POLICY["catlass_compliance"]
        self._dsl: str = _c["dsl"]
        self._enabled: bool = bool(_c["enable_catlass_call_check"])
        self._call_ns: str = _c["call_namespace"]
        self._call_prefix: str = self._call_ns + "."

    def applies(self, checker: "CodeChecker") -> bool:
        return checker.dsl == self._dsl and self._enabled

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return []

        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return []

        errors: List[Dict] = []

        # --- A. forward() must call torch.ops.catlass.xxx ---
        has_catlass_call = False
        for node in ast.walk(forward_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                call_name = _dotted_name(node.func)
                if call_name and call_name.startswith(self._call_prefix):
                    has_catlass_call = True
                    break

        if not has_catlass_call:
            errors.append(_err(
                0, "no_catlass_call",
                f"DSL 指定为 {checker.dsl}，但 ModelNew.forward() 中未找到任何 "
                f"{self._call_ns}.xxx 形式的调用。"
                f"代码可能使用了 torch 高层 API 替代 catlass kernel 实现。",
                "请确保 forward() 中通过 torch.ops.catlass.<op_name>(...) "
                "调用 catlass kernel，而非直接使用 torch 高层计算 API。"))

        policy = _ComputePolicy(
            hard_etype="torch_api_instead_of_kernel",
            soft_etype="torch_api_without_kernel",
            kernel_label="catlass kernel",
            skip_prefix=self._call_prefix,
        )
        errors.extend(
            _forbidden_compute_in_forward(forward_node, policy, has_catlass_call)
        )
        return errors


# AscendC anti-cheat messages — kept here (not the policy YAML) to match the
# other DSL compliance checks, which build their detail/suggestion in code.
# Keyed by the pattern ``name`` in code_checker.yaml's ``forbidden_patterns``.
# Only the two BLOCKING patterns that bypass ATen dispatch entirely (raw ACL /
# torch_npu builtins). Everything reaching dispatch is disabled at runtime by the
# compute gate (runtime_guard/), not by source spelling.
_ASCENDC_PATTERN_MESSAGES: dict = {
    'torch_npu_builtin_compute': (
        'AscendC wrapper 调用了 torch_npu.npu_* 内置算子，等价于委托给现成 NPU op。',
        '改为 torch.ops.npu.<custom_op>(...) 调用自定义 direct-invoke 算子，核心计算写在 ascendc_op 中。'),
    'aclnn_builtin_compute': (
        'AscendC host extension 调用了 aclnn 内置高层计算 API。',
        '不要用 aclnn 内置算子代替自定义 AscendC kernel。'),
}


class _SourcePatternComplianceCheck(_ComplianceCheck):
    """Shared skeleton for the TEXT/regex source-scan anti-cheat — for calls that
    bypass ATen dispatch (raw ``aclnn*`` ACL API / ``torch_npu.npu_*`` builtins)
    which neither an AST check nor the runtime compute gate can see, so they must
    be caught in the source text. Regex- not AST-based, because it also scans
    ``.cpp/.h/.asc/CMake`` (non-Python) files.

    Comments and string literals are stripped first, so a forbidden name that
    appears only in a comment/string (``// avoid aclnnMatmul(...)``) is not a
    false hit; only a genuine call site matches. A subclass supplies: which dsl it
    matches (``dsl_set``), which file suffixes/names to scan, the error-type
    prefix, and the ``(name, compiled_regex, detail, suggestion)`` patterns.
    """

    dsl_set: frozenset = frozenset()
    text_suffixes: frozenset = frozenset()
    text_filenames: frozenset = frozenset()
    etype_prefix: str = ""
    patterns: tuple = ()

    @staticmethod
    def _scan_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return not stripped.startswith(("#", "//", "/*", "*", "*/"))

    @staticmethod
    def _sanitize_line(line: str) -> str:
        """Strip string/char literals and comments from a line before the scan.
        Single-pass, string-state aware so a ``//`` inside a string is not a
        comment and a quote inside a comment does not open a string; handles
        ``#``/``//`` line comments + single-line ``/* ... */`` blocks. Spacing is
        left intact so spaced-dodge patterns still match real calls.
        """
        out: List[str] = []
        i = 0
        n = len(line)
        quote = None
        while i < n:
            c = line[i]
            if quote is not None:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
                continue
            if c in ("\"", "'"):
                quote = c
                i += 1
                continue
            if c == "#":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                end = line.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def applies(self, checker: "CodeChecker") -> bool:
        return checker.dsl in self.dsl_set

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        if not self._should_scan_file(checker):
            return []
        errors: List[Dict] = []
        for line_no, line in enumerate(code.splitlines(), 1):
            if not self._scan_line(line):
                continue
            scan_target = self._sanitize_line(line)
            if not scan_target.strip():
                continue
            for name, pattern, detail, suggestion in self.patterns:
                if pattern.search(scan_target):
                    errors.append(
                        _err(
                            line_no,
                            f"{self.etype_prefix}{name}",
                            detail,
                            suggestion,
                            snippet=line.rstrip(),
                        )
                    )
        return errors

    def _should_scan_file(self, checker: "CodeChecker") -> bool:
        source_path = getattr(checker, "_source_path", "") or ""
        if not source_path:
            return True
        name = os.path.basename(source_path)
        suffix = os.path.splitext(name)[1]
        return suffix in self.text_suffixes or name in self.text_filenames


class _AscendCComplianceCheck(_SourcePatternComplianceCheck):
    """AscendC-family (ascendc + ascendc_catlass) source-scan for calls that
    bypass ATen dispatch (raw ``aclnn*`` / ``torch_npu.npu_*``). Both are
    directory-backed C++ and share the same 'no raw stock-kernel' rule, so one
    scan covers both. Everything reaching dispatch (Python / C++ ``torch::*`` /
    ``at::*`` nested in the candidate's own custom op) is disabled at runtime by
    the compute gate (runtime_guard/) — both adapters emit ``guarded_call``, so
    ascendc and catlass are now fully symmetric. Pattern messages live in
    ``_ASCENDC_PATTERN_MESSAGES`` above.
    """

    name = "ascendc_compliance"
    etype_prefix = "ascendc_anti_cheat_"

    def __init__(self):
        _a = _POLICY["ascendc_anti_cheat"]
        # Cover both AscendC-family DSLs so catlass's .cpp is scanned too (closes
        # the earlier asymmetry where catlass raw-aclnn was neither statically
        # scanned nor runtime-gated).
        self.dsl_set = frozenset({_a["dsl"], _POLICY["catlass_compliance"]["dsl"]})
        self.text_suffixes = _ASCENDC_TEXT_SUFFIXES
        self.text_filenames = _ASCENDC_TEXT_FILENAMES
        self.patterns = tuple(
            (item["name"], re.compile(item["pattern"]),
             *_ASCENDC_PATTERN_MESSAGES[item["name"]])
            for item in _a["forbidden_patterns"]
        )


class _AutotuneComplianceCheck(_ComplianceCheck):
    """triton 系列：``@triton.autotune`` 装饰器必须包含 ``restore_value``
    参数（否则 benchmark 重跑会跨 config 污染输出）。
    """

    name = "autotune"

    def __init__(self):
        _a = _POLICY["autotune"]
        self._autotune_re = re.compile(
            rf"@{re.escape(_TRITON_MODULE_NAME)}\."
            rf"{re.escape(_a['decorator_attr'])}\s*\(",
            re.MULTILINE,
        )
        self._restore_value_re = re.compile(
            rf"{re.escape(_a['required_kwarg'])}\s*="
        )

    def applies(self, checker: "CodeChecker") -> bool:
        # ``@triton.autotune`` 只适用于 Triton DSL。
        return checker.dsl.startswith("triton")

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        errors: List[Dict] = []

        autotune_match = self._autotune_re.search(code)
        if not autotune_match:
            return errors

        autotune_line = code[:autotune_match.start()].count('\n') + 1

        paren_depth = 0
        start = autotune_match.end() - 1
        end = start
        for i in range(start, len(code)):
            if code[i] == '(':
                paren_depth += 1
            elif code[i] == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    end = i + 1
                    break
        autotune_block = code[start:end]

        if not self._restore_value_re.search(autotune_block):
            errors.append({
                "line": autotune_line,
                "error_type": "autotune_missing_restore_value",
                "detail": (
                    "@triton.autotune 装饰器缺少 restore_value 参数。"
                    "autotune benchmark 会对每个 config 反复执行 kernel，"
                    "不同 config 之间的输出会互相污染，导致验证失败。"
                ),
                "suggestion": (
                    "在 @triton.autotune(...) 中添加 restore_value=['输出指针参数名']，"
                    "列出 kernel 的所有输出指针参数。例如：\n"
                    "  @triton.autotune(\n"
                    "      configs=[...],\n"
                    "      key=[...],\n"
                    "      restore_value=['output_ptr'],  # 必须添加\n"
                    "  )"
                ),
                "code_snippet": "",
                "fix_strategy": "fix"
            })
            logger.warning(
                'CodeChecker: @triton.autotune at line %s missing restore_value', autotune_line
            )

        return errors


@dataclass(frozen=True)
class _AffinityUsage:
    has_scope: bool
    has_fixpipe: bool
    has_buffer_alloc: bool


def _find_triton_kernels(tree: ast.Module) -> List[ast.FunctionDef]:
    kernels = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_is_triton_decorator(decorator) for decorator in node.decorator_list):
            kernels.append(node)
    return kernels


def _affinity_api(func: ast.AST, al_alias: str, bl_alias: str,
                  only_apis: frozenset) -> Optional[str]:
    if not isinstance(func, ast.Attribute):
        return None
    if isinstance(func.value, ast.Name):
        if func.value.id == al_alias and func.attr in {"scope", "fixpipe"}:
            return func.attr
        if func.value.id == bl_alias and func.attr == "alloc":
            return "alloc"
        return None
    inner = func.value
    if not isinstance(inner, ast.Attribute) or not isinstance(inner.value, ast.Name):
        return None
    if inner.value.id == al_alias and func.attr in only_apis:
        return func.attr
    return None


def _collect_affinity_usage(kernels: List[ast.FunctionDef], al_alias: str,
                            bl_alias: str, only_apis: frozenset) -> _AffinityUsage:
    used = set()
    for kernel in kernels:
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call):
                continue
            api = _affinity_api(node.func, al_alias, bl_alias, only_apis)
            if api:
                used.add(api)
    return _AffinityUsage(
        has_scope="scope" in used,
        has_fixpipe="fixpipe" in used,
        has_buffer_alloc="alloc" in used,
    )


def _a5_affinity_errors(arch: str, usage: _AffinityUsage) -> List[Dict]:
    errors = []
    if not usage.has_scope:
        errors.append(_err(
            0, "a5_missing_scope",
            f"Target {arch} uses A5 hardware, but the kernel has no al.scope core partition.",
            "Partition Cube and Vector work with al.scope(core_mode=...).",
        ))
    if usage.has_scope and not usage.has_fixpipe:
        errors.append(_err(
            0, "a5_missing_fixpipe",
            f"Target {arch} uses al.scope but never transfers Cube results with al.fixpipe.",
            "Add al.fixpipe after Cube computation when moving L0C results to UB or L1.",
        ))
    if not usage.has_buffer_alloc:
        errors.append(_err(
            0, "a5_missing_bl_alloc",
            f"Target {arch} has no bl.alloc on-chip exchange buffer.",
            "Allocate the required UB/L1/L0 buffer with bl.alloc.",
        ))
    return errors


class _A5ComplianceCheck(_ComplianceCheck):
    """A5 (Ascend950) 硬件 + triton_ascend：含 tl.dot 的 kernel 必须
    使用 Cube/Vector 亲和接口 (al.scope / al.fixpipe / bl.alloc)。
    """

    name = "a5_compliance"

    def __init__(self):
        _a = _POLICY["a5_compliance"]
        self._arch_prefix: str = _a["arch_prefix"]
        self._dsl: str = _a["dsl"]
        self._enabled: bool = bool(_a["enable_triton_ascend_affinity_check"])
        self._al_alias: str = _a["aliases"]["al"]
        self._bl_alias: str = _a["aliases"]["bl"]
        self._only_apis: frozenset = frozenset(_a["only_apis"])

    @property
    def enabled(self) -> bool:
        """Whether A5 affinity enforcement is enabled by policy."""
        return self._enabled

    @staticmethod
    def _kernel_uses_tl_dot(kernel: ast.AST) -> bool:
        """Return whether a kernel contains ``tl.dot`` or its full name."""
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "dot":
                continue
            owner = func.value
            if isinstance(owner, ast.Name) and owner.id == "tl":
                return True
            if not isinstance(owner, ast.Attribute) or owner.attr != "language":
                continue
            if isinstance(owner.value, ast.Name) and owner.value.id == "triton":
                return True
        return False

    def applies(self, checker: "CodeChecker") -> bool:
        return (
            checker.arch.startswith(self._arch_prefix)
            and checker.dsl == self._dsl
        )

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:
        if not self._enabled:
            logger.info(
                "CodeChecker A5: arch=%s, dsl=%s; affinity enforcement disabled",
                checker.arch,
                checker.dsl,
            )
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        kernels = _find_triton_kernels(tree)
        if not kernels:
            return []
        if not any(self._kernel_uses_tl_dot(kernel) for kernel in kernels):
            logger.info(
                "CodeChecker A5: arch=%s, dsl=%s; no tl.dot, skipping affinity checks",
                checker.arch,
                checker.dsl,
            )
            return []
        usage = _collect_affinity_usage(
            kernels, self._al_alias, self._bl_alias, self._only_apis
        )
        return _a5_affinity_errors(checker.arch, usage)


# ===========================================================================
# CodeChecker class
# ===========================================================================


def _absolute_imports(tree: ast.Module) -> List[tuple[int, str]]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _import_error(line: int, module_name: str) -> Dict:
    error = _err(
        line,
        "import_error",
        f"Module '{module_name}' is unavailable in the current environment.",
        f"Check the spelling of '{module_name}' or declare it as a worker-owned runtime module.",
    )
    error["fix_strategy"] = "fix"
    return error


class CodeChecker:
    """
    代码检查器：在 Coder 生成代码后、Verifier 验证前，进行快速的纯静态检查

    检查流程：ast.parse → py_compile → import 验证 → 中文文本混入检测
    → DSL/arch 合规性检测。不调用 LLM，零额外成本。

    新增 per-DSL 合规检查的方法：实现一个 ``_<dsl>ComplianceCheck`` 子类
    （定义 ``applies(checker) / run(code, checker)``），在 ``_CHECKS`` 列
    表里追加一行；不要在 ``CodeChecker`` 类体内添加 per-DSL 字段或方法。
    """

    # Class-level singleton instances of each compliance check. State is
    # immutable (yaml policy frozensets / compiled regex) so sharing
    # across CodeChecker instances is safe.
    _triton_check = _TritonComplianceCheck()
    _catlass_check = _CatlassComplianceCheck()
    _ascendc_check = _AscendCComplianceCheck()
    _autotune_check = _AutotuneComplianceCheck()
    _a5_check = _A5ComplianceCheck()

    # All compliance checks. Iteration order shows up in error rendering.
    _CHECKS: list = [
        _triton_check, _catlass_check,
        _ascendc_check, _autotune_check, _a5_check,
    ]
    # Subset exposed via the ``_check_dsl_compliance`` public method
    # (called by autoresearch agent tools). Excludes autotune + A5,
    # which have their own dimensions (DSL prefix / arch+flag).
    _DSL_COMPLIANCE_CHECKS: list = [
        _triton_check, _catlass_check,
        _ascendc_check,
    ]

    # ------------------------------------------------------------------
    # Step 3: import 可用性检查
    # ------------------------------------------------------------------

    # Runtime modules that live on the eval target (NPU host), NOT on
    # the orchestrator that runs CodeChecker. Skip the find_spec gate
    # for them — a Windows / no-NPU orchestrator legitimately doesn't
    # have torch_npu / triton_ascend / etc. installed, and the kernel
    # is verified end-to-end by the remote worker anyway. Real typos
    # in user code surface there with a clear ImportError, not as a
    # silent reject here.
    _REMOTE_RUNTIME_MODULES = frozenset({
        "torch",
        "torch_npu",
        "triton_ascend",
        "tbe",
        "te",
        "acl",
        "aclnnop",
    })

    # Some DSLs import a generic frontend package whose runtime is nevertheless
    # backend-specific. Triton Ascend kernels spell their
    # imports as ``triton`` / ``triton.language`` even though that package only
    # exists in the remote Ascend environment.  Keep this policy keyed by DSL:
    # requiring it locally would make a Windows orchestrator reject kernels
    # that the remote worker can execute.
    _REMOTE_RUNTIME_MODULES_BY_DSL = {
        "triton_ascend": frozenset({"triton"}),
    }

    def __init__(self, backend: str, dsl: str, arch: str = "", config: Optional[dict] = None):
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.arch = arch.lower() if arch else ""
        # ``config`` accepted for caller-signature compat; policy 真源 is yaml.
        self.config = config or {}
        self._source_path = ""
        logger.info(
            'CodeChecker initialized: backend=%s, dsl=%s, arch=%s', self.backend, self.dsl, self.arch
        )

    # ------------------------------------------------------------------
    # Compat surface — autoresearch agent tools + tests expect these
    # names. They are *not* per-instance state; the values are pinned
    # by op/config/code_checker.yaml at module load.
    # ------------------------------------------------------------------

    @property
    def triton_decorators(self) -> frozenset:
        return _TRITON_DECORATORS

    @property
    def torch_compute_ops_hard(self) -> frozenset:
        return _TORCH_COMPUTE_OPS_HARD

    @property
    def torch_compute_ops_soft(self) -> frozenset:
        return _TORCH_COMPUTE_OPS_SOFT

    @property
    def torch_call_prefixes(self) -> frozenset:
        return _TORCH_CALL_PREFIXES

    @staticmethod
    def get_check_summary(errors: List[Dict]) -> str:
        """获取检查摘要（简短版本，用于日志）"""
        if not errors:
            return "代码检查通过"
        error_types = set(
            error.get("error_type", "unknown") for error in errors
        )
        return f"发现 {len(errors)} 个问题: {', '.join(error_types)}"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    @staticmethod
    def _is_python_source(source_name: str) -> bool:
        """Whether this handoff file runs the Python-only steps (ast / compile /
        import / stray-text). A known non-.py text source — the C++/AscendC/
        CMake files of a directory-backed DSL — skips them and goes straight to
        the DSL compliance scan. Empty / unknown path defaults to Python.
        """
        if not source_name:
            return True
        suffix = os.path.splitext(source_name)[1]
        if suffix == ".py":
            return True
        return not (suffix in _ASCENDC_TEXT_SUFFIXES
                    or source_name in _ASCENDC_TEXT_FILENAMES)

    # ------------------------------------------------------------------
    # Step 1: ast.parse 语法检查
    # ------------------------------------------------------------------

    @staticmethod
    def _check_python_syntax(code: str) -> List[Dict]:
        """
        使用 ast.parse() 进行语法检查：
        括号不匹配、缩进错误、关键字拼写等。

        注意：ast.parse 遇到第一个 SyntaxError 就会停止，
        因此这里只返回首个错误，后续可能还有其他问题需要在修复后再次检查。
        """
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_num = e.lineno or 0
            code_lines = code.split('\n')
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()

            error_msg = e.msg or "语法错误"
            if e.offset:
                error_msg += f"（第 {e.offset} 列）"

            errors.append({
                "line": line_num,
                "error_type": "syntax_error",
                "detail": f"Python 语法错误: {error_msg}",
                "suggestion": f"""请检查第 {line_num} 行的语法：
  - 检查括号、引号是否匹配
  - 检查缩进是否正确
  - 检查关键字拼写是否正确
  - 检查冒号、逗号等符号是否遗漏""",
                "code_snippet": code_snippet,
                "fix_strategy": "fix"
            })
            logger.warning('CodeChecker: Python syntax error at line %s: %s', line_num, error_msg)

        return errors

    # ------------------------------------------------------------------
    # Step 2: py_compile 编译检查
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_compile_temp_files(paths) -> None:
        for path in paths:
            if not path:
                continue
            try:
                os.unlink(path)
            except OSError as exc:
                logger.debug(
                    "failed to remove compile-check temp file %s: %s", path, exc
                )

    @staticmethod
    def _check_py_compile(code: str) -> List[Dict]:
        """
        使用 py_compile 进行编译级别检查。
        比 ast.parse 更严格，能捕获部分 ast.parse 遗漏的编译问题
        （如 SyntaxWarning 升级、重复关键字参数等）。
        """
        errors = []
        tmp_src = None
        tmp_pyc = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                tmp_src = f.name

            # 临时文件写入系统临时目录（Linux: /tmp, Windows: %TEMP%），不在当前工作目录。
            # 用独立的临时文件接收 .pyc 输出，避免往 __pycache__ 写入导致权限问题。
            fd, tmp_pyc = tempfile.mkstemp(suffix='.pyc')
            os.close(fd)

            py_compile.compile(tmp_src, cfile=tmp_pyc, doraise=True)
        except py_compile.PyCompileError as e:
            line_num = 0
            error_str = str(e)
            match = re.search(r'line (\d+)', error_str)
            if match:
                line_num = int(match.group(1))

            code_lines = code.split('\n')
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()

            errors.append({
                "line": line_num,
                "error_type": "compile_error",
                "detail": f"Python 编译错误: {error_str}",
                "suggestion": f"""请检查第 {line_num} 行附近的代码：
  - 检查是否有不合法的表达式或语法结构
  - 检查变量名、函数名是否合法
  - 检查是否有 Python 版本不兼容的写法""",
                "code_snippet": code_snippet,
                "fix_strategy": "fix"
            })
            logger.warning('CodeChecker: py_compile error at line %s: %s', line_num, error_str)
        except Exception as e:
            logger.warning('CodeChecker: py_compile check failed unexpectedly: %s', e)
        finally:
            CodeChecker._remove_compile_temp_files((tmp_src, tmp_pyc))

        return errors

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        """检查模块在当前环境中是否可用"""
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Step 4: 中文文本混入检测 —— regex 来自 op/config/code_checker.yaml
    # 的 stray_text.min_run / stray_text.unicode_ranges
    # ------------------------------------------------------------------

    @staticmethod
    def _check_stray_chinese(code: str) -> List[Dict]:
        """
        检测代码中混入的中文文本（LLM 常见问题）。

        规则：连续 >=3 个汉字出现在注释和字符串之外，视为误混入的中文描述。
        通过 tokenize 精确剥离注释和字符串，只扫描真正的代码 token。
        """
        import io
        import tokenize

        errors = []
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError):
            return errors

        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING):
                continue

            match = _STRAY_TEXT_RE.search(tok.string)
            if match:
                line_num = tok.start[0]
                chinese_text = match.group()
                errors.append({
                    "line": line_num,
                    "error_type": "stray_chinese_text",
                    "detail": f"代码中混入了中文文本 '{chinese_text}'，疑似未注释的中文描述",
                    "suggestion": (
                        f"第 {line_num} 行包含非代码的中文文本，请删除或改为注释（在行首加 #）。"
                        f"如果是有意使用的中文变量名，请忽略此警告。"
                    ),
                    "code_snippet": "",
                    "fix_strategy": "fix"
                })
                logger.warning(
                    "CodeChecker: stray Chinese text at line %s: '%s'", line_num, chinese_text
                )

        return errors

    # ------------------------------------------------------------------
    # 格式化输出
    # ------------------------------------------------------------------

    @staticmethod
    def _append_error_context(lines: List[str], code_lines: List[str],
                              error_line: int) -> None:
        start_line = max(1, error_line - 3)
        end_line = min(len(code_lines), error_line + 3)
        lines.append(f"  上下文（第 {start_line}-{end_line} 行）：")
        for context_line in range(start_line, end_line + 1):
            marker = ">>>" if context_line == error_line else "   "
            lines.append(
                f"  {marker} {context_line:4d} | {code_lines[context_line - 1]}"
            )

    @staticmethod
    def _format_errors(errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
        """格式化错误信息，便于传递给 Coder"""
        if not errors:
            return ""

        lines = [
            "## CodeChecker 静态检查报告",
            "",
            f"**发现 {len(errors)} 个问题，请修复后重新生成代码：**",
            ""
        ]

        for i, err in enumerate(errors, 1):
            error_line = err['line']
            lines.append(f"### 问题 {i}: 第 {error_line} 行 [{err.get('error_type', 'unknown')}]")
            lines.append(f"  {err['detail']}")

            if code_lines is not None and error_line > 0:
                CodeChecker._append_error_context(lines, code_lines, error_line)
            elif err.get('code_snippet'):
                lines.append(f"  出错代码: {err['code_snippet']}")

            if err.get('suggestion'):
                lines.append("  建议：")
                for sug_line in err['suggestion'].strip().split('\n'):
                    lines.append(f"    {sug_line}")

            lines.append("")

        lines.append("**注意：语法检查每次只能定位到首个错误，修复后可能还有后续问题，请仔细检查整段代码。**")

        return "\n".join(lines)

    @classmethod
    def a5_affinity_check_enabled(cls) -> bool:
        """Expose the configured A5 policy without leaking check internals."""
        return cls._a5_check.enabled

    @classmethod
    def worker_runtime_modules(cls, dsl: str = "") -> frozenset:
        """Return imports owned by the evaluation worker for one DSL."""
        normalized_dsl = (dsl or "").lower()
        return cls._REMOTE_RUNTIME_MODULES | cls._REMOTE_RUNTIME_MODULES_BY_DSL.get(
            normalized_dsl, frozenset()
        )

    def check(
        self, code: str, task_info: Optional[dict] = None
    ) -> Tuple[bool, str, List[Dict]]:
        """Run syntax and DSL compliance checks without invoking an LLM."""
        task_info = task_info or {}
        self._source_path = str(
            task_info.get("file") or task_info.get("path") or ""
        )
        if not code or not code.strip():
            return self._empty_code_result()
        if self._dsl_skips_static_checks():
            return True, "", []
        errors = self._python_source_errors(code)
        errors.extend(self._compliance_errors(code, errors))
        return self._check_result(code, errors)

    def _empty_code_result(self) -> Tuple[bool, str, List[Dict]]:
        logger.warning("CodeChecker: empty code provided")
        error = {
            "line": 0,
            "error_type": "empty_code",
            "detail": "代码为空，无法进行检查",
            "suggestion": "请生成有效的代码",
            "code_snippet": "",
            "fix_strategy": "rewrite",
        }
        return False, self._format_errors([error]), [error]

    def _dsl_skips_static_checks(self) -> bool:
        if not self.dsl:
            return False
        from op_autoresearch.op.verifier.adapters.factory import (
            get_dsl_adapter,
        )
        try:
            adapter = get_dsl_adapter(self.dsl)
        except ValueError:
            adapter = None
        if adapter is not None and adapter.static_check_via_python_ast:
            return False
        reason = "unknown DSL" if adapter is None else "not Python-based"
        logger.info(
            "CodeChecker: DSL '%s' is %s; skipping static Python checks",
            self.dsl,
            reason,
        )
        return True

    def _python_source_errors(self, code: str) -> List[Dict]:
        if not self._is_python_source(
            os.path.basename(self._source_path)
        ):
            return []
        errors = self._check_python_syntax(code)
        if not errors:
            errors.extend(self._check_py_compile(code))
        if not errors:
            errors.extend(self._check_imports(code))
        errors.extend(self._check_stray_chinese(code))
        return errors

    def _compliance_errors(
        self, code: str, existing_errors: List[Dict]
    ) -> List[Dict]:
        syntax_failed = any(
            error.get("error_type")
            in {"syntax_error", "compile_error"}
            for error in existing_errors
        )
        if syntax_failed:
            return []
        errors = []
        for check in self._CHECKS:
            if check.applies(self):
                errors.extend(check.run(code, self))
        return errors

    def _check_result(
        self, code: str, errors: List[Dict]
    ) -> Tuple[bool, str, List[Dict]]:
        message = (
            self._format_errors(errors, code.split("\n"))
            if errors
            else ""
        )
        if errors:
            logger.warning(
                "CodeChecker: found %s issue(s)", len(errors)
            )
            for error in errors:
                logger.warning(
                    "  Line %s: %s",
                    error["line"],
                    error["detail"],
                )
        else:
            logger.info("CodeChecker: all checks passed")
        return not errors, message, errors

    # ------------------------------------------------------------------
    # Public umbrella for autoresearch agent tools (op/autoresearch/
    # agent/tools.py). Runs ONLY the DSL anti-cheat subset (triton /
    # CATLASS) — excludes autotune and A5 which target other
    # dimensions.
    # ------------------------------------------------------------------

    def _check_dsl_compliance(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        for check in self._DSL_COMPLIANCE_CHECKS:
            if check.applies(self):
                errors.extend(check.run(code, self))
        return errors

    def _remote_runtime_modules(self) -> frozenset:
        """Modules whose availability is owned by the evaluation worker."""
        return self.worker_runtime_modules(self.dsl)

    def _check_imports(self, code: str) -> List[Dict]:
        """Report unavailable absolute imports, excluding worker-owned modules."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        checked = set()
        remote_modules = self._remote_runtime_modules()
        errors = []
        for line, module_name in _absolute_imports(tree):
            top_module = module_name.split(".")[0]
            if top_module in checked or top_module in remote_modules:
                continue
            checked.add(top_module)
            if self._is_module_available(top_module):
                continue
            errors.append(_import_error(line, module_name))
            logger.warning(
                "CodeChecker: import error at line %s: module '%s' not found",
                line,
                module_name,
            )
        return errors


# ---------------------------------------------------------------------------
# Back-compat module-level alias: ``op/agents/kernel_gen.py`` reads this
# at import time to pin its A5-affinity prompt branch. Defined after the
# class so it can resolve via ``CodeChecker._a5_check._enabled``.
# ---------------------------------------------------------------------------

_A5_ENABLE_AFFINITY_CHECK: bool = CodeChecker.a5_affinity_check_enabled()

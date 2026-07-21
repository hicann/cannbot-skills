# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test that code snippets in markdown files are syntactically valid.

- C++: tree-sitter-cpp for pure syntax parsing (no headers or SDK needed).
- Python: ast.parse() from stdlib.
- Bash/Shell: bash -n (syntax-only check).
"""

import ast
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import SKILLS_DIR

# Only check our own skills, not vendors
MD_FILES = sorted(SKILLS_DIR.rglob("*.md"))

# Regex to extract fenced code blocks tagged as cpp/c++
CPP_BLOCK_RE = re.compile(
    r"```(?:cpp|c\+\+)\s*\n(.*?)```",
    re.DOTALL,
)

# Minimum snippet length worth parsing (skip tiny fragments)
MIN_SNIPPET_LEN = 30

# Absolute path to bash, resolved once to avoid PATH-based resolution.
BASH_BIN = shutil.which("bash") or "/bin/bash"


def _collect_cpp_snippets():
    """Yield (file_path, block_index, code) for all C++ blocks."""
    for md_file in MD_FILES:
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for i, match in enumerate(CPP_BLOCK_RE.finditer(content)):
            code = match.group(1).strip()
            if len(code) >= MIN_SNIPPET_LEN:
                rel = md_file.relative_to(SKILLS_DIR)
                yield pytest.param(code, id=f"{rel}::block{i}")


# AscendC-specific keywords that tree-sitter-cpp doesn't understand.
# Replace them with valid C++ equivalents before parsing.
ASCENDC_REPLACEMENTS = [
    ("__aicore__", ""),
    ("__global__", ""),
    ("__gm__", ""),
    ("GM_ADDR", "void*"),
    ("ASCEND_IS_AIC", "(true)"),
    ("ASCEND_IS_AIV", "(true)"),
    ("GET_TILING_DATA", "// GET_TILING_DATA"),
    ("BEGIN_TILING_DATA_DEF", "// BEGIN_TILING_DATA_DEF"),
    ("END_TILING_DATA_DEF", "// END_TILING_DATA_DEF"),
    ("TILING_DATA_FIELD_DEF", "// TILING_DATA_FIELD_DEF"),
    ("REGISTER_TILING", "// REGISTER_TILING"),
]


# Patterns that indicate pseudocode rather than real C++ code.
PSEUDOCODE_MARKERS = [
    "Init()    //",     # Pseudocode method stubs
    "Process() //",     # Pseudocode method stubs
    "<OpName>",         # Template placeholders
    "for (each ",       # Pseudocode loops
    "if ASCEND_IS_AIC",  # Macro used as bare if (no parens in pseudocode)
    "...",              # Ellipsis in code = illustrative snippet
    "\u2192",           # Unicode arrow (→) in comments = pseudo
    "__in_pipe__",      # AscendC pipe annotations
    "__out_pipe__",     # AscendC pipe annotations
]


def _is_pseudocode(code: str) -> bool:
    """Heuristic: detect documentation pseudocode that isn't real C++."""
    return any(marker in code for marker in PSEUDOCODE_MARKERS)


def _preprocess_ascendc(code: str) -> str:
    """Replace AscendC-specific keywords with valid C++ equivalents."""
    for old, new in ASCENDC_REPLACEMENTS:
        code = code.replace(old, new)
    return code


try:
    from tree_sitter_languages import get_parser

    _parser = get_parser("cpp")
    HAS_TREE_SITTER = True
except ImportError:
    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser

        CPP_LANGUAGE = Language(tscpp.language())
        _parser = Parser(CPP_LANGUAGE)
        HAS_TREE_SITTER = True
    except ImportError:
        HAS_TREE_SITTER = False
        _parser = None


def _has_error_node(node) -> bool:
    """Recursively check if the AST contains ERROR nodes."""
    if node.type == "ERROR":
        return True
    return any(_has_error_node(child) for child in node.children)


def _count_error_nodes(node) -> int:
    """Count ERROR nodes in AST."""
    count = 1 if node.type == "ERROR" else 0
    for child in node.children:
        count += _count_error_nodes(child)
    return count


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter-cpp not installed")
class TestCppSnippetSyntax:
    @pytest.mark.parametrize("code", list(_collect_cpp_snippets()) if HAS_TREE_SITTER else [])
    def test_snippet_parses(self, code):
        """C++ snippet should parse without excessive syntax errors.

        We allow minor errors since snippets are often fragments
        (missing includes, ellipsis, pseudocode placeholders like
        <OpName>, etc). Flag only if >50% of top-level nodes are errors.
        """
        if _is_pseudocode(code):
            pytest.skip("pseudocode snippet")
        code = _preprocess_ascendc(code)
        tree = _parser.parse(bytes(code, "utf-8"))
        root = tree.root_node
        total = len(root.children)
        if total == 0:
            return  # empty parse, skip
        errors = _count_error_nodes(root)
        error_ratio = errors / max(total, 1)
        assert error_ratio < 0.5, (
            f"Too many syntax errors: {errors}/{total} nodes "
            f"({error_ratio:.0%})\n\n{code[:200]}..."
        )


# ---------------------------------------------------------------------------
# Python snippet syntax checking (ast.parse)
# ---------------------------------------------------------------------------

PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL,
)

# Patterns indicating pseudocode / partial snippets
# Regex: angle-bracket placeholders like <op_name>, <target-dir>
_PLACEHOLDER_RE = re.compile(r"<[a-zA-Z_][a-zA-Z0-9_-]*>")
_BRACE_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

PYTHON_PSEUDO_MARKERS = [
    "...",           # Ellipsis placeholders
]


def _has_placeholder(code: str) -> bool:
    """Check for <placeholder> or {placeholder} patterns in code."""
    return bool(_PLACEHOLDER_RE.search(code) or _BRACE_PLACEHOLDER_RE.search(code))


def _collect_python_snippets():
    """Yield (file_path, block_index, code) for all Python blocks."""
    for md_file in MD_FILES:
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for i, match in enumerate(PYTHON_BLOCK_RE.finditer(content)):
            code = match.group(1).strip()
            if len(code) >= MIN_SNIPPET_LEN:
                if (any(m in code.lower() for m in PYTHON_PSEUDO_MARKERS)
                        or _has_placeholder(code)):
                    continue
                rel = md_file.relative_to(SKILLS_DIR)
                yield pytest.param(code, id=f"{rel}::py{i}")


class TestPythonSnippetSyntax:
    @pytest.mark.parametrize("code", list(_collect_python_snippets()))
    def test_snippet_parses(self, code):
        """Python snippet should parse without syntax errors.

        Single-line fragments (e.g. bare `for` headers) are skipped
        since they're illustrative, not complete code.
        """
        if code.count("\n") == 0:
            pytest.skip("single-line fragment")
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f"Python syntax error at line {e.lineno}: {e.msg}\n\n"
                f"{code[:300]}..."
            )


# ---------------------------------------------------------------------------
# Bash/Shell snippet syntax checking (bash -n)
# ---------------------------------------------------------------------------

BASH_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)\s*\n(.*?)```",
    re.DOTALL,
)

# Patterns indicating pseudocode / partial snippets
BASH_PSEUDO_MARKERS = [
    "...",           # Ellipsis
    "$ ",            # Interactive shell prompts (not scripts)
    "(msdebug)",     # msDebug interactive session
    "(gdb)",         # GDB interactive session
]


def _collect_bash_snippets():
    """Yield (file_path, block_index, code) for all bash blocks."""
    for md_file in MD_FILES:
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for i, match in enumerate(BASH_BLOCK_RE.finditer(content)):
            code = match.group(1).strip()
            if len(code) >= MIN_SNIPPET_LEN:
                if (any(m in code.lower() for m in BASH_PSEUDO_MARKERS)
                        or _has_placeholder(code)):
                    continue
                rel = md_file.relative_to(SKILLS_DIR)
                yield pytest.param(code, id=f"{rel}::sh{i}")


_bash_snippets = list(_collect_bash_snippets())


@pytest.mark.skipif(not _bash_snippets, reason="no bash snippets found")
class TestBashSnippetSyntax:
    @pytest.mark.parametrize("code", _bash_snippets)
    def test_snippet_parses(self, code):
        """Bash snippet should parse without syntax errors (bash -n)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=True) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [BASH_BIN, "-n", f.name],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"Bash syntax error:\n{result.stderr}\n\n"
                    f"{code[:300]}..."
                )

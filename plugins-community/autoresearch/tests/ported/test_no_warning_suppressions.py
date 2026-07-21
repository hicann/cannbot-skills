# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Prevent warning bypasses from silently returning to the plugin."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "ar_tasks",
    "build",
    "dist",
    "node_modules",
}
_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".j2",
    ".js",
    ".json",
    ".jsonc",
    ".md",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
_LINT_CONFIG_NAMES = {
    ".flake8",
    ".pylintrc",
    ".ruff.toml",
    "pyproject.toml",
    "ruff.toml",
    "setup.cfg",
    "tox.ini",
}


def _joined(*parts: str) -> str:
    return "".join(parts)


_DIRECTIVES = {
    "Python line-linter bypass": re.compile(
        r"#\s*" + _joined("no", "qa") + r"\b", re.IGNORECASE
    ),
    "Python type-checker bypass": re.compile(
        r"#\s*type\s*:\s*" + _joined("ig", "nore") + r"\b", re.IGNORECASE
    ),
    "Python formatter bypass": re.compile(
        r"#\s*" + _joined("f", "mt") + r"\s*:\s*off\b", re.IGNORECASE
    ),
    "Pylint inline bypass": re.compile(
        r"#\s*pylint\s*:\s*" + _joined("dis", "able") + r"\b",
        re.IGNORECASE,
    ),
    "Python warning filter": re.compile(
        _joined("filter", "warnings") + r"\s*\(", re.IGNORECASE
    ),
    "coverage bypass": re.compile(
        r"#\s*pragma\s*:\s*" + _joined("no", " cover") + r"\b",
        re.IGNORECASE,
    ),
    "security-scanner bypass": re.compile(
        r"#\s*" + _joined("no", "sec") + r"\b", re.IGNORECASE
    ),
    "C/C++ linter bypass": re.compile(
        _joined("NO", "LINT") + r"(?:NEXTLINE|BEGIN|END)?\b"
    ),
    "Sonar bypass": re.compile(_joined("NO", "SONAR") + r"\b"),
    "compiler diagnostic bypass": re.compile(
        r"#\s*pragma\s+.*(?:diagnostic\s+ignored|diag_suppress)",
        re.IGNORECASE,
    ),
    "ShellCheck bypass": re.compile(
        r"shellcheck\s+" + _joined("dis", "able") + r"\b", re.IGNORECASE
    ),
    "ESLint bypass": re.compile(_joined("eslint", "-disable") + r"\b", re.IGNORECASE),
    "Semgrep bypass": re.compile(_joined("semgrep", ":ignore") + r"\b", re.IGNORECASE),
    "Cppcheck bypass": re.compile(
        _joined("cppcheck", "-suppress") + r"\b", re.IGNORECASE
    ),
    "Java warning bypass": re.compile(r"@" + _joined("Suppress", "Warnings") + r"\b"),
}
_CONFIG_BYPASS = re.compile(
    r"^\s*(?:ignore|extend-ignore|per-file-ignores|disable)\s*=",
    re.MULTILINE,
)


def _source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES or path.name in _LINT_CONFIG_NAMES:
            yield path


def test_repository_has_no_warning_suppressions() -> None:
    findings: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in _DIRECTIVES.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line}: {label}")
        if path.name in _LINT_CONFIG_NAMES:
            for match in _CONFIG_BYPASS.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(
                    f"{path.relative_to(ROOT)}:{line}: lint configuration bypass"
                )

    assert not findings, "Warning suppressions are forbidden:\n" + "\n".join(findings)

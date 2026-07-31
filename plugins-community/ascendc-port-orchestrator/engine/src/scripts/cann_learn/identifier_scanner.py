# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""C34a: deterministic identifier-leakage scanner.

Codex review of v1 design flagged "look for c310_impl substring" as too weak.
This module implements the v2 design's deterministic check:

  1. Build a denylist by extracting EVERY identifier-like token from the CANN
     files the learner read (identifiers, macros, namespace names, class
     names, template parameters, enum values, filenames, include paths).
  2. Build an allowlist of public AscendC API tokens from
     ASCENDC_API_CATALOG.md + public CANN headers (excluding internal/
     impl/ subdirs).
  3. denylist - allowlist = forbidden_set.
  4. Scan candidate output files for any forbidden_set token. Any hit = FAIL.

Threshold for PASS: leak_score == 0 (zero forbidden tokens in candidate output).

Limitations (acknowledged):
- Regex-based extraction; not full C++ AST. May miss exotic constructs.
  (Still much stronger than v1's single-substring heuristic.)
- Allowlist drawn from ASCENDC_API_CATALOG.md — must be kept up to date.
  Stale catalog → false positives (we'd reject valid public-API tokens).
- Comments-only names (e.g. `// internal: c310_impl`) are NOT extracted from
  CANN side; if they appear in candidate output, they bypass denylist. This is
  intentional — comment text is content the learner may legitimately re-write.
  Counterbalanced by C34c (copy-shape) which catches comment-text n-gram match.
"""
from __future__ import annotations
import logging

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Identifier matcher (C/C++ identifier characters)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Known C++ keywords + types that should NEVER be denylisted (they appear
# in everyone's code and would create catastrophic false positives if a CANN
# file happened to declare a variable named e.g. `int`).
_CPP_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "long", "mutable",
    "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
    "or", "or_eq", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
    "static", "static_assert", "static_cast", "struct", "switch", "template",
    "this", "thread_local", "throw", "true", "try", "typedef", "typeid",
    "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
    "wchar_t", "while", "xor", "xor_eq",
    # Common short tokens that can't reasonably be an "internal" name
    "i", "j", "k", "n", "m", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g",
    "T", "U", "V", "T1", "T2", "T3",
})


@dataclass
class ScanResult:
    leak_count: int
    leak_score: float  # 0.0 = clean, higher = worse (count / |forbidden_set|)
    leaks: list[tuple[str, str]]  # [(file_path, leaked_token), ...]
    forbidden_set_size: int
    allowlist_size: int

    @property
    def passed(self) -> bool:
        return self.leak_count == 0


def _is_meaningful_identifier(tok: str) -> bool:
    """Filter out C++ keywords AND short common English words (`to`, `is`, etc.)
    that appear in comments and code prose but are never CANN-internal symbols.

    Heuristic: if token is ≤ 3 chars and lowercase-only, treat as English word.
    Tokens with digits or uppercase letters at any position are kept regardless
    of length (catches e.g. `v0`, `kT`, `i32`).
    """
    if tok in _CPP_KEYWORDS:
        return False
    if len(tok) <= 3:
        # Lowercase-only short token = likely English word in comment, skip.
        # CANN-internal-shaped names virtually always have caps/digits/underscore.
        if tok.islower() and tok.isalpha():
            return False
    return True


def extract_identifiers(text: str) -> set[str]:
    """Extract all C/C++-identifier-shaped tokens from text. Strips C++ keywords
    AND short lowercase English words (likely from comments/prose, not symbols).
    """
    return {tok for tok in _IDENT_RE.findall(text) if _is_meaningful_identifier(tok)}


def extract_macros(text: str) -> set[str]:
    """Extract macro names from `#define X` lines."""
    out = set()
    for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        out.add(m.group(1))
    return out


def extract_include_paths(text: str) -> set[str]:
    """Extract include paths AND their components from `#include "X/Y/Z.h"`."""
    out = set()
    for m in re.finditer(r'^\s*#\s*include\s+["<]([^">]+)[">]', text, re.MULTILINE):
        path = m.group(1)
        out.add(path)
        # Also add path components (X, Y, Z) — those are nameable internal modules
        # that a candidate might reference verbatim
        parts = re.split(r"[/\\.]", path)
        out.update(p for p in parts if p and p not in {"h", "hpp", "cpp", "cc", "c"})
    return out


def extract_filename_tokens(filename: str) -> set[str]:
    """Filename without extension + its tokens (split by _ and -)."""
    name = re.sub(r"\.[a-z]+$", "", filename, flags=re.IGNORECASE)
    out = {name}
    out.update(name.split("_"))
    out.update(name.split("-"))
    return {t for t in out if t and t not in _CPP_KEYWORDS}


def build_denylist_from_files(file_paths: Iterable[Path]) -> set[str]:
    """Extract forbidden-token candidates from a set of CANN source files."""
    denylist: set[str] = set()
    for p in file_paths:
        skip_current_item = False
        try:
            text = p.read_text(errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        denylist |= extract_identifiers(text)
        denylist |= extract_macros(text)
        denylist |= extract_include_paths(text)
        denylist |= extract_filename_tokens(p.name)
    return denylist


def parse_ascendc_api_catalog(catalog_path: Path) -> set[str]:
    """Extract public AscendC API tokens from ASCENDC_API_CATALOG.md.

    The catalog uses markdown tables and code fences. Pull all tokens that
    look like API symbols (UpperCamel function names, snake_case macros,
    namespace-qualified `Namespace::Symbol`).
    """
    if not catalog_path.exists():
        return set()
    text = catalog_path.read_text(errors="replace")
    out: set[str] = set()
    # All identifier-like tokens — catalog only mentions public-facing names
    out |= extract_identifiers(text)
    # Plus namespace-qualified forms (collapse to both halves)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)", text):
        out.add(m.group(0))
        out.add(m.group(1))
        out.add(m.group(2))
    return out


def parse_public_headers(public_include_dir: Path) -> set[str]:
    """Extract public symbols from CANN's public headers (excluding internal/impl)."""
    if not public_include_dir.exists():
        return set()
    public_headers: list[Path] = []
    forbidden_dirs = {"internal", "impl", "_impl", "c310_impl", "_internal"}
    for header in public_include_dir.rglob("*.h"):
        # Skip if any path component is internal-marker
        if any(part in forbidden_dirs for part in header.parts):
            continue
        public_headers.append(header)
    return build_denylist_from_files(public_headers)


def scan_for_leaks(
    candidate_paths: Iterable[Path],
    forbidden_tokens: set[str],
) -> list[tuple[str, str]]:
    """For each candidate file, find all forbidden tokens that appear in it."""
    leaks: list[tuple[str, str]] = []
    for cand in candidate_paths:
        skip_current_item = False
        try:
            text = cand.read_text(errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        cand_tokens = extract_identifiers(text)
        for tok in cand_tokens:
            if tok in forbidden_tokens:
                leaks.append((str(cand), tok))
    return leaks


def scan(
    cann_files_read: Iterable[Path],
    candidate_output_paths: Iterable[Path],
    *,
    api_catalog_path: Path,
    public_include_dir: Path | None = None,
) -> ScanResult:
    """Run full C34a scan: build denylist - allowlist, scan candidates.

    Args:
        cann_files_read: CANN source files the learner READ (anything in here
            is forbidden unless on the allowlist).
        candidate_output_paths: candidate KB output files to scan.
        api_catalog_path: ASCENDC_API_CATALOG.md (KB, for public-API allowlist).
        public_include_dir: optional CANN_PATH/include dir for additional
            public-API allowlist. Internal/impl subdirs auto-excluded.

    Returns:
        ScanResult with leak_count, leak_score, list of leaks. passed() True iff
        leak_count == 0.
    """
    denylist_raw = build_denylist_from_files(cann_files_read)
    allowlist = parse_ascendc_api_catalog(api_catalog_path)
    if public_include_dir is not None:
        allowlist |= parse_public_headers(public_include_dir)
    forbidden = denylist_raw - allowlist - _CPP_KEYWORDS
    leaks = scan_for_leaks(candidate_output_paths, forbidden)
    return ScanResult(
        leak_count=len(leaks),
        leak_score=len(leaks) / max(1, len(forbidden)),
        leaks=leaks,
        forbidden_set_size=len(forbidden),
        allowlist_size=len(allowlist),
    )

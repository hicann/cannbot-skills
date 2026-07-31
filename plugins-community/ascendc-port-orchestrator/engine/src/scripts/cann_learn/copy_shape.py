# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""C34c: token n-gram contiguous-overlap detector — catches renamed-identifier copy.

Codex review point #6: 'Copy-not-learn missing as first-class. Names checked
(C34a), internal-symbol semantics (C34b), but neither catches close paraphrase
of loop structure, tiling constants, magic thresholds, branch predicates, or
source-shaped pseudocode.'

Approach: tokenize both candidate text and source text, normalize identifiers
to abstract tags (so renamed copy still matches), slide N-gram windows over
candidate, check each window against source token streams. >= threshold
contiguous matches = copy-shape.

What this catches that C34a misses:
- "Compute mean by summing then dividing" copied as `for (int j=0; j<R; j++) sum += a[j]; mean = sum/R;`
  even if identifiers renamed to `m`, `acc`, `cnt`
- Branch predicates: `if (x > THRESHOLD) ... else ...` structure preserved
  with renamed THRESHOLD constant
- Magic-number constants reused verbatim (1024, 0.044715, etc.)

What it does NOT catch:
- Pure conceptual restatement ("vendor uses a tree-reduction pattern")
  — that's the goal of learning, not copying
- Generic API call sequences (DataCopy + ReduceSum + Adds) shared by all
  reduction kernels — these will match harmlessly

Threshold: configurable, default 5% (5-gram windows where >5% of candidate
windows have a contiguous match in source). Below threshold = pass; at or
above = candidate REJECTED for copy-shape.

Limitations:
- Token normalization is regex-based, not real C++ AST
- Constant numbers are KEPT literal (catches magic-number copies); change
  this if it causes false positives
"""
from __future__ import annotations
import logging

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*"   # identifier
    r"|0[xX][0-9A-Fa-f]+"          # hex literal
    r"|\d+\.\d+(?:[eE][+-]?\d+)?"  # float literal
    r"|\d+"                         # int literal
    r"|->|::|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\.\.\."
    r"|[+\-*/%=<>!&|^~?:.,;()\[\]{}\\#@$])"
)

# C++ keywords are preserved as-is (loop structure markers — for/while/if/else
# matter for shape detection)
_PRESERVED_KEYWORDS = frozenset({
    "for", "while", "do", "if", "else", "switch", "case", "default",
    "return", "break", "continue", "goto", "throw", "try", "catch",
    "static", "const", "volatile", "register", "mutable",
    "class", "struct", "union", "enum", "namespace", "template",
    "typename", "typedef", "auto", "decltype", "constexpr",
    "true", "false", "nullptr", "this",
    "void", "bool", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned",
})


@dataclass
class CopyShapeResult:
    candidate_id: str
    score: float       # fraction of candidate N-grams that match source
    match_count: int
    total_windows: int
    threshold: float
    sample_matches: list[tuple[int, str]]  # [(window_index, ngram_text), ...]

    @property
    def passed(self) -> bool:
        return self.score < self.threshold


def normalize_tokens(text: str) -> list[str]:
    """Tokenize and normalize: identifiers (non-keywords) → IDENT tag,
    keep keywords + operators + literal numbers as-is.

    Effect: `for (int i = 0; i < N; i++)` and `for (int j = 0; j < count; j++)`
    both → `[for, (, int, IDENT, =, 0, ;, IDENT, <, IDENT, ;, IDENT, ++, )]`
    so renamed copies match.
    """
    tokens = _TOKEN_RE.findall(text)
    out = []
    for t in tokens:
        if t in _PRESERVED_KEYWORDS:
            out.append(t)
        elif t and t[0].isalpha() or (t and t[0] == "_"):
            out.append("IDENT")
        else:
            # Operator, punctuation, or numeric literal — keep verbatim
            out.append(t)
    return out


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def copy_shape_score(
    candidate_text: str,
    source_files: Iterable[Path],
    *,
    n: int = 5,
) -> tuple[float, list[tuple[int, str]]]:
    """Tokenize both, normalize identifiers, count N-gram contiguous matches.

    Returns (score, sample_matches). Score is fraction of candidate N-grams
    that appear contiguously in source. sample_matches is up to 10 examples
    for debug.
    """
    cand_tokens = normalize_tokens(candidate_text)
    cand_ngrams = _ngrams(cand_tokens, n)
    if not cand_ngrams:
        return (0.0, [])

    # Build source N-gram set across all source files
    source_ngrams: set[tuple[str, ...]] = set()
    for f in source_files:
        skip_current_item = False
        try:
            text = f.read_text(errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        src_tokens = normalize_tokens(text)
        source_ngrams.update(_ngrams(src_tokens, n))

    matches = 0
    sample_matches: list[tuple[int, str]] = []
    for i, ng in enumerate(cand_ngrams):
        if ng in source_ngrams:
            matches += 1
            if len(sample_matches) < 10:
                sample_matches.append((i, " ".join(ng)))
    return (matches / len(cand_ngrams), sample_matches)


def check(
    candidate_id: str,
    candidate_text: str,
    source_files: Iterable[Path],
    *,
    n: int = 5,
    threshold: float = 0.05,
) -> CopyShapeResult:
    """Run C34c on a single candidate.

    Default n=5 (5-gram windows), threshold=0.05 (5% match rate). At or above
    threshold → REJECTED.
    """
    score, samples = copy_shape_score(candidate_text, source_files, n=n)
    cand_tokens = normalize_tokens(candidate_text)
    total_windows = max(0, len(cand_tokens) - n + 1)
    return CopyShapeResult(
        candidate_id=candidate_id,
        score=score,
        match_count=int(score * total_windows) if total_windows else 0,
        total_windows=total_windows,
        threshold=threshold,
        sample_matches=samples,
    )

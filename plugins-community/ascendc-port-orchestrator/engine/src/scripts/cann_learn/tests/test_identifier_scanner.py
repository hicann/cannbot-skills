# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for C34a identifier-leakage scanner."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import identifier_scanner as ids  # noqa: E402


# ---------------------------------------------------------------------------
# Token extraction helpers
# ---------------------------------------------------------------------------
def test_extract_identifiers_basic():
    text = "auto x = MyClass::doStuff();"
    toks = ids.extract_identifiers(text)
    # Keywords stripped, identifiers kept
    assert "MyClass" in toks
    assert "doStuff" in toks
    assert "auto" not in toks  # keyword


def test_extract_identifiers_strips_keywords():
    text = "static const int n = 5;"
    toks = ids.extract_identifiers(text)
    assert "n" not in toks  # too short, in keyword set
    assert "static" not in toks
    assert "const" not in toks


def test_extract_macros_define():
    text = """
#define INTERNAL_LIMIT 1024
#define c310_impl_FOO 42
not_a_macro = 1;
"""
    macros = ids.extract_macros(text)
    assert macros == {"INTERNAL_LIMIT", "c310_impl_FOO"}


def test_extract_include_paths():
    text = """
#include "internal/c310_impl/normalize.h"
#include <ascendc/foo.h>
"""
    out = ids.extract_include_paths(text)
    assert "internal/c310_impl/normalize.h" in out
    assert "ascendc/foo.h" in out
    # path components also extracted
    assert "internal" in out
    assert "c310_impl" in out
    assert "normalize" in out
    assert "ascendc" in out
    # extensions excluded
    assert "h" not in out


def test_extract_filename_tokens():
    out = ids.extract_filename_tokens("normalize_c310_impl.h")
    # filename without extension
    assert "normalize_c310_impl" in out
    # underscore-split components
    assert "normalize" in out
    assert "c310" in out
    assert "impl" in out


# ---------------------------------------------------------------------------
# Denylist construction
# ---------------------------------------------------------------------------
def test_build_denylist_from_files(tmp_path):
    f = tmp_path / "c310_impl.h"
    f.write_text("""
#define NORMALIZE_INTERNAL_FLAG 0x1
namespace c310_impl {
class NormalizeVFImpl {
  void doInternalDispatch();
};
}
""")
    denylist = ids.build_denylist_from_files([f])
    # Identifiers
    assert "c310_impl" in denylist
    assert "NormalizeVFImpl" in denylist
    assert "doInternalDispatch" in denylist
    # Macro
    assert "NORMALIZE_INTERNAL_FLAG" in denylist
    # Filename tokens
    assert "c310" in denylist
    assert "impl" in denylist


# ---------------------------------------------------------------------------
# Allowlist (from API catalog)
# ---------------------------------------------------------------------------
def test_parse_api_catalog(tmp_path):
    cat = tmp_path / "ASCENDC_API_CATALOG.md"
    cat.write_text("""
# AscendC API Catalog

## DataMovement

- `DataCopy(LocalTensor, GlobalTensor)` — public API
- `Adds(dst, src, scalar)` — public

## Reduce

- `WholeReduceSum<T>` — public
- `BlockReduceSum<T>` — public

`AscendC::Sin(x)` returns sin.
""")
    allow = ids.parse_ascendc_api_catalog(cat)
    assert "DataCopy" in allow
    assert "WholeReduceSum" in allow
    assert "AscendC" in allow
    assert "Sin" in allow
    # Namespace-qualified form
    assert "AscendC::Sin" in allow


# ---------------------------------------------------------------------------
# End-to-end scan
# ---------------------------------------------------------------------------
def test_scan_clean_candidate_passes(tmp_path):
    cann = tmp_path / "cann_internal.h"
    cann.write_text("""
namespace c310_impl {
class NormalizeVFImpl { void privateDispatch(); };
}
""")
    cat = tmp_path / "ASCENDC_API_CATALOG.md"
    cat.write_text("DataCopy, Adds, Mul are public.\n")

    cand = tmp_path / "candidate.md"
    cand.write_text("""
[[HEADING]] Pattern: batched-reduce
Use `DataCopy` to load N rows then `WholeReduceSum` per row.
""".replace("[[HEADING]]", "#"))
    # Cand uses only public names; clean
    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    assert res.passed
    assert res.leak_count == 0


def test_scan_leaky_candidate_with_internal_class_fails(tmp_path):
    cann = tmp_path / "cann_internal.h"
    cann.write_text("""
namespace c310_impl {
class NormalizeVFImpl { };
}
""")
    cat = tmp_path / "ASCENDC_API_CATALOG.md"
    cat.write_text("DataCopy is public.\n")

    cand = tmp_path / "candidate.md"
    # LEAK: candidate references CANN-internal class verbatim
    cand.write_text("Use NormalizeVFImpl from c310_impl namespace.\n")

    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    assert not res.passed
    leaked_tokens = {tok for _, tok in res.leaks}
    assert "NormalizeVFImpl" in leaked_tokens
    assert "c310_impl" in leaked_tokens


def test_scan_macro_leak_detected(tmp_path):
    cann = tmp_path / "internal.h"
    cann.write_text("#define INTERNAL_DISPATCH_FLAG 1\n")
    cat = tmp_path / "cat.md"
    cat.write_text("\n")

    cand = tmp_path / "candidate.md"
    cand.write_text("Set INTERNAL_DISPATCH_FLAG=1 to enable.\n")

    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    assert not res.passed
    leaked_tokens = {tok for _, tok in res.leaks}
    assert "INTERNAL_DISPATCH_FLAG" in leaked_tokens


def test_scan_include_path_component_leak(tmp_path):
    cann = tmp_path / "x.h"
    cann.write_text('#include "internal/c310_impl/normalize.h"\n')
    cat = tmp_path / "cat.md"
    cat.write_text("\n")

    cand = tmp_path / "candidate.md"
    # LEAK: candidate names a CANN-internal directory component
    cand.write_text("This pattern adapts logic from c310_impl/normalize.\n")

    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    assert not res.passed
    leaked = {tok for _, tok in res.leaks}
    assert "c310_impl" in leaked


def test_scan_no_false_positive_on_public_api_use(tmp_path):
    cann = tmp_path / "internal.h"
    cann.write_text("""
namespace internal_mod {
  class InternalThing { };
  void DataCopy();  // CANN happens to also have a private DataCopy overload
}
""")
    cat = tmp_path / "cat.md"
    cat.write_text("DataCopy is public AscendC.\n")

    cand = tmp_path / "candidate.md"
    # Candidate uses DataCopy (public API) — should NOT be flagged even
    # though CANN internal headers also mention it
    cand.write_text("Use AscendC's `DataCopy` to load tiles.\n")

    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    assert res.passed, f"public-API use must not flag, got leaks={res.leaks}"


def test_scan_includes_public_headers_dir(tmp_path):
    """Allowlist also includes public CANN headers (excluding internal/)."""
    cann_internal = tmp_path / "cann_internal.h"
    cann_internal.write_text("class InternalSym;\n")
    cat = tmp_path / "cat.md"
    cat.write_text("(empty catalog)\n")

    public_dir = tmp_path / "public_headers"
    (public_dir / "ascendc").mkdir(parents=True)
    (public_dir / "ascendc" / "ops.h").write_text("class PublicOpsClass;\n")
    # Internal subdirs IGNORED
    (public_dir / "internal").mkdir()
    (public_dir / "internal" / "secret.h").write_text("class SecretFromInternal;\n")

    cand = tmp_path / "cand.md"
    # Uses PublicOpsClass (in public dir, allowed)
    cand.write_text("Use PublicOpsClass to do work.\n")

    res = ids.scan(
        cann_files_read=[cann_internal],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
        public_include_dir=public_dir,
    )
    assert res.passed
    assert "PublicOpsClass" not in {t for _, t in res.leaks}


def test_scan_includes_public_headers_excludes_internal_subdir(tmp_path):
    """Symbols in public_dir/internal/ are NOT auto-allowed."""
    cann_dir = tmp_path / "cann"
    cann_dir.mkdir()
    cann_internal_file = cann_dir / "x.h"
    cann_internal_file.write_text("class SecretFromInternal;\n")
    cat = tmp_path / "cat.md"
    cat.write_text("(empty)\n")

    public_dir = tmp_path / "public_headers"
    (public_dir / "internal").mkdir(parents=True)
    (public_dir / "internal" / "secret.h").write_text("class SecretFromInternal;\n")

    cand = tmp_path / "cand.md"
    cand.write_text("Use SecretFromInternal here.\n")

    res = ids.scan(
        cann_files_read=[cann_internal_file],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
        public_include_dir=public_dir,
    )
    # Symbol IS in public_dir but inside `internal/` subdir — excluded from
    # allowlist, treated as leak
    assert not res.passed
    assert "SecretFromInternal" in {t for _, t in res.leaks}


def test_scan_keyword_collision_not_flagged(tmp_path):
    """If CANN file has `int n` (keyword + short name), candidate using `n`
    should not be flagged.
    """
    cann = tmp_path / "internal.h"
    cann.write_text("int n; void func(int x);\n")
    cat = tmp_path / "cat.md"
    cat.write_text("\n")

    cand = tmp_path / "cand.md"
    cand.write_text("Loop counter `i` from 0 to N-1.\n")

    res = ids.scan(
        cann_files_read=[cann],
        candidate_output_paths=[cand],
        api_catalog_path=cat,
    )
    # `i`, `n`, `x` all in _CPP_KEYWORDS allowlist → not flagged
    assert res.passed

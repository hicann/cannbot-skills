# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for C34b public-API compile gate."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import compile_gate as cg  # noqa: E402


def test_build_public_only_sysroot_excludes_internal_dirs(tmp_path):
    """Internal/impl/c310_impl/_internal subdirs are filtered from sysroot."""
    cann = tmp_path / "include"
    cann.mkdir()
    (cann / "ascendc").mkdir()
    (cann / "ascendc" / "ops.h").write_text("// public\n")
    (cann / "internal").mkdir()
    (cann / "internal" / "secret.h").write_text("// internal\n")
    (cann / "ascendc" / "impl").mkdir()
    (cann / "ascendc" / "impl" / "private.h").write_text("// also internal\n")

    sysroot, excluded = cg.build_public_only_sysroot(cann)
    try:
        # public file present
        assert (sysroot / "ascendc" / "ops.h").exists()
        # internal/ excluded
        assert not (sysroot / "internal" / "secret.h").exists()
        # impl/ subdir excluded even when nested under public path
        assert not (sysroot / "ascendc" / "impl" / "private.h").exists()
        assert excluded == 2  # 2 files filtered
    finally:
        shutil.rmtree(sysroot)


def test_wrap_anchor_passes_through_function_decl():
    """Anchor that's already a function decl isn't double-wrapped."""
    code = "void my_func() { int x = 1; }"
    wrapped = getattr(cg, '_wrap_anchor_in_compile_unit')(code)
    # Only one `void my_func()` instance
    assert wrapped.count("void my_func") == 1


def test_wrap_anchor_wraps_bare_statements():
    """Bare statement (function call, not a decl) gets wrapped in stub function body."""
    code = "DataCopy(dst, src); WholeReduceSum(buf);"
    wrapped = getattr(cg, '_wrap_anchor_in_compile_unit')(code)
    assert "__cann_learn_anchor" in wrapped
    assert "DataCopy" in wrapped


def test_wrap_anchor_passes_through_int_decl():
    """`int x = 1;` is a valid TU-level decl — passes through without wrap."""
    code = "int x = 1;"
    wrapped = getattr(cg, '_wrap_anchor_in_compile_unit')(code)
    # Not wrapped — `int ` is a TU starter
    assert "__cann_learn_anchor" not in wrapped
    assert wrapped.strip() == code


def test_wrap_anchor_passes_through_includes():
    code = '#include "ascendc/ops.h"\nvoid foo();'
    wrapped = getattr(cg, '_wrap_anchor_in_compile_unit')(code)
    # Not double-wrapped — passed through
    assert wrapped == code or wrapped.startswith("#include")


def test_compile_anchor_compiler_unavailable_returns_failure(tmp_path):
    """If bishengir-clang isn't in PATH, gate fails-closed with explicit reason
    (NOT silently passing).
    """
    # Use a guaranteed-nonexistent binary name
    sysroot, _ = cg.build_public_only_sysroot(
        tmp_path / "include") if (tmp_path / "include").exists() else (tmp_path / "fakesys", 0)
    if not sysroot.exists():
        sysroot.mkdir()
    try:
        result = cg.compile_anchor(
            "void foo();",
            sysroot=sysroot,
            bishengir_bin="bishengir-clang-DEFINITELY-NOT-INSTALLED",
            candidate_id="t1",
        )
        assert result.compile_pass is False
        assert "COMPILER_UNAVAILABLE" in result.stderr_first_500
    finally:
        if sysroot.exists():
            shutil.rmtree(sysroot, ignore_errors=True)


def test_pass_rate_empty_dict_is_one():
    """No candidates = vacuously passing."""
    assert cg.pass_rate({}) == pytest.approx(1.0)


def test_pass_rate_mixed():
    results = {
        "a": cg.CompileResult("a", True, "", 0),
        "b": cg.CompileResult("b", False, "err", 0),
        "c": cg.CompileResult("c", True, "", 0),
    }
    assert cg.pass_rate(results) == pytest.approx(2 / 3)


def test_compile_anchor_with_real_compiler_basic(tmp_path):
    """If bishengir-clang IS available, syntax-only check on trivial code passes.
    SKIPPED in CI without compiler.
    """
    if not shutil.which(cg.DEFAULT_BISHENG_BIN) and not shutil.which("clang"):
        pytest.skip("no compiler in PATH")
    # Use clang as fallback for the gate's syntax check; bishengir is a clang variant
    bin_name = shutil.which(cg.DEFAULT_BISHENG_BIN) or shutil.which("clang")

    cann = tmp_path / "include"
    cann.mkdir()
    sysroot, _ = cg.build_public_only_sysroot(cann)
    try:
        # Trivial valid C++ snippet
        result = cg.compile_anchor(
            "int x = 1;",
            sysroot=sysroot,
            bishengir_bin=bin_name,
            candidate_id="trivial",
            extra_args=["--target=x86_64-linux"],  # avoid ascend target if using stock clang
        )
        # Should compile (parse) — even with empty sysroot, trivial expression parses
        assert result.compile_pass, f"trivial snippet should parse, got stderr: {result.stderr_first_500}"
    finally:
        shutil.rmtree(sysroot)


def test_compile_anchor_internal_include_fails(tmp_path):
    """Anchor that #includes an internal/ header fails parse against public sysroot.
    SKIPPED if no compiler.
    """
    if not shutil.which(cg.DEFAULT_BISHENG_BIN) and not shutil.which("clang"):
        pytest.skip("no compiler in PATH")
    bin_name = shutil.which(cg.DEFAULT_BISHENG_BIN) or shutil.which("clang")

    cann = tmp_path / "include"
    cann.mkdir()
    (cann / "internal").mkdir()
    (cann / "internal" / "private.h").write_text("class Secret {};\n")
    (cann / "public.h").write_text("class Pub {};\n")

    sysroot, excluded = cg.build_public_only_sysroot(cann)
    assert excluded == 1  # internal/private.h filtered
    try:
        # Anchor wants the internal header — should fail
        result = cg.compile_anchor(
            '#include "internal/private.h"\nvoid use() { Secret s; }',
            sysroot=sysroot,
            bishengir_bin=bin_name,
            candidate_id="leaky",
            extra_args=["--target=x86_64-linux"],
        )
        assert result.compile_pass is False
        # stderr should mention the missing file
        assert ("file not found" in result.stderr_first_500.lower()
                or "no such file" in result.stderr_first_500.lower()
                or "fatal error" in result.stderr_first_500.lower())

        # Compare: same anchor with public-only include parses fine
        result_clean = cg.compile_anchor(
            '#include "public.h"\nvoid use() { Pub p; }',
            sysroot=sysroot,
            bishengir_bin=bin_name,
            candidate_id="clean",
            extra_args=["--target=x86_64-linux"],
        )
        assert result_clean.compile_pass, f"public-only include should parse, got: {result_clean.stderr_first_500}"
    finally:
        shutil.rmtree(sysroot)


def test_gate_runs_multiple_candidates(tmp_path):
    """gate() runs over a dict of candidates, returns per-candidate result."""
    if not shutil.which(cg.DEFAULT_BISHENG_BIN) and not shutil.which("clang"):
        pytest.skip("no compiler in PATH")
    bin_name = shutil.which(cg.DEFAULT_BISHENG_BIN) or shutil.which("clang")

    cann = tmp_path / "include"
    cann.mkdir()

    candidates = {
        "trivial_pass": "int n = 42;",
        "syntax_error": "int n = ; broken",  # parser error
    }
    results = cg.gate(
        candidates,
        cann_include_dir=cann,
        bishengir_bin=bin_name,
        extra_clang_args=["--target=x86_64-linux"],
    )
    assert results["trivial_pass"].compile_pass
    assert not results["syntax_error"].compile_pass
    assert cg.pass_rate(results) == pytest.approx(0.5)

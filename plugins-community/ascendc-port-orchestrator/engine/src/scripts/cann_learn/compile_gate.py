# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""C34b: public-API-only compile gate.

Codex review of v1 design flagged: 'C34b would a future kw applying this pattern
produce CANN-internal symbols? — that's LLM judgment, not a gate.' v2 makes it
mechanical:

  1. Each candidate's `concrete-anchor` snippet must be self-contained (compilable
     as a function body or top-level decl).
  2. Build a sysroot containing CANN public headers MINUS internal/impl/c310_impl
     subdirectories.
  3. Run `bishengir-clang -fsyntax-only` (parse-only, no codegen) against that
     restricted sysroot.
  4. compile_pass == True iff the snippet parses under public-only headers.

If the snippet uses a CANN-internal symbol, the include resolution fails and
parsing errors → candidate REJECTED (it depends on internal API).

Limitations (acknowledged):
- bishengir-clang must be available in the environment. If not (e.g. running
  the scanner outside the build container), gate emits a status of
  COMPILER_UNAVAILABLE and skill caller refuses Mode 5 (preflight).
- Some patterns describe behavior across kernel passes (e.g. UB-layout strategy)
  that doesn't map to a single self-contained snippet. For those, the anchor IS
  a stylized API-call sequence wrapped in `void __cann_learn_anchor()`; if even
  that can't parse against public headers, the pattern is non-actionable in
  practice.
- This is parse-only — semantic errors / undefined references at link time are
  NOT caught. C34a (identifier denylist) covers the literal-symbol-leak case
  separately.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_INTERNAL_DIR_NAMES = frozenset({
    "internal", "impl", "_impl", "c310_impl", "_internal",
})

# Default include path — orchestrator passes the actual CANN_PATH from .ascendc_env
DEFAULT_BISHENG_BIN = "bishengir-clang"


@dataclass
class CompileResult:
    candidate_id: str
    compile_pass: bool
    stderr_first_500: str
    sysroot_path_excluded_count: int  # how many internal/ subdirs were filtered

    @property
    def passed(self) -> bool:
        return self.compile_pass


def build_public_only_sysroot(
    cann_include_dir: Path,
    *,
    extra_excluded_names: Iterable[str] = (),
) -> tuple[Path, int]:
    """Materialize a temp directory mirroring cann_include_dir but EXCLUDING
    any subdir named `internal`, `impl`, etc. Returns (path, count_excluded).

    Caller must clean up the temp dir (path.cleanup() if TemporaryDirectory or
    shutil.rmtree(path) if Path).

    NOTE: returned path is a Path, not a TemporaryDirectory — caller responsible
    for rmtree. This avoids forcing context-manager ergonomics on the scanner
    pipeline, which composes results across multiple candidates.
    """
    excluded_names = _INTERNAL_DIR_NAMES | set(extra_excluded_names)

    if not cann_include_dir.exists():
        raise FileNotFoundError(f"cann_include_dir not found: {cann_include_dir}")

    sysroot = Path(tempfile.mkdtemp(prefix="cann_learn_sysroot_"))
    excluded_count = 0
    # Walk and copy files only when none of the path components is excluded
    for src_file in cann_include_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(cann_include_dir)
        if any(part in excluded_names for part in rel.parts):
            excluded_count += 1
            continue
        dst_file = sysroot / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        # Symlink to save disk; if symlink fails (e.g. cross-fs), copy
        try:
            dst_file.symlink_to(src_file)
        except OSError:
            shutil.copy2(src_file, dst_file)
    return sysroot, excluded_count


def _wrap_anchor_in_compile_unit(anchor_code: str) -> str:
    """Wrap a candidate's anchor snippet so it parses as a valid C++ TU.

    Pattern: a stub function body. The snippet may already be a function, in
    which case we don't double-wrap.
    """
    s = anchor_code.strip()
    # Heuristic: if anchor starts with "void " / class / namespace / template /
    # `#include`, treat as TU-level. Otherwise wrap as function body.
    tu_starters = (
        "void ", "int ", "auto ", "class ", "struct ", "namespace ",
        "template ", "template<", "#include", "#define", "extern ",
    )
    if any(s.startswith(p) for p in tu_starters):
        return s
    return (
        "// auto-wrapped by C34b compile_gate\n"
        "void __cann_learn_anchor() {\n"
        + s + "\n"
        "}\n"
    )


def compile_anchor(
    anchor_code: str,
    *,
    sysroot: Path,
    bishengir_bin: str = DEFAULT_BISHENG_BIN,
    candidate_id: str = "anchor",
    extra_args: list[str] | None = None,
    timeout_sec: int = 30,
) -> CompileResult:
    """Run bishengir-clang -fsyntax-only against the anchor + public-only sysroot.

    Returns CompileResult; passed=True iff bishengir exited 0.
    """
    if not shutil.which(bishengir_bin):
        return CompileResult(
            candidate_id=candidate_id,
            compile_pass=False,
            stderr_first_500=f"COMPILER_UNAVAILABLE: {bishengir_bin} not found in PATH",
            sysroot_path_excluded_count=-1,
        )

    wrapped = _wrap_anchor_in_compile_unit(anchor_code)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cpp", delete=False
    ) as f:
        f.write(wrapped)
        anchor_file = Path(f.name)

    try:
        cmd = [
            bishengir_bin, "-fsyntax-only",
            "-isystem", str(sysroot),
            "--target=ascend-arch22",
            *(extra_args or []),
            str(anchor_file),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_sec,
            )
            return CompileResult(
                candidate_id=candidate_id,
                compile_pass=result.returncode == 0,
                stderr_first_500=(result.stderr or "")[:500],
                sysroot_path_excluded_count=0,  # caller knows from build_*_sysroot
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                candidate_id=candidate_id,
                compile_pass=False,
                stderr_first_500=f"TIMEOUT after {timeout_sec}s",
                sysroot_path_excluded_count=0,
            )
    finally:
        try:
            anchor_file.unlink()
        except OSError:
            pass


def gate(
    candidate_anchors: dict[str, str],
    *,
    cann_include_dir: Path,
    bishengir_bin: str = DEFAULT_BISHENG_BIN,
    extra_excluded_names: Iterable[str] = (),
    extra_clang_args: list[str] | None = None,
) -> dict[str, CompileResult]:
    """Run C34b compile gate over a dict of {candidate_id: anchor_code_str}.

    Builds public-only sysroot once, compiles each anchor against it, returns
    per-candidate CompileResult. pass_rate computable from results.values().
    """
    sysroot, excluded_count = build_public_only_sysroot(
        cann_include_dir, extra_excluded_names=extra_excluded_names,
    )
    try:
        out = {}
        for cand_id, anchor in candidate_anchors.items():
            r = compile_anchor(
                anchor,
                sysroot=sysroot,
                bishengir_bin=bishengir_bin,
                candidate_id=cand_id,
                extra_args=extra_clang_args,
            )
            r.sysroot_path_excluded_count = excluded_count
            out[cand_id] = r
        return out
    finally:
        # Best-effort cleanup
        try:
            shutil.rmtree(sysroot)
        except OSError:
            pass


def pass_rate(results: dict[str, CompileResult]) -> float:
    """Fraction of candidates whose anchor compiled clean."""
    if not results:
        return 1.0  # no candidates = vacuously passing
    return sum(1 for r in results.values() if r.passed) / len(results)

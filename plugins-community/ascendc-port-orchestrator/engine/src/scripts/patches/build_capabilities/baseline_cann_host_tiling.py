# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Baseline capability: DEBT-110 cann_stubs include + conditional host matmul-tiling link.

Behavior-neutral refactor (design §2C): the two DEBT-110 accretions that used to be
hard-coded in build_ascendc.py's f-string are relocated here as a BASELINE
CMakeContribution that always evaluates, WITH THEIR EXACT ORIGINAL FIRING CONDITIONS:

  1. cann_stubs include dir — added iff `<script_dir>/cann_stubs` exists on disk
     (exactly the original `if cann_stubs_dir.is_dir():` guard). Header-only shims
     that let independently generated op_host/<op>_tiling.h files compile against the
     build include path without a CANN host runtime install.

  2. host matmul-tiling link — the `find_library(nnopbase/opapi)` + `if(...)` block,
     emitted UNCHANGED as a raw cmake_fragment. It stays CONDITIONAL exactly as before:
     the link only fires at cmake-configure time when the CANN host libs are present
     (`if(CANN_NNOPBASE AND CANN_OPAPI)`); it is NOT made unconditional. On a
     clean-runtime lane the libs are absent and the DEBT-110 stub headers are used
     (CUBE precision deferred) — so non-CUBE / stub-only builds are unaffected.

This proves the CMakeContribution interface can express the existing accretions
(dogfooding), while the golden CMakeLists gate proves the emission is byte-identical.
"""
from __future__ import annotations

from .context import BuildContext
from .contribution import CMakeContribution

# Emitted verbatim after the base `target_link_libraries(pybind11_lib PRIVATE ...)`
# block. `${...}` are literal CMake variable refs (this is a plain string, not an
# f-string). Kept byte-for-byte identical to the pre-refactor inline text; ends with
# a trailing newline so it abuts the following `# Resolve torch ...` comment exactly.
DEBT110_LINK_FRAGMENT = (
    "# DEBT-110 / deformable precision (2026-06-29): conditionally link the CANN host\n"
    "# matmul-tiling libs (nnopbase + opapi) so a CUBE/MIX op_host TilingFunc gets a REAL\n"
    "# MatmulApiTiling::GetTiling (populated TCubeTiling) instead of the stub's zero-fill —\n"
    "# required for CUBE-op precision (host TCubeTiling lib enhancement, per the deformable\n"
    "# port_a3 await_user_decision). find_library keeps it CONDITIONAL: links the real libs\n"
    "# when the CANN host install is present, and falls back to the DEBT-110 stub headers\n"
    "# when absent (clean-runtime lane) — so non-CUBE / stub-only builds are unaffected.\n"
    "find_library(CANN_NNOPBASE nnopbase PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64)\n"
    "find_library(CANN_OPAPI opapi PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64)\n"
    "if(CANN_NNOPBASE AND CANN_OPAPI)\n"
    '  message("CANN host matmul-tiling libs found (${CANN_NNOPBASE} '
    '${CANN_OPAPI}) — linking real tiling (CUBE precision enabled)")\n'
    "  target_link_libraries(pybind11_lib PRIVATE ${CANN_NNOPBASE} ${CANN_OPAPI})\n"
    "else()\n"
    '  message("CANN host matmul-tiling libs NOT found in '
    '${ASCEND_CANN_PACKAGE_PATH}/lib64 — DEBT-110 stub headers '
    '(clean-runtime; CUBE precision deferred)")\n'
    "endif()\n"
)


def contribution(ctx: BuildContext) -> CMakeContribution:
    include_dirs: tuple = ()
    if ctx.script_dir is not None:
        cann_stubs_dir = ctx.script_dir / "cann_stubs"
        if cann_stubs_dir.is_dir():
            include_dirs = (cann_stubs_dir,)
    return CMakeContribution(
        include_dirs=include_dirs,
        cmake_fragment=DEBT110_LINK_FRAGMENT,
    )

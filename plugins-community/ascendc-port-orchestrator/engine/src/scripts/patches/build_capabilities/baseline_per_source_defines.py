# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Baseline capability: DEBT-20 + DEBT-20.1 per-source-file COMPILE_DEFINITIONS.

Behavior-neutral refactor (design §2C): the per_source_defines loader that used to
live inline in build_ascendc.py is relocated here and exposed as a BASELINE
CMakeContribution that ALWAYS evaluates. Its firing condition is unchanged — the
contribution's `defines` is exactly `load_per_source_defines(kernel_dir)` (empty when
build_overrides.json is absent/malformed, so the emitted CMake is unchanged).

Original DEBT-20/20.1 rationale is preserved verbatim below.

DEBT-20: FA Pattern B requires `-DASCENDC_MATMUL_AICORE` on the KFC kernel TU only;
setting it globally triggers a `kernel_kfc.h` transitive double-include collision.
DEBT-20.1 (2026-05-21T20:04Z): single `-DASCENDC_MATMUL_AICORE` lands in
kfc_register_obj.h:368 NO-SYNC branch (REGIST_MATMUL_OBJ collapses to InitCurObj).
Real KFC sync requires per-compile-pass defines:
    AIC pass: -DSPLIT_CORE_CUBE=1 -> KfcServer.Init + while-isRun loop (L262)
    AIV pass: -DSPLIT_CORE_VEC=1 -> KfcCommClient + CrossCoreWaitFlag (L322)
ascendc.cmake creates two OBJECT libraries (aic_obj, aiv_obj) from the same source
set; we use CMake's `set_source_files_properties(... TARGET_DIRECTORY ...)`
(cmake >= 3.18) to scope defines per sub-target.
"""
from __future__ import annotations

import json
from pathlib import Path

from .context import BuildContext
from .contribution import CMakeContribution


def load_per_source_defines(kernel_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load per-source-file compile defines from kernel/build_overrides.json.

    Schema (DEBT-20 + DEBT-20.1, 2026-05-21):
        {
            "per_source_defines": {
                "<source_filename.cpp>": ["MACRO1=value1", "MACRO2"],
                # OR (per-compile-pass form):
                "<source_filename.cpp>": {
                    "global": ["MACRO=value"],     # applies to BOTH passes
                    "aic": ["SPLIT_CORE_CUBE=1"],  # AIC compile pass only
                    "aiv": ["SPLIT_CORE_VEC=1"]    # AIV compile pass only
                }
            }
        }

    Returns: dict[source_filename, dict["global"|"aic"|"aiv", list[str]]].
    For legacy list form, all defines land under "global". Empty dict if file
    missing or malformed (build proceeds without overrides — non-breaking).
    """
    overrides_path = kernel_dir / "build_overrides.json"
    if not overrides_path.is_file():
        return {}
    try:
        data = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    raw = data.get("per_source_defines", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, list[str]]] = {}
    for fname, defines in raw.items():
        if not isinstance(fname, str):
            continue
        norm: dict[str, list[str]] = {"global": [], "aic": [], "aiv": []}
        if isinstance(defines, list):
            # Legacy form: all defines applied globally (both passes).
            valid = [d for d in defines if isinstance(d, str) and d]
            if valid:
                norm["global"] = valid
        elif isinstance(defines, dict):
            for pass_key in ("global", "aic", "aiv"):
                vals = defines.get(pass_key, [])
                if isinstance(vals, list):
                    norm[pass_key] = [d for d in vals if isinstance(d, str) and d]
        else:
            continue
        if norm["global"] or norm["aic"] or norm["aiv"]:
            result[fname] = norm
    return result


def needs_target_directory(per_source_defines: dict) -> bool:
    """True if any source uses per-compile-pass (aic/aiv) defines — requires
    cmake >= 3.18 for `set_source_files_properties(... TARGET_DIRECTORY ...)`.
    """
    return any(
        v.get("aic") or v.get("aiv") for v in per_source_defines.values()
    )


def contribution(ctx: BuildContext) -> CMakeContribution:
    """Baseline contribution: per-source COMPILE_DEFINITIONS from build_overrides.json."""
    return CMakeContribution(defines=load_per_source_defines(ctx.kernel_dir))

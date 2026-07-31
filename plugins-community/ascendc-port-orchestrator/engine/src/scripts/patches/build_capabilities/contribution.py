# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CMakeContribution — the additive build-input a capability returns, plus merge().

A capability factory `(BuildContext) -> CMakeContribution` describes what a kernel's
compile/link/codegen needs BEYOND the baseline template. Contributions are purely
additive; `merge()` folds any number of them into one that build_ascendc's template
injects at defined points. See BUILD_CAPABILITY_EXTENSION_DESIGN.md §2B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# BuildContext is only needed for typing PrebuildStep.action; import lazily-safe.
from .context import BuildContext


@dataclass(frozen=True)
class PrebuildStep:
    """A codegen / pre-generation action run BEFORE cmake configure.

    Covers op-framework msopgen-style pre-generation and IL/IR lowering codegen
    (the two independent consumers that justify the field, design §2B).
    """

    name: str
    action: Callable[[Optional[BuildContext]], None]

    def run(self, ctx: Optional[BuildContext] = None) -> None:
        self.action(ctx)


@dataclass(frozen=True)
class CMakeContribution:
    # Additional translation units for the AscendC kernel library.
    extra_sources: tuple[Path, ...] = ()
    # Additional AscendC include-directory entries.
    include_dirs: tuple[Path, ...] = ()
    # Additional libraries linked into the Python binding.
    link_libs: tuple[str, ...] = ()
    # Per-source or global compile definitions, grouped by filename and then
    # by the global, AIC, and AIV compiler passes.
    defines: dict = field(default_factory=dict)
    # raw CMake appended at a defined injection point (escape hatch)
    cmake_fragment: str = ""
    # codegen / pre-gen actions run BEFORE cmake configure, in declared order
    prebuild_steps: tuple[PrebuildStep, ...] = ()


_PASS_KEYS = ("global", "aic", "aiv")


def _union(items) -> tuple:
    """Order-preserving de-duplicated union across an iterable of sequences."""
    out: list = []
    for seq in items:
        for x in seq:
            if x not in out:
                out.append(x)
    return tuple(out)


def _merge_defines(contribs) -> dict:
    """Merge per-source defines: for each filename, concatenate global/aic/aiv
    lists across contributions, de-duplicated, order-preserving. Matches the
    per_source_defines shape so build_ascendc's emitter is unchanged."""
    merged: dict = {}
    for c in contribs:
        for fname, passes in c.defines.items():
            dst = merged.setdefault(fname, {"global": [], "aic": [], "aiv": []})
            for key in _PASS_KEYS:
                for val in passes.get(key, []):
                    if val not in dst[key]:
                        dst[key].append(val)
    return merged


def merge(contribs) -> CMakeContribution:
    """Fold contributions: sources ∪, includes ∪, links ∪, defines merged,
    fragments concatenated, prebuild_steps in declared order.
    """
    contribs = list(contribs)
    return CMakeContribution(
        extra_sources=_union(c.extra_sources for c in contribs),
        include_dirs=_union(c.include_dirs for c in contribs),
        link_libs=_union(c.link_libs for c in contribs),
        defines=_merge_defines(contribs),
        cmake_fragment="".join(c.cmake_fragment for c in contribs),
        prebuild_steps=tuple(s for c in contribs for s in c.prebuild_steps),
    )


def run_prebuild_steps(steps, ctx: Optional[BuildContext] = None) -> None:
    """Run pre-gen steps in order, before cmake configure."""
    for step in steps:
        step.run(ctx)

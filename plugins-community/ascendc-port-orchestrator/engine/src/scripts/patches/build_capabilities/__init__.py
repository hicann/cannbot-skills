# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""build_capabilities — pluggable build-input extension for build_ascendc.py.

Design: docs/design/BUILD_CAPABILITY_EXTENSION_DESIGN.md.

Layers:
  (B) CMakeContribution + PrebuildStep interface, merge(), run_prebuild_steps(),
      and the named-capability registry (this package).
  (C) Baseline contributions (DEBT-20/20.1, DEBT-110) that always apply — the
      behavior-neutral refactor, proven byte-identical by the golden gate.

Named capabilities (STEP 2 / Phase 2): `load_capabilities()` + `kfc_bootstrap`.

Canonical-boundary invariant (§2B): a capability contributes build INPUTS only; it
MUST NOT weaken or bypass any finalize/anti-cheat, arch35-wrap, entry-point, or
precision gate.
"""
from __future__ import annotations

from .context import BuildContext
from .contribution import (
    CMakeContribution,
    PrebuildStep,
    merge,
    run_prebuild_steps,
)
from .registry import Registry, registry
from .baseline import baseline_contributions
from .named import load_capabilities

__all__ = [
    "BuildContext",
    "CMakeContribution",
    "PrebuildStep",
    "merge",
    "run_prebuild_steps",
    "Registry",
    "registry",
    "baseline_contributions",
    "load_capabilities",
]

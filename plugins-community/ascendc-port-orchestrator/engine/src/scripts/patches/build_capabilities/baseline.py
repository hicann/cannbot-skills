# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Baseline contributions — the accretions that ALWAYS apply (design §2C).

These are NOT in the named registry (they are not opt-in): every build gets them,
with their exact original firing conditions. They dogfood the CMakeContribution
interface by expressing the pre-existing DEBT-20/20.1 and DEBT-110 behavior.
"""
from __future__ import annotations

from .context import BuildContext
from .contribution import CMakeContribution
from . import baseline_cann_host_tiling, baseline_per_source_defines


def baseline_contributions(ctx: BuildContext) -> list[CMakeContribution]:
    """Ordered so the merged include_dirs match the original emission order:
    per_source_defines contributes no includes; cann_host_tiling appends
    cann_stubs after the template's base includes.
    """
    return [
        baseline_per_source_defines.contribution(ctx),
        baseline_cann_host_tiling.contribution(ctx),
    ]

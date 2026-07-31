# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Vendored cann-bench ecosystem precision grader (生态 op-gen standard).

This package contains a VERBATIM, byte-identical copy of the real cann-bench precision
grader — the actual 生态 (ecosystem) op-gen grading engine — so a5_ops grades by the SAME
source of truth the ecosystem uses, NOT a re-implementation.

Source: gitcode.com/cann/cann-bench @ 007855b (master), tree `src/kernel_eval/utils/`.
Files `compare.py` + `thresholds.py` are COPIED BYTE-FOR-BYTE (md5-verified == clone); see
PROVENANCE.md. The complete import closure of the grader is exactly these two modules
(compare.py imports only stdlib + torch + `.thresholds`; thresholds.py imports only `typing`),
so this package is fully self-contained with ZERO external cann-bench imports.

The relative `from .thresholds import (...)` inside compare.py resolves WITHIN this package
(this dir is a package via this __init__.py), which is why the vendored files needed NO edit.

Public entry points (re-exported for the adapter):
    compare_tensors(output, golden, dtype, threshold=None, native_output=None, ...)
    CompareResult
    get_threshold / get_small_value_threshold / ...  (thresholds)

Do NOT edit compare.py / thresholds.py — they are the upstream standard. Adaptation logic
lives one level up in precision_cannbot_adapter.py.
"""
from .compare import (  # noqa: F401
    compare_tensors,
    compare_with_custom_threshold,
    CompareResult,
    SingleOutputResult,
)
from .thresholds import (  # noqa: F401
    get_threshold,
    get_small_value_threshold,
    get_small_value_error,
    get_cancel_boundary,
    get_cancel_zero_threshold,
    PRECISION_THRESHOLDS,
)

# SINGLE SOURCE OF TRUTH for which native_kind may RELAX compare.py's small-value/cancellation
# carve-out (codex01 round-2). Lives HERE (vendored-adjacent SSOT, NOT inside the byte-identical
# compare.py/thresholds.py) so every consumer imports the SAME frozenset instead of hardcoding the
# string. Only a TRUE CPU-same-precision native qualifies (mirrors cannbot: compare.py has no
# fp32-fallback concept; a different-precision baseline is native=None→strict). Everything else
# (untagged / bare-list / cpu_fp32_fallback / non-dict) is treated as native ABSENT (fail-closed).
NATIVE_CARVEOUT_WHITELIST = frozenset({"cpu_same_precision"})

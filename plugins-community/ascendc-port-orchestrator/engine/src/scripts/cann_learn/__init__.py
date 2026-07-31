# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CANN-source-learn carve-out (P0x v2 — DEBT-077).

Deterministic scanners + utilities for the aog-cann-learner sub-agent and
aog-knowledge-maintain Mode 5. Each scanner produces a numeric/boolean score
that the skill caller validates BEFORE merging any candidate KB entry.

Modules:
    identifier_scanner — C34a: deterministic CANN-vs-public-API denylist scan
    compile_gate       — C34b: bishengir-clang -fsyntax-only against public headers
    copy_shape         — C34c: token n-gram contiguous-overlap detection
    kb_overlap         — C35:  reason-code-based overlap with existing KB

See docs/design/KB_DESIGN_NOTES.md#cann-learn-agent-design-v2 for architecture + threat model.
"""
__version__ = "0.1.0-dev"

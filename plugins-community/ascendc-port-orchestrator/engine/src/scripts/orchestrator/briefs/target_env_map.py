# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Single source of truth for the target → env-key prefix map.

Both the briefs config layer (briefs/_common.py) and the capability gates
(a5_target_capability.py) resolve TARGET names to env-key prefixes. The map
lives HERE and both layers import it; it must never be copied again
(the anti-drift test pins both import sites to this single object). There is
no import-cycle obstacle: briefs/__init__.py is empty, this module is
stdlib-only, and top-level modules already import briefs submodules
(agent_dispatch.py imports briefs._common at module level).

The map is EXPLICIT, never derived by uppercasing: "310p" must resolve to
ASCEND310P_*, never the illegal "310P_*" shell identifier — a naive
f"{target.upper()}_..." lookup always misses and silently falls back to
A5_* (the DEBT-336 bug class).

tests/ut/test_port_destinations_310p.py::test_prefix_map_anti_drift_between_copies
pins both import sites to this single object.
"""
from __future__ import annotations

TARGET_ENV_PREFIXES = {
    "a5": "A5",
    "a3": "A3",
    "a2": "A2",
    "310p": "ASCEND310P",
}

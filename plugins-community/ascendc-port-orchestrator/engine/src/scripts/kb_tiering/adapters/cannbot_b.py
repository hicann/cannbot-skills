# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""cannbot b-tier (official-canonical) provider = the bundled `references/` KB.

cannbot's b-tier IS the a5_ops `references/` KB bundled verbatim (same KB_INDEX + OL/EC/PB
format). So the b-adapter IS main's `A5opsProvider` (vendored byte-identical as `adapter_a5ops.py`),
just pointed at the plugin-bundled `src/skills/references/KB_INDEX.md`. Read + promotion-gate-only
(`admit` rejects direct writes; `put` is the promote(c→b) landing) — matching design §2 (b =
release-controlled, read-only in deployment). This is retirement-clean: when a5_ops-core lands, the
re-sync is a code-swap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# adapter_a5ops.py (vendored) does a bare `from interface import …`; put kb_tiering/ on the path so
# that resolves, exactly as it does in the a5_ops layout.
_KB_TIERING = Path(__file__).resolve().parents[1]
if str(_KB_TIERING) not in sys.path:
    sys.path.insert(0, str(_KB_TIERING))

from adapter_a5ops import A5opsProvider  # noqa: E402  (vendored byte-identical from a5_ops)

# default bundled b-tier KB root (2026-07-05: relocated to <plugin_root>/kb/).
# cannbot_b.py lives at <plugin_root>/engine/src/scripts/kb_tiering/adapters/, so
# parents[5] == plugin_root.
_DEFAULT_REFERENCES = Path(__file__).resolve().parents[5] / "kb"


def make_cannbot_b(references_dir: Optional[str] = None, inbox: Optional[str] = None) -> A5opsProvider:
    """cannbot b-tier provider over the bundled references KB.

    Factory mirroring make_cannbot_c / make_autoport_c — for
    `build_arbiter(tmp, cannbot_c=…, cannbot_b=make_cannbot_b())`.
    """
    ref = Path(references_dir) if references_dir else _DEFAULT_REFERENCES
    return A5opsProvider(kb_index=ref / "KB_INDEX.md", inbox=Path(inbox) if inbox else None)

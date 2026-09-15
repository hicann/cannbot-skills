# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read-path bridge: c-tier brief injection over the Arbiter (the keyword→signature bridge).

Main's gap-map finding: the engine's brief injection is a *keyword filter returning MANY rows*
(`kb_inject_filtered`), whereas `Arbiter.resolve()` is *single best-jaccard*. A naive drop-in
returns MISS. So the bridge lives HERE (adapter/Arbiter side — NOT the frozen `interface.py`):
brief injection consumes each tier's `index_rows()` + a keyword filter, earlier tiers shadowing
later ones on the same hard-key (canonical-conflict still surfaces via the Arbiter separately).

OKF-only 迁移（2026-08）：legacy b-tier（内置索引 provider）已摘除，bundled 知识即 kb/okf；
`build_arbiter` 现在只组装 user-kb（c-tier），a-tier = 社区 skills 仍由引擎 CBA 路由单独注入。
`inject_for_brief` 是 brief 需要的 multi-row read。Hot-path integration into
`briefs/_common.py::kb_inject_filtered` is config-gated (only when a c-tier user_kb is active).
"""
from __future__ import annotations

from typing import Optional

from .interface import Arbiter, KBProvider, full_sig, jaccard
from .adapters.cannbot_c import make_cannbot_c, kb_write_root


def build_arbiter(user_kb_root: Optional[str] = None,
                  cannbot_c: Optional[KBProvider] = None) -> Arbiter:
    """Ordered Arbiter([c]) — c 是唯一本地 tier（b-tier 已随 OKF-only 摘除；a-tier = community
    skills, injected separately by the engine's CBA routing). Provider can be passed in (for
    tests) or built from the user_kb root.
    """
    c = cannbot_c or make_cannbot_c(user_kb_root)
    # wire the provider list into c's admit-gate delegation (tombstone check spans the tier set)
    configure_providers = getattr(c, "set_gate_providers", None)
    if callable(configure_providers):
        configure_providers([c])
    elif hasattr(c, "_providers"):
        setattr(c, "_providers", [c])
    return Arbiter([c])


def _row_text(row) -> str:
    """Normalized searchable text for an index row (works for Entry or dict rows)."""
    if hasattr(row, "claim"):                       # Entry
        return f"{row.id} {row.key} {row.claim} {' '.join(str(v) for v in row.scope.values())}"
    return " ".join(str(row.get(k, "")) for k in ("id", "key", "kind"))   # dict row (c-tier index)


def _row_key(row) -> str:
    return getattr(row, "key", None) or (row.get("key", "") if isinstance(row, dict) else "") or getattr(row, "id", "")


def inject_for_brief(arbiter: Arbiter, keywords: Optional[list[str]] = None,
                     min_kw_hit: float = 0.1) -> list:
    """Multi-row read for brief injection (the bridge — NOT single resolve).

    For each tier in precedence order, take its `index_rows()`, keep rows matching any keyword
    (substring OR signature-overlap), and merge with earlier tiers SHADOWING later ones on the
    same hard-key. Returns injectable rows tagged with their tier, c-first. Empty keywords →
    all rows (precedence-merged).
    """
    kw_sig = full_sig(" ".join(keywords)) if keywords else ""
    seen_keys: set = set()
    out: list = []
    for p in arbiter.providers:                     # ordered: c (index 0) shadows later tiers
        for row in p.index_rows():
            text = _row_text(row).lower()
            if keywords:
                hit = any(kw.lower() in text for kw in keywords) or (
                    kw_sig and jaccard(kw_sig, full_sig(text)) >= min_kw_hit)
                if not hit:
                    continue
            k = _row_key(row)
            if k and k in seen_keys:                 # earlier tier already provided this lesson
                continue
            if k:
                seen_keys.add(k)
            out.append({"tier": getattr(p, "tier", "?"), "row": row})
    return out

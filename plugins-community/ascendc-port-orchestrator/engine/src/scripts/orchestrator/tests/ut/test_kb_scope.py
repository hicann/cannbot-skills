# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""kb_scope 的直接覆盖（SoC-scope 谓词，DEBT-208/222）。

自 test_kb_inject_filtered.py 迁入（2026-08 OKF-only 迁移：该文件主体测的是
已删除的 legacy `kb_inject_filtered` / 旧索引解析 / legacy manifest 内的
DEBT-222 过滤，均随 OKF-only 退役；本文件保留其中仍有效的 kb_scope 行为覆盖，
并把路径形态更新到 OKF 布局）。条目级 scope 的端到端 pin 仍在
test_kw_brief_soc_scope_debt208.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from briefs.kb_scope import (  # noqa: E402
    kb_entry_soc_families,
    kb_file_applies_to_target,
    kb_file_soc_families,
)

# OKF patterns 下的真实 fixture：
#   gmm_swiglu_quant_a8w8_class_template.md — header blockquote 内 applies_to: soc=Ascend950PR/V351（a5-only）
#   hkv_patterns.md                         — header 无 applies_to（中性）
_A5_ONLY_TEMPLATE = "gmm_swiglu_quant_a8w8_class_template.md"
_NEUTRAL_TEMPLATE = "hkv_patterns.md"


@pytest.mark.parametrize("form", [
    "okf/reference/porter/patterns/gmm_swiglu_quant_a8w8_class_template.md",   # OKF canonical
    "patterns/domains/gmm_swiglu_quant_a8w8_class_template.md",         # raw classifier recommendation
    "domains/gmm_swiglu_quant_a8w8_class_template.md",                  # bare form
    "src/skills/references/target/ascendc/patterns/domains/gmm_swiglu_quant_a8w8_class_template.md",  # full/abs
])
def test_kb_file_applies_to_target_form_agnostic(form):
    """kb_file_applies_to_target resolves an a5-only template in EVERY path form
    the compose path could produce — not just the canonical one. A form-sensitive
    resolver would make the filter inert (theater) for the un-normalized forms.
    """
    assert kb_file_applies_to_target(form, "a5") is True     # a5 keeps
    assert kb_file_applies_to_target(form, "a3") is False    # a3 drops (a5-only)


def test_kb_file_applies_to_target_fail_open():
    # untagged / unknown file → keep (fail-open)
    assert kb_file_applies_to_target("okf/reference/porter/patterns/does_not_exist.md", "a3") is True
    # 已转卡、不在 reference/patterns 下的旧 domain 名 → 解析不到文件 → keep
    assert kb_file_applies_to_target("patterns/domains/sort.md", "a3") is True
    # neutral template（header 无 applies_to）→ kept for a3
    assert kb_file_applies_to_target(f"patterns/{_NEUTRAL_TEMPLATE}", "a3") is True
    # unknown target → keep (never silently drop)
    assert kb_file_applies_to_target(f"okf/reference/porter/patterns/{_A5_ONLY_TEMPLATE}", "whoknows") is True


def test_kb_entry_soc_families_reads_okf_cards(monkeypatch):
    """条目 scope 在 legacy target 树删除后仍可读：OKF 卡片以前文
    `original_id: PB-34` + `description:`/正文中的 `applies_to: soc=` 提供
    同样的机器可读 scope。把扫描目录限到 okf 子树，模拟 legacy 删除后的形态。
    """
    import briefs.kb_scope as ks

    monkeypatch.setattr(ks, "_KB_SCAN_DIRS", ("okf",))
    assert kb_entry_soc_families("PB-34") == {"V220"}
    # 未命中条目 → None（上游 fail-open 保留注入）
    assert kb_entry_soc_families("PB-99999") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_applies_to_survives_frontmatter_that_pushes_it_past_the_window():
    """REGRESSION (2026-09-05): frontmatter must not push `applies_to` out of the scan.

    `fa_class_template.md` is an A5-only template whose own body says handing it to a
    220x/A3 worker mis-scopes it. When the OKF migration added ~12 lines of frontmatter,
    its `applies_to: soc=` line moved to body-relative position 21 — past a plain
    `lines[:20]` window — so `kb_file_soc_families` returned None and
    `kb_file_applies_to_target(..., "a3")` FAILED OPEN (returned True).

    This is silent: no lint error, no other test failure, and the brief still renders.
    The fixture is deliberately THIS file (not a synthetic one and not a sibling whose
    `applies_to` happens to sit higher) because the bug only shows on a card whose
    frontmatter actually pushes the field past the window.
    """
    rel = "okf/reference/porter/patterns/fa_class_template.md"
    assert kb_file_soc_families(rel) == {"V351"}, (
        "fa_class_template declares soc=Ascend950PR; None here means the header scan "
        "lost it (most likely frontmatter grew past the window) and the SoC filter "
        "has silently degraded to fail-open"
    )
    assert kb_file_applies_to_target(rel, "a3") is False
    assert kb_file_applies_to_target(rel, "a5") is True


def test_header_zone_window_is_measured_from_end_of_frontmatter():
    """The window must not be consumed by frontmatter, however long it grows."""
    from briefs.kb_scope import _header_zone, _TEMPLATE_HEADER_LINES

    fm = ["---"] + [f"field_{i}: v" for i in range(30)] + ["---"]
    body = [f"body {i}" for i in range(_TEMPLATE_HEADER_LINES + 5)]
    zone = _header_zone(fm + body)
    assert zone[0] == "body 0", "scan must start at the first BODY line"
    assert len(zone) == _TEMPLATE_HEADER_LINES

    # no frontmatter at all -> window starts at line 0
    assert _header_zone(body)[0] == "body 0"

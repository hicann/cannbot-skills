# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate every tag→KB reference cited in SKILL.md §Step 2 tables exists in plugin kb/okf/.

OKF-only 布局：OL/PB/EC/P-P/CAND 引用按三级解析（卡文件名前缀 → frontmatter
`original_id:` → 正文并入提及）；裸路径相对 kb/ 校验。legacy `kb/target/ascendc/`
布局已退役，残留的 `patterns/...` 引用会被判 FAIL。

Run after SKILL.md edits to ensure the curated tables don't reference dead anchors.
Exit 0 if all references resolve, 2 if any missing.

Usage:
    python3 skills/aog-op-classify/validate_kb_paths.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[2]
_REFS = _PLUGIN_ROOT / "kb"
# OKF-only 迁移 (2026-08): legacy `kb/target/ascendc/` 布局已退役（OL/PB/EC 卡片化到
# kb/okf/runbooks/，domains 卡片化到 runbooks/operator-optimization/，未转卡 patterns
# 移到 kb/okf/reference/porter/patterns/）。校验以 kb/okf/ 为唯一真值。
_OKF = _REFS / "okf"
_SKILL_MD = _HERE.parent / "SKILL.md"

# ID prefix → OKF 卡文件名前缀（ol-114-two-pass-....md / p-p61-....md / cand-pp79-....md）
_ID_PREFIX = {"OL-": "ol", "PB-": "pb", "EC-": "ec", "P-P": "p-p", "CAND-": "cand"}


def extract_kb_paths(skill_md_text: str) -> list[str]:
    """Pull every `OL-XXX` / `PB-XXX` / `EC-XXX` / `P-PXX` / `CAND-*` reference
    out of the SKILL.md tag→KB tables. Also pull bare path references like
    `okf/runbooks/.../foo.md` (and legacy `patterns/...` stragglers, which now FAIL).
    """
    paths: set[str] = set()
    # OL/PB/EC/P-P/CAND-* anchor refs
    for m in re.finditer(r"(OL-\d+|PB-\d+|EC-\d+|P-P\d+|CAND-[A-Z0-9]+)", skill_md_text):
        paths.add(m.group(1))
    # Bare relative paths (foo/bar.md) — okf/ paths validate against kb/;
    # legacy patterns/ paths are extracted so they fail loudly instead of being ignored
    for m in re.finditer(r"`((?:okf|patterns)/[a-zA-Z0-9_/.\-]+\.md(?:#[A-Z0-9\-]+)?)`", skill_md_text):
        paths.add(m.group(1))
    return sorted(paths)


def _okf_corpus() -> dict[str, str]:
    """path → text for every card under kb/okf (read once, reused per anchor)."""
    corpus: dict[str, str] = {}
    if _OKF.is_dir():
        for p in sorted(_OKF.rglob("*.md")):
            try:
                corpus[str(p)] = p.read_text(errors="replace")
            except OSError:
                continue
    return corpus


def verify_anchor_exists(anchor: str, corpus: dict[str, str]) -> tuple[bool, str]:
    """OKF 布局校验：
    - Level 1: 存在 `<id 小写>-*.md` 卡（文件名前缀匹配）。
    - Level 2: 某卡 frontmatter `original_id: <ID>` 匹配（改名/换 slug 的卡）。
    - Level 3: ID 在 kb/okf 正文出现（旧条目被并入他卡 —— 知识仍在，但无独立卡）。
    三级皆无 → FAIL（死引用，需改 SKILL.md 或补卡）。
    - 裸路径（okf/...）→ 相对 kb/ 校验文件存在；带 #anchor 时查文件内容。
    """
    if "/" in anchor:  # bare path like okf/runbooks/.../foo.md
        path_part, _, frag = anchor.partition("#")
        target = _REFS / path_part
        if not target.exists():
            return False, (
                f"file not found under plugin kb/: {path_part} "
                f"(legacy patterns/... 路径已退役 — 域文件卡片化到 kb/okf/runbooks/，"
                f"未转卡 pattern 移到 kb/okf/reference/porter/patterns/)"
            )
        if frag and frag not in target.read_text(errors="replace"):
            return False, f"anchor `#{frag}` not found in {target.name}"
        return True, str(target)

    for prefix, _ in _ID_PREFIX.items():
        if anchor.startswith(prefix):
            break
    else:
        return False, f"unrecognized anchor format: {anchor}"

    # Level 1: filename prefix
    stem = anchor.lower()
    for p in corpus:
        if Path(p).name.startswith(stem + "-") or Path(p).name == stem + ".md":
            return True, p
    # Level 2: frontmatter original_id
    marker = f"original_id: {anchor}"
    for p, text in corpus.items():
        if marker in text:
            return True, f"{p} (via original_id)"
    # Level 3: content mention (条目并入他卡)
    for p, text in corpus.items():
        if anchor in text:
            return True, f"merged — no dedicated card; mentioned in {Path(p).name}"
    return False, f"no OKF card, original_id, or content mention for {anchor}"


def main() -> int:
    if not _SKILL_MD.exists():
        print(f"ERROR: SKILL.md not found at {_SKILL_MD}", file=sys.stderr)
        return 2
    if not _OKF.is_dir():
        print(f"ERROR: OKF KB root not found at {_OKF}", file=sys.stderr)
        return 2
    text = _SKILL_MD.read_text()
    anchors = extract_kb_paths(text)
    corpus = _okf_corpus()
    print(f"Found {len(anchors)} unique KB references in SKILL.md")
    failures: list[tuple[str, str]] = []
    for anchor in anchors:
        ok, detail = verify_anchor_exists(anchor, corpus)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {anchor:<25} {detail}")
        if not ok:
            failures.append((anchor, detail))
    if failures:
        print(f"\n{len(failures)} broken references:")
        for anchor, detail in failures:
            print(f"  - {anchor}: {detail}")
        return 2
    print("\nAll references valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

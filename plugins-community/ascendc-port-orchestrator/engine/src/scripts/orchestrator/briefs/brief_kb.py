#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""brief_kb — KB entry model + chip/op-filtered KB injection (KBEntry, kb_inject_filtered,
kb_manifest_block + parsing helpers), extracted from briefs/_common.py
(behavior-neutral god-file decomposition, 2026-07-05). Leaf module: stdlib-only, imports
nothing from _common. _common re-imports these so its importers + `from briefs._common import
kb_inject_filtered/KBEntry/...` are unaffected."""
from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
_FORCED_ARCH_TAGS = {"SIMT", "SIMD"}


@dataclass
class KBEntry:
    """One KB entry with parsed tags and resolved path."""
    entry_id: str
    path: str
    anchor: str
    hook: str
    tags: dict
    level: str


def kb_inject_filtered(
    target: dict,
    *,
    keywords: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list[KBEntry]:
    """Tag-aware KB filter — returns entries that apply to the target env.

    Replacement for ad-hoc keyword grep in worker/probe/optimizer briefs.
    Uses kb_schema.kb_entry_applies() for deterministic filter semantics.

    Args:
        target: target env dict (e.g. {"arch_family": "arch35", "soc": "Ascend950PR"})
        keywords: optional list of keywords to further narrow results
        categories: optional list of KB categories to restrict to
            (e.g. ["EC", "PB", "OL"]). Default: all categories.

    Returns:
        list of KBEntry sorted by specificity (most specific first).
        Empty list if no entries match.

    Example:
        target = _target_for_opgen_mode("port_a3_to_a5")
        entries = kb_inject_filtered(target, keywords=["DataCopy", "softmax"])
        for e in entries:
            print(f"{e.entry_id}: {e.hook}")
    """
    import sys as _sys
    from pathlib import Path as _Path

    _HERE = _Path(__file__).resolve().parent
    _SCRIPTS = _HERE.parent.parent  # src/scripts/
    if str(_SCRIPTS) not in _sys.path:
        _sys.path.insert(0, str(_SCRIPTS))

    try:
        from kb_schema import kb_entry_applies, sort_by_specificity, parse_tag_column, alias_match
    except ImportError:
        return []

    # 2026-07-05: KB relocated to <plugin_root>/kb/. _SCRIPTS == engine/src/scripts,
    # so _SCRIPTS.parent.parent == engine/ and its parent is <plugin_root>.
    _KB_INDEX = (
        _SCRIPTS.parent.parent.parent / "kb" / "KB_INDEX.md"
    )
    if not _KB_INDEX.is_file():
        return []

    rows = _parse_kb_index_rows(_KB_INDEX)
    applicable: list[KBEntry] = []

    for row in rows:
        if not kb_entry_applies(row["tags"], target):
            continue
        if categories and row.get("category", "") not in categories:
            continue
        if keywords:
            # alias_match: a keyword in any name family (V300/arch35/Ascend950PR)
            # recalls entries written under any other variant; word-boundary
            # matching guards the V300/V300x cross-chip collision (2026-05-28).
            search_text = f"{row['id']} {row['hook']} {' '.join(row['tags'].values())}"
            if not any(alias_match(kw, search_text) for kw in keywords):
                continue
        applicable.append(KBEntry(
            entry_id=row["id"],
            path=row["path"],
            anchor=row.get("anchor", ""),
            hook=row["hook"],
            tags=row["tags"],
            level=row.get("level", "L0"),
        ))

    # Sort by specificity (most specific first)
    tagged = [(e.tags, e) for e in applicable]
    return [e for _, e in sort_by_specificity(tagged)]


def _target_for_opgen_mode(mode: str) -> dict:
    """Return a default target dict for a given opgen_mode.

    Both supported modes produce arch35 AscendC kernels.
    """
    if mode in ("port_a3_to_a5", "backward"):
        return {"paradigm": "ascendc", "arch_family": "arch35"}
    return {"paradigm": "ascendc", "arch_family": "any"}


def _parse_kb_index_rows(kb_index_path: Path) -> list[dict]:
    """Parse KB_INDEX.md via the shared kb_schema.parse_kb_index_row, which
    tolerates BOTH the 3-col (`| [ID](p) | hook | L1 |`) and 4-col tagged
    formats. The previous local 4-col-only regex dropped every real 3-col
    row → brief injection recalled ~0 entries (NODE-21 Phase C was inert).
    """
    from kb_schema import parse_kb_index_row
    rows: list[dict] = []
    for line in kb_index_path.read_text(encoding="utf-8").split("\n"):
        parsed = parse_kb_index_row(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _detect_forced_architecture(workspace: Optional[Path]) -> Optional[str]:
    """Return the FORCED architecture ("SIMT" / "SIMD") if op_classification.json
    carries a forced-architecture marker, else None.

    Robust to the three marker forms above. Returns None on any missing /
    corrupt / unmarked classification (non-forced ops author normally).
    """
    if workspace is None:
        return None
    cls_path = workspace / "op_classification.json"
    if not cls_path.is_file():
        return None
    try:
        data = json.loads(cls_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # (1) explicit boolean keys
    if data.get("force_simt") is True:
        return "SIMT"
    if data.get("force_simd") is True:
        return "SIMD"
    # (2) explicit forced_arch string
    fa = data.get("forced_arch")
    if isinstance(fa, str) and fa.strip().upper() in _FORCED_ARCH_TAGS:
        return fa.strip().upper()
    # (3) forced-arch convention: a bare SIMT/SIMD tag in op_class_tags
    tags = data.get("op_class_tags") or []
    if isinstance(tags, list):
        tag_set = {str(t).strip().upper() for t in tags}
        forced = tag_set & _FORCED_ARCH_TAGS
        # If BOTH appear (shouldn't), don't guess — treat as non-forced.
        if len(forced) == 1:
            return next(iter(forced))
    return None


def kb_manifest_block(
    op: str,
    workspace: Optional[Path] = None,
    target: str = "a5",
    force_legacy_kb: bool = False,
) -> str:
    """KB Manifest section — uses op_taxonomy for deterministic lookup.

    ``force_legacy_kb`` is the per-brief escape hatch for a brief that curates
    exact legacy anchors which generic OKF retrieval cannot reproduce. The
    default remains OKF; this does not change the process-wide setting.

    Codex C1: Tier 1 Python template (90%) + Tier 2 KB lookup by op-class
    tag (10%). NO LLM in brief construction.

    P0aaj (2026-05-06): pass workspace path so op_taxonomy can run
    source-scan auto-tag inference for non-benchmark ops (cross-generation,
    backward, custom, sparse-attention) that aren't in the hand-curated OP_TAGS dict.

    P0abj (2026-05-08): pass target so the manifest's hardware-spec entry
    dispatches to the correct chip's ref doc. Pre-fix the manifest was
    target-blind and always loaded `hardware/target/ascend950pr.md` — A3/A2
    ops on DS env got A5 specs (wrong UB size, AIV count, atomics info).
    Defaults to "a5" for callers not yet updated; once all caller sites
    pass target, the default can be removed.
    """
    from briefs.op_taxonomy import lookup, validate_manifest_paths
    t = lookup(op, workspace=workspace, target=target)

    # Forced-architecture suppression: when the op's architecture is fixed at classification time
    # (forced-SIMT / forced-SIMD marker in op_classification.json), kw must
    # NOT be pointed at the SIMT_VS_SIMD decision tree — the choice is already
    # made; presenting the decision-tree guidance invites the override that
    # this fix prohibits. Drop the SIMT_VS_SIMD_DECISION reference from the
    # manifest so it is not loaded. The kw_brief `_forced_architecture_block`
    # carries the matching positive instruction ("implement the forced arch,
    # do NOT run the decision tree"). Non-forced ops are unaffected.
    # `_detect_forced_architecture` is LOCAL to this module (kept out of
    # kw_brief to avoid a _common↔kw_brief circular import — review 2026-06-16).
    if _detect_forced_architecture(workspace) is not None:
        t.kb_sections = [
            s for s in t.kb_sections if "SIMT_VS_SIMD_DECISION" not in s
        ]

    # P0abj followup (2026-05-09, user request): fail-fast if any manifest
    # entry is missing on disk. Without this, brief generation succeeds
    # silently and the agent gets file-not-found errors mid-Read after
    # the (expensive) spawn already happened. This re-raises with a
    # precise actionable message naming the missing path + references-root.
    validate_manifest_paths(t.kb_sections)

    # P0abq (2026-05-11): split manifest into two tiers to avoid triggering
    # model-server 500 errors from bulk-loading 5700+ lines of KB upfront.
    # Tier-1 (small index files, <500 lines each): load immediately.
    # Tier-2 (OPERATIONAL_KNOWLEDGE.md, 3859 lines): search on demand via
    # grep for op-specific OL entries — 95% of entries are irrelevant for
    # any single op class. Per-tag anchor references (e.g. OL-103 for
    # transcendental) are preserved as targeted reads.
    # P88 KB reorg: large-file detection works on substring match
    # ("OPERATIONAL_KNOWLEDGE.md" substring in "target/ascendc/OPERATIONAL_KNOWLEDGE.md").
    _TIER2_LARGE = {"OPERATIONAL_KNOWLEDGE.md"}

    # P88 KB reorg (2026-05-15): resolve legacy bare paths to new layout
    # BEFORE inserting into agent brief. Without this, agent sees stale
    # paths and Read/grep fails to find file. DS regression catch on
    # 22_Nonzero cold-start.
    from briefs.op_taxonomy import resolve_legacy_kb_path

    def _resolve_section(s: str) -> str:
        """Apply legacy path rewrite to either bare path or path#anchor form."""
        if "#" in s:
            base, anchor = s.split("#", 1)
            return f"{resolve_legacy_kb_path(base)}#{anchor}"
        return resolve_legacy_kb_path(s)

    resolved_sections = [_resolve_section(s) for s in t.kb_sections]

    # DEBT-222 (2026-07-17): honor each DOMAIN TEMPLATE's own machine-readable
    # `applies_to: soc=`. A domain template (patterns/domains/*.md) is arch-scoped
    # device knowledge; listing an a5-only template (cooperative / gmm / cube_vector
    # _fusion) in a 220x/a3 worker's manifest is the mis-delivery owner flagged
    # ("这些 template 连目标 soc 或者架构 tag 都没有 = 巨大的 bug"). This applies the
    # DEBT-208 principle ("honor applies_to: soc=") to the template layer, reusing
    # kb_scope (NOT a second parser). FAIL-OPEN by construction: a section is dropped
    # ONLY on a positive machine-readable exclusion (concrete soc= scope + known
    # target family outside it); untagged / soc=all / unknown target all keep the
    # template — this can only narrow an over-delivery, never invent an under-block.
    # Non-domain-template sections (HW spec, PATTERN_INDEX, OL/PB anchors) are
    # untouched. For target=a5 (the default) nothing is dropped — a5-only templates
    # cover a5 and neutral templates are soc=all — so a5 briefs stay byte-identical.
    try:
        from briefs.kb_scope import kb_file_applies_to_target

        def _domain_template_in_scope(sec: str) -> bool:
            base = sec.split("#", 1)[0]
            # Match a whole-file domain template in ANY path form the compose path
            # can emit — `…/domains/X.md` or bare `domains/X.md` (not the fa_class/
            # subdir, whose `domains/fa_class/X.md` has a slash after domains/).
            if not re.search(r"(?:^|/)domains/[^/]+\.md$", base):
                return True
            return kb_file_applies_to_target(base, target)

        resolved_sections = [s for s in resolved_sections if _domain_template_in_scope(s)]
    except Exception:
        pass  # fail-open: never break brief construction on the scope filter

    tier1 = []              # small index files — load all upfront
    tier2_file_anchors: dict[str, list[str]] = {}  # file → [anchor1, anchor2, ...]
    for s in resolved_sections:
        if "#" in s:
            base, anchor = s.split("#", 1)
            tier2_file_anchors.setdefault(base, []).append(anchor)
        elif any(large in s for large in _TIER2_LARGE):
            tier2_file_anchors.setdefault(s, [])
        else:
            tier1.append(s)

    tier1_md = "\n".join(f"  - {s}" for s in tier1) if tier1 else "  (none — all sections are search-on-demand)"

    # Build Tier 2 with specific anchor hints when available
    tier2_lines = []
    all_anchors = []
    for f in sorted(tier2_file_anchors):
        anchors = tier2_file_anchors[f]
        all_anchors.extend(anchors)
        if anchors:
            tier2_lines.append(f"  - {f}  # grep-read: {', '.join(anchors)}")
        else:
            tier2_lines.append(f"  - {f}")
    if all_anchors:
        # Add explicit anchor directory for grep convenience
        tier2_lines.append("")
        tier2_lines.append("  Targeted grep commands (run these instead of loading the whole file):")
        for a in all_anchors:
            tier2_lines.append(f"    grep -n '{a}' kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md")
    tier2_md = "\n".join(tier2_lines) if tier2_lines else "  (none)"

    tags_str = ', '.join(t.tags) if t.tags else 'UNTAGGED (default safe set only)'
    _manifest = f"""# KB MANIFEST — two-tier loading (P0abq: avoid bulk-load 500 errors)

Op-class tags: {tags_str}

## Tier 1 — LOAD IMMEDIATELY before Phase A (small index files, ~1500 lines total)
Paths are relative to kb/:
  - shared/ANTI_PRESSURE_PROTOCOLS.md   # MANDATORY — load before any decision
  - hardware/HIASCEND_DOC_URLS.md       # MANDATORY for AscendC ops — vendor docs URL registry.
    hiascend.com is JS-rendered, so use playwright (browser_navigate +
    browser_evaluate) rather than WebFetch. Before guessing CANN API behavior,
    consult this file first to find and fetch the canonical document page.
{tier1_md}

## Tier 2 — SEARCH ON DEMAND (large reference files — grep, don't bulk-load)
These files are large (OPERATIONAL_KNOWLEDGE.md is 3859 lines). Do NOT load them
in full. Instead, after reading Tier 1, identify the specific OL / PB / P-P
entries relevant to your op class from KB_INDEX.md, then grep for them:
  - `grep -n "OL-<N>" kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md`
  - Read ONLY the matched sections (typically 20-50 lines each)

Tier-2 files (search on demand):
{tier2_md}

If per-tag anchors reference specific sections (e.g. `OPERATIONAL_KNOWLEDGE.md#OL-103`),
read only those specific sections via grep, not the whole file.

If untagged, scan KB_INDEX.md §By Symptom for this op's failure pattern.

# ANTI-PRESSURE CHECKPOINT (cite at decision points)
Before emitting any handoff line, re-read the relevant Px from
`ANTI_PRESSURE_PROTOCOLS.md`:
- Before `→ orchestrator: done` → P1 + P7
- Before `→ orchestrator: PARTIAL_PERSIST` → P5 + P7
- Before spawning a sub-agent → P3 + P8
- Before writing "expected failure" / "structural ceiling" → P5
- Before `nohup &` / `Bash & disown` / direct Agent (skipping Skill) → P8
- Before manual workaround instead of fixing the script → P6
"""
    # c>b>a read-path: prepend user-local (c-tier) lessons AHEAD of the b-tier knowledge,
    # and append the CBA tier-a required-routes block AFTER. Both are config/workspace-gated
    # (empty unless active) → the default-b path stays byte-unchanged.
    #
    # OKF is the DEFAULT b-tier and is mutually exclusive with the legacy
    # KB_INDEX manifest. Empty/error retrieval is loud and never silently falls
    # back, otherwise a broken external knowledge dependency looks healthy.
    if _okf_enabled() and not force_legacy_kb:
        okf = _okf_reference_block(op, workspace, target)
        okf_body = okf if okf else (
            "# ⚠️ OKF 独占模式(默认开)但 knowledge-query 无返回——索引未 build / 无命中 / 检索失败。\n"
            "# 本次简报无 b 层知识,且**不回退**旧 KB_INDEX。请先 `engine/src/scripts/okf/okf_kb.sh build`\n"
            "# 并确认检索命中后再重跑;若确要用旧知识体系,显式设 `ASCENDC_PORT_OKF=0`。\n\n"
        )
        b_tier = okf_body + _kb_discipline_scaffold(target)
    else:
        b_tier = _manifest
    return (
        _c_tier_lessons_block(op, workspace, target)
        + b_tier
        + _cba_tier_a_routes_block(op, workspace)
    )


# ── §5.2 CBA tier-a routing (codex design) ──────────────────────────────────
# Topics whose authoritative source is a cannbot community skill (tier-a), NOT the
# plugin's bundled b-tier. When an op REQUIRES such a topic, the worker brief emits a
# MANDATORY route: invoke the named cannbot Skill + write a provenance marker; the
# CBA route gate (validation/cba_route_gate.py) fails the run if the Skill wasn't invoked.
# Per-op required routes are declared in workspace/{op}/.cba_required_routes.json
# (list of {"topic","skill","reference_hint"}). Empty/absent => no tier-a block (byte-identical brief).
def _cba_tier_a_routes_block(op, workspace=None) -> str:
    import json as _json
    if workspace is None:
        return ""
    rf = workspace / ".cba_required_routes.json"
    if not rf.exists():
        return ""
    try:
        routes = _json.loads(rf.read_text())
    except Exception:
        return ""
    if not routes:
        return ""
    _write_a_tier_load_record(op, workspace, routes)   # §5.2 C2: objective harness LOAD record
    lines = ["", "# CBA TIER-A ROUTES (MANDATORY — §5.2 c>b>a)",
             "以下 topic 的权威知识在 **cannbot 社区 skill（tier-a）**、不在自带 b-tier KB。"
             "生成中需要该 topic 时，**必须用 Skill 工具 invoke 指定的 cannbot skill**（仅 b-tier 不够）；"
             "invoke 后在 PROGRESS 写 provenance marker。**若该 Skill 未被 invoke，本 run 视为 CBA_MISSING_A_TIER（不放行）。**", ""]
    for r in routes:
        t = r.get("topic", "?")
        sk = r.get("skill", "?")
        hint = r.get("reference_hint", "")
        lines.append(f"- topic: `{t}`")
        lines.append(f"  - invoke Skill: **`{sk}`**" + (f"（参考 {hint}）" if hint else ""))
        lines.append(f"  - 写 marker: `CBA_USED tier=a topic={t} skill={sk}`")
        lines.append(f"  - 若 invoke 失败/不可用 → 停并报 `CBA_MISSING_A_TIER topic={t}`")
    return "\n".join(lines) + "\n"


def _write_a_tier_load_record(op, workspace, routes) -> None:
    """§5.2 C2 — objective harness-emitted LOAD record: which tier-a skills were
    SURFACED to the worker (surface == load). Written at brief-build time so the
    LOAD evidence is independent of the worker's self-report (USE evidence is the
    CBA_USED marker + the cba_route_gate transcript parse). Mirrors the finalize
    provenance-node contract: idempotent (preserves created_ts across re-briefs)
    and fail-open (never breaks brief construction).

    Schema (workspace/{op}/a_tier_manifest.json):
      {op, schema_version, created_ts, surfaced:[{topic,skill,kind,reason,reference_hint}]}
    Finalize merges a compact form into verification.json.a_tier_loaded."""
    import json as _json
    import datetime as _dt
    try:
        mf = workspace / "a_tier_manifest.json"
        created_ts = None
        if mf.exists():
            try:
                prev = _json.loads(mf.read_text())
                if isinstance(prev, dict):
                    created_ts = prev.get("created_ts")
            except Exception:
                created_ts = None
        if not created_ts:
            created_ts = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        surfaced = [{
            "topic": r.get("topic", "?"),
            "skill": r.get("skill", "?"),
            "kind": "REQUIRED",
            "reason": "cba-route",
            "reference_hint": r.get("reference_hint", ""),
        } for r in routes]
        mf.write_text(_json.dumps(
            {"op": op, "schema_version": 1, "created_ts": created_ts, "surfaced": surfaced},
            indent=2))
    except Exception:
        # fail-open: the LOAD record is additive; never block brief construction on it
        return


def _c_tier_lessons_block(op: str, workspace: Optional[Path] = None, target: str = "a5") -> str:
    """c-tier (user-local KB) lessons, injected AHEAD of the b-tier manifest (c>b>a precedence).

    The READ side of the c>b>a feedback loop: surfaces deployment-local lessons the running
    agent/user sedimented into the c-tier, at highest precedence. **Config-gated** — returns "" unless
    a c-tier user_kb is active (`kb_write_root()=='customer'`, i.e. `ASCENDC_PORT_USER_KB` set or the
    default `~/.ascendc-port/user_kb` exists), so the default-b path is byte-unchanged. Uses the
    read-bridge (multi-row keyword filter, NOT single resolve — the keyword→signature bridge).
    """
    import sys as _sys
    from pathlib import Path as _Path
    _SCRIPTS = _Path(__file__).resolve().parent.parent.parent  # src/scripts/
    if str(_SCRIPTS) not in _sys.path:
        _sys.path.insert(0, str(_SCRIPTS))
    try:
        from kb_tiering.adapters.cannbot_c import kb_write_root
        from kb_tiering.read_bridge import build_arbiter, inject_for_brief
    except ImportError:
        return ""
    try:
        if kb_write_root() != "customer":
            return ""
        arb = build_arbiter()                         # c (runtime-resolved) + bundled b
        kws = [op] + op.replace("_", " ").split()
        rows = [r for r in inject_for_brief(arb, keywords=kws) if r["tier"] == "customer"]
    except Exception:
        return ""                                     # read path must never break brief construction
    if not rows:
        return ""
    lines = ["# 用户本地 KB（c 层）— 相关经验（**最高优先 c>b>a**，来自本部署沉淀）", ""]
    for r in rows[:12]:                               # cap → keep brief bounded
        e = r["row"]
        lines.append(f"  - [{getattr(e, 'id', '')}] {getattr(e, 'claim', '')}")
    lines.append("  （纯经验为本地覆盖；若与官方 b 层正/反模式冲突，读侧浮出冲突交由证据裁决）")
    lines.append("")
    return "\n".join(lines) + "\n"


def _okf_enabled() -> bool:
    """Return whether OKF is the active b-tier source (default on)."""
    import os as _os

    return _os.environ.get("ASCENDC_PORT_OKF", "").strip().lower() not in (
        "0", "false", "off", "no",
    )


def _kb_discipline_scaffold(target: str = "a5") -> str:
    """Rules and target facts that stay mandatory for either knowledge format."""
    from briefs.op_taxonomy import TARGET_HW_SPEC_MAP

    norm = (target or "a5").lower()
    if norm.endswith("-ds"):
        norm = norm[:-3]
    hw = TARGET_HW_SPEC_MAP.get(norm, TARGET_HW_SPEC_MAP["a5"])
    return f"""## 必读(与知识来源无关,OKF/legacy 都要 — 编排纪律 + 本 target 硬件事实)
Paths relative to kb/:
  - shared/ALWAYS_LOADED_RULES.md       # MANDATORY — 开发必读规则(无条件加载)
  - shared/ANTI_PRESSURE_PROTOCOLS.md   # MANDATORY — 决策前必读
  - hardware/HIASCEND_DOC_URLS.md       # MANDATORY — vendor docs URL 表(playwright-fetch)
  - {hw}   # MANDATORY — 本 target({target})芯片规格(target 路由,OKF 不保证命中)

# ANTI-PRESSURE CHECKPOINT (cite at decision points)
Before emitting any handoff line, re-read the relevant Px from `ANTI_PRESSURE_PROTOCOLS.md`:
- Before `→ orchestrator: done` → P1 + P7
- Before `→ orchestrator: PARTIAL_PERSIST` → P5 + P7
- Before spawning a sub-agent → P3 + P8
- Before writing "expected failure" / "structural ceiling" → P5
- Before `nohup &` / `Bash & disown` / direct Agent (skipping Skill) → P8
- Before manual workaround instead of fixing the script → P6
"""


def _okf_reference_block(
    op: str,
    workspace: Optional[Path] = None,
    target: str = "a5",
) -> str:
    """Retrieve the ranked OKF cards used as the exclusive default b-tier.

    The retrieval engine is owned by the external ``cannbot-knowledge`` plugin.
    This adapter never raises; its caller turns an empty result into a loud
    marker instead of silently changing formats.
    """
    import importlib.util as _ilu
    import json as _json
    import subprocess as _sub
    from pathlib import Path as _Path

    if not _okf_enabled():
        return ""
    try:
        plugin_root = _Path(__file__).resolve().parents[5]
        kb_root = plugin_root / "kb" / "okf"
        index = kb_root / "search" / "okf.index.json"
        engine_file = plugin_root / "engine" / "src" / "scripts" / "okf" / "okf_engine.py"
        spec = _ilu.spec_from_file_location("okf_engine", str(engine_file))
        if spec is None or spec.loader is None:
            return ""
        engine = _ilu.module_from_spec(spec)
        spec.loader.exec_module(engine)
        knowledge_query = engine.knowledge_query_script()
        if knowledge_query is None or not (_Path(str(knowledge_query)).is_file() and index.is_file()):
            return ""
        try:
            from briefs.op_taxonomy import lookup as _lookup

            taxonomy = _lookup(op, workspace=workspace, target=target)
            query = " ".join(
                [op.replace("_", " ")] + list(getattr(taxonomy, "tags", []))[:6]
            )
        except Exception:
            query = op.replace("_", " ")
        result = _sub.run(
            [
                sys.executable, str(knowledge_query), "pipeline",
                "--recall", "bm25,tagtype", "--rerank", "bm25f",
                "--query", query, "--knowledge-root", str(kb_root),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return ""
        parsed = _json.loads(result.stdout)
        raw = parsed.get("hits", []) if isinstance(parsed, dict) else []
        hits = [hit for hit in raw if isinstance(hit, dict)][:5] if isinstance(raw, list) else []
    except Exception:
        return ""
    if not hits:
        return ""
    lines = [
        "# OKF 知识卡片（knowledge-query 检索 — b 层知识来源,独占;本次不含旧 KB_INDEX）",
        "knowledge-query 已按相关度排好序,直接读下列卡,**不要**再 grep `kb/target/**`：",
        "",
    ]
    for hit in hits:
        lines.append(
            "  - kb/okf/%s  # %s (kind=%s)"
            % (hit.get("path", ""), hit.get("title", ""), hit.get("kind", ""))
        )
    lines.append("")
    return "\n".join(lines) + "\n"

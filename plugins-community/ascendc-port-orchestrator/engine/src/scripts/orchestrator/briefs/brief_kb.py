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
"""brief_kb — KB brief section composition (kb_manifest_block + c>b>a read-path
blocks), extracted from briefs/_common.py
(behavior-neutral god-file decomposition, 2026-07-05). Leaf module: stdlib-only, imports
nothing from _common. _common re-imports these so its importers are unaffected.

OKF-only 迁移（2026-08）：legacy KB（旧索引 + target 目录）注入链路已整体退役
——`kb_inject_filtered` / `KBEntry` / 旧索引行解析 / legacy manifest 渲染 /
`ASCENDC_PORT_OKF` 开关 / `force_legacy_kb` 逃生门全部移除。OKF 检索是唯一
b-tier 路径，检索为空/失败时 fail-loud（响亮标记，不回退旧体系）。
`_detect_forced_architecture` 与 `_FORCED_ARCH_TAGS` 保留（kw/ko brief 的
forced-arch 块在用，与知识来源无关）。
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Optional

_HERE = Path(__file__).resolve()
_FORCED_ARCH_TAGS = {"SIMT", "SIMD"}

# Archived/deprecated OKF 卡只是历史留存（audit trail，安全规则 5 明确不删卡），
# 不得占用 brief 的 top-5 注入位（实测 pb-36-archived-deprecated-... 曾是 worker
# 读得最多的卡）。判定只用明确的退役标记约定，避免误伤正文里提及 "archived" 的
# 在役卡（如 OL-168 "treating archived kernel as starting material"）：
#   - 路径/文件名含 archived|deprecated 词元（命名约定，如 pb-36-archived-deprecated-...）；
#   - tags metadata 含 archived|deprecated 词元；
#   - 标题以 "[ARCHIVED...]"/"[...DEPRECATED...]" 方括号标记开头（归档卡的标题约定）。
_ARCHIVED_TOKEN_RE = re.compile(r"\b(?:archived|deprecated)\b", re.IGNORECASE)
_ARCHIVED_TITLE_RE = re.compile(r"^\s*\[[^\]]*(?:archived|deprecated)", re.IGNORECASE)


def _is_archived_okf_hit(hit: dict) -> bool:
    """True for archived/deprecated (history-only) OKF cards — excluded from brief injection."""
    if _ARCHIVED_TOKEN_RE.search(str(hit.get("path", ""))):
        return True
    tags = hit.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if any(_ARCHIVED_TOKEN_RE.search(str(t)) for t in tags):
        return True
    return bool(_ARCHIVED_TITLE_RE.search(str(hit.get("title", ""))))


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
) -> str:
    """KB section — OKF 检索是唯一 b-tier 知识来源。

    OKF-only 迁移（2026-08）：legacy manifest 渲染、`force_legacy_kb` 逃生门与
    `ASCENDC_PORT_OKF` 开关均已移除，不再有旧知识体系的回退分支。OKF 检索
    为空/失败时输出响亮标记（fail-loud），绝不静默回退。

    Codex C1: Tier 1 Python template (90%) + Tier 2 KB lookup by op-class
    tag (10%). NO LLM in brief construction.

    P0abj (2026-05-08): pass target so the discipline scaffold's hardware-spec
    entry dispatches to the correct chip's ref doc.
    """
    # c>b>a read-path: prepend user-local (c-tier) lessons AHEAD of the b-tier knowledge,
    # and append the CBA tier-a required-routes block AFTER. Both are config/workspace-gated
    # (empty unless active) → the default-b path stays byte-unchanged.
    #
    # OKF is the ONLY b-tier source. Empty/error retrieval is loud and never
    # silently falls back, otherwise a broken external knowledge dependency
    # looks healthy.
    okf = _okf_reference_block(op, workspace, target)
    okf_body = okf if okf else (
        "# ⚠️ OKF 检索无返回——索引未 build / 无命中 / 检索失败。\n"
        "# 本次简报无 b 层知识，且**无回退**（旧知识体系已退役）。"
        "请先 `engine/src/scripts/okf/okf_kb.sh build`\n"
        "# 并确认检索命中后再重跑。\n\n"
    )
    b_tier = okf_body + _kb_discipline_scaffold(target)
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
    agent/user sedimented into the c-tier, at highest precedence. Post OKF-only migration the
    bundled b-tier is gone, so `kb_write_root()` is always "customer"; the block still returns ""
    whenever the c-tier has no matching entries (empty/absent user_kb → byte-unchanged brief).
    Uses the read-bridge (multi-row keyword filter, NOT single resolve — the keyword→signature bridge).
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
  - shared/HIASCEND_DOC_URLS.md         # MANDATORY — vendor docs URL 表(playwright-fetch)
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


# knowledge-query reports a card's `path` relative to its CONTENT ROOT, and the roots are
# not uniform: `runbooks/...` is already okf-root-relative, but `reference/` cards come back
# bundle-relative (`porter/...`, `asc-devkit-vendored/...`). Concatenating "kb/okf/" verbatim
# therefore names a file that does not exist for every reference card. Measured 2026-09-05
# over 8 sampled queries x top-5: 7/40 slots pointed at nothing, and all 7 resolved once the
# `reference/` root was tried. Pre-existing, but this reorg gave those 546 cards frontmatter
# (description w2.0 / tags w2.5), which raised how often they reach the top 5.
_Path = Path
_PurePosixPath = PurePosixPath
# knowledge-query reports `runbooks/`/`ops/` paths relative to the okf root, but a
# `reference/` card relative to its BUNDLE. The first segment says which.
_OKF_ROOT_RELATIVE_TREES = ("runbooks", "ops")


def _local_path_card(local, root, rel):
    """The `kb/okf/...` name for an engine-resolved `local_path`, or None if it is not one.

    The whole `path` must be a suffix of `local_path`, not merely the basename:
    `porter/A/foo.md` and `reference/porter/B/foo.md` share a basename but are different
    cards, and swapping one for the other feeds the worker the wrong knowledge while every
    existence check still passes.
    """
    try:
        cand = _Path(str(local)).resolve()
        under = cand.relative_to(root).as_posix()
        if cand.is_file() and (under == rel or under.endswith("/" + rel)):
            return "kb/okf/%s" % under
    except (ValueError, OSError, RuntimeError, TypeError):
        return None      # outside kb/okf, unreadable, or a symlink loop
    return None


def _okf_hit_path(kb_root, hit: dict):
    """The plugin-root-relative path of a knowledge-query hit, or None if unverifiable.

    Every returned path is a confirmed existing FILE inside `kb/okf`, and it is the card
    the hit actually names. A brief line is an instruction to open a file, so both halves
    matter: a path that does not exist wastes the worker's turn, and a path that exists
    but belongs to a DIFFERENT card silently feeds it the wrong knowledge.

    Order (shape checks first — they must not be bypassable by a `local_path`):
      1. reject an absolute `path`, or one containing `..`. `Path("/a") / "/etc/x"` is
         silently `/etc/x`, and `..` escapes `kb/okf`.
      2. `hit["local_path"]` — the absolute path the ENGINE resolved. Used only when it is
         an existing file under `kb/okf` AND its basename matches `path`'s, so a stale or
         mismatched index entry cannot substitute one card for another.
      3. probe `kb/okf/reference/<rel>` then `kb/okf/<rel>` — `reference` first because a
         hit whose first segment is not a content root is bundle-relative, and only that
         order picks the right card when the same name exists under both.
      4. None. The caller says so rather than printing a guess.
    """
    rel = str(hit.get("path") or "").strip()
    try:
        root = _Path(kb_root).resolve()
    except (OSError, TypeError, ValueError):
        return None
    unsafe_rel = (
        rel.startswith("/")
        or "\\" in rel
        or ".." in _PurePosixPath(rel).parts
    )
    if not rel or unsafe_rel:
        return None

    local = hit.get("local_path")
    if local:
        # A `local_path` that is outside kb/okf, unreadable, or a symlink loop is not an
        # error here — it just means this shortcut does not apply and the probe below
        # decides. `_local_path_card()` returns None for all of those.
        resolved = _local_path_card(local, root, rel)
        if resolved:
            return resolved

    first = _PurePosixPath(rel).parts[0] if _PurePosixPath(rel).parts else ""
    bases = ("", "reference") if first in _OKF_ROOT_RELATIVE_TREES else ("reference", "")
    for base in bases:
        cand = (root / base / rel) if base else (root / rel)
        try:
            if cand.is_file() and cand.resolve().relative_to(root):
                return "kb/okf/%s%s" % (base + "/" if base else "", rel)
        except (ValueError, OSError, RuntimeError):
            continue  # escapes kb/okf via a symlink, or the stat failed
    return None


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
        hits = [hit for hit in raw if isinstance(hit, dict)] if isinstance(raw, list) else []
        # 先过滤已归档/废弃卡再截断 top-5：废弃卡只占位不提供有效指导，
        # 且不应把在役卡挤出注入位。全部被滤掉时走既有的响亮空标记路径。
        hits = [hit for hit in hits if not _is_archived_okf_hit(hit)][:5]
    except Exception:
        return ""
    if not hits:
        return ""
    lines = [
        "# OKF 知识卡片（knowledge-query 检索 — b 层知识来源,独占）",
        "knowledge-query 已按相关度排好序,直接读下列卡,不要全库扫描：",
        "",
    ]
    for hit in hits:
        shown = _okf_hit_path(kb_root, hit)
        lines.append(
            "  - %s  # %s (kind=%s)"
            % (shown or "(此条路径无法核实,已省略——不要打开任何文件来顶替它)",
               hit.get("title", ""), hit.get("kind", ""))
        )
    lines.append("")
    return "\n".join(lines) + "\n"

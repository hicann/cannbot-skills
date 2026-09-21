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
"""external_kb — 外部 cannbot-knowledge 知识仓（OKF v0.2）作为 b-tier 知识来源。

背景（issue #559 后续 / commit 223ee980）：插件内旧知识的 672 张 OKF v1 卡已
迁移至独立知识仓 cannbot-knowledge（`knowledge/` Bundle，OKF v0.2）。本模块是
引擎侧的唯一适配点：

- 配置：显式环境变量 `CANNBOT_KNOWLEDGE_ROOT` 优先；否则依次读取
  `workspace/.ascendc_env` 的 `CANNBOT_KNOWLEDGE_ROOT=`（用户填写的配置面，
  路径经 ASCENDC_ENV_PATH 定位）与当前项目向上最近的
  `.cannbot/knowledge.env`。后两者分别面向手工配置与 cannbot-knowledge 统一
  安装器写入的项目配置。三者都必须指向含 `knowledge/`、governance contract 与
  `.agents/skills/knowledge-query/` 的完整 checkout。
- port 插件不安装、更新或索引知识仓，也不回退插件内旧知识。未安装、配置无效、
  索引缺失或检索失败时返回空结果，由调用方 fail-loud。
- 检索：调用知识仓自带的 knowledge-query `search`（v5 JSON），按目标平台
  （a5→`950`）与来源平台（`a3`）各查一次后按 score 合并；`deprecated` 卡被过滤。
- 路径解析：运行时调用方使用新仓规范 `knowledge/...` 路径；
  `LEGACY_OKF_RELOCATED` 仅为旧 workspace/持久化 marker 提供向后兼容解析。

索引：检索依赖 `<root>/artifacts/indexes/knowledge.sqlite3`（不随 git 分发，
由知识仓 `knowledge_index.py build` 生成）。缺失时检索失败，调用方 fail-loud。
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:  # 与引擎共用 a5_orchestrator 日志树（含 stdout/file handler），巡检可直接 grep
    import logging_config as _logging_config

    _LOG = _logging_config.get_logger("external_kb")
except Exception:  # 独立导入场景（单测/脚本）退回模块 logger
    _LOG = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[5]          # briefs→orchestrator→scripts→src→engine→plugin
_PROJECT_KNOWLEDGE_ENV = Path(".cannbot") / "knowledge.env"

# 插件目标名 → 知识仓 Registry 平台 ID（platform-identification.md：A5/950PR 必须
# 归一为 `950`，不能传 `a5`）。
TARGET_TO_PLATFORM: dict[str, str] = {
    "a5": "950",
    "a3": "a3",
    "a2": "a2",
}


# 随 a3→a5 移植任务总是相关的一对平台：目标平台 + 来源平台（arch22/910C = a3）。
def platforms_for_target(target: str) -> list[str]:
    """检索平台集合：目标平台优先；a5 目标时附加来源平台 a3（跨代移植双语境）。"""
    norm = (target or "a5").lower()
    if norm.endswith("-ds"):
        norm = norm[:-3]
    primary = TARGET_TO_PLATFORM.get(norm, "950")
    out = [primary]
    if primary == "950" and "a3" not in out:
        out.append("a3")
    return out


# 插件内已删除卡片的旧相对路径 → 新仓 `knowledge/`
# 相对路径。迁移去向见 cannbot-skills commit 223ee980 与知识仓 logs/2026-09-16.md。
LEGACY_OKF_RELOCATED: dict[str, str] = {
    # runbooks/hardware → common/platforms/concepts（芯片规格）+ ops/ascendc/concepts（探针）
    "runbooks/hardware/target-ascend950pr.md": "common/platforms/concepts/target_ascend950pr.md",
    "runbooks/hardware/target-ascend910c.md": "common/platforms/concepts/target_ascend910c.md",
    "runbooks/hardware/target-ascend910b.md": "common/platforms/concepts/target_ascend910b.md",
    "runbooks/hardware/probe-2026-04-21-q-scalar-broadcast.md": (
        "ops/ascendc/concepts/scalar_broadcast_sync_bypass.md"
    ),
    "runbooks/hardware/probe-2026-04-21-q-instruction-cycles.md": (
        "ops/ascendc/concepts/sort_reduce_instruction_cycles.md"
    ),
    # reference/porter/handbook → concepts / guides
    "reference/porter/handbook/roofline_model.md": "ops/ascendc/concepts/roofline_model.md",
    "reference/porter/handbook/simd_development_reference.md": "ops/ascendc/guides/simd_development_reference.md",
    "reference/porter/handbook/simt_vs_simd_decision.md": "ops/ascendc/concepts/simt_vs_simd_decision.md",
    # reference/porter/patterns → ops/ascendc/examples
    "reference/porter/patterns/cube_vector_fusion.md": "ops/ascendc/examples/cube_vector_fusion.md",
    "reference/porter/patterns/fa_class_a3_mix_template.md": "ops/ascendc/examples/fa_class_a3_mix_template.md",
    "reference/porter/patterns/fa_class_template.md": "ops/ascendc/examples/fa_class_template.md",
    # reference/porter/playbook → ops/ascendc/guides/cross_gen_migration_guide
    "reference/porter/playbook/l2_register_based.md": (
        "ops/ascendc/guides/cross_gen_migration_guide/l2_register_based.md"
    ),
    "reference/porter/playbook/l5_register_based.md": (
        "ops/ascendc/guides/cross_gen_migration_guide/l5_register_based.md"
    ),
    "reference/porter/playbook/ops_nn_a5_artifact_layout.md": (
        "ops/ascendc/guides/cross_gen_migration_guide/ops_nn_a5_artifact_layout.md"
    ),
    # reference/porter/{precision,toolchain} → ops/ascendc/guides
    "reference/porter/precision/precision_standard_v2_1.md": "ops/ascendc/guides/precision/precision_standard_v2_1.md",
    "reference/porter/toolchain/msprof_agent_guide.md": "ops/ascendc/guides/msprof_agent_guide.md",
    # runbooks/field-notes → ops/ascendc/runbooks/{compilation,precision}
    "runbooks/field-notes/build/ec-59-phase-o5-re-measurement-disagrees-with-worker-pass.md": (
        "ops/ascendc/runbooks/compilation/ec_59_phase_o5_re_measurement_disagrees_with_worker_pass.md"
    ),
    "runbooks/field-notes/build/pb-34-matmulimpl-with-manual-crosscoresetflag-waitflag-m.md": (
        "ops/ascendc/runbooks/compilation/pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md"
    ),
    "runbooks/field-notes/build/pb-35-event-t-0-for-cube-internal-pipe-sync-mte1-m-m-fix.md": (
        "ops/ascendc/runbooks/compilation/pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md"
    ),
    "runbooks/field-notes/precision/ol-109-two-tier-precision-verdict.md": (
        "ops/ascendc/runbooks/precision/ol_109_two_tier_precision_verdict.md"
    ),
    # runbooks/operator-optimization → ops/ascendc/optimizations
    "runbooks/operator-optimization/fa-cross-core-sync-workspacequeue.md": (
        "ops/ascendc/optimizations/fa_cross_core_sync_workspacequeue.md"
    ),
    "runbooks/operator-optimization/ol-196-membase-vs-regbase-simd-vf-selection.md": (
        "ops/ascendc/optimizations/ol_196_membase_vs_regbase_simd_vf_selection.md"
    ),
}

# 前缀级迁移规则（静态表之外的同族卡片）：(旧相对前缀, 新 knowledge/ 相对前缀,
# 文件名是否 kebab→snake 重写)。按序匹配，命中即核验文件存在。
_RELOCATED_PREFIX_RULES: tuple[tuple[str, str, bool], ...] = (
    ("reference/porter/playbook/", "ops/ascendc/guides/cross_gen_migration_guide/", True),
    ("reference/porter/patterns/", "ops/ascendc/examples/", True),
    ("reference/porter/precision/", "ops/ascendc/guides/precision/", True),
    ("reference/porter/toolchain/", "ops/ascendc/guides/", True),
    ("runbooks/operator-optimization/", "ops/ascendc/optimizations/", True),
    ("runbooks/field-notes/build/", "ops/ascendc/runbooks/compilation/", True),
    ("runbooks/field-notes/precision/", "ops/ascendc/runbooks/precision/", True),
    ("runbooks/field-notes/perf/", "ops/ascendc/runbooks/performance/", True),
    ("runbooks/hardware/target-", "common/platforms/concepts/target_", True),
)

_ROOT_CACHE: dict[str, Optional[Path]] = {}


def _read_project_env_key(path: Path, key: str) -> str:
    """Read a shell-assignment value from the installer-owned project config.

    The knowledge installer writes simple ``export KEY=value`` assignments.  Parse
    those assignments instead of sourcing the file so project content cannot execute
    shell code inside the orchestrator process.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ""
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            return ""
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        for token in tokens:
            if not token.startswith(f"{key}="):
                continue
            return token.split("=", 1)[1]
    return ""


def _project_knowledge_root() -> str:
    """Return the nearest project installer configuration, if one exists."""
    try:
        project_start = os.environ.get("CANNBOT_PROJECT_ROOT", "")
        cwd = Path(project_start).expanduser() if project_start else Path.cwd()
        cwd = cwd.resolve()
    except (OSError, RuntimeError):
        return ""
    for parent in (cwd, *cwd.parents):
        env_file = parent / _PROJECT_KNOWLEDGE_ENV
        if not env_file.is_file():
            continue
        value = _read_project_env_key(env_file, "CANNBOT_KNOWLEDGE_ROOT")
        if not value:
            return ""
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = env_file.parent / candidate
        return str(candidate)
    return ""


def _ascendc_env_knowledge_root() -> str:
    """Return CANNBOT_KNOWLEDGE_ROOT from workspace/.ascendc_env, if filled.

    `.ascendc_env` 是用户填写的运行配置面（模板见 engine/workspace/
    .ascendc_env.template）：O0 预检要求在这里显式填写知识仓根，否则阻断
    启动。路径解析与 briefs._common.load_env 一致——优先 ASCENDC_ENV_PATH
    覆盖（launcher 会把它指向实际使用的 .ascendc_env）。
    """
    env_path = os.environ.get("ASCENDC_ENV_PATH", "")
    if not env_path:
        try:
            from briefs import _common as _bc

            env_path = str(_bc.DEFAULT_ASCENDC_ENV)
        except Exception:
            return ""
    try:
        text = Path(env_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^CANNBOT_KNOWLEDGE_ROOT=(.*)$", line)
        if not m:
            continue
        val = m.group(1).strip()
        single_quoted = val.startswith("'") and val.endswith("'")
        double_quoted = val.startswith('"') and val.endswith('"')
        if single_quoted or double_quoted:
            val = val[1:-1]
        return val
    return ""


def external_kb_root() -> Optional[Path]:
    """配置的外部知识仓根（校验过布局），未配置/无效时 None。

    解析顺序：进程环境变量 `CANNBOT_KNOWLEDGE_ROOT`（显式单次覆盖）→
    `workspace/.ascendc_env` 的 `CANNBOT_KNOWLEDGE_ROOT=`（用户配置面）→
    当前项目向上最近的 `.cannbot/knowledge.env`（安装器写入）。有效布局与
    cannbot-knowledge master 安装器一致：Bundle、Registry 和同 checkout 的
    knowledge-query 脚本必须同时存在。
    """
    cached = _ROOT_CACHE.get("root", None)
    if "root" in _ROOT_CACHE:
        return cached
    raw = (
        os.environ.get("CANNBOT_KNOWLEDGE_ROOT", "")
        or _ascendc_env_knowledge_root()
        or _project_knowledge_root()
    )
    root: Optional[Path] = None
    if raw:
        cand = Path(raw).expanduser()
        if (
            (cand / "knowledge" / "index.md").is_file()
            and (cand / "governance" / "schemas" / "registries.yaml").is_file()
            and (cand / ".agents" / "skills" / "knowledge-query" / "scripts" / "knowledge_query.py").is_file()
        ):
            try:
                root = cand.resolve()
            except OSError:
                root = None
    if root is not None:
        # Child harnesses and brief text use the same canonical variable.  Export
        # the installer-resolved project value without overriding an explicit one.
        os.environ.setdefault("CANNBOT_KNOWLEDGE_ROOT", str(root))
    _ROOT_CACHE["root"] = root
    return root


def external_knowledge_dir() -> Optional[Path]:
    """外部知识仓的 OKF Bundle 根（`<root>/knowledge`），未配置时 None。"""
    root = external_kb_root()
    return (root / "knowledge") if root is not None else None


def external_query_script() -> Optional[Path]:
    root = external_kb_root()
    if root is None:
        return None
    return root / ".agents" / "skills" / "knowledge-query" / "scripts" / "knowledge_query.py"


def external_index_file() -> Optional[Path]:
    root = external_kb_root()
    if root is None:
        return None
    return root / "artifacts" / "indexes" / "knowledge.sqlite3"


def resolve_kb_ref(rel: str) -> Optional[Path]:
    """把新仓规范路径（及兼容的旧引用）解析为真实文件。

    顺序：① `LEGACY_OKF_RELOCATED` 映射命中且外部仓内文件存在；
    ② 前缀迁移规则命中；③ 外部仓 `knowledge/<rel>` 直拼存在；
    ④ 找不到 → None。插件内旧知识不再是运行时来源。
    """
    rel_norm = rel.strip()
    for prefix in ("knowledge/", "kb/okf/", "okf/"):
        if rel_norm.startswith(prefix):
            rel_norm = rel_norm[len(prefix):]
            break
    knowledge = external_knowledge_dir()
    if knowledge is None:
        return None
    mapped = LEGACY_OKF_RELOCATED.get(rel_norm)
    if mapped:
        cand = knowledge / mapped
        if cand.is_file():
            return cand
    for old_prefix, new_prefix, snake in _RELOCATED_PREFIX_RULES:
        if rel_norm.startswith(old_prefix):
            name = rel_norm[len(old_prefix):]
            if "/" in name:  # 前缀规则只覆盖扁平卡片，不下钻子目录
                break
            if snake:
                name = name.replace("-", "_")
            cand = knowledge / new_prefix / name
            if cand.is_file():
                return cand
            break
    direct = knowledge / rel_norm
    if direct.is_file():
        return direct
    return None


def kb_ref_display(rel: str) -> str:
    """brief 文案用的可读路径：外部仓命中时给绝对路径。

    解析不到时返回原引用加响亮缺失标记，不静默指向插件内旧知识。
    """
    resolved = resolve_kb_ref(rel)
    if resolved is None:
        return f"{rel}（⚠️ 未找到：cannbot-knowledge 未安装、配置无效或未收录，不要凭记忆复述其内容）"
    return str(resolved)


def _search_command(
    query: str,
    platform: str,
    k: int,
    card_type: Optional[str] = None,
) -> Optional[list[str]]:
    """Build the fixed knowledge-query `search` argv (v5 JSON output).

    查询脚本与知识根经缓存的 resolver 解析；未配置时返回 None。
    """
    query_script = external_query_script()
    root = external_kb_root()
    if query_script is None or root is None:
        return None
    command = [
        sys.executable, str(query_script),
        "--knowledge-root", str(root), "search",
        "--query", query,
        "--platform", platform,
        "--domain", "ops",
        "--technology", "ascendc",
        "--include-shared",
        "--status", "all",
        "--match", "text",
        "-k", str(k),
        "--json",
    ]
    if card_type:
        command.extend(["--type", card_type])
    return command


def _run_search(
    query: str,
    platform: str,
    k: int,
    card_type: Optional[str] = None,
) -> list[dict]:
    """单次 knowledge-query search（v5 JSON）。任何失败返回 []（调用方判空后 fail-loud）。"""
    command = _search_command(query, platform, k, card_type)
    if command is None:
        return []
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        _LOG.warning(
            "knowledge-query search failed to run: query=%r platform=%s err=%s",
            query, platform, exc,
        )
        return []
    if result.returncode != 0:
        _LOG.warning(
            "knowledge-query search exited rc=%s: query=%r platform=%s stderr=%.300s",
            result.returncode, query, platform, result.stderr or "",
        )
        return []
    try:
        parsed = json.loads(result.stdout)
    except (ValueError, TypeError):
        _LOG.warning(
            "knowledge-query search returned non-JSON stdout: query=%r platform=%s stdout=%.300s",
            query, platform, result.stdout or "",
        )
        return []
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("results")
    if not isinstance(raw, list):
        raw = parsed.get("hits", [])
    return [hit for hit in raw if isinstance(hit, dict)] if isinstance(raw, list) else []


def _resolve_validated_hit(hit: dict, knowledge_resolved: Path) -> Optional[Path]:
    """核验一条命中的 local_path 在 Bundle 根内且真实存在；无效返回 None。"""
    local = hit.get("local_path")
    if not local:
        return None
    try:
        cand = Path(str(local)).resolve()
        cand.relative_to(knowledge_resolved)
    except (ValueError, OSError, RuntimeError, TypeError):
        return None
    return cand if cand.is_file() else None


def search_external_cards(
    query: str,
    target: str = "a5",
    per_platform_k: int = 10,
    card_type: Optional[str] = None,
) -> list[dict]:
    """检索外部知识仓，返回按 score 合并去重、过滤 deprecated 后的命中列表。

    每个命中是 knowledge-search.v5 的 result 字典（含绝对 `local_path`、`title`、
    `type`、`status`、`path` 概念 ID）。`local_path` 已核验位于 Bundle 根内且文件
    存在；核验不过的条目被丢弃（brief 行是「打开这个文件」的指令，路径必须真实）。
    配置缺失/索引缺失/查询失败时返回 []。
    """
    root = external_kb_root()
    knowledge = external_knowledge_dir()
    index = external_index_file()
    if any(item is None for item in (root, knowledge, index)):
        _LOG.warning(
            "external_kb not configured: root=%s knowledge=%s index=%s "
            "(check CANNBOT_KNOWLEDGE_ROOT / .cannbot/knowledge.env)",
            root, knowledge, index,
        )
        return []
    if not index.is_file():
        _LOG.warning(
            "external_kb index missing: %s (build it in the knowledge repo)", index
        )
        return []
    try:
        knowledge_resolved = knowledge.resolve()
    except OSError:
        return []

    merged: dict[str, dict] = {}
    for platform in platforms_for_target(target):
        for hit in _run_search(query, platform, per_platform_k, card_type):
            if str(hit.get("status", "")).strip().lower() == "deprecated":
                continue
            if _resolve_validated_hit(hit, knowledge_resolved) is None:
                continue
            key = str(hit.get("path") or hit.get("local_path"))
            prev = merged.get(key)
            if prev is None or float(hit.get("score") or 0.0) > float(prev.get("score") or 0.0):
                merged[key] = hit
    hits = sorted(merged.values(), key=lambda h: float(h.get("score") or 0.0), reverse=True)
    _LOG.info(
        "external_kb search done: query=%r target=%s platforms=%s hits=%d",
        query, target, platforms_for_target(target), len(hits),
    )
    return hits


def query_ref_displays(
    query: str,
    target: str = "a5",
    *,
    card_type: Optional[str] = None,
    limit: int = 5,
) -> list[str]:
    """把一个主题/API 需求解析为 knowledge-query 返回的真实卡片路径。

    旧 asc-devkit 目录簇没有一对一的新目录，不允许直接拼接或扫描。
    无命中时返回带查询条件的响亮缺失标记，让 worker 停止猜测。
    """
    hits = search_external_cards(
        query, target=target, per_platform_k=max(limit, 5), card_type=card_type
    )[:limit]
    paths = [str(hit.get("local_path") or "") for hit in hits]
    paths = [path for path in paths if path]
    if paths:
        return paths
    type_text = f", type={card_type}" if card_type else ""
    return [
        f"knowledge-query: {query} (platform={','.join(platforms_for_target(target))}"
        f"{type_text})（⚠️ 无命中/索引不可用，不要按旧目录猜测）"
    ]

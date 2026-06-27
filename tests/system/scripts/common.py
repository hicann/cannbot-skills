# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
common.py — ST 测试框架共享基础设施

消除 conftest.py 和 main.py 之间的重复代码，提供统一的路径常量、
配置加载和实体发现函数。所有框架模块都应从此模块导入路径和配置。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────
# 所有路径相对于当前文件 (tests/system/scripts/common.py)
# 框架根目录 = tests/system/
FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = FRAMEWORK_DIR / "config" / "st-test.config"
REPO_ROOT = FRAMEWORK_DIR.parent.parent  # 仓库根目录
EVALS_CASES_DIR = FRAMEWORK_DIR / "cases"  # 集中式 evals 存放目录
LOGS_DIR = FRAMEWORK_DIR / "logs"  # opencode session 导出 JSON 存放目录
SANDBOX_DIR = FRAMEWORK_DIR / "sandboxes"  # 沙箱隔离目录


def load_common_config() -> Dict[str, Any]:
    """加载 st-test.config，供 main.py 和 conftest.py 统一使用。

    Returns:
        配置字典。文件不存在或解析失败时返回默认值。
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.warning("配置解析失败 %s: %s", CONFIG_PATH, e)
    return {"skill_dirs": ["skills"], "skill_whitelist": [],
            "team_dirs": [], "team_whitelist": []}


def find_entity_path(
    entity_name: str,
    config_key: str,
    marker_files: tuple = ("SKILL.md",),
) -> Optional[Path]:
    """通用实体路径查找。

    在 config_key 对应的所有目录中查找 entity_name，
    并验证所有 marker_files 存在。

    Args:
        entity_name: 实体名称（目录名）
        config_key: 配置键名，如 "skill_dirs" 或 "team_dirs"
        marker_files: 标记文件名元组，如 ("SKILL.md",)

    Returns:
        实体目录的 Path，未找到时返回 None。
    """
    config = load_common_config()
    for dir_rel in config.get(config_key, []):
        candidate = REPO_ROOT / dir_rel / entity_name
        if candidate.exists() and candidate.is_dir():
            if all((candidate / m).exists() for m in marker_files):
                return candidate
    return None


def discover_all_entities(
    config_key: str,
    whitelist_key: str,
    marker_files: tuple = ("SKILL.md",),
) -> List[str]:
    """发现满足标记文件条件的所有实体。

    扫描 config_key 对应的所有目录，返回存在所有 marker_files 的子目录名列表。
    如果配置了 whitelist_key 白名单，仅返回白名单中的实体。

    Args:
        config_key: 配置键名，如 "skill_dirs" 或 "team_dirs"
        whitelist_key: 白名单配置键名，如 "skill_whitelist" 或 "team_whitelist"
        marker_files: 标记文件名元组

    Returns:
        排序后的实体名称列表
    """
    config = load_common_config()
    entities: Set[str] = set()
    whitelist = config.get(whitelist_key, [])
    for dir_rel in config.get(config_key, []):
        entity_dir = REPO_ROOT / dir_rel
        if not entity_dir.exists():
            continue
        for item in entity_dir.iterdir():
            if not item.is_dir():
                continue
            if whitelist and item.name not in whitelist:
                continue
            if all((item / m).exists() for m in marker_files):
                entities.add(item.name)
    return sorted(entities)


def discover_entities_with_evals(
    whitelist_key: str,
    target_type: Optional[str] = None,
) -> List[str]:
    """发现 cases/ 目录下有 evals.md 文件的实体。

    Args:
        whitelist_key: 白名单配置键名
        target_type: 可选过滤，"skill" 或 "team"。如为 None 则不过滤

    Returns:
        排序后的实体名称列表
    """
    from evals_parser import parse_evals_md

    config = load_common_config()
    entities: List[str] = []
    whitelist = config.get(whitelist_key, [])
    if not EVALS_CASES_DIR.exists():
        return entities
    for f in EVALS_CASES_DIR.iterdir():
        if not f.is_file() or not f.name.endswith("_evals.md"):
            continue
        candidate = f.name[: -len("_evals.md")]
        if whitelist and candidate not in whitelist:
            continue
        if target_type is not None:
            try:
                data = parse_evals_md(f)
            except Exception as e:
                logger.warning("Failed to parse evals file %s: %s", f, e)
                continue
            if not data or data.get("target_type") != target_type:
                continue
        entities.append(candidate)
    return sorted(entities)


def load_entity_evals_md(entity_name: str) -> Optional[Dict[str, Any]]:
    """从 cases/<entity_name>_evals.md 加载评测用例。"""
    from evals_parser import parse_evals_md

    evals_path = EVALS_CASES_DIR / f"{entity_name}_evals.md"
    return parse_evals_md(evals_path)

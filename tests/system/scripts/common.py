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


def _match_target_type(evals_file: Path, target_type: str) -> bool:
    """验证 evals 文件的 target_type 是否匹配。"""
    from evals_parser import parse_evals_md

    try:
        data = parse_evals_md(evals_file)
    except Exception as e:
        logger.warning("Failed to parse evals file %s: %s", evals_file, e)
        return False
    return bool(data and data.get("target_type") == target_type)


def _is_valid_evals_entity(item: Path, whitelist_set: Optional[Set[str]], target_type: Optional[str]) -> bool:
    """判断目录项是否为含 evals/evals.md 的有效实体。"""
    if not item.is_dir():
        return False
    if whitelist_set and item.name not in whitelist_set:
        return False
    evals_file = item / "evals" / "evals.md"
    if not evals_file.exists():
        return False
    if target_type is not None and not _match_target_type(evals_file, target_type):
        return False
    return True


def _scan_dir_for_evals(dir_rel: str, whitelist_set: Optional[Set[str]], target_type: Optional[str]) -> List[str]:
    """扫描单个目录，返回含 evals/evals.md 的实体名称列表。"""
    entity_dir = REPO_ROOT / dir_rel
    if not entity_dir.exists():
        return []
    return [
        item.name for item in entity_dir.iterdir()
        if _is_valid_evals_entity(item, whitelist_set, target_type)
    ]


def _collect_entity_with_evals(
    dir_keys: List[str],
    whitelist: List[str],
    target_type: Optional[str],
) -> List[str]:
    """遍历多个目录键搜索包含 evals/evals.md 的实体。"""
    entities: List[str] = []
    whitelist_set = set(whitelist) if whitelist else None
    for dir_key in dir_keys:
        for dir_rel in load_common_config().get(dir_key, []):
            entities.extend(_scan_dir_for_evals(dir_rel, whitelist_set, target_type))
    return entities


def discover_entities_with_evals(
    whitelist_key: str,
    target_type: Optional[str] = None,
) -> List[str]:
    """从 Skill/Team 目录下的 evals/evals.md 发现评测用例。

    扫描 skill_dirs/team_dirs 配置的目录，查找包含 evals/evals.md 的实体。

    Args:
        whitelist_key: 白名单配置键名
        target_type: 可选过滤，"skill" 或 "team"。如为 None 则不过滤

    Returns:
        排序后的实体名称列表
    """
    config = load_common_config()
    whitelist = config.get(whitelist_key, [])

    dir_keys = []
    if target_type == "skill":
        dir_keys = ["skill_dirs"]
    elif target_type == "team":
        dir_keys = ["team_dirs"]
    else:
        dir_keys = ["skill_dirs", "team_dirs"]

    return sorted(_collect_entity_with_evals(dir_keys, whitelist, target_type))


def load_entity_evals_md(entity_name: str, entity_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """从 {skill_dir}/{entity_name}/evals/evals.md 加载评测用例。

    Args:
        entity_name: 实体名称
        entity_type: 限定范围，'skill' 只扫描 skill_dirs，'team' 只扫描 team_dirs，
                     None 则扫描全部（默认）

    Returns:
        解析后的评测用例数据，未找到则返回 None
    """
    from evals_parser import parse_evals_md

    config = load_common_config()
    dir_keys = []
    if entity_type == "skill":
        dir_keys = ["skill_dirs"]
    elif entity_type == "team":
        dir_keys = ["team_dirs"]
    else:
        dir_keys = ["skill_dirs", "team_dirs"]

    for dir_key in dir_keys:
        for dir_rel in config.get(dir_key, []):
            evals_file = REPO_ROOT / dir_rel / entity_name / "evals" / "evals.md"
            if not evals_file.exists():
                continue
            try:
                return parse_evals_md(evals_file)
            except Exception as e:
                logger.warning("Failed to parse evals file %s: %s", evals_file, e)
                return None
    return None

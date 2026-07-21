#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging
import re
import sys
from pathlib import Path

import yaml

from _yaml_utils import YAMLFixSuggestion, parse_frontmatter

logger = logging.getLogger(__name__)

# 配置项
AGENTS_DIR = "./agents"
SKILLS_DIR = "./skills"
NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9\-]*$')  # 小写字母、数字、连字符
MAX_NAME_LENGTH = 64


def check_agent_file(file_path: Path) -> bool:
    """检查单个 agent .md 文件的规范.

    Args:
        file_path: agent .md 文件的路径.

    Returns:
        是否通过检查（True 表示通过，False 表示失败）.
    """
    passed = True
    try:
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析 YAML frontmatter
        data = parse_frontmatter(content)

        # 检查 name 字段
        if "name" not in data:
            logger.info(f"⚠️ {file_path}: 缺少 'name' 字段")
            passed = False
        else:
            name = str(data["name"]).strip()

            # 检查 name 格式：只能包含小写字母、数字、连字符
            if not NAME_PATTERN.match(name):
                logger.info(f"⚠️ {file_path}: name 格式错误，只能包含小写字母、数字和连字符，且必须以字母或数字开头: '{name}'")
                passed = False

            # 检查 name 长度
            if len(name) > MAX_NAME_LENGTH:
                logger.info(f"⚠️ {file_path}: name 长度超标（{len(name)}/{MAX_NAME_LENGTH}）")
                passed = False

            # 检查 name 与文件名一致性（文件名去掉 .md 后缀应与 name 一致）
            file_name_without_ext = file_path.stem  # 去掉 .md 后缀
            if name != file_name_without_ext:
                logger.info(f"⚠️ {file_path}: name '{name}' 与文件名不一致（文件名: '{file_name_without_ext}.md'）")
                passed = False

        # 检查 skills 字段（如果存在）
        if "skills" in data:
            skills = data["skills"]
            if not isinstance(skills, list):
                logger.info(f"⚠️ {file_path}: skills 字段必须是列表")
                passed = False
            else:
                # 获取所有可用的 skill 目录
                skills_path = Path(SKILLS_DIR)
                if skills_path.exists():
                    available_skills = {skill_dir.name for skill_dir in skills_path.iterdir() if skill_dir.is_dir()}

                    # 检查每个 skill 是否存在
                    for skill in skills:
                        skill_name = str(skill).strip()
                        if skill_name not in available_skills:
                            logger.info(f"⚠️ {file_path}: skill '{skill_name}' 在 {SKILLS_DIR} 目录下不存在")
                            passed = False
                else:
                    logger.info(f"⚠️  {file_path}: 无法验证 skills，{SKILLS_DIR} 目录不存在")
                    passed = False

    except YAMLFixSuggestion as e:
        logger.info(f"❌ {file_path}: YAML 解析失败 — 字段值包含未转义的保留字符")
        logger.info(f"   原始错误: {e.original_error}")
        if e.suggestion:
            logger.info(f"   💡 建议修复: {e.suggestion}")
        passed = False
    except yaml.YAMLError as e:
        logger.info(f"⚠️ {file_path}: YAML 解析失败 - {str(e)}")
        passed = False
    except ValueError as e:
        logger.info(f"⚠️ {file_path}: Frontmatter 解析失败 - {str(e)}")
        passed = False
    except Exception as e:
        logger.info(f"⚠️ {file_path}: 未知错误 - {str(e)}")
        passed = False

    return passed


def main():
    """检查 agent .md 文件.

    如果提供了命令行参数（文件路径列表），则只检查这些文件。
    这用于 pre-commit hook，只检查 staged 文件。

    如果没有提供参数，则检查整个 ./agents/ 目录下的所有 .md 文件。

    Exits:
        0: 所有文件检查通过.
        1: 部分文件不符合规范.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    agents_path = Path(AGENTS_DIR)

    # 如果提供了命令行参数（staged 文件列表），只检查这些文件
    if len(sys.argv) > 1:
        agent_files = [Path(file_path) for file_path in sys.argv[1:]]
        # 过滤出真正的 .md 文件
        agent_files = [f for f in agent_files if f.suffix == '.md' and f.exists()]
    else:
        # 否则检查整个目录（向后兼容）
        if not agents_path.exists():
            logger.info(f"ℹ️  {AGENTS_DIR} 目录不存在，跳过检查")
            sys.exit(0)

        agent_files = list(agents_path.glob("*.md"))

    if not agent_files:
        logger.info(f"ℹ️  没有找到需要检查的 agent .md 文件")
        sys.exit(0)

    # 检查每个文件
    all_passed = True
    for file in agent_files:
        if not check_agent_file(file):
            all_passed = False

    # 输出结果并返回对应状态码
    if all_passed:
        logger.info("✅ 所有 agent .md 文件检查通过")
        sys.exit(0)
    else:
        logger.info("\n❌ 部分 agent .md 文件不符合规范，请修复后重新提交")
        sys.exit(1)


if __name__ == "__main__":
    main()

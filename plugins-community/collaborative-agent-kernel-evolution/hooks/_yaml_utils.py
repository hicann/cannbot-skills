# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""YAML frontmatter 解析工具 — 供 skills_linter / agents_linter 共用."""
import yaml


class YAMLFixSuggestion(Exception):
    """YAML 解析失败但可自动修复时抛出，携带修复建议."""

    def __init__(self, original_error: str, suggestion: str):
        self.original_error = original_error
        self.suggestion = suggestion
        super().__init__(original_error)


def parse_frontmatter(content: str) -> dict:
    """从 Markdown 文件中提取 YAML frontmatter.

    Args:
        content: Markdown 文件的完整内容.

    Returns:
        解析后的 YAML frontmatter 字典.

    Raises:
        ValueError: 当文件不以 '---' 开始或缺少闭合分隔符时抛出.
        YAMLFixSuggestion: YAML 解析失败但可通过加引号修复时抛出（携带建议）.
        yaml.YAMLError: YAML 解析失败且无法自动修复时抛出.
    """
    lines = content.split('\n')
    if not lines or lines[0].strip() != '---':
        raise ValueError("Markdown file does not start with YAML frontmatter delimiter '---'")

    # 找到第二个 '---' 分隔符
    frontmatter_lines = []
    found_closing = False
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            found_closing = True
            break
        frontmatter_lines.append(lines[i])

    if not found_closing:
        raise ValueError("Markdown file does not have closing YAML frontmatter delimiter '---'")

    frontmatter_text = '\n'.join(frontmatter_lines)

    # 尝试直接解析
    try:
        data = yaml.safe_load(frontmatter_text)
        if not isinstance(data, dict):
            raise ValueError("YAML frontmatter must be a mapping (dict), but got "
                             f"{type(data).__name__ if data is not None else 'None'}")
        return data
    except yaml.YAMLError as original_exc:
        # YAML 解析失败 — 对所有字段尝试 auto-fix 以生成修复建议，但不静默通过
        fixed_lines = []
        fix_applied = []  # 记录修复了哪些字段
        for line in frontmatter_lines:
            stripped = line.strip()
            # 检测 "key: value" 格式且值未加引号的行
            if ':' in stripped and not stripped.startswith('#'):
                key_part, _, val_part = stripped.partition(':')
                val = val_part.strip()
                if val and not val.startswith(('"', "'", '>', '|', '[', '{', '&', '*')):
                    # 用 yaml.dump 生成安全的标量表示
                    key = key_part.strip()
                    safe_scalar = yaml.dump(
                        {key: val},
                        default_flow_style=False,
                        allow_unicode=True,
                    ).strip()
                    if safe_scalar != stripped:
                        line = safe_scalar
                        fix_applied.append(safe_scalar)
            fixed_lines.append(line)

        if not fix_applied:
            # 没有可修复的字段 — 抛原始错误
            raise

        fixed_frontmatter_text = '\n'.join(fixed_lines)
        # auto-fix 后再次解析；若解析为 dict 则抛出带建议的异常，
        # 解析仍失败（yaml.YAMLError）则异常自然向上传播。
        fixed_data = yaml.safe_load(fixed_frontmatter_text)
        if isinstance(fixed_data, dict):
            # auto-fix 能修复 — 抛出带建议的异常（linting 仍不通过）
            suggestion_lines = '\n'.join(f"      {s}" for s in fix_applied)
            suggestion = f"将包含保留字符的值用引号包裹:\n{suggestion_lines}"
            raise YAMLFixSuggestion(str(original_exc), suggestion) from original_exc
        # auto-fix 结果不是 dict 或其他情况 — 抛原始错误
        raise

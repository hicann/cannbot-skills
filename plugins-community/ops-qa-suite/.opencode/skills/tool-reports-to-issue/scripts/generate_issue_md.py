#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GitCode Issue MD 文档生成器

提供统一的接口生成符合 GitCode Issue 模板格式的 MD 文档。
其他 skill 可以调用此脚本生成 issue 报告。
"""

import argparse
from pathlib import Path


BUG_REPORT_TEMPLATE = (
    "Thanks for sending an issue! Please fill in the following template "
    "to help quickly solve your problem.\n\n"
    "### Describe the current behavior / 问题描述\n\n{description}\n\n"
    "### Environment / 环境信息\n\n{environment}\n\n"
    "### Steps to reproduce the issue / 重现步骤\n\n{steps}\n\n"
    "### Describe the expected behavior / 预期结果\n\n{expected}\n\n"
    "### Related log / screenshot / 日志 / 截图\n\n{logs}\n\n"
    "### Special notes for this issue/备注 (Optional / 选填)\n\n{notes}"
)

DOCUMENTATION_TEMPLATE = (
    "Thanks for sending an issue! Please fill in the following template "
    "to help quickly solve your problem.\n\n"
    "### Document Link（文档链接）\n\n{doc_link}\n\n"
    "### Issues Section（问题文档片段）\n\n{issues_section}\n\n"
    "### Existing Issues（存在的问题）\n\n{existing_issues}"
)

REQUIREMENT_TEMPLATE = (
    "Thanks for sending an requirement! Please fill in the following template "
    "to help quickly solve your problem.\n\n"
    "### Backgroud（背景信息）\n\n{background}\n\n"
    "### Origin（信息来源）\n\n{origin}\n\n"
    "### Benefit / Necessity （价值/作用）\n\n{benefit}\n\n"
    "### Design（设计方案）\n\n{design}"
)

QUESTION_TEMPLATE = """Welcome to ask questions and discuss with other members.

### 问题描述

{question}"""


def generate_title(template_type: str, summary: str, include_ai_tag: bool = True) -> str:
    """生成符合 GitCode 模板格式的标题
    
    Args:
        template_type: 模板类型 (bug-report, documentation, requirement, question, blank)
        summary: 问题简述
        include_ai_tag: 是否包含 [AI 识别] 标记
    
    Returns:
        标题字符串
    """
    ai_tag = "[AI 识别] " if include_ai_tag else ""
    
    templates = {
        'bug-report': f'[Bug-Report|缺陷反馈]: {ai_tag}{summary}',
        'documentation': f'[Documentation|文档反馈]: {ai_tag}{summary}',
        'requirement': f'[Requirement|需求建议]: {ai_tag}{summary}',
        'question': f'[Question|问题咨询]: {ai_tag}{summary}',
        'blank': f'{ai_tag}{summary}',
    }
    
    return templates.get(template_type.lower(), templates['blank'])


def generate_description(template_type: str, **kwargs) -> str:
    """生成符合 GitCode 模板格式的 Description
    
    Args:
        template_type: 模板类型
        **kwargs: 模板字段内容
    
    Returns:
        Description 字符串
    """
    templates = {
        'bug-report': BUG_REPORT_TEMPLATE,
        'documentation': DOCUMENTATION_TEMPLATE,
        'requirement': REQUIREMENT_TEMPLATE,
        'question': QUESTION_TEMPLATE,
    }
    
    template = templates.get(template_type.lower())
    if template is None:
        return kwargs.get('content', '')
    
    if template_type.lower() == 'bug-report':
        return template.format(
            description=kwargs.get('description', ''),
            environment=kwargs.get('environment', ''),
            steps=kwargs.get('steps', ''),
            expected=kwargs.get('expected', ''),
            logs=kwargs.get('logs', ''),
            notes=kwargs.get('notes', ''),
        )
    elif template_type.lower() == 'documentation':
        return template.format(
            doc_link=kwargs.get('doc_link', ''),
            issues_section=kwargs.get('issues_section', ''),
            existing_issues=kwargs.get('existing_issues', ''),
        )
    elif template_type.lower() == 'requirement':
        return template.format(
            background=kwargs.get('background', ''),
            origin=kwargs.get('origin', ''),
            benefit=kwargs.get('benefit', ''),
            design=kwargs.get('design', ''),
        )
    elif template_type.lower() == 'question':
        return template.format(
            question=kwargs.get('question', ''),
        )
    
    return kwargs.get('content', '')


def generate_issue_md(
    repo: str,
    template_type: str,
    summary: str,
    description: str,
    labels: str | None = None,
    output_dir: str = './issues',
    include_ai_tag: bool = True,
    issue_suffix: str | None = None,
) -> str:
    """生成 GitCode Issue MD 文档
    
    Args:
        repo: 仓库名称
        template_type: 模板类型
        summary: 问题简述
        description: Description 内容
        labels: 标签
        output_dir: 输出目录
        include_ai_tag: 是否包含 [AI 识别] 标记
        issue_suffix: Issue 文件名后缀（如 variable_not_defined），用于区分不同类型
    
    Returns:
        输出文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    title = generate_title(template_type, summary, include_ai_tag)
    
    label_str = labels if labels else ('bug-report' if template_type == 'bug-report' else '')
    
    # 文件名：repo_issue_suffix_issue.md 或 repo_issue.md
    if issue_suffix:
        filename = f'{repo}_{issue_suffix}_issue.md'
    else:
        filename = f'{repo}_issue.md'
    
    content = [
        f'# {title}',
        '',
        f'**标签**: `{label_str}`',
        '',
        '---',
        '',
        description,
        '',
        '---',
        '',
        f'**提交地址**: https://gitcode.com/CANN/{repo}/issues/new',
    ]
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='GitCode Issue MD 文档生成器')
    parser.add_argument('--repo', required=True, help='仓库名称')
    parser.add_argument('--template', default='bug-report', 
                        choices=['bug-report', 'documentation', 'requirement', 'question', 'blank'],
                        help='模板类型')
    parser.add_argument('--summary', required=True, help='问题简述')
    parser.add_argument('--description', default='', help='问题描述')
    parser.add_argument('--environment', default='', help='环境信息')
    parser.add_argument('--steps', default='', help='重现步骤')
    parser.add_argument('--expected', default='', help='预期结果')
    parser.add_argument('--logs', default='', help='日志/截图')
    parser.add_argument('--notes', default='', help='备注')
    parser.add_argument('--labels', default='', help='标签')
    parser.add_argument('--output-dir', default='./issues', help='输出目录')
    parser.add_argument('--no-ai-tag', action='store_true', help='不包含 [AI 识别] 标记')
    
    args = parser.parse_args()
    
    if args.template == 'bug-report':
        description = generate_description(
            'bug-report',
            description=args.description,
            environment=args.environment,
            steps=args.steps,
            expected=args.expected,
            logs=args.logs,
            notes=args.notes,
        )
    else:
        description = args.description
    
    output_file = generate_issue_md(
        repo=args.repo,
        template_type=args.template,
        summary=args.summary,
        description=description,
        labels=args.labels,
        output_dir=args.output_dir,
        include_ai_tag=not args.no_ai_tag,
    )
    
    print(f'Issue MD 文档已生成: {output_file}')


if __name__ == '__main__':
    main()
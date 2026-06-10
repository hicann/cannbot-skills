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
"""Markdown 断链扫描与修复工具

扫描仓库 Markdown 文件中的断链，支持：
- 自动检测断链类型
- 自动修复路径错误
- 创建修复 PR
"""

import os
import re
import subprocess
import argparse
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.gitcode.com/api/v5"


@dataclass
class BrokenLink:
    """断链信息"""
    readme_path: str
    link_text: str
    link_path: str
    target_file: str
    broken_type: str  # 'path_error', 'file_not_exist', 'line_break', 'external', 'anchor_not_exist'
    fix_strategy: str  # 'fix_path', 'delete_link', 'merge_line', 'skip'
    line_number: int = 0  # 问题所在行号
    fixed: bool = False


def get_token():
    """从 git credential store 获取 GitCode token"""
    git_cmd = shutil.which('git') or 'git'
    result = subprocess.run(
        [git_cmd, "credential", "fill"],
        input="protocol=https\nhost=gitcode.com\n\n",
        capture_output=True,
        text=True
    )
    lines = result.stdout.strip().split("\n")
    token = None
    for line in lines:
        if line.startswith("password="):
            token = line.split("=", 1)[1]
            break
    return token


def scan_markdown_files(repo_path: Path, scope: str = "all") -> List[Path]:
    """扫描 Markdown 文件"""
    if scope == "readme":
        return list(repo_path.rglob("README.md"))
    return list(repo_path.rglob("*.md"))


def check_anchor_exists(file_path: Path, anchor: str) -> bool:
    """检查锚点是否存在（支持 Markdown 和 HTML 格式）
    
    Args:
        file_path: 目标文件路径
        anchor: 锚点名称
        
    Returns:
        锚点是否存在
    """
    try:
        content = file_path.read_text()
        
        # Markdown 格式锚点: # anchor 或 <a name="anchor"></a>
        # 1. Markdown heading: # Heading 或 ## Heading
        #    锚点格式: heading (小写，空格替换为-)
        md_anchor_pattern = re.compile(
            r'^#+\s+' + re.escape(anchor.replace('-', ' ').replace('-', ' ')),
            re.IGNORECASE | re.MULTILINE
        )
        
        # 2. HTML anchor: <a name="anchor"></a>
        html_anchor_pattern = re.compile(
            r'<a\s+name\s*=\s*["\']' + re.escape(anchor) + r'["\']',
            re.IGNORECASE
        )
        
        # 3. Markdown 自动锚点: {#anchor}
        md_custom_anchor_pattern = re.compile(
            r'\{#' + re.escape(anchor) + r'\}',
            re.IGNORECASE
        )
        
        # 检查所有格式
        if html_anchor_pattern.search(content):
            return True
        
        if md_custom_anchor_pattern.search(content):
            return True
        
        # Markdown heading 锚点需要特殊处理
        # 标题 "环境准备" 对应锚点 "环境准备" 或 "prepare&install"
        # 需要检查标题文本是否匹配锚点
        heading_pattern = re.compile(r'^#+\s+([^\n]+)', re.MULTILINE)
        for match in heading_pattern.finditer(content):
            heading_text = match.group(1).strip()
            # 标题锚点: 移除特殊字符，空格替换为-，转小写
            heading_anchor = re.sub(r'[^\w\s-]', '', heading_text)
            heading_anchor = heading_anchor.replace(' ', '-').lower()
            
            if heading_anchor == anchor.lower() or heading_text == anchor:
                return True
        
        return False
        
    except Exception as e:
        # 如果无法读取文件，假设锚点存在（避免误判）
        return True


def check_link_validity(readme_path: Path, link: str) -> Tuple[bool, str, str]:
    """检查链接有效性
    
    Returns:
        (is_valid, broken_type, fix_strategy)
    """
    # 外部链接跳过
    if link.startswith("http://") or link.startswith("https://"):
        return True, "external", "skip"
    
    # 分离锚点
    anchor = None
    file_link = link
    if '#' in link:
        parts = link.split('#', 1)
        file_link = parts[0]
        anchor = parts[1]
    
    # 解析相对路径
    if file_link.startswith("./"):
        target_path = readme_path.parent / file_link[2:]
    elif file_link.startswith("../"):
        parts = file_link.split("/")
        target_path = readme_path.parent
        for part in parts:
            if part == "..":
                target_path = target_path.parent
            elif part and part != ".":
                target_path = target_path / part
    elif file_link.startswith("examples/") or file_link.startswith("docs/"):
        target_path = readme_path.parent / file_link
    else:
        target_path = readme_path.parent / file_link
    
    try:
        target_path = Path(os.path.normpath(str(target_path)))
    except Exception:
        return False, "path_error", "delete_link"
    
    # 检查文件是否存在
    if target_path.exists():
        # 文件存在，检查锚点（如果有）
        if anchor:
            # 检查锚点是否存在
            if check_anchor_exists(target_path, anchor):
                return True, "", ""
            else:
                # 文件存在但锚点不存在
                return False, "anchor_not_exist", "skip"
        else:
            # 无锚点，文件存在，链接有效
            return True, "", ""
    
    # 检查是否是换行导致的断链
    if "\n" in link or link.endswith("/"):
        return False, "line_break", "merge_line"
    
    # 文件不存在
    return False, "file_not_exist", "delete_link"


def scan_broken_links(repo_path: Path, scope: str = "all") -> List[BrokenLink]:
    """扫描断链"""
    broken_links = []
    md_files = scan_markdown_files(repo_path, scope)
    
    for md_file in md_files:
        try:
            content = md_file.read_text()
            original_content = content
            
            # 移除代码块（防止代码中的方括号被误识别为链接）
            content_no_codeblock = re.sub(r'```[\s\S]*?```', '', content)
            
            pattern = r'\[([^\]]+)\]\(((?:[^()]*|\([^()]*\))*)\)'
            
            # 在移除代码块后的内容中查找链接
            for match in re.finditer(pattern, content_no_codeblock):
                link_text = match.group(1)
                link = match.group(2)
                
                if link.startswith("http://") or link.startswith("https://") or link.startswith("#"):
                    continue
                
                is_valid, broken_type, fix_strategy = check_link_validity(md_file, link)
                
                if not is_valid:
                    # 在原始内容中查找对应的行号（基于链接文本）
                    original_pattern = re.escape(link_text) + r'\]\(' + re.escape(link)
                    original_match = re.search(original_pattern, original_content)
                    
                    if original_match:
                        # 计算原始文件中的行号
                        match_position = original_match.start()
                        line_number = original_content[:match_position].count('\n') + 1
                    else:
                        # 如果在原始内容中找不到，使用移除代码块后的位置估算
                        match_position = match.start()
                        line_number = content_no_codeblock[:match_position].count('\n') + 1
                    
                    # 计算目标文件路径
                    if link.startswith("./"):
                        target = str(md_file.parent / link[2:])
                    else:
                        target = str(md_file.parent / link)
                    
                    broken_links.append(BrokenLink(
                        readme_path=str(md_file),
                        link_text=link_text,
                        link_path=link,
                        target_file=target,
                        broken_type=broken_type,
                        fix_strategy=fix_strategy,
                        line_number=line_number
                    ))
        
        except Exception as e:
            logger.debug("Failed to scan broken links in %s: %s", md_file, e)
            continue
    
    return broken_links


def fix_broken_link(readme_path: Path, broken: BrokenLink, dry_run: bool = False) -> bool:
    """修复断链"""
    try:
        content = readme_path.read_text()
        
        if broken.broken_type == "line_break":
            # 合并换行链接
            fixed_link = broken.link_path.replace("\n", "").replace("/", "/")
            pattern = re.escape(broken.link_path.split("\n")[0])
            new_content = re.sub(
                r'\[([^\]]+)\]\(' + pattern + r'[^\)]*\)',
                f'[{broken.link_text}]({fixed_link})',
                content
            )
        
        elif broken.broken_type == "path_error":
            # 尝试修正路径（需要具体情况具体分析）
            # 这里仅处理常见的 arch35 层级问题
            if "examples/test_" in broken.link_path and "arch35" not in broken.link_path:
                fixed_link = broken.link_path.replace("examples/", "examples/arch35/")
                new_content = content.replace(broken.link_path, fixed_link)
            else:
                new_content = content
        
        elif broken.broken_type == "file_not_exist":
            # 删除不存在链接的行
            pattern = r'\|[^\|]*\[' + re.escape(broken.link_text) + r'\]\([^\)]*\)[^\|]*\|[^\|]*\|'
            new_content = re.sub(pattern, '', content)
            # 清理多余的空行
            new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)
        
        else:
            return False
        
        if not dry_run:
            readme_path.write_text(new_content)
        
        return True
    
    except Exception as e:
        logger.error(f"修复失败: {e}", exc_info=True)
        return False


def generate_report(repo_path: Path, broken_links: List[BrokenLink], 
                    output_dir: str = "reports/broken-link-fixer") -> str:
    """生成断链报告"""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    repo_name = repo_path.name
    filename = f"{repo_name}_broken_link_report_{timestamp}.md"
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 分类统计
    type_counts = {}
    for link in broken_links:
        type_counts[link.broken_type] = type_counts.get(link.broken_type, 0) + 1
    
    content = f"""# Markdown 断链扫描报告

## 基本信息

| 项目 | 值 |
|------|-----|
| **仓库** | {repo_name} |
| **扫描时间** | {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} |
| **扫描范围** | README.md |
| **断链总数** | {len(broken_links)} |

## 断链分类统计

| 断链类型 | 数量 | 可自动修复 |
|---------|:---:|:---:|
| 路径错误 | {type_counts.get('path_error', 0)} | ✅ |
| 链接换行 | {type_counts.get('line_break', 0)} | ✅ |
| 文件不存在 | {type_counts.get('file_not_exist', 0)} | ❌ |
| 锚点不存在 | {type_counts.get('anchor_not_exist', 0)} | ❌ |
| 外部链接失效 | {type_counts.get('external', 0)} | ❌ |

## 详细断链列表

### 可自动修复

| README | 行号 | 链接文字 | 链接路径 | 断链类型 |
|--------|:---:|---------|---------|---------|
"""
    
    for link in broken_links:
        if link.fix_strategy in ['fix_path', 'merge_line']:
            rel_path = Path(link.readme_path).relative_to(repo_path)
            content += (
                f"| {rel_path} | {link.line_number} | {link.link_text} "
                f"| `{link.link_path}` | {link.broken_type} |\n"
            )
    
    content += "\n### 需人工处理\n\n| README | 行号 | 链接文字 | 链接路径 | 断链类型 |\n|--------|:---:|---------|---------|---------|\n"
    
    for link in broken_links:
        if link.fix_strategy in ['delete_link', 'skip']:
            rel_path = Path(link.readme_path).relative_to(repo_path)
            content += (
                f"| {rel_path} | {link.line_number} | {link.link_text} "
                f"| `{link.link_path}` | {link.broken_type} |\n"
            )
    
    content += f"""
## 修复建议

1. 使用 `fixer-broken-link --fix` 自动修复可修复的断链
2. 手动处理需要补充文档的断链

---
生成时间: {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    filepath.write_text(content)
    return str(filepath)


def create_fix_pr(repo_path: Path, title: str, token: str, 
                  head: str = None, base: str = "master") -> Dict:
    """创建修复 PR"""
    # 获取当前分支
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True,
        cwd=str(repo_path)
    )
    current_branch = result.stdout.strip() or "master"
    
    # 获取 remotes
    result = subprocess.run(
        ["git", "remote", "-v"],
        capture_output=True, text=True,
        cwd=str(repo_path)
    )
    
    remotes = {}
    for line in result.stdout.strip().split("\n"):
        if "(fetch)" in line:
            parts = line.split()
            remote_name = parts[0]
            remote_url = parts[1]
            
            if "gitcode.com" in remote_url:
                if remote_url.startswith("https://"):
                    path = remote_url.split("gitcode.com/")[-1].rstrip("/")
                elif remote_url.startswith("git@"):
                    path = remote_url.split(":")[-1].rstrip("/")
                else:
                    continue
                
                if path.endswith(".git"):
                    path = path[:-4]
                
                parts = path.split("/")
                if len(parts) >= 2:
                    remotes[remote_name] = {
                        'owner': parts[0],
                        'repo': parts[1]
                    }
    
    # 确定目标仓库
    if 'origin' in remotes:
        target_owner = remotes['origin']['owner']
        target_repo = remotes['origin']['repo']
    else:
        raise Exception("无法确定目标仓库")
    
    # 确定源分支
    head_branch = head or current_branch
    if 'fork' in remotes:
        fork_info = remotes['fork']
        head_branch = f"{fork_info['owner']}:{head_branch}"
        fork_path = f"{fork_info['owner']}/{fork_info['repo']}"
    else:
        fork_path = None
    
    # 创建 PR
    url = f"{API_BASE}/repos/{target_owner}/{target_repo}/pulls"
    
    data = {
        "access_token": token,
        "title": title,
        "head": head_branch,
        "base": base,
    }
    
    if fork_path:
        data["fork_path"] = fork_path
    
    resp = requests.post(url, json=data)
    
    if resp.status_code != 201:
        raise Exception(f"PR 创建失败: {resp.status_code} - {resp.text}")
    
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Markdown 断链扫描与修复")
    parser.add_argument("--repo", required=True, help="仓库名或路径")
    parser.add_argument("--scope", default="readme", choices=["all", "readme"], 
                        help="扫描范围")
    parser.add_argument("--fix", action="store_true", help="自动修复断链")
    parser.add_argument("--create-pr", action="store_true", help="创建修复 PR")
    parser.add_argument("--dry-run", action="store_true", help="模拟修复")
    parser.add_argument("--pr-title", default=None, help="PR 标题")
    parser.add_argument("--output-dir", default="reports/broken-link-fixer", 
                        help="报告输出目录")
    
    args = parser.parse_args()
    
    # 检测仓库路径
    repo_path = Path(args.repo)
    if not repo_path.exists():
        repo_path = Path.cwd() / args.repo
        if not repo_path.exists():
            print(f"错误: 仓库路径不存在: {args.repo}")
            return 1
    
    print(f"扫描仓库: {repo_path}")
    print(f"扫描范围: {args.scope}")
    
    # 扫描断链
    broken_links = scan_broken_links(repo_path, args.scope)
    
    if not broken_links:
        print("✅ 未发现断链")
        return 0
    
    print(f"发现 {len(broken_links)} 个断链")
    
    # 分类统计
    fixable = [l for l in broken_links if l.fix_strategy in ['fix_path', 'merge_line']]
    manual = [l for l in broken_links if l.fix_strategy in ['delete_link', 'skip']]
    
    print(f"  可自动修复: {len(fixable)}")
    print(f"  需人工处理: {len(manual)}")
    
    # 生成报告
    report_file = generate_report(repo_path, broken_links, args.output_dir)
    print(f"\n报告文件: {report_file}")
    
    # 自动修复
    if args.fix and fixable:
        print(f"\n开始修复 {len(fixable)} 个断链...")
        
        fixed_count = 0
        for link in fixable:
            readme_path = Path(link.readme_path)
            if fix_broken_link(readme_path, link, args.dry_run):
                link.fixed = True
                fixed_count += 1
                rel_path = readme_path.relative_to(repo_path)
                logger.info(f"修复成功: {rel_path}: {link.link_text}")
        
        print(f"\n修复完成: {fixed_count}/{len(fixable)}")
        
        if args.dry_run:
            print("[dry-run] 未实际修改文件")
    
    # 创建 PR
    if args.create_pr and args.fix and not args.dry_run:
        token = get_token()
        if not token:
            print("错误: 无法获取 GitCode token")
            return 1
        
        pr_title = args.pr_title or f"fix: 修复 README 断链问题（共 {len(fixable)} 处）"
        
        try:
            pr = create_fix_pr(repo_path, pr_title, token)
            print(f"\n✅ PR #{pr['number']} 已创建")
            print(f"   URL: {pr.get('html_url', 'N/A')}")
        except Exception as e:
            print(f"\n❌ PR 创建失败: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
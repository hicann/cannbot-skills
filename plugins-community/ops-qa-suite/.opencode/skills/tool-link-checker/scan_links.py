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
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# Add scripts directory to path for repo_detector import
_scripts_dir = Path(__file__).resolve().parent.parent.parent / 'scripts'
sys.path.insert(0, str(_scripts_dir))
from repo_detector import get_repo_root as _detect_repo_root

# Module-level variable to store detection method from repo_detector
_detection_method = None

# Excluded directories during markdown file scanning
EXCLUDED_DIRS = {'third_party', '.git', '.opencode', 'build', 'build_out', '__pycache__'}


def _should_check_link(link: str) -> bool:
    """检查链接是否需要验证
    
    Args:
        link: 链接路径
        
    Returns:
        是否需要检查该链接
    """
    if link.startswith('http'):
        return False
    
    is_md_or_code = link.endswith('.md') or link.endswith('.cpp') or link.endswith('.h')
    is_relative = '../' in link or './' in link
    
    if not (is_md_or_code or is_relative):
        return False
    
    is_image = (link.endswith('.png') or link.endswith('.jpg') or
                link.endswith('.jpeg') or link.endswith('.gif') or
                link.endswith('.svg'))
    
    return not is_image


def find_all_md_links(md_file):
    """查找markdown文件中的所有链接"""
    links = []
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            # 匹配markdown链接格式 [text](link)
            pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            for i, line in enumerate(lines, 1):
                matches = re.findall(pattern, line)
                for text, link in matches:
                    # 只检查.md结尾的链接或相对路径链接，排除外部链接
                    if _should_check_link(link):
                        links.append({
                            'file': md_file,
                            'text': text,
                            'link': link,
                            'line': i
                        })
    except (UnicodeDecodeError, OSError):
        pass
    return links


def resolve_link(md_file, link):
    """解析相对路径链接为绝对路径"""
    md_dir = Path(md_file).parent
    try:
        # 移除URL参数和锚点
        if '?' in link:
            link = link.split('?')[0]
        if '"' in link:  # 处理markdown图片语法中的title
            link = link.split('"')[0].strip()
        
        # 处理相对路径
        target = md_dir / link
        # 解析路径并规范化
        target = target.resolve()
        return str(target)
    except Exception:
        return None


def check_link_exists(target_path, repo_root):
    """检查链接目标是否存在"""
    # 移除锚点部分
    if '#' in target_path:
        target_path = target_path.split('#')[0]
    
    # 直接检查文件是否存在
    try:
        exists = Path(target_path).exists()
        return exists
    except Exception:
        return False


def categorize_broken_link(link_info):
    """分类断链类型"""
    link = link_info['link_path']
    resolved = link_info.get('resolved_path', '')
    
    # 检查是否是路径错误
    if '../index/index_fill' in link:
        return '外部链接(index仓库)'
    if re.match(r'../ops-\w+/', link):
        return '外部链接(其他仓库)'
    if 'docs/zh/context/quick_install' in link or 'docs/zh/context/build' in link:
        return '路径错误(context应为install)'
    if '//zh' in link:
        return '路径错误(双斜杠)'
    if 'docs/context/' in link and 'docs/zh/context/' not in link:
        return '路径错误(缺少zh层级)'
    
    # 检查文件类型
    if link.endswith('.cpp') or link.endswith('.h'):
        return '源码文件不存在'
    if link.endswith('.md'):
        return '文档文件不存在'
    
    return '其他问题'


def parse_args():
    parser = argparse.ArgumentParser(description='Markdown断链扫描')
    parser.add_argument('repo_name', type=str, nargs='?', default=None,
                        help='仓库名称（ops-math/ops-nn/ops-transformer/ops-cv）')
    parser.add_argument('--repo_root', type=str, default=None,
                        help='仓库根目录绝对路径')
    return parser.parse_args()


def get_repo_root(repo_name, repo_root_arg):
    """推断仓库根目录

    使用 repo_detector 进行智能检测：
    1. 用户通过 --repo_root 指定的绝对路径
    2. 当前目录直接在 ops-* 仓库中
    3. 当前目录包含 ops-* 子目录
    4. 向上遍历父目录查找

    Returns:
        Path: 仓库根目录
    """
    global _detection_method
    try:
        path, method = _detect_repo_root(repo_name, repo_root_arg)
        _detection_method = method
        return path
    except ValueError:
        if repo_root_arg:
            _detection_method = 'legacy_fallback (repo_root_arg)'
            return Path(repo_root_arg)
        if repo_name:
            fallback = Path.cwd() / repo_name
            if fallback.exists():
                _detection_method = 'legacy_fallback (cwd/repo_name)'
                return fallback
        raise


def main():
    args = parse_args()
    repo_root = get_repo_root(args.repo_name, args.repo_root)
    repo_name = args.repo_name or repo_root.name
    
    if not repo_root.exists():
        print(f"错误: 仓库路径不存在: {repo_root}")
        print(f"提示: 请确保当前工作目录下存在 {repo_name}/ 子目录")
        return
    
    # 查找所有md文件
    md_files = []
    for root, dirs, files in os.walk(repo_root):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    print(f"找到 {len(md_files)} 个markdown文件")
    print(f"检测方法: {_detection_method or 'unknown'}")
    
    # 检查所有链接
    broken_links = []
    external_links = []
    checked_count = 0
    
    repo_root_str = str(repo_root)
    
    for md_file in md_files:
        links = find_all_md_links(md_file)
        for link_info in links:
            checked_count += 1
            link = link_info['link']
            target_path = resolve_link(md_file, link)
            
            error_type = categorize_broken_link({
                'link_path': link,
                'resolved_path': target_path
            })
            
            if '外部链接' in error_type:
                external_links.append({
                    'source_file': md_file.replace(repo_root_str, '.'),
                    'line': link_info['line'],
                    'link_text': link_info['text'],
                    'link_path': link,
                    'error_type': error_type
                })
            elif target_path and not check_link_exists(target_path, repo_root_str):
                broken_links.append({
                    'source_file': md_file.replace(repo_root_str, '.'),
                    'line': link_info['line'],
                    'link_text': link_info['text'],
                    'link_path': link,
                    'resolved_path': target_path.replace(repo_root_str, '.') if target_path else '无法解析',
                    'error_type': error_type
                })
    
    print(f"\n检查了 {checked_count} 个链接")
    print(f"发现 {len(broken_links)} 个断链")
    print(f"发现 {len(external_links)} 个外部链接\n")
    
    # 按错误类型分组
    error_types = defaultdict(list)
    for bl in broken_links:
        error_types[bl['error_type']].append(bl)
    
    # 输出结果
    print("=" * 80)
    print("断链详细清单:")
    print("=" * 80)
    
    for i, bl in enumerate(broken_links, 1):
        print(f"\n{i}. 文件: {bl['source_file']}")
        print(f"   行号: {bl['line']}")
        print(f"   链接文本: {bl['link_text']}")
        print(f"   链接路径: {bl['link_path']}")
        print(f"   解析路径: {bl['resolved_path']}")
        print(f"   错误类型: {bl['error_type']}")
    
    # 输出分类统计
    print("\n" + "=" * 80)
    print("断链分类统计:")
    print("=" * 80)
    for error_type, links in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{error_type}: {len(links)}个")
        for link in links[:3]:  # 只显示前3个示例
            print(f"  - {link['source_file']}:{link['line']} -> {link['link_path']}")
        if len(links) > 3:
            print(f"  ... 还有 {len(links)-3} 个类似问题")
    
    # 保存结果到文件
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    output_file = f'/tmp/{repo_name}_broken_links_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{repo_name}仓库断链报告\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"扫描时间: {timestamp}\n")
        f.write(f"断链总数: {len(broken_links)}\n")
        f.write(f"外部链接数: {len(external_links)}\n")
        f.write(f"检查链接数: {checked_count}\n")
        f.write(f"扫描md文件数: {len(md_files)}\n")
        f.write("=" * 80 + "\n\n")
        
        # 按错误类型分组输出
        for error_type, links in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"\n{error_type}: {len(links)}个\n")
            f.write("-" * 80 + "\n")
            for link in links:
                f.write(f"文件: {link['source_file']}\n")
                f.write(f"行号: {link['line']}\n")
                f.write(f"链接文本: {link['link_text']}\n")
                f.write(f"链接路径: {link['link_path']}\n")
                f.write(f"解析路径: {link['resolved_path']}\n\n")
        
        # 输出外部链接
        if external_links:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"外部链接: {len(external_links)}个\n")
            f.write("-" * 80 + "\n")
            for link in external_links[:10]:  # 只显示前10个
                f.write(f"{link['source_file']}:{link['line']} -> {link['link_path']}\n")
    
    print(f"\n完整报告已保存到: {output_file}")
    print(f"断链率: {len(broken_links)/checked_count*100:.2f}%")


if __name__ == '__main__':
    main()
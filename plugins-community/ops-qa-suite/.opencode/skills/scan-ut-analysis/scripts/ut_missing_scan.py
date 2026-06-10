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
"""
UT 缺失扫描脚本

用法:
    python ut_missing_scan.py --repo ops-nn --output reports/ops-nn_ut_missing_report.json
    python ut_missing_scan.py --repo ops-math --op-list reports/op_list/ops-math_operator_list.md
"""

import os
import re
import json
import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'scripts'))
from repo_detector import get_repo_root, get_reports_output_dir
from repo_discovery import RepoDiscovery
from config_loader import get_structure_dirs, get_ut_structure, get_architecture_dirs

STRUCTURE = get_structure_dirs()
UT_DIRS = get_ut_structure()
ARCH_DIRS = get_architecture_dirs()


def get_category_order(repo_type, repo_root=None):
    if repo_root:
        return RepoDiscovery.discover_categories(repo_root)
    return []


def get_category_sort_key(cat, repo_type, repo_root=None):
    order = get_category_order(repo_type, repo_root)
    if cat.startswith('experimental/'):
        subcat = cat.split('/')[1]
        base_order = order.index(subcat) if subcat in order else 999
        return (1, base_order, cat)
    elif cat in order:
        return (0, order.index(cat), cat)
    else:
        return (2, 999, cat)


def parse_operator_list(op_list_file):
    ops = []
    with open(op_list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('|') and not line.startswith('| 序号') and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5 and parts[1].isdigit():
                    ops.append({
                        'idx': int(parts[1]),
                        'category': parts[2],
                        'name': parts[3],
                        'path': parts[4].strip('`')
                    })
    return ops


def check_ir_proto(op_dir):
    op_graph_dir = os.path.join(op_dir, STRUCTURE.get('graph', 'op_graph'))
    
    if not os.path.exists(op_graph_dir):
        return False
    
    if os.path.isdir(op_graph_dir):
        for f in os.listdir(op_graph_dir):
            if f.endswith('_proto.h'):
                return True
    
    return False


def check_infershape(op_dir):
    op_host_dir = os.path.join(op_dir, STRUCTURE['host'])
    ut_host_dir = os.path.join(op_dir, UT_DIRS['host_ut'])
    
    if not os.path.exists(op_host_dir):
        return False, False, False
    
    sources = []
    if os.path.isdir(op_host_dir):
        for f in os.listdir(op_host_dir):
            if f.endswith('_infershape.cpp') and '_arch' not in f:
                sources.append(os.path.join(op_host_dir, f))
    
    for arch in ARCH_DIRS:
        arch_dir = os.path.join(op_host_dir, arch)
        if os.path.exists(arch_dir) and os.path.isdir(arch_dir):
            for f in os.listdir(arch_dir):
                if '_infershape' in f and f.endswith('.cpp'):
                    sources.append(os.path.join(arch_dir, f))
    
    if os.path.isdir(op_host_dir):
        for f in os.listdir(op_host_dir):
            if '_infershape_arch' in f and f.endswith('.cpp'):
                sources.append(os.path.join(op_host_dir, f))
    
    uts = []
    if os.path.exists(ut_host_dir) and os.path.isdir(ut_host_dir):
        for f in os.listdir(ut_host_dir):
            if f.startswith('test_') and '_infershape' in f and f.endswith('.cpp'):
                uts.append(os.path.join(ut_host_dir, f))
        for arch in ARCH_DIRS:
            arch_ut_dir = os.path.join(ut_host_dir, arch)
            if os.path.exists(arch_ut_dir) and os.path.isdir(arch_ut_dir):
                for f in os.listdir(arch_ut_dir):
                    if f.startswith('test_') and '_infershape' in f and f.endswith('.cpp'):
                        uts.append(os.path.join(arch_ut_dir, f))
    
    has_ir_proto = check_ir_proto(op_dir)
    
    return len(sources) > 0, len(uts) > 0, has_ir_proto


def check_tiling(op_dir):
    op_host_dir = os.path.join(op_dir, STRUCTURE['host'])
    ut_host_dir = os.path.join(op_dir, UT_DIRS['host_ut'])
    
    if not os.path.exists(op_host_dir):
        return False, False
    
    public_sources = []
    arch_sources = []
    
    if os.path.isdir(op_host_dir):
        for f in os.listdir(op_host_dir):
            if f.endswith('_tiling.cpp'):
                if '_arch' not in f:
                    public_sources.append(os.path.join(op_host_dir, f))
                else:
                    arch_sources.append(os.path.join(op_host_dir, f))
    
    for arch in ARCH_DIRS:
        arch_dir = os.path.join(op_host_dir, arch)
        if os.path.exists(arch_dir) and os.path.isdir(arch_dir):
            for f in os.listdir(arch_dir):
                if '_tiling' in f and f.endswith('.cpp'):
                    arch_sources.append(os.path.join(arch_dir, f))
    
    sources = public_sources + arch_sources
    
    public_uts = []
    arch_uts = []
    
    if os.path.exists(ut_host_dir) and os.path.isdir(ut_host_dir):
        for f in os.listdir(ut_host_dir):
            if f.startswith('test_') and '_tiling' in f and f.endswith('.cpp'):
                if '_arch' not in f:
                    public_uts.append(os.path.join(ut_host_dir, f))
                else:
                    arch_uts.append(os.path.join(ut_host_dir, f))
        for arch in ARCH_DIRS:
            arch_ut_dir = os.path.join(ut_host_dir, arch)
            if os.path.exists(arch_ut_dir) and os.path.isdir(arch_ut_dir):
                for f in os.listdir(arch_ut_dir):
                    if f.startswith('test_') and '_tiling' in f and f.endswith('.cpp'):
                        arch_uts.append(os.path.join(arch_ut_dir, f))
    
    uts = public_uts + arch_uts
    
    has_src = len(sources) > 0
    has_ut = len(uts) > 0
    
    return has_src, has_ut


def check_kernel(op_dir):
    op_kernel_dir = os.path.join(op_dir, STRUCTURE['kernel'])
    ut_kernel_dir = os.path.join(op_dir, UT_DIRS['kernel_ut'])
    
    if not os.path.exists(op_kernel_dir):
        return False, False
    
    sources = []
    if os.path.isdir(op_kernel_dir):
        for root, _, files in os.walk(op_kernel_dir):
            for f in files:
                if f.endswith('.cpp') and '_def.cpp' not in f and 'tilingdata' not in f:
                    sources.append(os.path.join(root, f))
    
    uts = []
    if os.path.exists(ut_kernel_dir):
        for root, _, files in os.walk(ut_kernel_dir):
            for f in files:
                if f.startswith('test_') and f.endswith('.cpp'):
                    uts.append(os.path.join(root, f))
    
    return len(sources) > 0, len(uts) > 0


def check_api(op_dir):
    op_api_dir = os.path.join(op_dir, STRUCTURE['api'])
    op_host_api_dir = os.path.join(op_dir, STRUCTURE['host'], STRUCTURE['api'])
    ut_api_dir = os.path.join(op_dir, UT_DIRS['api_ut'])
    ut_host_api_dir = os.path.join(op_dir, UT_DIRS['host_ut'], STRUCTURE['api'])
    
    sources = []
    if os.path.exists(op_api_dir) and os.path.isdir(op_api_dir):
        for f in os.listdir(op_api_dir):
            if f.startswith('aclnn_') and f.endswith('.cpp'):
                sources.append(os.path.join(op_api_dir, f))
    if os.path.exists(op_host_api_dir) and os.path.isdir(op_host_api_dir):
        for f in os.listdir(op_host_api_dir):
            if f.startswith('aclnn_') and f.endswith('.cpp'):
                sources.append(os.path.join(op_host_api_dir, f))
    
    uts = []
    if os.path.exists(ut_api_dir) and os.path.isdir(ut_api_dir):
        for f in os.listdir(ut_api_dir):
            if f.startswith('test_aclnn_') and f.endswith('.cpp'):
                uts.append(os.path.join(ut_api_dir, f))
    if os.path.exists(ut_host_api_dir) and os.path.isdir(ut_host_api_dir):
        for f in os.listdir(ut_host_api_dir):
            if f.startswith('test_aclnn_') and f.endswith('.cpp'):
                uts.append(os.path.join(ut_host_api_dir, f))
    
    return len(sources) > 0, len(uts) > 0


def scan_repository(repo_root, op_list_file, repo_type):
    operators = parse_operator_list(op_list_file)
    logger.info(f"解析到 {len(operators)} 个算子")
    
    results = []
    stats = {
        'infershape': {'has_source': 0, 'missing': 0},
        'tiling': {'has_source': 0, 'missing': 0},
        'kernel': {'has_source': 0, 'missing': 0},
        'api': {'has_source': 0, 'missing': 0}
    }
    category_stats = defaultdict(lambda: defaultdict(lambda: {'has_source': 0, 'missing': 0}))
    
    for i, op in enumerate(operators):
        op_dir = os.path.join(repo_root, op['path'])
        if not os.path.exists(op_dir):
            continue
        
        result = {'name': op['name'], 'category': op['category'], 'path': op['path'], 'missing': []}
        
        has_src, has_ut, has_ir_proto = check_infershape(op_dir)
        if has_src and has_ir_proto:
            stats['infershape']['has_source'] += 1
            category_stats[op['category']]['infershape']['has_source'] += 1
            if not has_ut:
                stats['infershape']['missing'] += 1
                category_stats[op['category']]['infershape']['missing'] += 1
                result['missing'].append('infershape')
        
        has_src, has_ut = check_tiling(op_dir)
        if has_src:
            stats['tiling']['has_source'] += 1
            category_stats[op['category']]['tiling']['has_source'] += 1
            if not has_ut:
                stats['tiling']['missing'] += 1
                category_stats[op['category']]['tiling']['missing'] += 1
                result['missing'].append('tiling')
        
        has_src, has_ut = check_kernel(op_dir)
        if has_src:
            stats['kernel']['has_source'] += 1
            category_stats[op['category']]['kernel']['has_source'] += 1
            if not has_ut:
                stats['kernel']['missing'] += 1
                category_stats[op['category']]['kernel']['missing'] += 1
                result['missing'].append('kernel')
        
        has_src, has_ut = check_api(op_dir)
        if has_src:
            stats['api']['has_source'] += 1
            category_stats[op['category']]['api']['has_source'] += 1
            if not has_ut:
                stats['api']['missing'] += 1
                category_stats[op['category']]['api']['missing'] += 1
                result['missing'].append('api')
        
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{len(operators)}")
    
    sorted_results = sorted(
        results,
        key=lambda x: (get_category_sort_key(x['category'], repo_type, repo_root),
                       x['name']))
    
    report_data = {
        'stats': stats,
        'category_stats': dict(category_stats),
        'results': sorted_results,
        'repo_type': repo_type
    }
    
    return report_data


def main():
    parser = argparse.ArgumentParser(description='UT 缺失扫描脚本')
    parser.add_argument('--repo', required=True,
                        help='仓库类型（支持任意 ops-* 仓库，如 ops-math、ops-nn、ops-transformer、ops-cv 等）')
    parser.add_argument('--repo-root', default=None,
                        help='仓库根目录（默认根据 repo 类型推断）')
    parser.add_argument('--op-list', default=None,
                        help='算子列表文件路径（默认根据 repo 类型推断）')
    parser.add_argument('--output', default=None,
                        help='输出 JSON 文件路径（默认根据 repo 类型推断）')
    
    args = parser.parse_args()
    
    # 动态验证仓库名（支持任意 ops-* 仓库）
    if not args.repo.startswith('ops-'):
        print(f"错误: 仓库名应以 'ops-' 开头，当前为: {args.repo}", file=sys.stderr)
        sys.exit(1)
    
    repo_root, detection_method = get_repo_root(args.repo, args.repo_root)
    
    date_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d')
    
    reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo)
    
    if args.op_list:
        op_list_file = args.op_list
    else:
        op_list_file = reports_dir / date_str / args.repo / 'operator_list.json'
    
    if args.output:
        output_file = args.output
    else:
        output_file = reports_dir / date_str / args.repo / 'ut-analysis-guide_data.json'
    
    print(f"仓库类型: {args.repo}")
    print(f"仓库根目录: {repo_root} (检测方式: {detection_method})")
    print(f"Reports 目录: {reports_dir} (检测方式: {reports_method})")
    print(f"算子列表: {op_list_file}")
    print(f"输出文件: {output_file}")
    print()
    
    report_data = scan_repository(repo_root, op_list_file, args.repo)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n扫描完成，结果已保存: {output_file}")
    
    stats = report_data['stats']
    print("\n统计摘要:")
    print("| UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |")
    print("|---------|----------|---------|----------|")
    for t in ['infershape', 'tiling', 'kernel', 'api']:
        s = stats[t]
        ratio = s['missing'] / s['has_source'] * 100 if s['has_source'] > 0 else 0
        print(f"| {t} | {s['has_source']} | {s['missing']} | {ratio:.1f}% |")


if __name__ == '__main__':
    main()
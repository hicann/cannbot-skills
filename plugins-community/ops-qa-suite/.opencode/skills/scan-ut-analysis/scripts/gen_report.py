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
import json
import argparse
from datetime import datetime, timezone


def get_category_sort_key(cat, categories_order):
    if not categories_order:
        return (2, 999, cat)
    
    if cat.startswith('experimental/'):
        subcat = cat.split('/')[1]
        base_order = categories_order.index(subcat) if subcat in categories_order else 999
        return (1, base_order, cat)
    elif cat in categories_order:
        return (0, categories_order.index(cat), cat)
    else:
        return (2, 999, cat)


def generate_report(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data['stats']
    category_stats = data['category_stats']
    results = data['results']
    repo_type = data.get('repo_type', 'unknown')
    
    categories_order = sorted(category_stats.keys())
    
    missing_ops = [r for r in results if r['missing']]
    
    report = []
    
    report.append(f"# {repo_type} UT 缺失分析报告（基础分析）")
    report.append("")
    report.append(f"**分析日期**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}")
    report.append("")
    report.append(f"**仓库类型**: {repo_type}")
    report.append("")
    report.append("**分析方法**: 基础分析（仅检查目录结构）")
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## 统计摘要")
    report.append("")
    report.append("| UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |")
    report.append("|---------|----------|---------|----------|")
    for t in ['infershape', 'tiling', 'kernel', 'api']:
        s = stats[t]
        ratio = s['missing'] / s['has_source'] * 100 if s['has_source'] > 0 else 0
        report.append(f"| {t} | {s['has_source']} | {s['missing']} | {ratio:.1f}% |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## 按分类统计")
    report.append("")
    report.append("| 分类 | UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |")
    report.append("|------|---------|----------|---------|----------|")
    
    categories = sorted(category_stats.keys(), key=lambda x: get_category_sort_key(x, categories_order))
    for cat in categories:
        for t in ['infershape', 'tiling', 'kernel', 'api']:
            s = category_stats[cat].get(t, {'has_source': 0, 'missing': 0})
            if s['has_source'] > 0:
                ratio = s['missing'] / s['has_source'] * 100
                report.append(f"| {cat} | {t} | {s['has_source']} | {s['missing']} | {ratio:.1f}% |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## 缺失详情")
    report.append("")
    report.append(f"**共 {len(missing_ops)} 个算子有 UT 缺失**")
    report.append("")
    report.append("| 序号 | 分类 | 算子 | 路径 | 缺失类型 |")
    report.append("|------|------|------|------|----------|")
    
    sorted_missing = sorted(missing_ops,
                         key=lambda x: (get_category_sort_key(x['category'], categories_order), x['name']))
    for idx, r in enumerate(sorted_missing, 1):
        missing_str = ', '.join(r['missing'])
        report.append(f"| {idx} | {r['category']} | {r['name']} | `{r['path']}` | {missing_str} |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## 分析说明")
    report.append("")
    report.append("### 分析方法")
    report.append("")
    report.append("本报告采用**基础分析模式**，仅检查目录结构：")
    report.append("")
    report.append("- 对比源文件目录与 UT 文件目录")
    report.append("- 有源文件则必须有对应 UT")
    report.append("- 不深入分析源文件实现逻辑")
    report.append("")
    report.append("### 可能误判的场景")
    report.append("")
    report.append("基础分析可能误判以下情况：")
    report.append("")
    report.append("1. **infershape 纯模板调用**")
    report.append("   - 源文件行数 <= 25 行")
    report.append("   - 无自定义 InferDataType 函数")
    report.append("   - 只调用公共模板函数（InferShape4Elewise 等）")
    report.append("   - 此类 infershape 不需要独立 UT")
    report.append("")
    report.append("2. **tiling 极简实现**")
    report.append("   - 源文件行数 < 60 行")
    report.append("   - 无实际 tiling 计算逻辑")
    report.append("   - 仅做简单参数传递")
    report.append("")
    report.append("### 详细分析建议")
    report.append("")
    report.append("如需精确判断 infershape/tiling 是否真的需要独立 UT，建议：")
    report.append("")
    report.append("- 深入分析源文件实现逻辑")
    report.append("- 检查源文件行数")
    report.append("- 检查是否调用公共模板函数")
    report.append("- 检查是否有自定义计算逻辑")
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## 建议")
    report.append("")
    report.append("### 优先级排序")
    report.append("")
    report.append("根据缺失数量和重要性，建议按以下优先级补充 UT：")
    report.append("")
    
    total_missing = sum(s['missing'] for s in stats.values())
    
    priorities = []
    if stats['kernel']['missing'] > 0:
        ratio = stats['kernel']['missing'] / stats['kernel']['has_source'] * 100
        priorities.append(f"1. **kernel UT（缺失 {stats['kernel']['missing']} 个）**")
        priorities.append("   - 涉及实际计算逻辑")
        priorities.append("   - 计算正确性最关键")
        priorities.append(f"   - 缺失比例 {ratio:.1f}%")
    
    if stats['api']['missing'] > 0:
        ratio = stats['api']['missing'] / stats['api']['has_source'] * 100
        priorities.append(f"2. **api UT（缺失 {stats['api']['missing']} 个）**")
        priorities.append("   - 涉及参数校验和接口测试")
        priorities.append(f"   - 缺失比例 {ratio:.1f}%")
    
    if stats['infershape']['missing'] > 0:
        ratio = stats['infershape']['missing'] / stats['infershape']['has_source'] * 100
        priorities.append(f"3. **infershape UT（缺失 {stats['infershape']['missing']} 个）**")
        priorities.append("   - 部分可能为纯模板调用")
        priorities.append(f"   - 缺失比例 {ratio:.1f}%")
    
    if stats['tiling']['missing'] > 0:
        ratio = stats['tiling']['missing'] / stats['tiling']['has_source'] * 100
        priorities.append(f"4. **tiling UT（缺失 {stats['tiling']['missing']} 个）**")
        priorities.append("   - 部分可能为极简实现")
        priorities.append(f"   - 缺失比例 {ratio:.1f}%")
    
    report.extend(priorities)
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("**报告生成完成**")
    report.append("")
    report.append(f"- JSON 数据文件: {json_file}")
    report.append(f"- Markdown 报告: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"报告已生成: {output_file}")
    print(f"报告共 {len(report)} 行")


def main():
    parser = argparse.ArgumentParser(description='UT 缺失报告生成脚本')
    parser.add_argument('--input', required=True,
                        help='输入 JSON 文件路径')
    parser.add_argument('--output', default=None,
                        help='输出 Markdown 文件路径（默认 reports/{date}/{repo}/ut-analysis-guide_report_{time}.md）')
    
    args = parser.parse_args()
    
    if args.output:
        output_file = args.output
    else:
        import re
        match = re.search(r'reports/(\d{8})/([^/]+)/', args.input)
        if match:
            date_str, repo = match.groups()
            time_str = datetime.now(tz=timezone.utc).strftime('%H%M%S')
            output_file = f'reports/{date_str}/{repo}/ut-analysis-guide_report_{time_str}.md'
        else:
            output_file = args.input.replace('.json', '.md')
    
    generate_report(args.input, output_file)


if __name__ == '__main__':
    main()
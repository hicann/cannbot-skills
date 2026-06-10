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

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_scan_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_issues(issues):
    """分类确定性说明不一致问题"""
    type_a = []
    type_b = []
    type_c = []
    type_d = []
    other = []
    
    for item in issues['deterministic_inconsistent']:
        reason = item.get('reason', '')
        enable_method = item.get('enable_method', '')
        has_impl = item.get('has_deterministic_impl', False)
        op_dir = item.get('op_dir', '')
        link_path = item.get('link_path', '')
        impl_file = item.get('impl_file', '') or ''
        
        row = {
            '接口名': item['interface_name'],
            '算子目录': op_dir,
            'aclnn文档路径': link_path,
            '代码文件路径': impl_file,
            'op_api_list说明': item['deterministic_a2a3'],
            'aclnn文档说明': (item.get('aclnn_doc_value', 'N/A') or 'N/A')[:80],
            '文档方法': enable_method or '-',
            '代码实现': '有' if has_impl else '无',
            '代码方法': item.get('actual_method', '-') or '-',
            '问题原因': reason,
            '状态': '待修复',
            '负责人': '',
            '修复计划': '',
            '修复日期': '',
            '备注': '',
        }
        
        if '文档声明通过' in reason and '代码中未使用' in reason:
            type_a.append(row)
        elif 'op_api_list声明' in reason and ('aclnn文档声明' in reason or '不支持' in reason):
            type_b.append(row)
        elif '文档声明"确定性实现"' in reason and '代码中有' in reason:
            type_c.append(row)
        elif 'aclnn文档缺少' in reason:
            type_d.append(row)
        else:
            other.append(row)
    
    return type_a, type_b, type_c, type_d, other


def create_link_broken_sheet(issues, repo):
    data = []
    for item in issues['link_broken']:
        op_dir = item.get('op_dir', '')
        data.append({
            '接口名': item['interface_name'],
            '算子目录': op_dir,
            '链接路径': item['link_path'],
            '正确路径建议': f'{repo}/{op_dir}/docs/{item["interface_name"]}.md' if op_dir else '-',
            '状态': item['link_status'],
            '修复状态': '待修复',
            '负责人': '',
            '修复计划': '',
            '修复日期': '',
            '备注': '',
        })
    return data


def create_missing_interfaces_sheet(scan_data, repo):
    data = []
    for item in scan_data['missing_in_doc']:
        op_dir = item.get('op_dir', '')
        doc_path = item.get('doc_path', '')
        rel_path = ''
        repo_marker = f'/{repo}/'
        if doc_path and repo_marker in doc_path:
            rel_path = doc_path.split(repo_marker)[1] if repo_marker in doc_path else doc_path
        data.append({
            '接口名': item['interface_name'],
            '算子目录': op_dir,
            'aclnn文档相对路径': f'../../{rel_path}' if rel_path else '-',
            'aclnn文档绝对路径': doc_path,
            '修复状态': '待补充',
            '负责人': '',
            '修复计划': '',
            '修复日期': '',
            '备注': '',
        })
    return data


def generate_excel(json_path, output_path, repo):
    scan_data = load_scan_data(json_path)
    issues = scan_data['issues']
    
    type_a, type_b, type_c, type_d, other = categorize_issues(issues)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_data = {
            '问题类型': [
                '类型A: 文档声称支持但代码不支持',
                '类型B: op_api_list与aclnn文档不一致',
                '类型C: 文档声明确定性但代码有配置',
                '类型D: aclnn文档缺少确定性说明',
                '链接断链',
                '接口遗漏',
                '其他确定性问题',
                '总计',
            ],
            '问题数量': [
                len(type_a),
                len(type_b),
                len(type_c),
                len(type_d),
                len(issues['link_broken']),
                len(scan_data['missing_in_doc']),
                len(other),
                len(type_a) + len(type_b) + len(type_c) + len(type_d) + 
                len(issues['link_broken']) + len(scan_data['missing_in_doc']) + len(other),
            ],
            '优先级': ['高', '中', '中', '低', '低', '低', '低', '-'],
            '修复建议': [
                '补充代码实现或修改文档',
                '同步两处文档',
                '如果代码支持配置，文档改为"支持开启"',
                '补充aclnn文档',
                '修复链接路径',
                '补充到op_api_list',
                '按具体情况处理',
                '-',
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='汇总', index=False)
        
        if type_a:
            df_a = pd.DataFrame(type_a)
            df_a.to_excel(writer, sheet_name='类型A-文档声称支持', index=False)
        
        if type_b:
            df_b = pd.DataFrame(type_b)
            df_b.to_excel(writer, sheet_name='类型B-文档不一致', index=False)
        
        if type_c:
            df_c = pd.DataFrame(type_c)
            df_c.to_excel(writer, sheet_name='类型C-文档声明确定性', index=False)
        
        if type_d:
            df_d = pd.DataFrame(type_d)
            df_d.to_excel(writer, sheet_name='类型D-文档缺失', index=False)
        
        link_data = create_link_broken_sheet(issues, repo)
        if link_data:
            df_link = pd.DataFrame(link_data)
            df_link.to_excel(writer, sheet_name='链接断链', index=False)
        
        missing_data = create_missing_interfaces_sheet(scan_data, repo)
        if missing_data:
            df_missing = pd.DataFrame(missing_data)
            df_missing.to_excel(writer, sheet_name='接口遗漏', index=False)
        
        if other:
            df_other = pd.DataFrame(other)
            df_other.to_excel(writer, sheet_name='其他问题', index=False)
    
    print(f"Excel 文件已生成: {output_path}")
    print(f"\nSheet 列表:")
    print(f"  - 汇总: 问题类型统计")
    print(f"  - 类型A-文档声称支持: {len(type_a)} 个问题")
    print(f"  - 类型B-文档不一致: {len(type_b)} 个问题")
    print(f"  - 类型C-文档声明确定性: {len(type_c)} 个问题")
    print(f"  - 类型D-文档缺失: {len(type_d)} 个问题")
    print(f"  - 链接断链: {len(link_data)} 个问题")
    print(f"  - 接口遗漏: {len(missing_data)} 个问题")
    if other:
        print(f"  - 其他问题: {len(other)} 个问题")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成问题跟踪 Excel 文件')
    parser.add_argument('--repo', required=True,
                        help='仓库名（支持任意 ops-* 仓库）')
    parser.add_argument('--json', default=None,
                        help='JSON 数据文件路径')
    parser.add_argument('--output', default=None,
                        help='输出 Excel 文件路径')
    
    args = parser.parse_args()
    
    date_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d')
    time_str = datetime.now(tz=timezone.utc).strftime('%H%M%S')
    
    if args.json:
        json_path = args.json
    else:
        json_path = f'reports/{date_str}/{args.repo}/op-api-list-validation_data.json'
    
    if args.output:
        output_path = args.output
    else:
        output_path = f'reports/{date_str}/{args.repo}/op-api-list-validation_tracking_{time_str}.xlsx'
    
    generate_excel(json_path, output_path, args.repo)
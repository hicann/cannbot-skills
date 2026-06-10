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
算子接口列表一致性扫描脚本

用于验证仓库级算子接口列表文档（docs/zh/op_api_list.md）与实际 aclnn 接口实现的一致性：
- 接口名是否与实际实现一致
- 接口链接是否能正常跳转到 aclnn API 文档
- 接口说明是否与功能实现一致
- 确定性说明是否与实际实现一致

用法:
    python op_api_list_scan.py --repo ops-nn
    python op_api_list_scan.py --repo ops-math --output \
        reports/op-api-list-validation/ops-math_op_api_list_validation_report.md
"""

import argparse
import os
import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)

_scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts')
sys.path.insert(0, os.path.abspath(_scripts_dir))
from config_loader import get_structure_dirs, get_op_api_list_path

STRUCTURE = get_structure_dirs()
OP_API_LIST_DOC = get_op_api_list_path()


def parse_markdown_table(content):
    """解析 Markdown 表格"""
    lines = content.split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.strip().startswith('|'):
            in_table = True
            table_lines.append(line)
        elif in_table and not line.strip().startswith('|'):
            break
    
    if len(table_lines) < 2:
        return [], []
    
    # 解析表头
    header = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
    
    # 解析数据行（跳过分隔行）
    rows = []
    for _, line in enumerate(table_lines[2:], start=2):
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(cells) >= len(header):
            rows.append(cells)
    
    return header, rows


def find_missing_interfaces(repo_root, listed_interface_names, listed_op_dirs):
    """排查接口遗漏
    
    遍历仓库所有算子目录，检查是否有aclnn实现但未列入op_api_list。
    
    Args:
        repo_root: 仓库根目录
        listed_interface_names: op_api_list中已列出的接口名集合
        listed_op_dirs: op_api_list中已列出的算子目录集合
    
    Returns:
        tuple: (missing_required, optional_missing)
        - missing_required: 必须添加的接口（有aclnn手动实现）
        - optional_missing: 可选添加的接口（ACLNNTYPE=aclnn/aclnn_inner）
    """
    missing_required = []
    optional_missing = []
    
    skip_dirs = ['experimental', 'docs', 'tests', 'examples', 'scripts', 'tools', '.git', 'cmake', 'build', 'reports']
    
    for item in os.listdir(repo_root):
        category_path = os.path.join(repo_root, item)
        
        if not os.path.isdir(category_path):
            continue
        
        if item in skip_dirs or item.startswith('.') or item.startswith('_'):
            continue
        
        for op_name in os.listdir(category_path):
            op_dir = os.path.join(category_path, op_name)
            
            if not os.path.isdir(op_dir):
                continue
            
            op_path = f"{item}/{op_name}"
            
            if is_experimental_op(op_path):
                continue
            
            if op_path in listed_op_dirs:
                continue
            
            aclnn_type = get_aclnn_type(op_dir)
            is_aclnn_manual = has_aclnn_manual(op_dir)
            
            if is_aclnn_manual:
                interface_name = f"aclnn{''.join(word.capitalize() for word in op_name.split('_'))}"
                
                if interface_name not in listed_interface_names:
                    missing_required.append({
                        'op_path': op_path,
                        'interface_name': interface_name,
                        'reason': '有aclnn手动实现但未列入op_api_list',
                    })
            
            if aclnn_type in ['aclnn', 'aclnn_inner']:
                interface_name = f"aclnn{''.join(word.capitalize() for word in op_name.split('_'))}"
                
                if interface_name not in listed_interface_names:
                    optional_missing.append({
                        'op_path': op_path,
                        'interface_name': interface_name,
                        'aclnn_type': aclnn_type,
                        'reason': f'ACLNNTYPE={aclnn_type}，可选登记',
                    })
    
    return missing_required, optional_missing


def parse_op_api_list_file(op_api_list_file):
    """解析 op_api_list.md 文件"""
    with open(op_api_list_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    header, rows = parse_markdown_table(content)
    
    # 列顺序：接口名、说明、确定性说明(A2/A3)、确定性说明(A5)
    interfaces = []
    
    for row in rows:
        if len(row) < 3:
            continue
        
        interface_cell = row[0]
        
        # 从链接中提取接口名和路径
        link_match = re.search(r'\[([^\]]+)\]\(([^\)]+)\)', interface_cell)
        if link_match:
            interface_name = link_match.group(1)
            link_path = link_match.group(2)
        else:
            interface_name = interface_cell
            link_path = ''
        
        # 提取接口说明
        description = row[1] if len(row) > 1 else ''
        
        # 提取确定性说明
        deterministic_a2a3 = row[2] if len(row) > 2 else ''
        deterministic_a5 = row[3] if len(row) > 3 else ''
        
        # 从链接中提取算子目录路径（比接口名推断更准确）
        op_dir = None
        if link_path:
            path_match = re.search(r'\.\./([^/]+)/([^/]+)/docs/', link_path)
            if path_match:
                op_dir = f"{path_match.group(1)}/{path_match.group(2)}"
        
        # 如果链接无法提取，再用接口名推断
        if not op_dir:
            op_name = interface_to_op_name(interface_name)
            if op_name:
                op_dir = op_name
        
        interfaces.append({
            'interface_name': interface_name,
            'link_path': link_path,
            'description': description,
            'deterministic_a2a3': deterministic_a2a3,
            'deterministic_a5': deterministic_a5,
            'op_dir': op_dir,
        })
    
    return interfaces


def interface_to_op_name(interface_name):
    """将接口名转换为算子目录名"""
    # aclnnGridSampler2D -> grid_sample
    # aclnnUpsampleBilinear2dAA -> upsample_bilinear2d_aa
    
    if not interface_name.startswith('aclnn'):
        return None
    
    # 去掉 aclnn 前缀
    name = interface_name[5:]
    
    # PascalCase to snake_case
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append('_')
        result.append(char.lower())
    
    return ''.join(result)


def find_op_dir_by_interface(repo_root, interface_name):
    """根据接口名查找算子目录"""
    op_name = interface_to_op_name(interface_name)
    
    if not op_name:
        return None
    
    # 搜索算子目录
    for root, dirs, _ in os.walk(repo_root):
        for dir_name in dirs:
            if dir_name == op_name:
                return os.path.join(root, dir_name)
    
    return None


def is_experimental_op(op_path):
    """判断是否为 experimental 目录下的算子"""
    return op_path and ('experimental/' in op_path or op_path.startswith('experimental'))


def get_aclnn_type(op_dir):
    """获取 ACLNNTYPE 参数"""
    aclnn_type = None
    
    top_cmake = os.path.join(op_dir, 'CMakeLists.txt')
    if os.path.exists(top_cmake):
        with open(top_cmake, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'ACLNNTYPE\s+(\w+)', content)
            if match:
                aclnn_type = match.group(1)
    
    host_cmake = os.path.join(op_dir, 'op_host', 'CMakeLists.txt')
    if os.path.exists(host_cmake) and not aclnn_type:
        with open(host_cmake, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'ACLNNTYPE\s+(\w+)', content)
            if match:
                aclnn_type = match.group(1)
    
    return aclnn_type


def has_aclnn_manual(op_dir):
    """检查是否有手动实现的 aclnn 接口文件"""
    op_api_dir = os.path.join(op_dir, 'op_api')
    if os.path.exists(op_api_dir):
        for f in os.listdir(op_api_dir):
            if f.startswith('aclnn_') and (f.endswith('.cpp') or f.endswith('.h')):
                return True
    
    host_op_api_dir = os.path.join(op_dir, 'op_host', 'op_api')
    if os.path.exists(host_op_api_dir):
        for _, _, files in os.walk(host_op_api_dir):
            for f in files:
                if f.startswith('aclnn_') and (f.endswith('.cpp') or f.endswith('.h')):
                    return True
    
    return False


def need_aclnn_doc(op_dir):
    """判断是否需要 aclnn 文档
    
    判断规则：
    - ACLNNTYPE=aclnn 或 aclnn_inner → 必须有
    - ACLNNTYPE=aclnn_exclude + 有 aclnn_xxx 文件 → 必须有
    - 其他情况 → 无需
    
    Returns:
        tuple: (bool, str) - 是否需要文档, aclnn来源(auto/manual/none)
    """
    aclnn_type = get_aclnn_type(op_dir)
    aclnn_manual = has_aclnn_manual(op_dir)
    
    if aclnn_type in ['aclnn', 'aclnn_inner']:
        return True, 'auto'
    elif aclnn_manual:
        return True, 'manual'
    else:
        return False, 'none'


def check_aclnn_doc_exists(op_dir, interface_name):
    """检查 aclnn 文档是否存在"""
    docs_dir = os.path.join(op_dir, 'docs')
    
    if not os.path.exists(docs_dir):
        return False, None
    
    # 文档名通常是 aclnn{OpName}.md
    doc_name = f"{interface_name}.md"
    doc_path = os.path.join(docs_dir, doc_name)
    
    if os.path.exists(doc_path):
        return True, doc_path
    
    # 也检查其他可能的文档名
    for f in os.listdir(docs_dir):
        if f.startswith('aclnn') and f.endswith('.md'):
            return True, os.path.join(docs_dir, f)
    
    return False, None


def check_link_validity(repo_root, link_path):
    """检查链接是否有效"""
    # 链接格式通常是 ../../activation/gelu/docs/aclnnGelu.md
    # 转换为相对路径
    
    if not link_path:
        return False, 'link_empty'
    
    # 提取路径
    path_match = re.search(r'\.\./([^/]+)/([^/]+)/docs/([^\.]+\.md)', link_path)
    if not path_match:
        return False, 'link_format_error'
    
    category = path_match.group(1)
    op_name = path_match.group(2)
    doc_name = path_match.group(3)
    
    actual_path = os.path.join(repo_root, category, op_name, 'docs', doc_name)
    
    if os.path.exists(actual_path):
        return True, 'valid'
    
    return False, 'link_broken'


def parse_product_support_table(content):
    """从 aclnn 文档中解析产品支持情况表格
    
    返回：
    - dict: {产品名称: 是否支持(True/False)}
    """
    products = {}
    
    # 查找产品支持情况表格
    table_match = re.search(r'## 产品支持情况\s*\n+(.*?)\n##', content, re.DOTALL)
    if not table_match:
        # 尝试另一种模式
        table_match = re.search(r'产品支持情况[^\n]*\n+(.*?)(?:\n##|\n\n##)', content, re.DOTALL)
    
    if not table_match:
        return products
    
    table_content = table_match.group(1)
    
    # 解析表格行
    lines = table_content.split('\n')
    for line in lines:
        if line.strip().startswith('|') and not line.strip().startswith('|:'):
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(cells) >= 2:
                # 提取产品名称（去掉 <term> 标签）
                product_name = cells[0]
                product_name = re.sub(r'<term>|</term>', '', product_name).strip()
                
                # 判断是否支持
                support_status = cells[1].strip()
                is_supported = '√' in support_status or '支持' in support_status
                
                products[product_name] = is_supported
    
    return products


def extract_deterministic_section(doc_path):
    """从 aclnn 文档中提取约束说明章节的确定性内容
    
    返回：
    - dict: {产品型号: 确定性说明} 或 {'all': 确定性说明}（未区分型号时）
    - 如果没有确定性说明，返回 None
    - 如果约束说明明确写了"无"，返回 {'all': '无'}
    
    支持的格式：
    1. "- 确定性计算：" + 子项列表
    2. "* 确定性说明：" + 内容
    3. 直接的 "默认支持确定性计算。" 或 "默认确定性实现。"
    4. "无。" 表示无约束说明
    """
    if not doc_path or not os.path.exists(doc_path):
        return None
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    constraint_match = re.search(r'## 约束说明\s*\n+(.*?)(?:\n## |\n##调用|\Z)', content, re.DOTALL)
    if not constraint_match:
        return None
    
    constraint_content = constraint_match.group(1)
    
    # 检查是否是"无"的情况
    if re.match(r'^无[。\s]*$', constraint_content.strip()):
        return {'all': '无'}
    
    lines = constraint_content.split('\n')
    deterministic_lines = []
    found_deterministic_header = False
    
    # 新增：检查是否有直接的确定性说明（无标题格式）
    first_non_empty = None
    for line in lines:
        stripped = line.strip()
        if stripped:
            first_non_empty = stripped
            break
    
    # 如果第一行直接包含确定性说明（如 "默认支持确定性计算。"）
    deterministic_phrases = [
        '默认支持确定性计算', '默认确定性实现', '默认为确定性实现'
    ]
    is_direct_desc = first_non_empty and (
        any(phrase in first_non_empty for phrase in deterministic_phrases) or
        re.match(r'^(?:aclnn)?\w+.*(?:确定性实现|支持确定性计算)', first_non_empty)
    )
    if is_direct_desc:
        # 直接提取这一行作为确定性说明
        simple_desc_match = re.search(r'(aclnn[^。\n]*(?:确定性实现|支持确定性计算)[^。\n]*)', first_non_empty)
        if simple_desc_match:
            return {'all': simple_desc_match.group(1).strip()}
        # 如果不包含aclnn接口名，将整个内容作为说明
        if '确定性' in first_non_empty:
            return {'all': first_non_empty.rstrip('。')}
    
    for _, line in enumerate(lines):
        stripped = line.strip()
        
        # 支持多种标题格式："- 确定性计算"、"* 确定性说明"、"确定性计算"
        header_patterns = [
            '- 确定性计算', '-确定性计算', '* 确定性说明',
            '*确定性说明', '确定性计算', '确定性说明'
        ]
        if any(stripped.startswith(pattern) for pattern in header_patterns):
            found_deterministic_header = True
            # 如果标题行本身包含说明内容，也提取
            has_aclnn = 'aclnn' in stripped
            has_deterministic = '确定性' in stripped
            has_keywords = '默认' in stripped or '支持' in stripped
            if has_aclnn or (has_deterministic and has_keywords):
                deterministic_lines.append(stripped)
            continue
        
        if found_deterministic_header:
            # 检查是否结束
            is_list_item = stripped.startswith('-')
            has_aclnn = 'aclnn' in stripped
            has_term = 'term>' in stripped
            has_deterministic = '确定性' in stripped
            if is_list_item and not (has_aclnn or has_term or has_deterministic):
                break
            if stripped.startswith('##') or stripped.startswith('调用'):
                break
            if stripped and 'aclnn' in stripped:
                deterministic_lines.append(stripped)
            elif stripped and 'term>' in stripped:
                deterministic_lines.append(stripped)
            elif stripped and ('确定性' in stripped or '支持' in stripped):
                deterministic_lines.append(stripped)
    
    if not deterministic_lines:
        return None
    
    deterministic_content = ' '.join(deterministic_lines)
    
    result = {}
    
    aclnn_deterministic_match = re.search(r'(acl(?:nn)?\w*[^。\n]*默认(?:确定性实现|支持确定性计算)[^。\n]*)', deterministic_content)
    if aclnn_deterministic_match:
        result['all'] = aclnn_deterministic_match.group(1).strip()
    
    product_lines = re.findall(r'<term>([^<]+)</term>[：:\s]*([^-]+)', deterministic_content)
    
    if product_lines:
        for product, desc in product_lines:
            product = product.strip()
            desc = desc.strip()
            if 'aclnn' in desc or 'acl' in desc or '确定性' in desc:
                result[product] = desc
    
    if not result:
        simple_match = re.search(r'(acl(?:nn)?\w+[^。\n]+(?:确定性|支持)[^。\n]*)', deterministic_content)
        if simple_match:
            result['all'] = simple_match.group(1).strip()
        elif 'acl' in deterministic_content or 'aclnn' in deterministic_content:
            aclnn_match = re.search(r'(acl(?:nn)?\w+[^\n]*)', deterministic_content)
            if aclnn_match:
                result['all'] = aclnn_match.group(1).strip()
    
    if not result:
        return None
    
    return result


def check_deterministic_in_doc(doc_path):
    """从 aclnn 文档中提取确定性说明（按产品型号区分）
    
    返回：
    - (deterministic_by_product, product_support)
    - deterministic_by_product: dict {产品型号: 确定性说明} 或 {'all': 说明}
    - product_support: dict {产品名称: 是否支持}
    """
    if not doc_path or not os.path.exists(doc_path):
        return None, {}
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析产品支持情况表格
    product_support = parse_product_support_table(content)
    
    # 提取确定性说明（按产品型号）
    deterministic_by_product = extract_deterministic_section(doc_path)
    
    return deterministic_by_product, product_support


VALID_DET_TERMINISTIC_PATTERNS = [
    (r'aclnn\w+默认确定性实现', '默认确定性实现'),
    (r'aclnn\w+默认非确定性实现,支持通过\w+开启确定性', '默认非确定性实现,支持配置开启'),
    (r'aclnn\w+默认非确定性实现,不支持通过\w+开启确定性', '默认非确定性实现,不支持配置开启'),
]

VALID_OP_LIST_DET_TERMINISTIC = [
    '默认确定性实现',
    '默认非确定性实现,支持配置开启',
    '默认非确定性实现,不支持配置开启',
]


def check_a2a3_a5_consistency(deterministic_a2a3, deterministic_a5, deterministic_by_product):
    """检查A2/A3与A5确定性说明是否一致
    
    如果两者不一致，aclnn文档需要特殊说明两者的不同。
    
    Args:
        deterministic_a2a3: op_api_list中A2/A3列的确定性说明
        deterministic_a5: op_api_list中A5列的确定性说明
        deterministic_by_product: aclnn文档中的确定性说明（按产品）
    
    Returns:
        tuple: (is_consistent, reason)
    """
    if not deterministic_a2a3 or not deterministic_a5:
        return True, '缺少确定性说明，跳过一致性检查'
    
    if deterministic_a2a3.strip() == '-' or deterministic_a5.strip() == '-':
        return True, '产品不支持，跳过一致性检查'
    
    if deterministic_a2a3 == deterministic_a5:
        return True, 'A2/A3与A5确定性说明一致'
    
    if 'all' in deterministic_by_product:
        return False, 'A2/A3与A5确定性说明不一致，但aclnn文档未区分产品型号'
    
    a2a3_keys = get_product_key_for_column('A2/A3')
    a5_keys = get_product_key_for_column('Ascend 950')
    
    has_a2a3_specific = False
    has_a5_specific = False
    
    for product in deterministic_by_product.keys():
        if match_product_by_column(product, a2a3_keys):
            has_a2a3_specific = True
        if match_product_by_column(product, a5_keys):
            has_a5_specific = True
    
    if not has_a2a3_specific or not has_a5_specific:
        return False, 'A2/A3与A5确定性说明不一致，但aclnn文档未完整区分产品型号'
    
    return True, 'A2/A3与A5确定性说明不一致，aclnn文档已区分产品型号'


def check_deterministic_format_deterministic(doc_desc, op_list_desc):
    """验证确定性话术规范性
    
    Args:
        doc_desc: aclnn文档中的确定性说明
        op_list_desc: op_api_list表格中的确定性说明
    
    Returns:
        tuple: (is_valid, reason)
    """
    if not doc_desc:
        return False, 'aclnn文档缺少确定性说明'
    
    doc_matched = False
    doc_pattern_type = None
    
    for pattern, list_pattern in VALID_DET_TERMINISTIC_PATTERNS:
        if re.search(pattern, doc_desc):
            doc_matched = True
            doc_pattern_type = list_pattern
            break
    
    if not doc_matched:
        return False, f'aclnn文档话术不规范: {doc_desc[:50]}'
    
    if op_list_desc not in VALID_OP_LIST_DET_TERMINISTIC:
        return False, f'op_api_list话术不规范: {op_list_desc}'
    
    return True, '话术规范'


def parse_deterministic_type(desc):
    """解析确定性说明的类型
    
    返回：
    - 'deterministic': 默认确定性实现
    - 'non_deterministic_support': 默认非确定性，支持配置开启
    - 'non_deterministic_not_support': 默认非确定性，不支持配置开启
    - 'no_constraint': 无约束说明（约束说明为"无"）
    - None: 无法识别
    """
    if not desc:
        return None
    
    if desc.strip() == '无' or desc.strip() == '无。':
        return 'no_constraint'
    
    deterministic_patterns = [
        '默认确定性实现',
        '默认为确定性实现',
        '默认支持确定性计算',
        '支持确定性计算',
    ]
    for pattern in deterministic_patterns:
        if pattern in desc:
            return 'deterministic'
    
    if re.search(r'aclnn\w+默认(?:确定性实现|支持确定性计算)', desc):
        return 'deterministic'
    
    if '不支持配置开启' in desc or '不支持通过aclrtCtxSetSysParamOpt开启确定性' in desc or '暂不支持确定性实现' in desc:
        return 'non_deterministic_not_support'
    
    if '支持配置开启' in desc or '支持通过aclrtCtxSetSysParamOpt开启确定性' in desc or '支持开启' in desc:
        return 'non_deterministic_support'
    
    if '默认非确定性实现' in desc:
        if '支持' in desc and '开启' in desc:
            return 'non_deterministic_support'
        return 'non_deterministic_not_support'
    
    return None


def normalize_product_name(name):
    """标准化产品名称，用于匹配"""
    name = name.strip()
    name = re.sub(r'<term>|</term>', '', name)
    name = name.replace('系列产品', '').replace('推理产品', '').replace('训练产品', '')
    name = name.replace('/', '').replace(' ', '')
    return name.lower()


def get_product_key_for_column(column_name):
    """根据 op_api_list 列名获取对应的产品匹配关键词"""
    column_name = column_name.strip().lower()
    
    if 'a2' in column_name or 'a3' in column_name:
        return ['a2', 'a3', 'atlas a2', 'atlas a3']
    if '950' in column_name:
        return ['950', 'ascend 950']
    if 'a5' in column_name:
        return ['a5']
    
    return []


def match_product_by_column(doc_product, column_product_keys):
    """根据列名关键词匹配文档中的产品"""
    doc_norm = normalize_product_name(doc_product)
    
    for key in column_product_keys:
        if key in doc_norm:
            return True
    
    return False


def analyze_deterministic(op_api_list_value, deterministic_by_product, product_support, column_name='A2/A3'):
    """分析确定性说明是否一致（基于固定话术验证）
    
    Args:
        op_api_list_value: op_api_list.md 表格中对应产品的确定性说明
        deterministic_by_product: dict {产品型号: 确定性说明} 或 {'all': 说明}
        product_support: dict {产品名称: 是否支持}
        column_name: op_api_list 列名（如 'A2/A3' 或 'Ascend 950')
    
    Returns:
        tuple: (is_consistent, reason, details)
    """
    details = {
        'op_list_type': None,
        'doc_type': None,
        'doc_desc': None,
        'matched_product': None,
        'format_valid': None,
    }
    
    if not op_api_list_value or op_api_list_value.strip() == '-':
        if op_api_list_value.strip() == '-':
            return True, '产品不支持，跳过检查', details
        return False, 'op_api_list缺少确定性说明', details
    
    if not deterministic_by_product:
        return False, 'aclnn文档缺少确定性说明（需要补充）', details
    
    doc_desc = None
    if 'all' in deterministic_by_product:
        doc_desc = deterministic_by_product['all']
        details['matched_product'] = 'all'
    else:
        column_keys = get_product_key_for_column(column_name)
        for doc_product, desc in deterministic_by_product.items():
            if match_product_by_column(doc_product, column_keys):
                doc_desc = desc
                details['matched_product'] = doc_product
                break
    
    if not doc_desc:
        for doc_product, desc in deterministic_by_product.items():
            doc_desc = desc
            details['matched_product'] = doc_product
            break
    
    if not doc_desc:
        return False, 'aclnn文档缺少确定性说明', details
    
    details['doc_desc'] = doc_desc
    
    is_valid, reason = check_deterministic_format_deterministic(doc_desc, op_api_list_value)
    details['format_valid'] = is_valid
    
    if not is_valid:
        return False, reason, details
    
    return True, '一致（话术规范）', details


def scan_repository(repo_root, repo_type):
    """扫描仓库"""
    op_api_list_file = os.path.join(repo_root, 'docs/zh/op_api_list.md')
    
    if not os.path.exists(op_api_list_file):
        return {'error': f'op_api_list.md 文件不存在: {op_api_list_file}'}
    
    interfaces = parse_op_api_list_file(op_api_list_file)
    logger.info(f"解析到 {len(interfaces)} 个接口")
    
    results = []
    issues = {
        'interface_not_found': [],
        'aclnn_doc_missing': [],
        'aclnn_doc_not_needed': [],
        'link_broken': [],
        'description_inconsistent': [],
        'deterministic_inconsistent': [],
        'experimental_skipped': [],
        'a2a3_a5_inconsistent': [],
        'missing_in_doc': [],
        'optional_missing': [],
    }
    
    stats = {
        'total': len(interfaces),
        'passed': 0,
        'failed': 0,
        'by_check': {
            'interface_exists': {'total': 0, 'passed': 0},
            'aclnn_doc': {'total': 0, 'passed': 0},
            'link': {'total': 0, 'passed': 0},
            'description': {'total': 0, 'passed': 0},
            'deterministic_a2a3': {'total': 0, 'passed': 0},
            'deterministic_a5': {'total': 0, 'passed': 0},
        }
    }
    
    for interface in interfaces:
        result = {
            'interface_name': interface['interface_name'],
            'op_dir': interface.get('op_dir'),
            'checks': {},
            'status': 'passed',
        }
        
        # 跳过 experimental 目录下的算子接口（生态开发者提供，不检查）
        if is_experimental_op(interface.get('op_dir')):
            result['checks']['interface_exists'] = 'skipped_experimental'
            result['status'] = 'skipped'
            stats['total'] -= 1
            issues['experimental_skipped'].append(interface)
            results.append(result)
            continue
        
        # 检查1：接口对应的算子是否存在（使用链接路径）
        stats['by_check']['interface_exists']['total'] += 1
        op_dir_path = os.path.join(repo_root, interface['op_dir']) if interface.get('op_dir') else None
        
        if not interface.get('op_dir') or not os.path.exists(op_dir_path):
            result['checks']['interface_exists'] = 'not_found'
            result['status'] = 'failed'
            issues['interface_not_found'].append(interface)
            stats['by_check']['interface_exists']['passed'] += 0
            op_dir_path = None
        else:
            result['checks']['interface_exists'] = 'found'
            result['op_dir'] = op_dir_path
            stats['by_check']['interface_exists']['passed'] += 1
        
        # 检查2：aclnn 文档是否存在（基于 ACLNNTYPE 判断）
        if op_dir_path:
            stats['by_check']['aclnn_doc']['total'] += 1
            need_doc, doc_source = need_aclnn_doc(op_dir_path)
            result['aclnn_doc_source'] = doc_source
            
            if not need_doc:
                result['checks']['aclnn_doc'] = 'not_needed'
                issues['aclnn_doc_not_needed'].append(interface)
                stats['by_check']['aclnn_doc']['passed'] += 1
            else:
                doc_exists, doc_path = check_aclnn_doc_exists(op_dir_path, interface['interface_name'])
                
                if not doc_exists:
                    result['checks']['aclnn_doc'] = 'missing'
                    result['status'] = 'failed'
                    issues['aclnn_doc_missing'].append({**interface, 'doc_source': doc_source})
                else:
                    result['checks']['aclnn_doc'] = 'exists'
                    result['doc_path'] = doc_path
                    stats['by_check']['aclnn_doc']['passed'] += 1
        
        # 检查3：链接是否有效
        stats['by_check']['link']['total'] += 1
        link_valid, link_status = check_link_validity(repo_root, interface['link_path'])
        
        if not link_valid:
            result['checks']['link'] = link_status
            if link_status != 'link_empty':
                result['status'] = 'failed'
                issues['link_broken'].append({**interface, 'link_status': link_status})
        else:
            result['checks']['link'] = 'valid'
            stats['by_check']['link']['passed'] += 1
        
# 检查4：确定性说明一致性（A2/A3）- 基于固定话术验证
        if op_dir_path and interface['deterministic_a2a3']:
            stats['by_check']['deterministic_a2a3']['total'] += 1
            
            deterministic_by_product, product_support = check_deterministic_in_doc(result.get('doc_path'))
            
            is_consistent, reason, details = analyze_deterministic(
                interface['deterministic_a2a3'],
                deterministic_by_product,
                product_support,
                column_name='A2/A3'
            )
            
            if not is_consistent:
                result['checks']['deterministic_a2a3'] = f'inconsistent: {reason}'
                result['status'] = 'failed'
                issues['deterministic_inconsistent'].append({
                    **interface,
                    'reason': reason,
                    'column': 'A2/A3',
                    'doc_deterministic': deterministic_by_product,
                    'doc_product_support': product_support,
                    'details': details,
                })
            else:
                result['checks']['deterministic_a2a3'] = 'consistent'
                stats['by_check']['deterministic_a2a3']['passed'] += 1
        
        # 检查5：确定性说明一致性（Ascend 950）- 基于固定话术验证
        if op_dir_path and interface.get('deterministic_a5'):
            stats['by_check']['deterministic_a5']['total'] += 1
            
            deterministic_by_product, product_support = check_deterministic_in_doc(result.get('doc_path'))
            
            is_consistent, reason, details = analyze_deterministic(
                interface['deterministic_a5'],
                deterministic_by_product,
                product_support,
                column_name='Ascend 950'
            )
            
            if not is_consistent:
                result['checks']['deterministic_a5'] = f'inconsistent: {reason}'
                if result['status'] != 'failed':
                    result['status'] = 'failed'
                issues['deterministic_inconsistent'].append({
                    **interface,
                    'reason': reason,
                    'column': 'Ascend 950',
                    'doc_deterministic': deterministic_by_product,
                    'doc_product_support': product_support,
                    'details': details,
                })
            else:
                result['checks']['deterministic_a5'] = 'consistent'
                stats['by_check']['deterministic_a5']['passed'] += 1
        
        # 检查6：A2/A3与A5确定性说明一致性检查
        if interface.get('deterministic_a2a3') and interface.get('deterministic_a5'):
            if result.get('doc_path'):
                deterministic_by_product, _ = check_deterministic_in_doc(result.get('doc_path'))
                
                is_consistent, reason = check_a2a3_a5_consistency(
                    interface['deterministic_a2a3'],
                    interface['deterministic_a5'],
                    deterministic_by_product
                )
                
                if not is_consistent:
                    result['checks']['a2a3_a5_consistency'] = f'inconsistent: {reason}'
                    issues['a2a3_a5_inconsistent'].append({
                        **interface,
                        'reason': reason,
                        'a2a3': interface['deterministic_a2a3'],
                        'a5': interface['deterministic_a5'],
                    })
        
        results.append(result)
        
        if result['status'] == 'passed':
            stats['passed'] += 1
        else:
            stats['failed'] += 1
    
    # 遗漏排查：检查是否有接口未列入 op_api_list
    listed_interface_names = set(interface['interface_name'] for interface in interfaces)
    listed_op_dirs = set(interface.get('op_dir') for interface in interfaces if interface.get('op_dir'))
    
    missing_required, optional_missing = find_missing_interfaces(repo_root, listed_interface_names, listed_op_dirs)
    
    if missing_required:
        issues['missing_in_doc'] = missing_required
        stats['failed'] += len(missing_required)
    
    if optional_missing:
        issues['optional_missing'] = optional_missing
    
    return {
        'stats': stats,
        'issues': issues,
        'results': results,
        'repo_type': repo_type,
        'scan_time': datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'missing_in_doc': missing_required,
        'optional_missing': optional_missing,
    }


def generate_report(scan_data, output_path):
    """生成 Markdown 报告"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    repo_type = scan_data['repo_type']
    stats = scan_data['stats']
    issues = scan_data['issues']
    results = scan_data['results']
    
    report = f"""# {repo_type} 算子接口列表一致性扫描报告

**扫描时间**: {scan_data['scan_time']}  
**仓库路径**: `{repo_type}/`  
**扫描范围**: docs/zh/op_api_list.md（全量验证）

---

## 一、扫描概览

### 1.1 统计摘要

| 检查项 | 扫描数 | 通过数 | 失败数 |
|-------|-------|-------|-------|
| 接口名一致性 | {stats['by_check']['interface_exists']['total']} | {stats['by_check']['interface_exists']['passed']} | {stats['by_check']['interface_exists']['total'] - stats['by_check']['interface_exists']['passed']} |
| aclnn文档存在性 | {stats['by_check']['aclnn_doc']['total']} | {stats['by_check']['aclnn_doc']['passed']} | {stats['by_check']['aclnn_doc']['total'] - stats['by_check']['aclnn_doc']['passed']} |
| 链接跳转有效性 | {stats['by_check']['link']['total']} | {stats['by_check']['link']['passed']} | {stats['by_check']['link']['total'] - stats['by_check']['link']['passed']} |
| 确定性说明一致性(A2/A3) | {stats['by_check']['deterministic_a2a3']['total']} | {stats['by_check']['deterministic_a2a3']['passed']} | {stats['by_check']['deterministic_a2a3']['total'] - stats['by_check']['deterministic_a2a3']['passed']} |
| 确定性说明一致性(Ascend 950) | {stats['by_check']['deterministic_a5']['total']} | {stats['by_check']['deterministic_a5']['passed']} | {stats['by_check']['deterministic_a5']['total'] - stats['by_check']['deterministic_a5']['passed']} |
| **总计** | **{stats['total']}** | **{stats['passed']}** | **{stats['failed']}** |

### 1.2 问题分类统计

| 问题类型 | 数量 | 占比 |
|---------|------|------|
| 接口对应的算子不存在 | {len(issues['interface_not_found'])} | {len(issues['interface_not_found'])/max(stats['total'],1)*100:.1f}% |
| aclnn文档缺失 | {len(issues['aclnn_doc_missing'])} | {len(issues['aclnn_doc_missing'])/max(stats['total'],1)*100:.1f}% |
| 无需aclnn文档 | {len(issues['aclnn_doc_not_needed'])} | {len(issues['aclnn_doc_not_needed'])/max(stats['total'],1)*100:.1f}% |
| 链接断链 | {len(issues['link_broken'])} | {len(issues['link_broken'])/max(stats['total'],1)*100:.1f}% |
| 确定性说明不一致 | {len(issues['deterministic_inconsistent'])} | {len(issues['deterministic_inconsistent'])/max(stats['total'],1)*100:.1f}% |
| A2/A3与A5一致性问题 | {len(issues['a2a3_a5_inconsistent'])} | {len(issues['a2a3_a5_inconsistent'])/max(stats['total'],1)*100:.1f}% |
| 接口遗漏（未列入op_api_list） | {len(issues['missing_in_doc'])} | {len(issues['missing_in_doc'])/max(stats['total'],1)*100:.1f}% |
| 可选登记接口（仅提醒） | {len(issues['optional_missing'])} | - |
| 跳过experimental算子 | {len(issues['experimental_skipped'])} | - |

---

## 二、问题详情

"""
    
    # 接口对应的算子不存在
    if issues['interface_not_found']:
        report += "### 2.1 接口对应的算子不存在\n\n"
        report += "| 序号 | 接口名 | 推断算子名 |\n"
        report += "|:---:|-------|-----------|\n"
        for i, item in enumerate(issues['interface_not_found'], 1):
            report += f"| {i} | {item['interface_name']} | {item.get('op_dir', 'N/A')} |\n"
        report += "\n"
    
    # aclnn 文档缺失
    if issues['aclnn_doc_missing']:
        report += "### 2.2 aclnn文档缺失\n\n"
        report += "| 序号 | 接口名 | 算子路径 | aclnn来源 |\n"
        report += "|:---:|-------|---------|----------|\n"
        for i, item in enumerate(issues['aclnn_doc_missing'], 1):
            doc_source = item.get('doc_source', 'N/A')
            source_desc = '自动生成' if doc_source == 'auto' else '手动实现' if doc_source == 'manual' else doc_source
            report += f"| {i} | {item['interface_name']} | `{item.get('op_dir', 'N/A')}` | {source_desc} |\n"
        report += "\n"
    
    # 无需 aclnn 文档
    if issues['aclnn_doc_not_needed']:
        report += "### 2.3 无需aclnn文档 ✅ 非问题\n\n"
        report += "| 序号 | 接口名 | 算子路径 | 说明 |\n"
        report += "|:---:|-------|---------|------|\n"
        for i, item in enumerate(issues['aclnn_doc_not_needed'], 1):
            aclnn_type = item.get('op_dir', 'N/A')
            report += (
                f"| {i} | {item['interface_name']} | `{aclnn_type}` | "
                f"ACLNNTYPE=aclnn_exclude且无aclnn实现 |\n"
            )
        report += "\n"
    
    # 链接断链
    if issues['link_broken']:
        report += "### 2.4 链接断链\n\n"
        report += "| 序号 | 接口名 | 链接路径 | 状态 |\n"
        report += "|:---:|-------|---------|------|\n"
        for i, item in enumerate(issues['link_broken'], 1):
            report += f"| {i} | {item['interface_name']} | `{item['link_path']}` | {item['link_status']} |\n"
        report += "\n"
    
    # 确定性说明不一致（仅基于文档判断）
    if issues['deterministic_inconsistent']:
        report += "### 2.5 确定性说明不一致\n\n"
        report += "| 序号 | 接口名 | 列名 | op_api_list说明 | aclnn文档确定性说明 | 原因 |\n"
        report += "|:---:|-------|------|----------------|---------------------|------|\n"
        for i, item in enumerate(issues['deterministic_inconsistent'], 1):
            column = item.get('column', 'A2/A3')
            op_list_value = item.get('deterministic_a2a3') if column == 'A2/A3' else item.get('deterministic_a5', '-')
            doc_deterministic = item.get('doc_deterministic', {})
            if doc_deterministic:
                if 'all' in doc_deterministic:
                    doc_desc = doc_deterministic['all']
                else:
                    matched_product = item.get('details', {}).get('matched_product', 'N/A')
                    doc_desc = doc_deterministic.get(matched_product, str(doc_deterministic)[:60])
            else:
                doc_desc = '缺少确定性说明'
            doc_desc_display = doc_desc[:60] + '...' if len(str(doc_desc)) > 60 else str(doc_desc)
            report += (
                f"| {i} | {item['interface_name']} | {column} | "
                f"`{op_list_value}` | `{doc_desc_display}` | {item['reason']} |\n"
            )
        report += "\n"
    
    # A2/A3与A5一致性检查
    if issues['a2a3_a5_inconsistent']:
        report += "### 2.6 A2/A3与A5确定性说明不一致 ⚠️\n\n"
        report += "以下接口的A2/A3和A5确定性说明不一致，但aclnn文档未区分产品型号。\n\n"
        report += "| 序号 | 接口名 | A2/A3说明 | A5说明 | 原因 |\n"
        report += "|:---:|-------|----------|--------|------|\n"
        for i, item in enumerate(issues['a2a3_a5_inconsistent'], 1):
            report += f"| {i} | {item['interface_name']} | `{item['a2a3']}` | `{item['a5']}` | {item['reason']} |\n"
        report += "\n"
    
    # 跳过的 experimental 算子接口
    if issues['experimental_skipped']:
        report += "### 2.7 跳过的experimental算子接口 ✅ 生态开发者提供\n\n"
        report += f"共跳过 {len(issues['experimental_skipped'])} 个 experimental 目录下的算子接口（生态开发者提供，不检查）。\n\n"
        report += "| 序号 | 接口名 | 算子路径 |\n"
        report += "|:---:|-------|---------|\n"
        for i, item in enumerate(issues['experimental_skipped'], 1):
            report += f"| {i} | {item['interface_name']} | `{item.get('op_dir', 'N/A')}` |\n"
        report += "\n"
    
    # 接口遗漏（必须添加）
    if issues['missing_in_doc']:
        report += "### 2.8 接口遗漏 ⚠️ 未列入op_api_list\n\n"
        report += f"共发现 {len(issues['missing_in_doc'])} 个接口有aclnn手动实现但未列入 op_api_list.md 文档。\n\n"
        report += "| 序号 | 接口名 | 算子路径 | 遗漏原因 |\n"
        report += "|:---:|-------|---------|---------|\n"
        for i, item in enumerate(issues['missing_in_doc'], 1):
            report += f"| {i} | {item['interface_name']} | `{item['op_path']}` | {item['reason']} |\n"
        report += "\n"
    
    # 可选登记接口（仅提醒）
    if issues['optional_missing']:
        report += "### 2.9 可选登记接口 ✅ 仅提醒\n\n"
        report += f"共发现 {len(issues['optional_missing'])} 个接口 ACLNNTYPE 为 aclnn/aclnn_inner，用户可自行选择是否登记。\n\n"
        report += "| 序号 | 接口名 | 算子路径 | ACLNNTYPE | 说明 |\n"
        report += "|:---:|-------|---------|----------|------|\n"
        for i, item in enumerate(issues['optional_missing'], 1):
            report += (
                f"| {i} | {item['interface_name']} | `{item['op_path']}` | "
                f"{item['aclnn_type']} | 可选登记，需上传aclnn文档 |\n"
            )
        report += "\n"
    
    report += "---\n\n**报告生成时间**: " + scan_data['scan_time'] + "\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")
    return report


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from repo_detector import get_repo_root, get_reports_output_dir


def main():
    parser = argparse.ArgumentParser(description='算子接口列表一致性扫描脚本')
    parser.add_argument('--repo', required=True,
                        help='仓库类型（支持任意 ops-* 仓库，如 ops-math、ops-nn、ops-transformer、ops-cv 等）')
    parser.add_argument('--repo-root', default=None,
                        help='仓库根目录（默认根据 repo 类型推断）')
    parser.add_argument('--output', default=None,
                        help='输出 Markdown 报告路径')
    parser.add_argument('--json', default=None,
                        help='输出 JSON 数据路径')
    
    args = parser.parse_args()
    
    if not args.repo.startswith('ops-'):
        print(f"错误: 仓库名应以 'ops-' 开头，当前为: {args.repo}", file=sys.stderr)
        sys.exit(1)
    
    repo_root, detection_method = get_repo_root(args.repo, args.repo_root)
    repo_root = str(repo_root)
    
    date_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d')
    time_str = datetime.now(tz=timezone.utc).strftime('%H%M%S')
    
    reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo)
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(reports_dir / date_str / args.repo / f'op-api-list-validation_report_{time_str}.md')
    
    if args.json:
        json_path = args.json
    else:
        json_path = str(reports_dir / date_str / args.repo / 'op-api-list-validation_data.json')
    
    print(f"仓库类型: {args.repo}")
    print(f"仓库根目录: {repo_root}")
    print(f"检测方法: {detection_method}")
    print(f"Reports 目录: {reports_dir} (检测方式: {reports_method})")
    print(f"输出报告: {output_path}")
    print()
    
    scan_data = scan_repository(repo_root, args.repo)
    
    if 'error' in scan_data:
        print(f"错误: {scan_data['error']}")
        return
    
    # 保存 JSON 数据
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scan_data, f, indent=2, ensure_ascii=False)
    print(f"JSON数据已保存: {json_path}")
    
    # 生成报告
    generate_report(scan_data, output_path)
    
    # 打印摘要
    stats = scan_data['stats']
    print("\n扫描摘要:")
    print(f"总接口数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"通过率: {stats['passed']/stats['total']*100:.1f}%")
    
    if scan_data['missing_in_doc']:
        print(f"\n实际存在但未列入op_api_list: {len(scan_data['missing_in_doc'])} 个")


if __name__ == '__main__':
    main()
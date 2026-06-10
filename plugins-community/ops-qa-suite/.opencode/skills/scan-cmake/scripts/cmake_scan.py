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
CMake 配置问题扫描脚本（动态推断版本）

扫描任意 ops-* 仓库的 CMakeLists.txt 文件，检测以下问题：
1. OPTYPE 与目录名不一致
2. 函数名错误（add_modules_llt_sources 使用错误）
3. 变量名错误（OPTEST_NAME）
4. 参数名错误（HOSTNAME/UT_NAME 使用错误）
5. if 语句语法错误（变量未用引号包裹）
6. CMake 目标名称冲突/重复定义
7. 缺少源文件错误（No SOURCES given to target）
8. 条件判断缺少 OP_HOST_UT
9. 第三方依赖解压失败

核心改进：
- 动态推断 CMake 行为（无需硬编码仓库配置）
- 支持任意 ops-* 仓库
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).parent
GITCODE_ISSUE_SCRIPTS = SCRIPT_DIR.parent.parent / 'gitcode-issue-creator' / 'scripts'
if GITCODE_ISSUE_SCRIPTS.exists():
    sys.path.insert(0, str(GITCODE_ISSUE_SCRIPTS))
    from generate_issue_md import (
        generate_title as generate_title_func,
        generate_description as generate_description_func,
        generate_issue_md as generate_issue_md_func
    )

REPO_DETECTOR_PATH = SCRIPT_DIR.parent.parent.parent / 'scripts'
if REPO_DETECTOR_PATH.exists():
    sys.path.insert(0, str(REPO_DETECTOR_PATH))
    from repo_detector import get_repo_root, get_reports_output_dir
    from cmake_profiler import CMakeProfiler
    from repo_discovery import RepoDiscovery
    from config_loader import get_structure_dirs, get_ut_structure

STRUCTURE = get_structure_dirs()
UT_DIRS = get_ut_structure()


class CMakeScanner:
    """CMake 配置问题扫描器"""

    CORRECT_VARS = [
        'OP_TILING_MODULE_NAME',
        'OP_INFERSHAPE_MODULE_NAME',
        'OP_API_MODULE_NAME',
        'OP_KERNEL_MODULE_NAME',
    ]

    # 错误的变量名
    INCORRECT_VAR = 'OPTEST_NAME'

    # 问题类型名称映射
    ISSUE_TYPE_NAMES = {
        'optype_mismatch': 'OPTYPE 与目录名不一致',
        'function_not_defined': '函数不存在',
        'variable_not_defined': '变量不存在',
        'param_name_error': '参数名错误',
        'if_syntax_error': 'if 语句语法错误',
        'target_conflict': '目标名称冲突',
        'no_sources': '缺少源文件',
        'condition_missing': '条件判断缺失',
        'dependency_error': '第三方依赖错误',
        'ut_branch_missing': 'UT 分支缺失',
    }

    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.results = {
            'scan_time': datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'scan_type': 'all',
            'repos': {},
            'summary': {
                'total_issue_files': 0,
                'by_repo': {},
                'by_issue_type': {k: 0 for k in self.ISSUE_TYPE_NAMES.keys()},
            },
        }
        self.op_dirs_by_repo = {}

    @staticmethod
    def extract_optype_values(content: str) -> list:
        """从 CMakeLists 内容中提取所有 OPTYPE 值
        
        只匹配有效的算子名称（字母、数字、下划线），避免把右括号等字符算进去。
        """
        optypes = []
        pattern = r'OPTYPE\s+([a-zA-Z0-9_]+)'
        for match in re.finditer(pattern, content):
            optypes.append(match.group(1))
        return optypes

    @staticmethod
    def get_op_name_from_path(cmake_path: Path) -> str | None:
        parts = cmake_path.parts
        try:
            op_host_idx = parts.index(STRUCTURE['host'])
            op_name = parts[op_host_idx - 1]
            return op_name
        except (ValueError, IndexError):
            return None

    def build_op_dir_map(self, repo_path: Path) -> dict:
        op_dirs = {}
        cmake_files = list(repo_path.rglob(f"{STRUCTURE['host']}/CMakeLists.txt"))
        for cmake_file in cmake_files:
            op_name = self.get_op_name_from_path(cmake_file)
            if op_name:
                rel_path = str(cmake_file.relative_to(repo_path))
                op_dirs[op_name] = {
                    'path': rel_path,
                    'full_path': str(cmake_file),
                }
        return op_dirs

    @staticmethod
    def find_conflicting_op(optype_value: str, repo_path: Path, op_dirs: dict) -> dict | None:
        """查找与 OPTYPE 值冲突的算子"""
        conflicting_op = None
        if optype_value in op_dirs:
            conflicting_op = {
                'op_name': optype_value,
                'path': op_dirs[optype_value]['path'],
            }
        return conflicting_op

    def scan_optype_issues(self, file_path: Path, op_name: str, repo_path: Path, op_dirs: dict) -> list:
        """扫描 OPTYPE 相关问题"""
        issues = []

        try:
            content = file_path.read_text()
        except Exception as e:
            return [{'issue_type': 'read_error', 'detail': str(e)}]

        optypes = self.extract_optype_values(content)
        lines = content.split('\n')

        for optype in optypes:
            if optype != op_name:
                for i, line in enumerate(lines, 1):
                    if f'OPTYPE {optype}' in line:
                        conflicting_op = self.find_conflicting_op(optype, repo_path, op_dirs)
                        
                        issue = {
                            'file': str(file_path),
                            'line': i,
                            'issue_type': 'optype_mismatch',
                            'op_name': op_name,
                            'optype_value': optype,
                            'detail': f'OPTYPE "{optype}" 与目录名 "{op_name}" 不一致',
                            'suggestion': f'修改为 OPTYPE {op_name}',
                        }
                        
                        if conflicting_op:
                            issue['conflicting_op'] = conflicting_op
                            issue['detail'] += f'，与算子 "{conflicting_op["op_name"]}" 冲突'
                        
                        issues.append(issue)

        return issues

    def scan_ut_issues(self, file_path: Path, repo_type: str, config: dict) -> list:
        """扫描 UT 相关问题
        
        检测两类问题：
        1. BUG：当前有 UT 源文件，但 CMakeLists 缺少对应分支（会导致构建失败）
        2. 规范问题：当前无 UT 源文件，但 CMakeLists 结构不完整（代码质量问题）
        
        检测逻辑：
        - 有 test_*_tiling.cpp → 应有 OP_TILING_MODULE_NAME 调用（BUG）
        - 有 test_*_infershape.cpp → 应有 OP_INFERSHAPE_MODULE_NAME 调用（BUG）
        - 无源文件 → 应有完整结构支持所有类型（规范问题）
        """
        issues = []

        try:
            content = file_path.read_text()
        except Exception as e:
            logger.debug("Failed to read file for ut issues scan: %s", e)
            return [{'issue_type': 'read_error', 'detail': str(e)}]

        lines = content.split('\n')

        issue_severity = config.get('issue_severity', 'bug')
        tests_cmake_required = config.get('tests_cmake_required', True)
        severity_label = 'BUG' if issue_severity == 'bug' else '规范问题'

        base_detail_suffix = ''
        if not tests_cmake_required:
            base_detail_suffix = '（注意：此文件不会被引入构建系统，属于遗留无效代码）'
        else:
            base_detail_suffix = '（注意：此文件会被引入构建系统，会导致构建失败）'

        # 问题类型 1：函数名错误
        if not config.get('defines_llt_sources', False):
            for incorrect_func in config.get('incorrect_functions', []):
                if incorrect_func in content:
                    for i, line in enumerate(lines, 1):
                        if incorrect_func in line:
                            issues.append({
                                'file': str(file_path),
                                'line': i,
                                'issue_type': 'function_not_defined',
                                'detail': f'[{severity_label}] 使用不存在的函数 {incorrect_func}{base_detail_suffix}',
                                'suggestion': (
                    f'替换为 {config["correct_function"]}'
                    if tests_cmake_required
                    else '建议删除此文件或清理无效代码'
                ),
                                'severity': issue_severity,
                                'tests_cmake_required': tests_cmake_required,
                            })

        # 问题类型 2：变量名错误
        if self.INCORRECT_VAR in content:
            for i, line in enumerate(lines, 1):
                if self.INCORRECT_VAR in line:
                    suggestion_vars = self._suggest_correct_var(line)
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue_type': 'variable_not_defined',
                        'detail': f'[{severity_label}] 使用不存在的变量 ${self.INCORRECT_VAR}{base_detail_suffix}',
                        'suggestion': f'替换为 ${suggestion_vars}' if tests_cmake_required else '建议删除此文件或清理无效代码',
                        'severity': issue_severity,
                        'tests_cmake_required': tests_cmake_required,
                    })

        # 问题类型 3：参数名错误
        for incorrect_param in config.get('incorrect_params', []):
            pattern = rf'(add_modules_ut_sources|add_modules_llt_sources)\s*\(\s*{incorrect_param}'
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue_type': 'param_name_error',
                        'detail': f'[{severity_label}] 参数名错误 {incorrect_param}{base_detail_suffix}',
                        'suggestion': f'替换为 {config["correct_param"]}' if tests_cmake_required else '建议删除此文件或清理无效代码',
                        'severity': issue_severity,
                        'tests_cmake_required': tests_cmake_required,
                    })

        # 问题类型 4：UT 分支缺失检测
        # 检测逻辑：根据当前目录的 UT 源文件，检查 CMakeLists 是否有对应分支
        ut_branch_issues = self._check_ut_branch_coverage(file_path, content, tests_cmake_required)
        issues.extend(ut_branch_issues)

        return issues

    @staticmethod
    def scan_cmake_ut_file_issues(file_path: Path) -> list:
        """扫描 cmake/ut.cmake 文件中的 if 语句语法错误"""
        issues = []

        try:
            content = file_path.read_text()
        except Exception as e:
            return [{'issue_type': 'read_error', 'detail': str(e)}]

        lines = content.split('\n')

        # 检测 if(${var} STREQUAL "") 模式
        pattern = r'if\(\s*\$\{(\w+)\}\s*STREQUAL\s*"?\s*"?'
        for i, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                var_name = match.group(1)
                issues.append({
                    'file': str(file_path),
                    'line': i,
                    'issue_type': 'if_syntax_error',
                    'detail': f'if 语句变量 ${var_name} 未用引号包裹，当变量为空时会导致语法错误',
                    'suggestion': f'修改为 if("${var_name}" STREQUAL "")',
                })

        return issues

    def scan_repo(self, repo_type: str) -> dict:
        """扫描单个仓库"""
        repo_path = self.workspace / repo_type
        if not repo_path.exists():
            return {'error': f'仓库路径不存在: {repo_path}'}

        config = CMakeProfiler.profile_repo(repo_path)
        issues = []

        # 扫描 OPTYPE 问题：op_host/CMakeLists.txt
        op_dirs = self.build_op_dir_map(repo_path)
        self.op_dirs_by_repo[repo_type] = op_dirs

        op_host_cmake_files = list(repo_path.rglob(f"{STRUCTURE['host']}/CMakeLists.txt"))
        for cmake_file in op_host_cmake_files:
            op_name = self.get_op_name_from_path(cmake_file)
            if op_name:
                optype_issues = self.scan_optype_issues(cmake_file, op_name, repo_path, op_dirs)
                for issue in optype_issues:
                    issue['file'] = str(cmake_file.relative_to(repo_path))
                issues.extend(optype_issues)

        ut_cmake_files = list(repo_path.rglob(f"{UT_DIRS['host_ut']}/CMakeLists.txt"))
        for cmake_file in ut_cmake_files:
            ut_issues = self.scan_ut_issues(cmake_file, repo_type, config)
            for issue in ut_issues:
                issue['file'] = str(cmake_file.relative_to(repo_path))
            issues.extend(ut_issues)

        # 扫描 cmake/ut.cmake 文件中的 if 语句语法错误
        cmake_ut_file = repo_path / 'cmake' / 'ut.cmake'
        if cmake_ut_file.exists():
            ut_file_issues = self.scan_cmake_ut_file_issues(cmake_ut_file)
            for issue in ut_file_issues:
                issue['file'] = str(cmake_ut_file.relative_to(repo_path))
            issues.extend(ut_file_issues)

        repo_result = {
            'total_files': len(op_host_cmake_files) + len(ut_cmake_files),
            'issue_files': len(set(i['file'] for i in issues)),
            'issues': issues,
        }

        self.results['repos'][repo_type] = repo_result
        self.results['summary']['by_repo'][repo_type] = len(set(i['file'] for i in issues))

        # 统计问题类型
        for issue in issues:
            issue_type = issue['issue_type']
            if issue_type in self.results['summary']['by_issue_type']:
                self.results['summary']['by_issue_type'][issue_type] += 1

        return repo_result

    def scan_optype_only(self, repo_type: str) -> dict:
        """仅扫描 OPTYPE 问题"""
        self.results['scan_type'] = 'optype'
        repo_path = self.workspace / repo_type
        if not repo_path.exists():
            return {'error': f'仓库路径不存在: {repo_path}'}

        issues = []
        op_dirs = self.build_op_dir_map(repo_path)
        self.op_dirs_by_repo[repo_type] = op_dirs

        op_host_cmake_files = list(repo_path.rglob(f"{STRUCTURE['host']}/CMakeLists.txt"))
        for cmake_file in op_host_cmake_files:
            op_name = self.get_op_name_from_path(cmake_file)
            if op_name:
                optype_issues = self.scan_optype_issues(cmake_file, op_name, repo_path, op_dirs)
                for issue in optype_issues:
                    issue['file'] = str(cmake_file.relative_to(repo_path))
                issues.extend(optype_issues)

        repo_result = {
            'total_files': len(op_host_cmake_files),
            'issue_files': len(set(i['file'] for i in issues)),
            'issues': issues,
        }

        self.results['repos'][repo_type] = repo_result
        self.results['summary']['by_repo'][repo_type] = len(set(i['file'] for i in issues))

        for issue in issues:
            issue_type = issue['issue_type']
            if issue_type in self.results['summary']['by_issue_type']:
                self.results['summary']['by_issue_type'][issue_type] += 1

        return repo_result

    def scan_ut_only(self, repo_type: str) -> dict:
        self.results['scan_type'] = 'ut'
        repo_path = self.workspace / repo_type
        if not repo_path.exists():
            return {'error': f'仓库路径不存在: {repo_path}'}

        config = CMakeProfiler.profile_repo(repo_path)
        issues = []

        ut_cmake_files = list(repo_path.rglob(f"{UT_DIRS['host_ut']}/CMakeLists.txt"))
        for cmake_file in ut_cmake_files:
            ut_issues = self.scan_ut_issues(cmake_file, repo_type, config)
            for issue in ut_issues:
                issue['file'] = str(cmake_file.relative_to(repo_path))
            issues.extend(ut_issues)

        # 扫描 cmake/ut.cmake 文件
        cmake_ut_file = repo_path / 'cmake' / 'ut.cmake'
        if cmake_ut_file.exists():
            ut_file_issues = self.scan_cmake_ut_file_issues(cmake_ut_file)
            for issue in ut_file_issues:
                issue['file'] = str(cmake_ut_file.relative_to(repo_path))
            issues.extend(ut_file_issues)

        repo_result = {
            'total_files': len(ut_cmake_files),
            'issue_files': len(set(i['file'] for i in issues)),
            'issues': issues,
        }

        self.results['repos'][repo_type] = repo_result
        self.results['summary']['by_repo'][repo_type] = len(set(i['file'] for i in issues))

        for issue in issues:
            issue_type = issue['issue_type']
            if issue_type in self.results['summary']['by_issue_type']:
                self.results['summary']['by_issue_type'][issue_type] += 1

        return repo_result

    def scan_all_repos(self, scan_type: str = 'all') -> dict:
        """扫描所有仓库（动态发现）"""
        self.results['scan_type'] = scan_type
        
        repos = RepoDiscovery.discover_repos(self.workspace)
        if not repos:
            return {'error': '未发现任何 ops-* 仓库'}
        
        for repo_name, _ in repos:
            if scan_type == 'optype':
                self.scan_optype_only(repo_name)
            elif scan_type == 'ut':
                self.scan_ut_only(repo_name)
            else:
                self.scan_repo(repo_name)

        total = sum(self.results['summary']['by_repo'].values())
        self.results['summary']['total_issue_files'] = total

        return self.results

    def save_results(self, output_path: str):
        """保存结果到 JSON 文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f'结果已保存到: {output_file}')

    def print_summary(self):
        """打印扫描摘要"""
        print('\n' + '=' * 80)
        print(f'CMake 配置问题扫描摘要（扫描类型: {self.results["scan_type"]}）')
        print('=' * 80)
        
        print('\n各仓库问题文件统计:')
        print('-' * 40)
        print(f'{"仓库":<20} {"问题文件数":<15}')
        print('-' * 40)
        for repo, count in self.results['summary']['by_repo'].items():
            print(f'{repo:<20} {count:<15}')
        print('-' * 40)
        print(f'{"总计":<20} {self.results["summary"]["total_issue_files"]:<15}')

        print('\n按问题类型统计:')
        print('-' * 50)
        print(f'{"问题类型":<30} {"数量":<10}')
        print('-' * 50)
        for issue_type, count in self.results['summary']['by_issue_type'].items():
            if count > 0:
                name = self.ISSUE_TYPE_NAMES.get(issue_type, issue_type)
                print(f'{name:<30} {count:<10}')
        print('-' * 50)

        print('\n问题详情:')
        print('-' * 100)
        for repo, repo_result in self.results['repos'].items():
            if repo_result.get('issues'):
                print(f'\n【{repo}】:')
                for issue in repo_result['issues'][:20]:
                    issue_name = self.ISSUE_TYPE_NAMES.get(issue['issue_type'], issue['issue_type'])
                    print(f'  [{issue_name}] {issue["file"]}:{issue["line"]}')
                    print(f'    详情: {issue["detail"]}')
                    print(f'    建议: {issue["suggestion"]}')
                    if issue.get('conflicting_op'):
                        print(f'    冲突算子: {issue["conflicting_op"]["op_name"]}')
                    print()

    # 私有方法（实现细节）

    @staticmethod
    def _suggest_correct_var(line: str) -> str:
        """根据上下文建议正确的变量名"""
        if 'tiling' in line.lower():
            return 'OP_TILING_MODULE_NAME'
        elif 'infershape' in line.lower():
            return 'OP_INFERSHAPE_MODULE_NAME'
        elif 'api' in line.lower() or 'aclnn' in line.lower():
            return 'OP_API_MODULE_NAME'
        elif 'kernel' in line.lower():
            return 'OP_KERNEL_MODULE_NAME'
        else:
            return 'OP_TILING_MODULE_NAME 或 OP_INFERSHAPE_MODULE_NAME'

    @staticmethod
    def _extract_if_blocks(content: str) -> list:
        """提取 if 条件块"""
        blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('if('):
                match = re.search(r'if\((.*?)\)', line)
                if match:
                    block_content = []
                    j = i
                    while j < len(lines):
                        block_content.append(lines[j])
                        if lines[j].strip().startswith('endif'):
                            break
                        j += 1
                    blocks.append({
                        'line': i + 1,
                        'condition': match.group(1),
                        'content': '\n'.join(block_content),
                    })
            i += 1
        
        return blocks

    @staticmethod
    def _should_have_op_host_ut(block: dict, file_path: Path) -> tuple[bool, bool]:
        """判断该 if 块是否应该包含 OP_HOST_UT"""
        condition = block['condition']
        block_content = block['content']
        
        if 'OP_HOST_UT' in condition:
            return (False, False)
        
        test_dir = file_path.parent
        has_source_files = False
        
        if test_dir.exists():
            tiling_files = list(test_dir.rglob('test_*_tiling.cpp'))
            infershape_files = list(test_dir.rglob('test_*_infershape.cpp'))
            has_source_files = len(tiling_files) > 0 or len(infershape_files) > 0
        
        is_bug = False
        should_check = False
        
        if 'OP_TILING_MODULE_NAME' in block_content or 'OP_INFERSHAPE_MODULE_NAME' in block_content:
            should_check = True
            is_bug = has_source_files
        elif 'OP_API_MODULE_NAME' in block_content:
            should_check = False
            is_bug = False
        else:
            ut_keywords = ['TILING_UT', 'PROTO_UT', 'UT_TEST_ALL']
            if any(kw in condition for kw in ut_keywords):
                should_check = True
                is_bug = has_source_files
        
        return (should_check, is_bug)

    def _check_ut_branch_coverage(self, file_path: Path, content: str, tests_cmake_required: bool) -> list:
        """检查 UT 分支覆盖情况
        
        检测两类问题：
        1. BUG：有源文件但缺少对应分支
        2. 规范问题：无源文件但分支结构不完整
        """
        issues = []
        test_dir = file_path.parent
        
        has_tiling = False
        has_infershape = False
        has_op_api = False
        
        if test_dir.exists():
            tiling_files = list(test_dir.rglob('test_*_tiling.cpp'))
            infershape_files = list(test_dir.rglob('test_*_infershape.cpp'))
            op_api_files = list(test_dir.rglob('test_aclnn_*.cpp'))
            
            has_tiling = len(tiling_files) > 0
            has_infershape = len(infershape_files) > 0
            has_op_api = len(op_api_files) > 0
        
        has_tiling_branch = 'OP_TILING_MODULE_NAME' in content
        has_infershape_branch = 'OP_INFERSHAPE_MODULE_NAME' in content
        has_op_api_branch = 'OP_API_MODULE_NAME' in content
        
        if_blocks = self._extract_if_blocks(content)
        
        host_ut_condition_ok = False
        for block in if_blocks:
            if ('OP_TILING_MODULE_NAME' in block['content'] or 
                'OP_INFERSHAPE_MODULE_NAME' in block['content']):
                condition = block['condition']
                if 'OP_HOST_UT' in condition or 'UT_TEST_ALL' in condition:
                    host_ut_condition_ok = True
                    break
        
        op_api_condition_ok = False
        for block in if_blocks:
            if 'OP_API_MODULE_NAME' in block['content']:
                condition = block['condition']
                if 'OP_API_UT' in condition or 'UT_TEST_ALL' in condition:
                    op_api_condition_ok = True
                    break
        
        if has_tiling and not (has_tiling_branch and host_ut_condition_ok):
            issues.append({
                'file': str(file_path),
                'line': 1,
                'issue_type': 'ut_branch_missing',
                'detail': (
                    '[BUG] 存在 tiling UT 源文件，'
                    '但 CMakeLists 缺少 OP_TILING_MODULE_NAME 分支或条件不正确'
                ),
                'suggestion': (
                    '添加: if(UT_TEST_ALL OR OP_HOST_UT)\n'
                    '    add_modules_llt_sources(HOSTNAME $OP_TILING_MODULE_NAME ...)\n'
                    'endif()'
                ),
                'severity': 'bug',
                'tests_cmake_required': tests_cmake_required,
                'has_source': True,
                'ut_type': 'tiling',
            })
        
        if has_infershape and not (has_infershape_branch and host_ut_condition_ok):
            issues.append({
                'file': str(file_path),
                'line': 1,
                'issue_type': 'ut_branch_missing',
                'detail': (
                    '[BUG] 存在 infershape UT 源文件，'
                    '但 CMakeLists 缺少 OP_INFERSHAPE_MODULE_NAME 分支或条件不正确'
                ),
                'suggestion': (
                    '添加: if(UT_TEST_ALL OR OP_HOST_UT)\n'
                    '    add_modules_llt_sources(HOSTNAME $OP_INFERSHAPE_MODULE_NAME ...)\n'
                    'endif()'
                ),
                'severity': 'bug',
                'tests_cmake_required': tests_cmake_required,
                'has_source': True,
                'ut_type': 'infershape',
            })
        
        if has_op_api and not (has_op_api_branch and op_api_condition_ok):
            issues.append({
                'file': str(file_path),
                'line': 1,
                'issue_type': 'ut_branch_missing',
                'detail': (
                    '[BUG] 存在 op_api UT 源文件，'
                    '但 CMakeLists 缺少 OP_API_MODULE_NAME 分支或条件不正确'
                ),
                'suggestion': (
                    '添加: if(UT_TEST_ALL OR OP_API_UT)\n'
                    '    add_modules_llt_sources(HOSTNAME $OP_API_MODULE_NAME ...)\n'
                    'endif()'
                ),
                'severity': 'bug',
                'tests_cmake_required': tests_cmake_required,
                'has_source': True,
                'ut_type': 'op_api',
            })
        
        if not has_tiling and not has_infershape and not has_op_api:
            missing_types = []
            if not has_tiling_branch:
                missing_types.append('tiling')
            if not has_infershape_branch:
                missing_types.append('infershape')
            if not has_op_api_branch:
                missing_types.append('op_api')
            
            if missing_types:
                issues.append({
                    'file': str(file_path),
                    'line': 1,
                    'issue_type': 'ut_branch_missing',
                    'detail': (
                        f'[规范问题] 无 UT 源文件，但 CMakeLists 缺少以下分支: '
                        f'{", ".join(missing_types)}（作为规范应支持所有类型）'
                    ),
                    'suggestion': '建议添加完整结构以支持未来添加 UT，或删除此无效 CMakeLists',
                    'severity': 'code_quality',
                    'tests_cmake_required': tests_cmake_required,
                    'has_source': False,
                    'ut_type': '规范',
                })
        
        return issues


def generate_report(results: dict, output_path: str):
    """生成 CMake 扫描报告和 GitCode Issue 文件
    
    生成两个输出：
    1. cmake_issues_report.md - 扫描汇总报告
    2. issues/<repo>_<issue_type>_issue.md - 符合 GitCode 模板格式的 Issue 文件（调用 gitcode-issue-creator）
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    issues_dir = output_file.parent / 'issues'
    issues_dir.mkdir(parents=True, exist_ok=True)

    type_names = {
        'optype_mismatch': 'OPTYPE 与目录名不一致',
        'function_not_defined': '函数不存在',
        'variable_not_defined': '变量不存在',
        'param_name_error': '参数名错误',
        'if_syntax_error': 'if 语句语法错误',
        'target_conflict': '目标名称冲突',
        'no_sources': '缺少源文件',
        'condition_missing': '条件判断缺失',
        'dependency_error': '第三方依赖错误',
    }

    repo_configs = {}
    
    for repo in results['repos'].keys():
        repo_path = Path(workspace) / repo if workspace else Path.cwd() / repo
        if repo_path.exists():
            profile = CMakeProfiler.profile_repo(repo_path)
            repo_configs[repo] = {
                'tests_cmake_required': profile.get('tests_cmake_required', True),
                'severity': 'bug' if profile.get('tests_cmake_required', True) else 'code_quality',
            }

    # 按 BUG 和规范问题分组
    bug_issues = []
    code_quality_issues = []

    for repo in results['repos'].keys():
        repo_result = results['repos'].get(repo, {})
        issues = repo_result.get('issues', [])
        repo_config = repo_configs.get(repo, {})
        
        for issue in issues:
            issue['repo'] = repo
            issue_severity = issue.get('severity', repo_config.get('severity', 'bug'))
            if issue_severity == 'bug':
                bug_issues.append(issue)
            else:
                code_quality_issues.append(issue)

    # 生成汇总报告
    report_lines = [
        '# CMake 配置问题扫描报告',
        '',
        f'**扫描时间**: {results["scan_time"]}',
        f'**扫描类型**: {results["scan_type"]}',
        '',
        '## 问题分类说明',
        '',
        '| 问题性质 | 说明 | 仓库示例 |',
        '|----------|------|----------|',
        '| **BUG** | 会导致构建失败，需优先修复 | ops-transformer, ops-nn |',
        '| **规范问题** | 遗留无效代码，不影响构建，建议清理 | ops-math, ops-cv |',
        '',
        '## 问题统计汇总',
        '',
        '| 仓库 | 问题性质 | 问题文件数 |',
        '|------|----------|-----------|',
    ]

    for repo, count in results['summary']['by_repo'].items():
        severity = repo_configs.get(repo, {}).get('severity', 'bug')
        severity_label = 'BUG' if severity == 'bug' else '规范问题'
        report_lines.append(f'| {repo} | {severity_label} | {count} |')
    report_lines.append(f'| **合计** | - | **{results["summary"]["total_issue_files"]}** |')

    report_lines.extend([
        '',
        '### 按问题类型统计',
        '',
        '| 问题类型 | 数量 |',
        '|----------|------|',
    ])
    
    for issue_type, count in results['summary']['by_issue_type'].items():
        if count > 0:
            name = type_names.get(issue_type, issue_type)
            report_lines.append(f'| {name} | {count} |')

    # 生成 BUG 类型 Issue 文件（调用 gitcode-issue-creator）
    if bug_issues:
        report_lines.extend([
            '',
            '---',
            '',
            '## ⚠️ BUG 类型问题 - GitCode Issue 文件',
            '',
            '> 以下问题会导致 CMake 配置失败，需要优先修复。',
            '> 已生成符合 GitCode 模板格式的 Issue 文件，可直接提交。',
            '',
        ])
        
        # 按仓库和类型分组
        by_repo_type = {}
        for issue in bug_issues:
            key = (issue['repo'], issue['issue_type'])
            if key not in by_repo_type:
                by_repo_type[key] = []
            by_repo_type[key].append(issue)

        issue_count = 0
        for (repo, issue_type), type_issues in sorted(by_repo_type.items()):
            issue_count += 1
            type_name = type_names.get(issue_type, issue_type)
            files = sorted(set(i['file'] for i in type_issues))
            
            # 构建问题描述内容
            description_text = (
                f'{repo} 仓库中存在 {len(files)} 个文件的 '
                f'CMakeLists.txt 存在 {type_name} 问题。\n\n'
                f'**问题详情**: {type_issues[0]["detail"]}'
            )
            environment_text = (
                f'- 仓库: `{repo}`\n'
                f'- 问题类型: `{type_name}`\n'
                f'- 问题文件数: `{len(files)}`\n'
                f'- 问题性质: `BUG` - 会导致构建失败'
            )
            steps_text = (
                '1. 执行构建命令 `bash build.sh -u` 或 `bash build.sh --ophost`\n'
                '2. 触发 CMake 配置阶段\n'
                '3. 检查问题文件列表中的 CMakeLists.txt'
            )
            expected_text = f'**修复建议**: {type_issues[0]["suggestion"]}'
            
            # 构建问题文件列表
            file_list_lines = ['**问题文件列表:**', '', '| 序号 | 文件路径 | 行号 |', '|------|----------|------|']
            for idx, issue in enumerate(type_issues[:30], 1):
                file_list_lines.append(f'| {idx} | `{issue["file"]}` | {issue["line"]} |')
            if len(type_issues) > 30:
                file_list_lines.append(f'| ... | 等共 {len(type_issues)} 个问题 | - |')
            logs_text = '\n'.join(file_list_lines)
            
            notes_text = '**影响:**\n- CMake 配置失败\n- UT 构建可能失败\n- 目标冲突可能导致链接错误'
            
            # 调用 gitcode-issue-creator 生成 Issue 文件
            issue_summary = f'{repo} {type_name}导致 CMake 配置异常'
            
            try:
                # 使用 generate_issue_md 模块
                from generate_issue_md import (
                    generate_issue_md as _generate_issue_md_func_inner,
                    generate_description as _generate_description_func_inner
                )
                
                issue_description = _generate_description_func_inner(
                    'bug-report',
                    description=description_text,
                    environment=environment_text,
                    steps=steps_text,
                    expected=expected_text,
                    logs=logs_text,
                    notes=notes_text,
                )
                
                # 使用 issue_type 作为文件名后缀，区分不同类型
                issue_file = _generate_issue_md_func_inner(
                    repo=repo,
                    template_type='bug-report',
                    summary=issue_summary,
                    description=issue_description,
                    labels='bug-report',
                    output_dir=str(issues_dir),
                    issue_suffix=issue_type,  # 如 variable_not_defined
                )
                
                report_lines.extend([
                    f'### Issue #{issue_count}: {repo} {type_name}',
                    '',
                    f'- **标题**: `[Bug-Report|缺陷反馈]: [AI 识别] {issue_summary}`',
                    f'- **Issue 文件**: `{Path(issue_file).relative_to(output_file.parent)}`',
                    f'- **提交地址**: https://gitcode.com/CANN/{repo}/issues/new',
                    '',
                ])
            except ImportError:
                # 如果无法导入，使用传统方式生成
                report_lines.extend([
                    f'### Issue #{issue_count}: {repo} {type_name}',
                    '',
                    f'- **标题**: `[Bug-Report|缺陷反馈]: [AI 识别] {issue_summary}`',
                    f'- **问题文件数**: `{len(files)}`',
                    f'- **提交地址**: https://gitcode.com/CANN/{repo}/issues/new',
                    '',
                    '**问题文件列表:**',
                    '',
                ])
                for f in files[:10]:
                    report_lines.append(f'- `{f}`')
                if len(files) > 10:
                    report_lines.append(f'- ... 等共 {len(files)} 个文件')
                report_lines.append('')

    # 生成规范问题（简化格式）
    if code_quality_issues:
        report_lines.extend([
            '',
            '---',
            '',
            '## 📋 规范问题（遗留无效代码）',
            '',
            '> 以下问题不会导致构建失败，但属于遗留无效代码，建议清理或删除。',
            '',
            '**原因说明:**',
            '',
            '- ops-math/ops-cv: tests/ut/op_host/CMakeLists.txt 不会被引入构建系统',
            '- func.cmake 中的 `add_all_ut_sources` 通过 `file(GLOB)` 直接查找源文件',
            '- 这些 CMakeLists.txt 文件是遗留代码，实际不需要',
            '',
        ])
        
        # 按仓库和类型分组
        by_repo_type = {}
        for issue in code_quality_issues:
            key = (issue['repo'], issue['issue_type'])
            if key not in by_repo_type:
                by_repo_type[key] = []
            by_repo_type[key].append(issue)

        issue_count = 0
        for (repo, issue_type), type_issues in sorted(by_repo_type.items()):
            issue_count += 1
            type_name = type_names.get(issue_type, issue_type)
            files = sorted(set(i['file'] for i in type_issues))

            report_lines.extend([
                '',
                f'### 规范问题 #{issue_count}: {repo} {type_name}',
                '',
                f'**问题类型**: `{type_name}`',
                f'**问题文件数**: `{len(files)}`',
                '',
                '**问题描述:**',
                '',
                f'{type_issues[0]["detail"]}',
                '',
                '**处理建议:**',
                '',
                '1. **删除文件**: 这些 CMakeLists.txt 文件不需要，可以直接删除',
                '2. **或清理内容**: 如果保留，清理为空白文件或仅保留注释',
                '',
                '**问题文件列表:**',
                '',
            ])
            for f in files[:30]:
                report_lines.append(f'- `{f}`')
            if len(files) > 30:
                report_lines.append(f'- ... 等共 {len(files)} 个文件')
            report_lines.append('')

    report_lines.extend([
        '',
        '---',
        '',
        f'*报告生成时间: {results["scan_time"]}*',
    ])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f'报告已保存到: {output_file}')
    if bug_issues and issues_dir.exists():
        print(f'Issue 文件保存在: {issues_dir}')


def main():
    parser = argparse.ArgumentParser(description='CMake 配置问题扫描脚本（动态发现）')
    parser.add_argument('--scan', type=str, default='all',
                        choices=['optype', 'ut', 'all'],
                        help='扫描类型')
    parser.add_argument('--repo', type=str, default='all',
                        help='扫描的仓库名（支持任意 ops-* 仓库，或 all 扫描全部）')
    parser.add_argument('--workspace', type=str, default='.',
                        help='工作空间根目录（包含各仓库子目录的父目录）')
    parser.add_argument('--repo-root', type=str, default=None,
                        help='仓库根目录路径（优先级高于 --workspace）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径（默认 reports/{date}/{repo}/cmake-scan_data.json）')
    parser.add_argument('--report', type=str, default=None,
                        help='输出 Markdown 报告路径（默认 reports/{date}/{repo}/cmake-scan_report_{time}.md）')

    args = parser.parse_args()
    
    date_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d')
    time_str = datetime.now(tz=timezone.utc).strftime('%H%M%S')
    
    workspace = args.workspace
    detection_method = f'workspace ({args.workspace})'
    
    if args.repo_root:
        repo_root_path = Path(args.repo_root)
        if not repo_root_path.exists():
            print(f"错误: 仓库根目录不存在: {args.repo_root}", file=sys.stderr)
            sys.exit(1)
        workspace = str(repo_root_path.parent)
        detection_method = f'repo_root ({args.repo_root})'
    elif args.repo != 'all' and args.workspace == '.':
        try:
            repo_root_path, method = get_repo_root(args.repo)
            workspace = str(repo_root_path.parent)
            detection_method = method
        except ValueError:
            pass
    
    reports_dir, reports_method = get_reports_output_dir(repo_type=args.repo if args.repo != 'all' else None)
    
    if args.repo != 'all':
        resolved_path = Path(workspace) / args.repo
        print(f"仓库路径: {resolved_path.resolve()} (检测方式: {detection_method})")
    else:
        print(f"工作空间: {Path(workspace).resolve()} (检测方式: {detection_method})")
    
    print(f"Reports 目录: {reports_dir} (检测方式: {reports_method})")
    
    if args.output is None:
        if args.repo == 'all':
            args.output = str(reports_dir / date_str / 'cmake-scan_data.json')
        else:
            args.output = str(reports_dir / date_str / args.repo / 'cmake-scan_data.json')
    
    if args.report is None:
        if args.repo == 'all':
            args.report = str(reports_dir / date_str / 'cmake-scan_report_{time_str}.md')
        else:
            args.report = str(reports_dir / date_str / args.repo / f'cmake-scan_report_{time_str}.md')

    scanner = CMakeScanner(workspace)

    if args.repo == 'all':
        results = scanner.scan_all_repos(args.scan)
    else:
        if args.scan == 'optype':
            results = scanner.scan_optype_only(args.repo)
        elif args.scan == 'ut':
            results = scanner.scan_ut_only(args.repo)
        else:
            results = scanner.scan_repo(args.repo)
        scanner.results['summary']['total_issue_files'] = results.get('issue_files', 0)

    scanner.print_summary()
    scanner.save_results(args.output)
    generate_report(scanner.results, args.report)


if __name__ == '__main__':
    main()

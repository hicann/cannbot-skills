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
CMake 行为推断模块

功能：
1. 分析仓库的 CMakeLists.txt 推断正确的函数名
2. 推断正确的参数名（HOSTNAME vs UT_NAME）
3. 判断 tests CMake 是否会被引入构建系统
4. 推断问题严重程度（bug vs code_quality）

推断方法：
1. 扫描仓库中的 CMakeLists.txt 文件
2. 统计函数和参数使用频率
3. 分析构建系统结构
4. 基于统计结果推断正确配置

用法：
    from cmake_profiler import CMakeProfiler
    
    profile = CMakeProfiler.profile_repo(repo_root)
    print(f"正确函数: {profile['correct_function']}")
    print(f"正确参数: {profile['correct_param']}")
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CMakeProfiler:
    """CMake 行为推断器"""
    
    # 需要检测的 CMake 函数
    FUNCTION_PATTERNS = [
        r'add_modules_ut_sources\s*\(',
        r'add_modules_llt_sources\s*\(',
        r'add_opapi_ut_sources\s*\(',
    ]
    
    # 需要检测的参数名
    PARAM_PATTERNS = [
        r'UT_NAME\s*',
        r'HOSTNAME\s*',
    ]
    
    # 标准 CMake 目录
    CMAKE_DIRS = [
        'op_host/CMakeLists.txt',
        'tests/ut/op_host/CMakeLists.txt',
        'CMakeLists.txt',
    ]
    
    @classmethod
    def profile_repo(cls, repo_root: Path) -> Dict:
        """
        分析仓库 CMake 行为
        
        Returns:
            Dict: {
                'correct_function': 正确的函数名,
                'incorrect_functions': 错误的函数名列表,
                'correct_param': 正确的参数名,
                'incorrect_params': 错误的参数名列表,
                'tests_cmake_required': tests CMake 是否必需,
                'defines_llt_sources': 是否定义 add_modules_llt_sources,
                'severity': 问题严重程度,
                'confidence': 推断置信度,
            }
        """
        if not repo_root.exists():
            return cls._get_default_profile()
        
        # 统计函数和参数使用频率
        function_counts = {}
        param_counts = {}
        
        # 扫描所有 CMakeLists.txt 文件
        cmake_files = list(repo_root.rglob('CMakeLists.txt'))
        
        for cmake_file in cmake_files:
            try:
                content = cmake_file.read_text(encoding='utf-8')
                
                # 统计函数使用
                for func_pattern in cls.FUNCTION_PATTERNS:
                    matches = re.findall(func_pattern, content)
                    func_name = re.search(r'(\w+)\s*\(', func_pattern).group(1)
                    function_counts[func_name] = function_counts.get(func_name, 0) + len(matches)
                
                # 统计参数使用
                for param_pattern in cls.PARAM_PATTERNS:
                    matches = re.findall(param_pattern, content)
                    param_name = re.search(r'(\w+)', param_pattern).group(1)
                    param_counts[param_name] = param_counts.get(param_name, 0) + len(matches)
                    
            except Exception as e:
                logger.debug("Failed to read cmake file: %s", e)
                continue
        
        # 推断正确配置
        correct_function = cls._infer_correct_function(function_counts)
        correct_param = cls._infer_correct_param(param_counts)
        tests_required = cls._infer_tests_required(repo_root)
        defines_llt = cls._infer_defines_llt_sources(repo_root)
        severity = cls._infer_severity(tests_required)
        
        # 计算置信度
        confidence = cls._calculate_confidence(function_counts, param_counts)
        
        # 推断错误配置（相反的）
        incorrect_functions = []
        if correct_function == 'add_modules_ut_sources':
            if function_counts.get('add_modules_llt_sources', 0) > 0:
                incorrect_functions.append('add_modules_llt_sources')
        
        incorrect_params = []
        if correct_param == 'UT_NAME':
            if param_counts.get('HOSTNAME', 0) > 0:
                incorrect_params.append('HOSTNAME')
        elif correct_param == 'HOSTNAME':
            if param_counts.get('UT_NAME', 0) > 0:
                incorrect_params.append('UT_NAME')
        
        return {
            'correct_function': correct_function,
            'incorrect_functions': incorrect_functions,
            'correct_param': correct_param,
            'incorrect_params': incorrect_params,
            'tests_cmake_required': tests_required,
            'defines_llt_sources': defines_llt,
            'severity': severity,
            'confidence': confidence,
            'function_counts': function_counts,
            'param_counts': param_counts,
        }
    
    @classmethod
    def get_cmake_issues(cls, repo_root: Path) -> List[Dict]:
        """
        扫描 CMake 问题
        
        Returns:
            List[Dict]: 问题列表
        """
        profile = cls.profile_repo(repo_root)
        issues = []
        
        # 扫描所有 tests/ut/op_host/CMakeLists.txt
        ut_cmake_files = list(repo_root.rglob('tests/ut/op_host/CMakeLists.txt'))
        
        for cmake_file in ut_cmake_files:
            try:
                content = cmake_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                # 检查错误函数
                for incorrect_func in profile['incorrect_functions']:
                    for i, line in enumerate(lines, 1):
                        if incorrect_func in line:
                            issues.append({
                                'file': str(cmake_file.relative_to(repo_root)),
                                'line': i,
                                'issue_type': 'function_not_defined',
                                'detail': f'使用不存在的函数 {incorrect_func}',
                                'severity': profile['severity'],
                            })
                
                # 检查错误参数
                for incorrect_param in profile['incorrect_params']:
                    for i, line in enumerate(lines, 1):
                        if incorrect_param in line:
                            issues.append({
                                'file': str(cmake_file.relative_to(repo_root)),
                                'line': i,
                                'issue_type': 'param_name_error',
                                'detail': f'参数名错误 {incorrect_param}',
                                'severity': profile['severity'],
                            })
                            
            except Exception as e:
                logger.debug("Failed to get cmake issues: %s", e)
                continue
        
        return issues
    
    @classmethod
    def _infer_correct_function(cls, counts: Dict) -> str:
        """推断正确的函数名（使用频率最高的）"""
        ut_count = counts.get('add_modules_ut_sources', 0)
        llt_count = counts.get('add_modules_llt_sources', 0)
        
        if llt_count > ut_count:
            return 'add_modules_llt_sources'
        return 'add_modules_ut_sources'
    
    @classmethod
    def _infer_correct_param(cls, counts: Dict) -> str:
        """推断正确的参数名（使用频率最高的）"""
        ut_count = counts.get('UT_NAME', 0)
        hostname_count = counts.get('HOSTNAME', 0)
        
        if hostname_count > ut_count:
            return 'HOSTNAME'
        return 'UT_NAME'
    
    @classmethod
    def _infer_tests_required(cls, repo_root: Path) -> bool:
        """推断 tests/ut/op_host/CMakeLists.txt 是否会被引入构建"""
        
        top_cmake = repo_root / 'CMakeLists.txt'
        if top_cmake.exists():
            try:
                content = top_cmake.read_text(encoding='utf-8')
                
                if re.search(r'add_subdirectory\s*\(\s*\$\{.*tests', content):
                    return True
                if re.search(r'add_subdirectory\s*\(\s*tests', content):
                    return True
                
                if re.search(r'add_all_ut_sources', content):
                    return False
                    
            except Exception as e:
                logger.debug("Failed to read top CMakeLists: %s", e)
                pass
        
        cmake_dir = repo_root / 'cmake'
        if cmake_dir.exists():
            for cmake_file in cmake_dir.glob('*.cmake'):
                try:
                    content = cmake_file.read_text(encoding='utf-8')
                    
                    if re.search(r'file\s*\(\s*GLOB', content):
                        return False
                        
                except Exception as e:
                    logger.debug("Failed to read cmake config: %s", e)
                    continue
        
        tests_cmake = repo_root / 'tests' / 'ut' / 'op_host' / 'CMakeLists.txt'
        if tests_cmake.exists():
            return True
        
        return False
    
    @classmethod
    def _infer_defines_llt_sources(cls, repo_root: Path) -> bool:
        """推断仓库是否定义了 add_modules_llt_sources 函数"""
        
        cmake_dir = repo_root / 'cmake'
        if cmake_dir.exists():
            for cmake_file in cmake_dir.glob('*.cmake'):
                try:
                    content = cmake_file.read_text(encoding='utf-8')
                    
                    if re.search(r'function\s*\(\s*add_modules_llt_sources', content):
                        return True
                        
                except Exception as e:
                    logger.debug("Failed to read cmake llt sources: %s", e)
                    continue
        
        return False
    
    @classmethod
    def _infer_severity(cls, tests_required: bool) -> str:
        """推断问题严重程度"""
        if tests_required:
            return 'bug'
        return 'code_quality'
    
    @classmethod
    def _calculate_confidence(cls, func_counts: Dict, param_counts: Dict) -> float:
        """计算推断置信度"""
        total_func = sum(func_counts.values())
        total_param = sum(param_counts.values())
        
        if total_func == 0 and total_param == 0:
            return 0.0
        
        if total_func >= 10 and total_param >= 10:
            return 0.9
        elif total_func >= 5 or total_param >= 5:
            return 0.7
        else:
            return 0.5
    
    @classmethod
    def _get_default_profile(cls) -> Dict:
        """获取默认配置（无数据时）"""
        return {
            'correct_function': 'add_modules_ut_sources',
            'incorrect_functions': ['add_modules_llt_sources'],
            'correct_param': 'UT_NAME',
            'incorrect_params': ['HOSTNAME'],
            'tests_cmake_required': True,
            'defines_llt_sources': False,
            'severity': 'bug',
            'confidence': 0.0,
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CMake 行为推断工具')
    parser.add_argument('--repo', type=str, required=True,
                        help='仓库路径')
    parser.add_argument('--issues', action='store_true',
                        help='显示 CMake 问题')
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo)
    if not repo_root.exists():
        print(f"仓库路径不存在: {repo_root}")
        return 1
    
    profile = CMakeProfiler.profile_repo(repo_root)
    
    print(f"=== {repo_root.name} CMake 行为推断 ===")
    print(f"推断置信度: {profile['confidence']:.1%}")
    print()
    print(f"正确函数: {profile['correct_function']}")
    print(f"错误函数: {profile['incorrect_functions']}")
    print(f"正确参数: {profile['correct_param']}")
    print(f"错误参数: {profile['incorrect_params']}")
    print(f"tests CMake 必需: {profile['tests_cmake_required']}")
    print(f"定义 llt_sources: {profile['defines_llt_sources']}")
    print(f"问题严重程度: {profile['severity']}")
    print()
    print(f"函数使用统计: {profile['function_counts']}")
    print(f"参数使用统计: {profile['param_counts']}")
    
    if args.issues:
        issues = CMakeProfiler.get_cmake_issues(repo_root)
        print()
        print(f"发现 {len(issues)} 个 CMake 问题:")
        for issue in issues[:10]:
            print(f"  [{issue['severity']}] {issue['file']}:{issue['line']}")
            print(f"    {issue['detail']}")
    
    return 0


if __name__ == '__main__':
    exit(main())
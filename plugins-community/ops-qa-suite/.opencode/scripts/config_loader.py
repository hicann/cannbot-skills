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
配置加载器 - 从 repo_config.yaml 读取配置

提供统一的配置访问接口，所有脚本应使用此加载器获取配置值。
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def deep_merge(base: Dict, override: Optional[Dict]) -> Dict:
    """深合并配置字典，override 的值覆盖 base 的值"""
    if override is None:
        return deepcopy(base)
    
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


CONFIG_PATH = Path(__file__).parent.parent / "config" / "repo_config.yaml"


DEFAULT_CONFIG = {
    'repository': {
        'pattern': 'ops-',
        'markers': ['CMakeLists.txt', 'docs', 'op_host', 'op_kernel', 'common', 'cmake'],
        'min_markers_required': 2,
        'excluded_dirs': [
            'tests', 'docs', 'common', 'cmake', 'examples',
            'op_host', 'op_kernel', 'op_api', 'op_graph',
            '.git', '.github', '.opencode', 'build', 'framework',
        ],
    },
    'gitcode': {
        'clone_template': 'https://gitcode.com/cann/{repo}.git',
        'issue_template': 'https://gitcode.com/cann/{repo}/issues/new',
        'api_base': 'https://api.gitcode.com/api/v5',
        'web_base': 'https://gitcode.com/cann',
    },
    'structure': {
        'directories': {
            'host': 'op_host',
            'kernel': 'op_kernel',
            'kernel_aicpu': 'op_kernel_aicpu',
            'api': 'op_api',
            'graph': 'op_graph',
            'docs': 'docs',
        },
        'ut_structure': {
            'host_ut': 'tests/ut/op_host',
            'kernel_ut': 'tests/ut/op_kernel',
            'api_ut': 'tests/ut/op_api',
            'aicpu_ut': 'tests/ut/aicpu_op_kernel',
        },
        'architecture_dirs': ['arch20', 'arch32', 'arch35'],
    },
    'docs': {
        'zh_root': 'docs/zh',
        'repo_docs': {
            'op_list': 'docs/zh/op_list.md',
            'op_api_list': 'docs/zh/op_api_list.md',
            'readme': 'README.md',
        },
    },
}


_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """
    加载配置文件（带缓存），与默认配置深合并
    
    Returns:
        Dict: 配置字典，包含 repository, gitcode, structure, docs 等配置
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        _config_cache = deep_merge(DEFAULT_CONFIG, user_config)
    else:
        _config_cache = deepcopy(DEFAULT_CONFIG)
    
    return _config_cache


def get_repo_pattern() -> str:
    """获取仓库名匹配模式"""
    return load_config()['repository']['pattern']


def get_repo_markers() -> list:
    """获取仓库特征标记列表"""
    return load_config()['repository']['markers']


def get_min_markers() -> int:
    """获取最小特征匹配数"""
    return load_config()['repository']['min_markers_required']


def get_excluded_dirs() -> list:
    """获取排除目录列表"""
    return load_config()['repository']['excluded_dirs']


def get_gitcode_clone_template() -> str:
    """获取 GitCode 克隆 URL 模板"""
    return load_config()['gitcode']['clone_template']


def get_gitcode_issue_template() -> str:
    """获取 GitCode Issue URL 模板"""
    return load_config()['gitcode']['issue_template']


def get_gitcode_api_base() -> str:
    """获取 GitCode API 基础 URL"""
    return load_config()['gitcode']['api_base']


def get_gitcode_web_base() -> str:
    """获取 GitCode Web 基础路径"""
    return load_config()['gitcode']['web_base']


def build_clone_url(repo_name: str) -> str:
    """构建克隆 URL"""
    return get_gitcode_clone_template().format(repo=repo_name)


def build_issue_url(repo_name: str) -> str:
    """构建 Issue URL"""
    return get_gitcode_issue_template().format(repo=repo_name)


def build_web_url(repo_name: str) -> str:
    """构建 Web URL"""
    return f"{get_gitcode_web_base()}/{repo_name}"


def get_structure_dirs() -> Dict[str, str]:
    """获取标准目录结构"""
    return load_config()['structure']['directories']


def get_ut_structure() -> Dict[str, str]:
    """获取 UT 目录结构"""
    return load_config()['structure']['ut_structure']


def get_architecture_dirs() -> list:
    """获取架构子目录列表"""
    return load_config()['structure']['architecture_dirs']


def get_op_list_path() -> str:
    """获取 op_list.md 路径"""
    return load_config()['docs']['repo_docs']['op_list']


def get_op_api_list_path() -> str:
    """获取 op_api_list.md 路径"""
    return load_config()['docs']['repo_docs']['op_api_list']


def reload_config() -> Dict[str, Any]:
    """重新加载配置（清除缓存）"""
    global _config_cache
    _config_cache = None
    return load_config()


if __name__ == '__main__':
    config = load_config()
    print("=== 配置内容 ===")
    print(f"仓库模式: {get_repo_pattern()}")
    print(f"GitCode Clone: {get_gitcode_clone_template()}")
    print(f"GitCode API: {get_gitcode_api_base()}")
    print(f"架构目录: {get_architecture_dirs()}")
    print()
    print(f"示例克隆 URL: {build_clone_url('ops-math')}")
    print(f"示例 Issue URL: {build_issue_url('ops-math')}")
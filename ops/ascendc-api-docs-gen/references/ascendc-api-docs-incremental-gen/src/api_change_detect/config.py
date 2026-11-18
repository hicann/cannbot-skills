# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""API Change Detect 配置。"""

import os
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """可配置的搜索路径。

    所有路径相对于 REPO_ROOT。修改这些值即可适配不同的代码仓库结构。
    """

    # ── 文档 ──
    doc_dir: str = "docs/zh/api"
    # 文档扫描是否递归子目录
    doc_recursive: bool = True
    # 排除的文档文件名模式（非 API 文档）
    doc_exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "README.md",
            "*列表.md",
            "概述.md",
            "通用说明*.md",
            "附录.md",
            "预留接口.md",
            "接口变更说明.md",
            "Ascend-C-API列表.md",
            "Layout和层次化表述法.md",
            "tensor_api_list.md",
        ]
    )

    # ── include 目录（声明文件搜索）──
    include_dirs: list[str] = field(
        default_factory=lambda: [
            "include/basic_api",
            "include/adv_api",
            "include/simt_api",
            "include/tensor_api",
            "include/c_api",
            "include/aicpu_api",
            "include/utils",
        ]
    )

    # ── impl 目录（实现文件搜索）──
    impl_basic: str = "impl/basic_api"
    impl_adv: str = "impl/adv_api"
    impl_simt: str = "impl/simt_api"
    impl_tensor: str = "impl/tensor_api"
    impl_capi: str = "impl/c_api"
    impl_utils: str = "impl/utils"

    # ── 测试目录 ──
    test_dir: str = "tests/api"
    # 按 API 类型映射测试子目录
    test_dir_map: dict = field(
        default_factory=lambda: {
            "basic_api": "tests/api/basic_api",
            "adv_api": "tests/api/adv_api",
            "simt_api": "tests/api/simt_api",
            "tensor_api": "tests/api/tensor_api",
            "c_api": "tests/api/c_api",
            "aicpu_api": "tests/api/aicpu_api",
            "utils": "tests/api/utils",
            # #17 修复：补充缺失的测试目录
            "common": "tests/api/common",
            "reg_compute_api": "tests/api/reg_compute_api",
        }
    )

    # ── 示例目录 ──
    examples_dir: str = "examples"

    # ── 文档扫描的文件扩展名 ──
    doc_extensions: list[str] = field(default_factory=lambda: [".md"])

    # ── 测试文件扩展名 ──
    test_extensions: list[str] = field(default_factory=lambda: [".cpp", ".cc", ".py"])

    # ── 变更检测关注的目录前缀 ──
    changed_file_prefixes: list[str] = field(
        default_factory=lambda: ["include/", "impl/", "tests/", "examples/", "docs/"]
    )

    # ── API 变更判定：只有这些目录变更才算 API 受影响 ──
    # docs/ 和 examples/ 的变更不计入 API 变更
    api_change_prefixes: list[str] = field(default_factory=lambda: ["include/", "impl/", "tests/", "examples/"])

    # ── 变更检测排除的后缀 ──
    changed_file_exclude_suffixes: list[str] = field(
        default_factory=lambda: ["_doc.json", ".pyc", ".DS_Store", ".json"]
    )

    # ── 目录遍历时排除的目录名 ──
    walk_exclude_dirs: list[str] = field(default_factory=lambda: [".git", ".svn", "__pycache__"])

    # ── 目录遍历时排除的后缀 ──
    walk_exclude_dir_suffixes: list[str] = field(default_factory=lambda: ["_重组"])


# ── 默认配置 ──
# 代码仓库根目录
# 代码仓库根目录：默认取环境变量，未设置时用本仓内的 examples 目录占位
REPO_ROOT = os.environ.get("ASCDEVKIT_REPO_ROOT", "")

# 默认路径配置（适配 asc-devkit 仓库结构）
DEFAULT_PATH_CONFIG = PathConfig()

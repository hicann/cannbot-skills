# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch3 实现原理 — 严格按四维表提取。

四维表映射:
- adv_api: 文件(impl)✅ 函数(被调基础API名)✅ 代码段(调用行)✅ 注释段❌
- 其他 6 种: 全部无
"""

import os

from config import BASE_API_LIST, CH_IMPL_PRINCIPLE
from evidence import Evidence

CHAPTER_NAME = CH_IMPL_PRINCIPLE


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取。仅 adv_api 有内容。"""
    if api_type != "adv_api":
        return []
    evidence = []
    evidence.extend(_file_dimension(api_name, api_type, impl_files))
    evidence.extend(_function_dimension(api_name, impl_files, export_root))
    evidence.extend(_code_dimension(api_name, impl_files, extractor))
    # 注释段：无
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(api_name, api_type, impl_files):
    """文件维度：包含被调 API 的 impl 文件。"""
    evidence = []
    for impl in impl_files:
        evidence.append(
            Evidence(
                dimension="file",
                file=impl,
                line=1,
                content=impl,
                metadata={"type": "impl_file"},
            )
        )
    return evidence


def _function_dimension(api_name, impl_files, base_path):
    """函数维度：被调用的基础 API 名。从调用行提取。"""
    evidence = []
    seen = set()
    import re

    for impl in impl_files:
        full_path = os.path.join(base_path, impl)
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            text = content.decode("utf-8", errors="replace")
            for base_api in BASE_API_LIST:
                if base_api == api_name:
                    continue
                pattern = re.compile(rf"\b{re.escape(base_api)}\s*[<(]")
                if pattern.search(text) and base_api not in seen:
                    seen.add(base_api)
                    # 找第一次出现的位置
                    m = pattern.search(text)
                    if m:
                        line_num = text[: m.start()].count("\n") + 1
                        evidence.append(
                            Evidence(
                                dimension="function",
                                file=impl,
                                line=line_num,
                                content=f"{base_api} (位于 {impl})",
                                metadata={"type": "impl_function", "name": base_api},
                            )
                        )
        except (FileNotFoundError, PermissionError):
            print("file error!")
            continue
    return evidence


def _code_dimension(api_name, impl_files, e):
    """代码段维度：impl 内部调用行。"""
    calls = e.extract_impl_internal_calls(api_name, impl_files)
    return calls

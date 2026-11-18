#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Debug incremental_update_index 流程。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_PATH_CONFIG

from api_change_detect import (
    get_commit_id,
    incremental_update_index,
    load_index,
)

# 路径按脚本位置推导，可用环境变量覆盖（不写死本机绝对路径）
_DATA_DIR = os.environ.get("ASCDEVKIT_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OLD_DIR = os.path.join(_DATA_DIR, "asc-devkit-master")
NEW_DIR = os.path.join(_DATA_DIR, "asc-devkit-changed")
INDEX_PATH = os.path.join(OLD_DIR, "api_reverse_index.json")

# 加载旧索引
print("加载旧索引...")
old_index_raw = load_index(INDEX_PATH)
saved_fingerprint = old_index_raw.pop("__fingerprint__", None)

# 重建 old_api_file_map
old_api_file_map = {}
for f, apis in old_index_raw.items():
    for api in apis:
        old_api_file_map.setdefault(api, set()).add(f)

# 检查 Abs 在 old_api_file_map 中的文档路径
print("\n检查 Abs 在 old_api_file_map 中的文档:")
abs_docs = [f for f in old_api_file_map["Abs"] if f.startswith("docs/") and f.endswith(".md")]
print("  文档数量:", len(abs_docs))
for doc in abs_docs:
    print("  ", doc)

# 执行增量更新
print("\n执行增量更新...")
new_index, new_api_map, summary = incremental_update_index(
    OLD_DIR, NEW_DIR, old_index_raw, old_api_file_map, DEFAULT_PATH_CONFIG
)

# 检查 Abs 在 new_api_map 中的文档路径
print("\n检查 Abs 在 new_api_map 中的文档:")
if "Abs" in new_api_map:
    abs_docs_new = [f for f in new_api_map["Abs"] if f.startswith("docs/") and f.endswith(".md")]
    print("  文档数量:", len(abs_docs_new))
    for doc in abs_docs_new:
        print("  ", doc)

    # 对比差异
    old_docs_set = set(abs_docs)
    new_docs_set = set(abs_docs_new)
    missing = old_docs_set - new_docs_set
    if missing:
        print("\n  ❌ 丢失的文档:")
        for doc in missing:
            print("  ", doc)
else:
    print("  ❌ Abs 不在 new_api_map 中！")

# 检查索引条目数
print("\n索引条目数对比:")
print("  old:", len(old_index_raw))
print("  new:", len(new_index) - 1)  # 减去fingerprint


# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""pytest 配置：runner 目录下的测试。"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: 标记耗时测试（LLM 调用），默认跳过",
    )
    config.addinivalue_line(
        "markers",
        "dispatch: 标记 subagent dispatch 测试（需要 LLM）",
    )


def pytest_collection_modifyitems(config, items):
    """默认跳过 slow/dispatch 标记的测试，除非显式 -k 选择或 --runslow。"""
    if config.getoption("-k") or config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --runslow 或 -k 显式选择")
    for item in items:
        if "slow" in item.keywords or "dispatch" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="运行 slow 标记的测试",
    )

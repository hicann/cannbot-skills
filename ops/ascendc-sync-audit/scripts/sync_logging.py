#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""sync_audit/ascendc_flow_analyzer/case_retriever 共用的 logging 初始化。

info 级输出到 stdout（保持 CLI 输出契约），warning/error 输出到 stderr。
"""

import logging
import sys


def stream_handler(stream, level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(message)s'))
    return handler


def init_logging(logger: logging.Logger, stderr_logger: logging.Logger) -> None:
    """配置 info→stdout、warning+→stderr 的双通道日志。"""
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(stream_handler(sys.stdout, logging.INFO))
    stderr_logger.setLevel(logging.WARNING)
    stderr_logger.propagate = False
    stderr_logger.addHandler(stream_handler(sys.stderr, logging.WARNING))
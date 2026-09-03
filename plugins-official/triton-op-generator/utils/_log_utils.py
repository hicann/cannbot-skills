#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""logger 公共配置：INFO/DEBUG → stdout，WARNING+ → stderr。

供 utils/ 下的批量脚本复用，避免各自重复定义一份 _setup_logger。
"""
import logging
import sys

# (输出流, 该 handler 的最低级别, 上限级别——None 表示不设上限)
_HANDLER_SPECS = (
    (sys.stdout, logging.DEBUG, logging.WARNING),
    (sys.stderr, logging.WARNING, None),
)


def setup_logger(logger: logging.Logger, level: int = logging.INFO) -> logging.Logger:
    """按上述分流规则配置 logger；已配置过则原样返回，可重复调用。"""
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")

    for stream, low, high in _HANDLER_SPECS:
        handler = logging.StreamHandler(stream)
        handler.setLevel(low)
        if high is not None:
            handler.addFilter(lambda record, limit=high: record.levelno < limit)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

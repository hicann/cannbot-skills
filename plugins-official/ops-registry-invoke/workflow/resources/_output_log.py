# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared logging helper for ops-registry-invoke workflow CLI scripts.

Provides :func:`get_logger`, which returns a ``logging.Logger`` that emits bare
messages (no level/time prefix) so byte output is identical to the previous
``sys.stdout.write`` / ``sys.stderr.write`` calls. INFO records go to stdout,
WARNING-and-above go to stderr. Streams are resolved dynamically on every emit
so ``contextlib.redirect_stdout`` / ``redirect_stderr`` in tests are honored.
"""

from __future__ import annotations

import logging
import sys


class _DynamicStreamHandler(logging.Handler):
    # ==== construction ===========================================================================

    def __init__(self, get_stream, low: int, high: int) -> None:
        super().__init__()
        self._get_stream = get_stream
        self._low = low
        self._high = high
        self.setFormatter(logging.Formatter("%(message)s"))

    # ==== public API =============================================================================

    def emit(self, record: logging.LogRecord) -> None:
        if self._low <= record.levelno <= self._high:
            self._get_stream().write(self.format(record) + "\n")


def get_logger(name: str) -> logging.Logger:
    """Return a configured, idempotent logger that mirrors bare stream writes."""
    logger = logging.getLogger(name)
    if not getattr(logger, "_output_log_configured", False):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(_DynamicStreamHandler(lambda: sys.stdout, logging.INFO, logging.INFO))
        logger.addHandler(_DynamicStreamHandler(lambda: sys.stderr, logging.WARNING, logging.CRITICAL))
        setattr(logger, "_output_log_configured", True)
    return logger

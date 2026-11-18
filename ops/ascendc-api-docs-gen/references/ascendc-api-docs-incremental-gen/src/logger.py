# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logging configuration for the project.

日志落盘策略集中在这里：每个 logger 一个按大小滚动的文件 handler，可选再挂一份
控制台输出。所有 handler 的级别/格式装配都走 _attach_handler()，避免各处重复
「setLevel + setFormatter + addHandler」三连。
"""

import builtins
import logging
import os
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

_log_dir_env = os.environ.get("CANN_PLUGIN_LOG_DIR")
if not _log_dir_env:
    _log_dir_env = str(Path(__file__).resolve().parent.parent / "data" / "logs")
LOG_DIR = Path(_log_dir_env)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_loggers = {}

# 落盘参数与日志格式：集中成常量，handler 装配与格式定义不再分散在函数体里
LOG_LINE_TEMPLATE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_TIME_TEMPLATE = "%Y-%m-%d %H:%M:%S"
FILE_MAX_BYTES = 5 * 1024 * 1024
FILE_BACKUP_COUNT = 3


def build_formatter() -> logging.Formatter:
    """构造本项目统一的日志格式化器。"""
    return logging.Formatter(fmt=LOG_LINE_TEMPLATE, datefmt=LOG_TIME_TEMPLATE)


def attach_handler(logger: logging.Logger, handler: logging.Handler, level: int) -> None:
    """把 handler 按指定级别与统一格式挂到 logger 上（唯一的装配入口）。"""
    handler.setLevel(level)
    handler.setFormatter(build_formatter())
    logger.addHandler(handler)


def make_file_handler(log_name: str) -> RotatingFileHandler:
    """按日志名构造滚动文件 handler（目录来自 LOG_DIR）。"""
    return RotatingFileHandler(
        filename=str(LOG_DIR / f"{log_name}.log"),
        maxBytes=FILE_MAX_BYTES,
        backupCount=FILE_BACKUP_COUNT,
        encoding="utf-8",
    )


def _prepare_logger(name: str, level: int) -> logging.Logger:
    """取出（或创建）logger 并设置级别与传播策略。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def setup_logger(
    name: str = "vCLI",
    level: int = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """Setup and return a logger with optional console output and file output."""
    cached = _loggers.get(name)
    if cached is not None:
        return cached

    logger = _prepare_logger(name, level)
    if console:
        attach_handler(logger, logging.StreamHandler(sys.stdout), level)
    attach_handler(logger, make_file_handler(name), logging.DEBUG)

    _loggers[name] = logger
    return logger


def setup_namespace_logger(prefix: str) -> logging.Logger:
    """为命名空间设置层级日志器，该命名空间下所有 getLogger(__name__) 的日志自动输出到文件。"""
    cached = _loggers.get(prefix)
    if cached is not None:
        return cached

    logger = logging.getLogger(prefix)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    attach_handler(logger, make_file_handler(prefix), logging.DEBUG)

    _loggers[prefix] = logger
    return logger


def get_logger(name: str = "vCLI") -> logging.Logger:
    """Get or create a logger with the given name."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def _emit(level_method: str, msg: str, logger_name: str) -> None:
    """按级别名把消息交给对应 logger 的方法（info/warning/... 共用）。"""
    getattr(get_logger(logger_name), level_method)(msg)


def info(msg: str, logger_name: str = "vCLI"):
    """Log info message."""
    _emit("info", msg, logger_name)


def debug(msg: str, logger_name: str = "vCLI"):
    """Log debug message."""
    _emit("debug", msg, logger_name)


def warning(msg: str, logger_name: str = "vCLI"):
    """Log warning message."""
    _emit("warning", msg, logger_name)


def error(msg: str, logger_name: str = "vCLI"):
    """Log error message."""
    _emit("error", msg, logger_name)


def critical(msg: str, logger_name: str = "vCLI"):
    """Log critical message."""
    _emit("critical", msg, logger_name)


def _framed(title: str, mark: str, count: int, logger_name: str) -> None:
    """用重复的 mark 把标题夹在中间输出（section / subsection 共用）。"""
    logger = get_logger(logger_name)
    line = mark * count
    for part in (line, title, line):
        logger.info(part)


def section(title: str, logger_name: str = "vCLI"):
    """Log a section divider with title."""
    _framed(title, "=", 60, logger_name)


def subsection(title: str, logger_name: str = "vCLI"):
    """Log a subsection title."""
    _framed(title, "-", 20, logger_name)


def highlight(msg: str, logger_name: str = "vCLI"):
    """Log highlighted message."""
    _emit("info", msg, logger_name)


def log_result(msg: str, logger_name: str = "vCLI"):
    """Log result message."""
    _emit("info", msg, logger_name)


def separator(logger_name: str = "vCLI"):
    """Log a separator line."""
    get_logger(logger_name).info("-" * 60)


@contextmanager
def redirect_stdout_to_log(logger_name: str = "vCLI"):
    """Context manager that redirects builtins.print to the logger.

    Only bare print() calls are affected. Rich's console.print() writes
    directly to its own file handle and is not redirected.
    """
    original_print = builtins.print

    def _print_to_log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if msg.strip():
            get_logger(logger_name).info(msg)

    builtins.print = _print_to_log
    try:
        yield
    finally:
        builtins.print = original_print

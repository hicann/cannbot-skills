#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Backward-compatible entry point for pypto-op-lint."""

import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pypto_op_lint as _pkg  # noqa: E402
from pypto_op_lint import observability as _observability  # noqa: E402

LOGS_DIR = _observability.LOGS_DIR
LOGS_EVENTS_FILE = _observability.LOGS_EVENTS_FILE


def _sync_runtime_state() -> None:
    global LOGS_DIR, LOGS_EVENTS_FILE
    _observability.LOGS_DIR = LOGS_DIR
    _observability.LOGS_EVENTS_FILE = LOGS_EVENTS_FILE


def __getattr__(name: str) -> Any:
    if hasattr(_pkg, name):
        return getattr(_pkg, name)
    raise AttributeError(name)


def _wrap(name: str):
    target = getattr(_pkg, name)

    def wrapped(*args, **kwargs):
        _sync_runtime_state()
        return target(*args, **kwargs)

    wrapped.__name__ = getattr(target, "__name__", name)
    wrapped.__doc__ = getattr(target, "__doc__", None)
    return wrapped


_build_context = _wrap("_build_context")
_run_checks = _wrap("_run_checks")
hook_post_edit = _wrap("hook_post_edit")
hook_post_bash = _wrap("hook_post_bash")
hook_pre_edit_backup = _wrap("hook_pre_edit_backup")
hook_stop = _wrap("hook_stop")
_parse_front_matter = _wrap("_parse_front_matter")
_validate_doc_schema = _wrap("_validate_doc_schema")
parse_front_matter = _wrap("parse_front_matter")
validate_doc_schema = _wrap("validate_doc_schema")
main = _wrap("main")

if __name__ == "__main__":
    raise SystemExit(main())

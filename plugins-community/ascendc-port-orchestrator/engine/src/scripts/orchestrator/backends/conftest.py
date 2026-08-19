# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Configure backend-test isolation with fake OpenCode and Claude binaries.

Tests exercise command building, not the live runtime. The G5 runtime self-check probes
the real OpenCode binary and behavioural safety net, so it is skipped here through the
documented escape hatch; that self-check is unit-tested in tests/ut/test_opencode_runtime.py.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_runtime_check(monkeypatch):
    monkeypatch.setenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", "1")

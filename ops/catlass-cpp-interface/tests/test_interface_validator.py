# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "catlass-cpp-interface/scripts/validate_interface.py"
SPEC = importlib.util.spec_from_file_location("interface_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validator)


def test_completed_interface_passes_with_generic_sections() -> None:
    text = """# Example API

## 接口签名

```cpp
Tensor example(Tensor input, bool enabled);
```

## 支持范围

输入为浮点 Tensor，支持任意合法 shape。

## 边界行为

空输入返回空输出，非法 dtype 返回参数错误。
"""
    assert validator.validate_interface(text) == []


def test_empty_or_unfinished_interface_fails() -> None:
    assert validator.validate_interface("   ")
    for marker in ("TODO", "TBD", "待填写", "待补充", "待确认"):
        assert validator.validate_interface(f"接口说明\n{marker}：补齐行为")


def test_required_sections_must_exist_and_contain_content() -> None:
    complete = """## 接口签名
接口内容
## 支持范围
范围内容
## 边界行为
边界内容
"""
    section_content = {
        "接口签名": "接口内容",
        "支持范围": "范围内容",
        "边界行为": "边界内容",
    }
    for title, content in section_content.items():
        missing = complete.replace(f"## {title}\n{content}\n", "")
        errors = validator.validate_interface(missing)
        assert any(f"missing required section: {title}" in error for error in errors)

    empty = complete.replace("## 支持范围\n范围内容", "## 支持范围\n")
    errors = validator.validate_interface(empty)
    assert any("section must not be empty: 支持范围" in error for error in errors)


def test_cli_rejects_missing_interface_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "docs/api.md"
        with (
            patch.object(sys, "argv", ["validate_interface.py", "--api", str(path)]),
            redirect_stdout(StringIO()) as output,
        ):
            assert validator.main() == 1
        assert "interface file not found" in output.getvalue()

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

from .helpers import load_lint_module


def test_parse_front_matter_returns_empty_when_missing():
    mod = load_lint_module()
    meta, body = mod.parse_front_matter("# SPEC\n")
    assert not meta
    assert body == "# SPEC\n"


def test_parse_front_matter_extracts_values():
    mod = load_lint_module()
    text = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
---
# SPEC
"""
    meta, body = mod.parse_front_matter(text)
    assert meta["op_name"] == "demo"
    assert meta["supported_dtypes"] == ["bfloat16"]
    assert body.startswith("# SPEC")


def test_validate_doc_schema_requires_fields_by_doc_type():
    mod = load_lint_module()
    spec_errors = mod.validate_doc_schema("SPEC", {"op_name": "x"})
    design_errors = mod.validate_doc_schema("DESIGN", {"op_name": "x"})
    api_errors = mod.validate_doc_schema("API_REPORT", {})
    assert any("supported_dtypes" in err for err in spec_errors)
    assert any("dynamic_axes" in err for err in design_errors)
    assert any("op_name" in err for err in api_errors)

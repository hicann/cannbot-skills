#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression for scalar/mapped synthesis dtype and real CSV validation."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_blackbox_cases import validate_literal_synthesis
from test_skills import fixture, run_script


class SynthesisDtypeTests(unittest.TestCase):
    def check(self, dtype, inputs):
        validate_literal_synthesis(
            "BB-dtype", "extreme_inputs[0]", {"synthesize": {"dtype": dtype}}, inputs
        )

    def test_scalar_applies_to_all_tensors_not_attributes(self):
        values = {
            "q": {"shape": [2], "dtype": "float16"},
            "k": {"shape": [2], "dtype": "float16"},
            "scale": 0.5,
            "mask": None,
        }
        self.check("float16", values)
        values["k"]["dtype"] = "float32"
        with self.assertRaisesRegex(ValueError, "dtype for k"):
            self.check("float16", values)

    def test_mapping_allows_mixed_dtypes_and_partial_constraints(self):
        values = {
            "q": {"shape": [2], "dtype": "float16"},
            "mask": {"shape": [2], "dtype": "float32"},
        }
        self.check({"q": "float16", "mask": "float32"}, values)
        self.check({"q": "float16"}, values)
        self.check(None, values)
        self.check({}, values)
        for dtype in ({"missing": "float16"}, {"mask": "float16"}):
            with self.assertRaises(ValueError):
                self.check(dtype, values)

    def test_tensor_list_members(self):
        values = {"xs": {"tensors": [{"shape": [1], "dtype": "float16"}]}}
        self.check("float16", values)
        self.check({"xs": "float16"}, values)
        values["xs"]["tensors"].append({"shape": [1], "dtype": "float32"})
        with self.assertRaises(ValueError):
            self.check("float16", values)

    def test_invalid_formats_report_contract_error(self):
        for dtype in ([], ["float16"], 0, False, "", "unknown", {"x": 3}):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(ValueError, "synthesize.dtype"),
            ):
                self.check(dtype, {"x": {"shape": [1], "dtype": "float16"}})

    def test_real_blackbox_cli_accepts_scalar_and_mapping(self):
        import yaml

        for dtype in ("float32", {"x": "float32", "y": "float32"}):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                fixture(work)
                spec_path = work / "design/doc/spec.yaml"
                spec = yaml.safe_load(spec_path.read_text())
                spec["extreme_inputs"][0]["synthesize"] = {"dtype": dtype}
                spec_path.write_text(yaml.safe_dump(spec))
                result = run_script("build_blackbox_cases.py", work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("validated", result.stdout)

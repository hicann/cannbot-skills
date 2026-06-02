#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for the TTK CSV validator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ttk_validate_csv.py"
SPEC = importlib.util.spec_from_file_location("ttk_validate_csv", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TTKValidateCSVTest(unittest.TestCase):
    def test_count_outer_elements_ignores_singleton_tuple_trailing_comma(self) -> None:
        self.assertEqual(1, MODULE.count_outer_elements('("float16",)'))
        self.assertEqual(2, MODULE.count_outer_elements('("float16","float32")'))
        self.assertEqual(2, MODULE.count_outer_elements("((1, 2),(3, 4))"))

    def test_tuple_length_consistency_rejects_single_dtype_for_two_shapes(self) -> None:
        rows = [
            {
                "input_shapes": "((1, 2),(3, 4))",
                "input_dtypes": '("float16",)',
            }
        ]

        self.assertFalse(MODULE.check_tuple_length_consistency(rows, ["input_dtypes"], "input_shapes"))

    def test_validate_returns_false_when_expected_header_is_missing(self) -> None:
        headers = [col for col in MODULE.KERNEL_COLUMNS if col != "precision_tolerances"]
        values = {col: "None" for col in headers}
        values.update(
            {
                "testcase_name": "case1",
                "op_name": "demo",
                "input_shapes": "((1, 2),)",
                "input_dtypes": '("float16",)',
                "output_shapes": "((1, 2),)",
                "output_dtypes": '("float16",)',
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "cases.csv"
            csv_path.write_text(
                ",".join(headers) + "\n" + ",".join(values[col] for col in headers) + "\n",
                encoding="utf-8",
            )

            self.assertFalse(MODULE.validate(csv_path))


if __name__ == "__main__":
    unittest.main()

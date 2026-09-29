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

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/catlass-cpp-reference/scripts"))

from generate_definition import generate
from validate_reference import validate


class ReferenceToolTests(unittest.TestCase):
    def test_generate_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "reference.py"
            template = root / "definition.template.json"
            output = root / "definition.json"
            source.write_text("def reference(x):\n    return x\n", encoding="utf-8")
            template.write_text(
                json.dumps({"name": "linear_attention", "reference": ""}),
                encoding="utf-8",
            )
            generate(source, template, output)
            self.assertEqual([], validate(source, output))

    def test_manual_reference_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "reference.py"
            definition = root / "definition.json"
            source.write_text("return 1\n", encoding="utf-8")
            definition.write_text(
                json.dumps({"reference": "return 2\n"}), encoding="utf-8"
            )
            self.assertTrue(validate(source, definition))


if __name__ == "__main__":
    unittest.main()

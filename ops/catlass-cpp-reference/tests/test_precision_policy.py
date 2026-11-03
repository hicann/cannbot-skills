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
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "catlass-cpp-reference/scripts/validate_precision_policy.py"
SPEC = importlib.util.spec_from_file_location("precision_policy_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validator)


class PrecisionPolicyTests(unittest.TestCase):
    def policy(self) -> dict:
        return json.loads(
            (ROOT / "catlass-cpp-reference/templates/precision-policy.json").read_text(
                encoding="utf-8"
            )
        )

    def write_policy(
        self, policy: dict
    ) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        policy_path = root / "reference/precision-policy.json"
        policy_path.parent.mkdir()
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        return policy_path, temp_dir

    def test_template_policy_passes(self) -> None:
        policy = self.policy()
        policy_path, temp_dir = self.write_policy(policy)
        self.addCleanup(temp_dir.cleanup)
        self.assertEqual([], validator.validate(policy_path))

    def test_missing_ratio_fails(self) -> None:
        policy = self.policy()
        policy.pop("region_matched_ratio")
        policy_path, temp_dir = self.write_policy(policy)
        self.addCleanup(temp_dir.cleanup)
        self.assertTrue(
            any(
                "region_matched_ratio" in error
                for error in validator.validate(policy_path)
            )
        )

    def test_invalid_dtype_threshold_fails(self) -> None:
        policy = self.policy()
        policy["dtype"]["float32"]["rtol"] = -1
        policy_path, temp_dir = self.write_policy(policy)
        self.addCleanup(temp_dir.cleanup)
        self.assertTrue(
            any("float32.rtol" in error for error in validator.validate(policy_path))
        )


if __name__ == "__main__":
    unittest.main()

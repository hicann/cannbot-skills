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
"""Regression tests for S2P0 platform ownership in whitebox prompts."""

from __future__ import annotations

import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
S2P0_PROMPTS = [
    SKILL_DIR / "references" / "prompts" / "S2P0-scout-kernel.md",
    SKILL_DIR / "references" / "prompts" / "S2P0-scout-tiling.md",
    SKILL_DIR / "references" / "prompts" / "S2P0-scout-verify.md",
]


class S2P0PlatformPromptTest(unittest.TestCase):
    def test_s2p0_prompts_delegate_platform_mapping_to_npu_arch(self) -> None:
        for path in S2P0_PROMPTS:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")

                self.assertIn("npu-arch", text)
                self.assertIn("本文件不维护", text)
                self.assertIn("platform.npu_arch_macro", text)
                self.assertIn("platform.arch_dir", text)
                self.assertNotIn("## 平台映射", text)
                self.assertNotIn("| NpuArch | __NPU_ARCH__ | Regbase 支持 | 代表芯片 |", text)
                self.assertNotIn("DAV_3510", text)
                self.assertNotIn("DAV_2201", text)
                self.assertNotIn("DAV_2002", text)
                self.assertNotIn("Ascend950", text)
                self.assertNotIn("Ascend910B", text)
                self.assertNotIn("Ascend310P", text)
                self.assertNotIn("arch35", text)

    def test_s2p0_prompts_define_structured_failure_modes(self) -> None:
        scout_t = (SKILL_DIR / "references" / "prompts" / "S2P0-scout-tiling.md").read_text(encoding="utf-8")
        scout_k = (SKILL_DIR / "references" / "prompts" / "S2P0-scout-kernel.md").read_text(encoding="utf-8")
        scout_verify = (SKILL_DIR / "references" / "prompts" / "S2P0-scout-verify.md").read_text(encoding="utf-8")

        self.assertIn("status: file_not_found", scout_t)
        self.assertIn("searched_paths", scout_t)
        self.assertIn("status: file_not_found", scout_k)
        self.assertIn("dispatch_mode: none", scout_k)
        self.assertIn("key_count: 0", scout_k)
        self.assertIn("verification.status", scout_verify)
        self.assertIn("partial", scout_verify)
        self.assertIn("request_incremental_read", scout_verify)


if __name__ == "__main__":
    unittest.main()

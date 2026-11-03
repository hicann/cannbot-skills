# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression coverage for overlapping migrated Operator family prefixes."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


script = Path(__file__).resolve().parents[1] / "scripts/record_knowledge.py"
spec = importlib.util.spec_from_file_location("catlass_knowledge", script)
knowledge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knowledge)


class FamilyLookupTest(unittest.TestCase):
    def test_forward_query_excludes_backward_and_keeps_explicit_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = root / knowledge.ATTENTION
            attention.mkdir(parents=True)
            for name, status in (
                ("flash_attention_forward.md", "stable"),
                ("flash_attention_grad_backward.md", "stable"),
                ("sparse_flash_mla_interface.md", "draft"),
            ):
                (attention / name).write_text(
                    f"---\ntitle: {name}\ndescription: fixture\nstatus: {status}\nplatforms: [a2]\n---\n# Fixture\n",
                    encoding="utf-8",
                )

            def query(family, arch="a2"):
                return knowledge.query(
                    root, "operators", [], family, arch, None, True, False
                )["results"]

            self.assertEqual(
                [item["title"] for item in query("flash_attention")],
                ["flash_attention_forward.md"],
            )
            self.assertEqual(
                [item["title"] for item in query("flash_attention_grad")],
                ["flash_attention_grad_backward.md"],
            )
            self.assertEqual(query("sparse_flash_mla")[0]["status"], "draft")
            self.assertEqual(query("sparse_flash_mla", "ascend950"), [])
            self.assertEqual(len(query(None)), 2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for local-source links created by sync_devkit.py."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SYNC_DEVKIT = ROOT / "ops/pypto-docs-search/scripts/sync_devkit.py"


class SyncDevkitPathTests(unittest.TestCase):
    def test_local_sources_resolve_with_relative_and_absolute_paths(self):
        for relative in (True, False):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                sources = {
                    "docs": workdir / "src/docs/zh",
                    "ops": workdir / "gym/src/pypto_gym/ops",
                    "tests": workdir / "gym/tests/ops",
                }
                for source in sources.values():
                    source.mkdir(parents=True)
                (sources["docs"] / "api").mkdir()

                cache = (
                    Path(".cache/pypto-devkit")
                    if relative
                    else workdir / ".cache/pypto-devkit"
                )
                pypto_src = Path("src") if relative else workdir / "src"
                gym_src = Path("gym") if relative else workdir / "gym"
                env = os.environ.copy()
                env.update(
                    {
                        "PYPTO_DEVKIT_DIR": str(cache),
                        "PYPTO_SRC": str(pypto_src),
                        "PYPTO_GYM_SRC": str(gym_src),
                    }
                )

                result = subprocess.run(
                    [sys.executable, str(SYNC_DEVKIT)],
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifest = json.loads((workdir / cache / "MANIFEST.json").read_text())
                for name, source in sources.items():
                    link = workdir / cache / name
                    self.assertTrue(link.is_symlink(), name)
                    self.assertTrue(
                        link.exists(), f"{name} link is broken: {os.readlink(link)}"
                    )
                    self.assertEqual(link.resolve(), source.resolve())
                    self.assertEqual(manifest[name]["mode"], "symlink")


if __name__ == "__main__":
    unittest.main()

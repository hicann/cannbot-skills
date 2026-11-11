#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the public GE memory tuning helper."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mem_tuning.sh"


def write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class MemoryTuningInputPathTest(unittest.TestCase):
    def test_dump_path_is_passed_as_data(self):
        with tempfile.TemporaryDirectory(prefix="mem-tuning-path-") as temporary:
            root = Path(temporary)
            imas_dir = root / "imas"
            log_dir = root / "logs"
            imas_dir.mkdir()
            log_dir.mkdir()

            marker = root / "injected"
            dump = root / "dump';:>injected;#.txt"
            dump.write_text("controlled dump input\n", encoding="utf-8")
            runner_marker = root / "runner-called"
            runner = root / "fake-ut"
            write_executable(
                runner,
                "#!/bin/sh\n"
                'printf "%s\\n" "$GE_MT_DUMP_FILE" > "$UT_MARKER"\n'
                'printf "%s\\n" "topo_mode[BFS]"\n'
                'exit 0\n',
            )
            write_executable(
                imas_dir / "imas",
                "#!/bin/sh\n"
                "printf '%s\\n' "
                "'reuse_rate:100 after_reuse:10 theory_min:8 no_reuse:0 "
                "total:20 atomic:0 continuous:0 streams:1'\n",
            )
            write_executable(
                root / "bc",
                "#!/bin/sh\n"
                "cat >/dev/null\n"
                "printf '%s\\n' '10'\n",
            )

            env = os.environ.copy()
            env.update(
                {
                    "GE_MT_IMAS_DIR": str(imas_dir),
                    "GE_MT_LOG_DIR": str(log_dir),
                    "GE_MT_MODES": "0",
                    "GE_MT_MOPS": "default",
                    "UT_MARKER": str(runner_marker),
                    "PATH": f"{root}:{env['PATH']}",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPT), str(dump), str(runner)],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists(), "dump path was interpreted as shell code")
            self.assertTrue(runner_marker.is_file(), "UT runner was not invoked")
            self.assertEqual(
                runner_marker.read_text(encoding="utf-8").strip(), str(dump.resolve())
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for the Kernel debug API static scanner."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from scripts.scan_debug_apis import EXCLUDED_APIS, scan_text  # noqa: E402


class ScanDebugApisTest(unittest.TestCase):
    def test_detects_supported_api_families(self):
        source = """
        #include \"utils/debug/asc_printf.h\"
        #include \"utils/debug/asc_assert.h\"
        #include \"utils/debug/asc_dump.h\"
        #include \"utils/debug/asc_time.h\"
        AscendC::printf(\"x=%f\\n\", x);
        ascendc_assert(idx < limit, \"idx=%u\\n\", idx);
        if (isnan(x)) { __trap(); }
        asc_dump_ubuf<float>(ubuf, 100, 16);
        auto start = clock();
        asc_time_stamp(0x10000);
        """

        result = scan_text(source, "kernel.asc")

        self.assertEqual(
            result["supported"],
            ["printf", "assert", "__trap", "asc_dump", "clock", "asc_time_stamp"],
        )
        self.assertEqual(result["excluded"], [])
        self.assertEqual(result["files"], ["kernel.asc"])

    def test_reports_excluded_profiling_apis(self):
        source = "asc_prof_start(); TRACE_START(foo); asc_mark_stamp<PIPE_S>(1);"

        result = scan_text(source, "profiling.asc")

        self.assertEqual(result["supported"], [])
        self.assertEqual(result["excluded"], sorted(EXCLUDED_APIS & {
            "asc_prof_start", "TRACE_START", "asc_mark_stamp"
        }))

    def test_accepts_kernel_operator_as_umbrella_header(self):
        source = """
        #include \"kernel_operator.h\"
        AscendC::printf(\"value=%d\\n\", value);
        ascendc_assert(value >= 0);
        """

        result = scan_text(source, "kernel_operator.asc")

        self.assertEqual(result["missing_headers"], {})

    def test_cli_can_fail_for_excluded_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "bad.asc"
            source_path.write_text("asc_prof_stop();\n", encoding="utf-8")
            command = [
                sys.executable,
                str(SKILL_ROOT / "scripts" / "scan_debug_apis.py"),
                "--json",
                "--fail-on-excluded",
                str(source_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(report["files"][0]["excluded"], ["asc_prof_stop"])

    def test_ignores_excluded_api_names_in_documentation(self):
        source = "This skill excludes asc_prof_start and TRACE_START from its scope."

        result = scan_text(source, "SKILL.md")

        self.assertEqual(result["excluded"], [])


if __name__ == "__main__":
    unittest.main()

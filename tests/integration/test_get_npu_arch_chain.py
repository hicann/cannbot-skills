#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A/B source chain tests for get_npu_arch.py.

Validates the layered acquisition chain (issue #587 fix):
- full-soc-version: 首选 asys Chip Info，备选 DSMI dsmi_get_chip_info
- NpuArch: 首选 asys Arch Info，备选 ini（SoC_version 精确匹配）
- short-soc-version / CCE_AIV_version / variant_dir: from ini

All A/B paths are exercised with mocks: A-only, B-only (A degraded),
A+B consistent, A+B inconsistent (must warn), both degraded (must fail).
No real NPU environment required.
"""

import os
import sys
import unittest
from unittest.mock import patch

SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ops", "ascendc-env-check", "scripts"
)
sys.path.insert(0, SCRIPT_DIR)

import get_npu_arch as g

ASYS_HW_WITH_ARCH = """
 +-------------------------------+--------------------------------------+
 | Group of 1 Device             | INFORMATION                          |
 +===============================+======================================+
 | NPU Count                     | 1                                    |
 | Chip Info                     | Ascend 950PR_9579 V100               |
 | Arch Info                     | 3510                                 |
 +-------------------------------+--------------------------------------+
"""

ASYS_ARCH_LINE = " | Arch Info                     | 3510                                 |\n"
ASYS_CHIP_LINE = " | Chip Info                     | Ascend 950PR_9579 V100               |\n"
ASYS_HW_NO_ARCH = ASYS_HW_WITH_ARCH.replace(ASYS_ARCH_LINE, "")

ASYS_HW_A3 = (ASYS_HW_WITH_ARCH
              .replace("Ascend 950PR_9579 V100", "Ascend 910_9382 V220")
              .replace(ASYS_ARCH_LINE, ASYS_ARCH_LINE.replace("| 3510 ", "| 2201 ")))


class TestAsysParsing(unittest.TestCase):
    """Layer A: asys output parsing (Chip Info / Arch Info / NPU Count)."""

    def test_parse_chip_info_full_soc(self):
        full_soc = g._parse_full_soc_from_chip_info("Ascend 950PR_9579 V100")
        self.assertEqual(full_soc, "Ascend950PR_9579")

    def test_parse_chip_info_no_version(self):
        full_soc = g._parse_full_soc_from_chip_info("Ascend 910B3")
        self.assertEqual(full_soc, "Ascend910B3")

    def test_parse_asys_field(self):
        val, _ = g._parse_asys_field(ASYS_HW_WITH_ARCH, "Chip Info")
        self.assertEqual(val, "Ascend 950PR_9579 V100")

    def test_parse_arch_info(self):
        val, _ = g._parse_asys_field(ASYS_HW_WITH_ARCH, "Arch Info")
        self.assertEqual(val, "3510")

    def test_arch_info_absent(self):
        val, _ = g._parse_asys_field(ASYS_HW_NO_ARCH, "Arch Info")
        self.assertIsNone(val)

    def test_npu_count(self):
        val, _ = g._parse_asys_field(ASYS_HW_WITH_ARCH, "NPU Count")
        self.assertEqual(val, "1")


class TestFullSocChain(unittest.TestCase):
    """full-soc-version：首选 asys，备选 DSMI fallback。"""

    def test_a_asys_primary(self):
        with patch.object(g, "_find_asys", return_value="/fake/asys"), \
             patch.object(g, "_run_asys_hardware", return_value=ASYS_HW_WITH_ARCH), \
             patch.object(g, "probe_full_soc_via_dsmi") as dsmi:
            dsmi.return_value = "Ascend950PR_9579"
            full_soc, source = g.probe_full_soc()
        self.assertEqual(full_soc, "Ascend950PR_9579")
        self.assertEqual(source, "asys")
        dsmi.assert_not_called()  # A 成功时 B 不触发

    def test_b_dsmi_fallback_when_asys_missing(self):
        with patch.object(g, "_find_asys", return_value=None), \
             patch.object(g, "probe_full_soc_via_dsmi", return_value="Ascend910_9382"):
            full_soc, source = g.probe_full_soc()
        self.assertEqual(full_soc, "Ascend910_9382")
        self.assertEqual(source, "dsmi")

    def test_b_dsmi_fallback_when_asys_output_bad(self):
        with patch.object(g, "_find_asys", return_value="/fake/asys"), \
             patch.object(g, "_run_asys_hardware", return_value="garbage output"), \
             patch.object(g, "probe_full_soc_via_dsmi", return_value="Ascend910B3"):
            full_soc, source = g.probe_full_soc()
        self.assertEqual(full_soc, "Ascend910B3")
        self.assertEqual(source, "dsmi")

    def test_both_fail(self):
        with patch.object(g, "_find_asys", return_value=None), \
             patch.object(g, "probe_full_soc_via_dsmi", return_value=None):
            self.assertIsNone(g.probe_full_soc())


class TestIniLayer(unittest.TestCase):
    """Layer B for NpuArch + short-soc-version / CCE_AIV_version / variant_dir from ini."""

    INI_CONTENT = (
        "[version]\n"
        "SoC_version=Ascend950PR_9579\n"
        "Short_SoC_version=Ascend950\n"
        "CCEC_AIV_version=dav-c310-vec\n"
        "NpuArch=3510\n"
        "\n"
        "[SoCInfo]\n"
        "ai_core_cnt=28\n"
    )

    def _fake_ini_dir(self, tmpdir):
        cfg = os.path.join(tmpdir, "x86_64-linux", "data", "platform_config")
        os.makedirs(cfg)
        with open(os.path.join(cfg, "Ascend950PR_9579.ini"), "w") as f:
            f.write(self.INI_CONTENT)
        with open(os.path.join(cfg, "Ascend910B3.ini"), "w") as f:
            f.write("[version]\nSoC_version=Ascend910B3\nShort_SoC_version=Ascend910B\n"
                    "CCEC_AIV_version=dav-c220-vec\nNpuArch=2201\n")
        # 诱饵：前缀相同但 SoC_version 不同，精确匹配必须跳过
        with open(os.path.join(cfg, "Ascend910B3-1.ini"), "w") as f:
            f.write("[version]\nSoC_version=Ascend910B3-1\nShort_SoC_version=Ascend910B\n"
                    "CCEC_AIV_version=dav-c220-vec\nNpuArch=2201\n")
        return tmpdir

    def test_exact_soc_match_skips_prefix_trap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self._fake_ini_dir(tmpdir)
            ini = g.find_ini_for_soc(tmpdir, "Ascend910B3")
            self.assertIsNotNone(ini)
            fields = g.read_ini_fields(ini)
            self.assertEqual(fields["SoC_version"], "Ascend910B3")  # 未误中 910B3-1

    def test_no_ini_match(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self._fake_ini_dir(tmpdir)
            self.assertIsNone(g.find_ini_for_soc(tmpdir, "Ascend999"))

    def test_variant_dir_from_aiv(self):
        self.assertEqual(g.variant_dir_from_aiv("dav-c310-vec"), "dav_c310")
        self.assertEqual(g.variant_dir_from_aiv("dav-c220-vec"), "dav_c220")
        self.assertEqual(g.variant_dir_from_aiv("other"), "other")


class TestProbeAll(unittest.TestCase):
    """End-to-end chain assembly with mocks."""

    def _run(self, asys_output, dsmi_result):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ini_dir = os.path.join(tmpdir, "x86_64-linux", "data", "platform_config")
            os.makedirs(ini_dir)
            with open(os.path.join(ini_dir, "Ascend910_9382.ini"), "w") as f:
                f.write("[version]\nSoC_version=Ascend910_9382\nShort_SoC_version=Ascend910_93\n"
                        "CCEC_AIV_version=dav-c220-vec\nNpuArch=2201\n")
            with patch.object(g, "_find_asys", return_value="/fake/asys"), \
                 patch.object(g, "_run_asys_hardware", return_value=asys_output), \
                 patch.object(g, "probe_full_soc_via_dsmi", return_value=dsmi_result), \
                 patch.object(g, "get_cann_home", return_value=tmpdir), \
                 patch.object(g, "get_arch_dir", return_value="x86_64-linux"):
                return g.probe_all()

    def test_chain_a3_asys_with_arch(self):
        """A3 机器：asys 出 full-soc-version + Arch Info，ini 佐证。"""
        r = self._run(ASYS_HW_A3, None)
        self.assertEqual(r["full_soc"], "Ascend910_9382")
        self.assertEqual(r["full_soc_source"], "asys")
        self.assertEqual(r["npu_arch"], "2201")
        self.assertEqual(r["npu_arch_source"], "asys")
        self.assertEqual(r["short_soc"], "Ascend910_93")  # A3 判定的关键断言
        self.assertEqual(r["variant_dir"], "dav_c220")

    def test_chain_a3_asys_no_arch_ini_fallback(self):
        """A3 机器 + asys 无 Arch Info 字段：NpuArch 由 ini 兜底。"""
        asys_no_arch = ASYS_HW_A3.replace(ASYS_ARCH_LINE.replace("| 3510 ", "| 2201 "), "")
        r = self._run(asys_no_arch, None)
        self.assertEqual(r["npu_arch"], "2201")
        self.assertEqual(r["npu_arch_source"], "ini")

    def test_chain_npu_arch_conflict_warns(self):
        """asys 报 2201、ini 报 3510 不一致：必须告警，不得静默。"""
        bad_asys = ASYS_HW_A3.replace(ASYS_ARCH_LINE.replace("| 3510 ", "| 2201 "),
                                     ASYS_ARCH_LINE)
        r = self._run(bad_asys, None)
        self.assertEqual(r["npu_arch"], "3510")  # asys 优先
        self.assertTrue(any("NpuArch 不一致" in w for w in r["warnings"]))

    def test_chain_both_sources_fail(self):
        """双降级：无 asys 无 DSMI，明确失败不崩。"""
        with patch.object(g, "_find_asys", return_value=None), \
             patch.object(g, "probe_full_soc_via_dsmi", return_value=None):
            r = g.probe_all()
        self.assertIsNone(r["full_soc"])
        self.assertTrue(any("探测失败" in w for w in r["warnings"]))

    def test_chain_degraded_still_reports_npu_count(self):
        """full-soc-version 探测失败早退时，仍采集 npu_count（asys NPU Count 可用则输出）。"""
        asys_no_chip = ASYS_HW_WITH_ARCH.replace(ASYS_CHIP_LINE, "")
        r = self._run(asys_no_chip, None)
        self.assertIsNone(r["full_soc"])
        self.assertEqual(r["npu_count"], 1)

    def test_chain_dsmi_full_soc_fallback(self):
        """asys 整体不可用，DSMI 出 full-soc-version，ini 出 NpuArch。"""
        r = self._run(None, "Ascend910_9382")
        self.assertEqual(r["full_soc"], "Ascend910_9382")
        self.assertEqual(r["full_soc_source"], "dsmi")
        self.assertEqual(r["npu_arch"], "2201")
        self.assertEqual(r["npu_arch_source"], "ini")


class TestReportFormat(unittest.TestCase):
    """Output format contract: full-soc-version / short-soc-version naming."""

    def test_report_contains_version_suffix_terms(self):
        r = {
            "full_soc": "Ascend950PR_9579", "full_soc_source": "asys",
            "npu_arch": "3510", "npu_arch_source": "ini", "short_soc": "Ascend950",
            "ccec_aiv_version": "dav-c310-vec", "variant_dir": "dav_c310",
            "ini_path": "/x/Ascend950PR_9579.ini", "npu_count": 1, "warnings": [],
        }
        report = g._format_report(r)
        self.assertIn("full-soc-version=Ascend950PR_9579", report)
        self.assertIn("short-soc-version=Ascend950", report)
        self.assertIn("dav-3510", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)

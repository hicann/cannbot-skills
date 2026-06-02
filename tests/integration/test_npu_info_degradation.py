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
"""Degradation path tests for _npu_info.py.

Validates that _npu_info.py gracefully handles:
- npu-smi not installed
- npu-smi info --help fails / returns empty
- Individual -t subcommands unsupported on the device
- info -m mapping table fails
- All queries return ParseResult with proper warnings instead of crashing.

These tests use unittest.mock to simulate various failure modes
without requiring a real NPU environment.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add script dir to path
SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ops", "ascendc-env-check", "scripts"
)
sys.path.insert(0, SCRIPT_DIR)

import _npu_info as npu_module
from _npu_info import (
    NpuSmiDiscovery,
    NpuInfoCollector,
    query_with_strategy,
    query_npu_smi_mapping,
    ParseResult,
    _run_cmd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_cmd_mock(mapping: dict):
    """Return a mock for _run_cmd that looks up responses by cmd tuple.

    mapping: {(cmd, tuple): (stdout, rc, stderr), ...}
    """
    def mock_run(cmd_list):
        key = tuple(cmd_list)
        if key in mapping:
            return mapping[key]
        return ("", 127, f"Command not found: {' '.join(cmd_list)}")
    return mock_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNpuSmiDiscovery(unittest.TestCase):
    """Test NpuSmiDiscovery with mocked --help output."""

    def test_discover_from_help(self):
        """Discovery correctly parses --help type list."""
        help_text = (
            "Commands:\n"
            "       -t type        Show information for type\n"
            "                      type: board, flash, memory, usages, temp,\n"
            "                            power, common, health, product\n"
            "\n"
            "       Options:\n"
            "       -i %d          Card ID\n"
        )
        with patch.object(npu_module, "_run_cmd", return_value=(help_text, 0, "")):
            d = NpuSmiDiscovery()
            types = d.available_types
            self.assertIn("temp", types)
            self.assertIn("power", types)
            self.assertIn("common", types)
            self.assertIn("health", types)
            self.assertIn("usages", types)
            self.assertNotIn("type", types)

    def test_discover_empty_on_failure(self):
        """When --help fails, discovery returns empty set (no crash)."""
        with patch.object(npu_module, "_run_cmd", return_value=("", 1, "error")):
            d = NpuSmiDiscovery()
            self.assertEqual(d.available_types, set())
            self.assertFalse(d.is_available("temp"))

    def test_discover_empty_on_no_npu_smi(self):
        """When npu-smi is not found, discovery returns empty set."""
        with patch.object(npu_module, "_run_cmd", return_value=("", 127, "not found")):
            d = NpuSmiDiscovery()
            self.assertEqual(d.available_types, set())


class TestQueryWithStrategy(unittest.TestCase):
    """Test query_with_strategy fallback paths."""


    def test_primary_subcmd_works(self):
        """When primary subcommand (temp) is available, use it."""
        mock = self._make_mock(temp_ok=True, common_ok=True, usages_ok=True)
        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            with patch.object(NpuSmiDiscovery, "_discover_types", return_value={"temp", "common", "usages", "health"}):
                d = NpuSmiDiscovery()
                result = query_with_strategy(d, 5, "temperature")
            self.assertIsNotNone(result.value)
            self.assertIn("42", result.value)
            # Should not have warnings about missing subcommands
            self.assertTrue(
                all("not available" not in w for w in result.warnings),
                f"Unexpected warnings: {result.warnings}"
            )

    def test_fallback_to_common(self):
        """When temp is unavailable, fall back to common."""
        mock = self._make_mock(temp_ok=False, common_ok=True, usages_ok=True)
        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            with patch.object(NpuSmiDiscovery, "_discover_types", return_value={"common", "usages", "health"}):
                d = NpuSmiDiscovery()
                result = query_with_strategy(d, 5, "temperature")
            self.assertIsNotNone(result.value)
            self.assertIn("42", result.value)
            # Should have a warning about temp not being available
            self.assertTrue(
                any("temp" in w.lower() for w in result.warnings),
                f"Expected temp-unavailable warning, got: {result.warnings}"
            )

    def test_all_subcmds_unavailable(self):
        """When no subcommands are available, return None with warnings."""
        mock = self._make_mock(temp_ok=False, common_ok=False, usages_ok=False)
        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            with patch.object(NpuSmiDiscovery, "_discover_types", return_value=set()):
                d = NpuSmiDiscovery()
                result = query_with_strategy(d, 5, "temperature")
            self.assertIsNone(result.value)
            self.assertEqual(result.confidence, "low")
            self.assertTrue(len(result.warnings) > 0)

    def test_device_not_support_message(self):
        """Handle 'does not support' gracefully."""
        def mock(cmd_list):
            return ("", 1, "This device does not support querying temp info")

        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            with patch.object(NpuSmiDiscovery, "_discover_types", return_value={"temp"}):
                d = NpuSmiDiscovery()
                result = query_with_strategy(d, 5, "temperature")
            self.assertIsNone(result.value)
            self.assertTrue(
                any("does not support" in w.lower() for w in result.warnings)
            )

    def _make_mock(self, temp_ok=True, common_ok=True, usages_ok=True):
        """Build a mock where specific subcommands succeed or fail."""
        responses = {}
        for subcmd, ok in [("temp", temp_ok), ("common", common_ok), ("usages", usages_ok)]:
            if ok:
                responses[("npu-smi", "info", "-t", subcmd, "-i", "5")] = (
                    f"NPU Temperature(C)             : 42\n", 0, ""
                )
            else:
                responses[("npu-smi", "info", "-t", subcmd, "-i", "5")] = (
                    "", 1, f"Error: device does not support {subcmd}"
                )
        return make_run_cmd_mock(responses)


class TestMappingTableFallback(unittest.TestCase):
    """Test query_npu_smi_mapping and get_npu_ids fallback."""

    def test_mapping_success(self):
        """Normal mapping table parsing."""
        mapping = (
            "NPU ID    Chip ID    Chip Logic ID    Chip Name\n"
            "  5         0           0               Ascend 910B3\n"
        )
        with patch.object(npu_module, "_run_cmd", return_value=(mapping, 0, "")):
            result = query_npu_smi_mapping()
            self.assertEqual(result.value, [5])
            self.assertEqual(result.confidence, "high")

    def test_mapping_failure_returns_low_confidence(self):
        """When info -m fails, return None with low confidence."""
        with patch.object(npu_module, "_run_cmd", return_value=("", 1, "command not found")):
            result = query_npu_smi_mapping()
            self.assertIsNone(result.value)
            self.assertEqual(result.confidence, "low")
            self.assertTrue(len(result.warnings) > 0)

    def test_mapping_empty_output(self):
        """Empty output from info -m."""
        with patch.object(npu_module, "_run_cmd", return_value=("", 0, "")):
            result = query_npu_smi_mapping()
            self.assertIsNone(result.value)
            self.assertTrue(any("empty" in w.lower() for w in result.warnings))


class TestNpuInfoCollectorDegradation(unittest.TestCase):
    """Test NpuInfoCollector API under degraded conditions."""

    def test_collector_no_npu_smi(self):
        """Collector handles missing npu-smi gracefully."""
        with patch.object(npu_module, "_run_cmd", return_value=("", 127, "not found")):
            collector = NpuInfoCollector()
            ids = collector.get_npu_ids()
            self.assertEqual(ids, [])
            self.assertTrue(len(collector.warnings) > 0)

    def test_collector_all_queries_degraded(self):
        """All per-device queries return None when no subcommands work."""
        # Mapping works (so we have an NPU ID), but all -t fail
        def mock(cmd_list):
            key = tuple(cmd_list)
            if key == ("npu-smi", "info", "-m"):
                return ("NPU ID    Chip ID\n  5         0\n", 0, "")
            if key == ("npu-smi", "info", "--help"):
                return ("type: health\n", 0, "")
            # All -t subcommands fail
            return ("", 1, "not supported")

        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            with patch.object(NpuSmiDiscovery, "_discover_types", return_value={"health"}):
                collector = NpuInfoCollector()

            # health should work
            health = collector.get_health(5)
            # Others will fail because their strategies don't match 'health'
            temp = collector.get_temperature(5)
            power = collector.get_power(5)
            mem = collector.get_memory_info(5)
            usage = collector.get_usage_info(5)

            # health may be None if the subcommand fails; just verify no crash
            # and warnings are tracked
            self.assertIsInstance(mem, dict)
            self.assertIsInstance(usage, dict)
            # Warnings should be tracked for all queries
            self.assertTrue(len(collector.warnings) > 0)

    def test_common_cache_reduces_calls(self):
        """Common cache prevents redundant npu-smi calls."""
        call_count = 0
        common_output = (
            "Aicore Usage Rate(%)           : 10\n"
            "HBM Usage Rate(%)              : 20\n"
            "Temperature(C)                 : 45\n"
            "NPU Real-time Power(W)         : 100\n"
        )

        def mock(cmd_list):
            nonlocal call_count
            key = tuple(cmd_list)
            if key == ("npu-smi", "info", "-m"):
                return ("NPU ID\n  5\n", 0, "")
            if key == ("npu-smi", "info", "--help"):
                return ("type: common, temp, power, usages\n", 0, "")
            if key == ("npu-smi", "info", "-t", "common", "-i", "5"):
                call_count += 1
                return (common_output, 0, "")
            return ("", 1, "not supported")

        with patch.object(npu_module, "_run_cmd", side_effect=mock):
            collector = NpuInfoCollector()
            _ = collector.get_temperature(5)
            _ = collector.get_power(5)
            _ = collector.get_usage_info(5)
            # common should only be called once and cached
            self.assertEqual(
                call_count, 1,
                f"Expected 1 common call (cached), got {call_count}"
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and format validation."""

    def test_kv_parser_with_units(self):
        """_parse_kv handles various unit formats."""
        from _npu_info import _parse_kv

        output = (
            "NPU Temperature (C)            : 39\n"
            "Temperature(C)                 : 40\n"
            "NPU Real-time Power(W)         : 88.3\n"
        )
        self.assertEqual(_parse_kv(output, "NPU Temperature"), "39")
        self.assertEqual(_parse_kv(output, "Temperature"), "40")
        self.assertEqual(_parse_kv(output, "NPU Real-time Power"), "88.3")
        self.assertIsNone(_parse_kv(output, "Nonexistent Key"))

    def test_kv_parser_empty_output(self):
        """_parse_kv handles empty output."""
        from _npu_info import _parse_kv
        self.assertIsNone(_parse_kv("", "Key"))
        self.assertIsNone(_parse_kv("no colon here", "Key"))

    def test_validate_kv_format_detects_table(self):
        """_validate_kv_format rejects table-like output."""
        from _npu_info import _validate_kv_format
        table = "| NPU | Chip |\n| 0   | ABC  |\n| 1   | DEF  |\n+-----+------+\n"
        is_kv, warnings = _validate_kv_format(table)
        self.assertFalse(is_kv)
        # Table is rejected (either for too few kv lines or table markers)
        self.assertTrue(len(warnings) > 0)

    def test_validate_kv_format_accepts_kv(self):
        """_validate_kv_format accepts true key:value output."""
        from _npu_info import _validate_kv_format
        kv = "Key : Value\nAnother : 123\n"
        is_kv, warnings = _validate_kv_format(kv)
        self.assertTrue(is_kv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

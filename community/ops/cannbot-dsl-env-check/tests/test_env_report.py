#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exercise environment probes without importing device libraries."""

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/env_check.py"
SPEC = importlib.util.spec_from_file_location("env_check", SCRIPT)
env_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env_check)


class ProbeTests(unittest.TestCase):
    def test_import_probe_is_independent_of_npu_probe(self):
        response = subprocess.CompletedProcess(
            [], 0, '{"module_file": "/wheel/module.py", "version": "1"}', ""
        )
        with patch.object(env_check, "run", return_value=response) as run:
            result, context = env_check.check_l0()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(context["version"], "1")
        statement = run.call_args.args[0][2]
        self.assertIn("import json, cannbotdsl", statement)
        self.assertNotIn("import torch", statement)

    def test_runtime_probe_reports_devices(self):
        response = subprocess.CompletedProcess(
            [],
            0,
            '{"code": "OK", "visible_device_count": 1, "devices": [{"name": "test-npu"}]}',
            "",
        )
        with patch.object(env_check, "run", return_value=response) as run:
            result, context = env_check.check_l3()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(context["visible_device_count"], 1)
        self.assertIn("import torch_npu", run.call_args.args[0][2])

    def test_unavailable_runtime_keeps_diagnostic(self):
        response = subprocess.CompletedProcess(
            [], 13, '{"code": "NPU_UNAVAILABLE"}', ""
        )
        with patch.object(env_check, "run", return_value=response):
            result, _ = env_check.check_l3()
        self.assertEqual(result["code"], "NPU_UNAVAILABLE")
        self.assertNotEqual(result["status"], "PASS")

    def test_missing_tools_return_unavailable_inventory(self):
        with (
            patch.object(env_check.shutil, "which", return_value=None),
            patch.dict(env_check.os.environ, {}, clear=True),
        ):
            inventory = env_check.inspect_host_npu()
        self.assertEqual(inventory["source"], "unavailable")
        self.assertIsNone(inventory["detected_count"])
        self.assertTrue(inventory["warnings"])

    def test_smi_deduplicates_devices_and_keeps_health(self):
        outputs = [
            subprocess.CompletedProcess([], 0, "0 first\n0 duplicate\n1 second\n", ""),
            subprocess.CompletedProcess([], 0, "Health : OK\n", ""),
            subprocess.CompletedProcess([], 0, "Health : Warning\n", ""),
        ]
        inventory = {
            "source": "unavailable",
            "detected_count": None,
            "devices": [],
            "warnings": [],
        }
        with (
            patch.object(env_check.shutil, "which", return_value="/sbin/npu-smi"),
            patch.object(env_check, "run", side_effect=outputs),
        ):
            env_check.inspect_npu_smi(inventory)
        self.assertEqual(inventory["detected_count"], 2)
        self.assertEqual([d["health"] for d in inventory["devices"]], ["OK", "Warning"])

#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CLI regression tests using synthetic RT2 log fragments."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "addr_extract.py"


def launch(op="Add_1", addrs=("0x1000", "0x2000", "0x3000"), sizes=(16, 16, 16)):
    return (
        "I (123):2026-10-11-09:00:00.000 [executor_tracer.cc:100]123 "
        f"OnExecuteEvent:[KernelTrace][LaunchKernelWithHandle_{op}]"
        f"Input/Output addresses: {' '.join(addrs)} , "
        f"Input/Output sizes: {' '.join(map(str, sizes))}\n"
    )


def counts(op="Add_1", inputs=2, outputs=1):
    return (
        f"[KernelTrace][LaunchKernelWithHandle_{op}]Launch args-args addresses info: "
        f"ArgsInputsAddr(address/length): 0x4000/{inputs * 8}, "
        f"ArgsOutputsAddr(address/length): 0x5000/{outputs * 8}\n"
    )


class AddressExtractionTest(unittest.TestCase):
    def run_log(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "runtime fragment.log"
            log.write_text(text, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(log)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def addresses(self, output):
        return re.findall(
            r"^\s+((?:input|output|addr)_\d+)\s+(0x[0-9a-f]+)\s+\(([^)]+)\)",
            output,
            re.MULTILINE,
        )

    def test_missing_counts_preserves_addresses_without_inventing_roles(self):
        output = self.run_log(launch())
        self.assertEqual(
            self.addresses(output),
            [
                ("addr_0", "0x1000", "16B"),
                ("addr_1", "0x2000", "16B"),
                ("addr_2", "0x3000", "16B"),
            ],
        )
        self.assertIn("ArgsInputsAddr", output)
        self.assertIn("Add_1", output)

    def test_known_counts_preserves_input_output_roles(self):
        output = self.run_log(counts() + launch())
        self.assertEqual(
            self.addresses(output),
            [
                ("input_0", "0x1000", "16B"),
                ("input_1", "0x2000", "16B"),
                ("output_0", "0x3000", "16B"),
            ],
        )
        self.assertNotIn("ArgsInputsAddr", output)

    def test_explicit_zero_inputs_is_not_missing_metadata(self):
        output = self.run_log(
            counts(inputs=0, outputs=2)
            + launch(addrs=("0x1000", "0x2000"), sizes=(16, 16))
        )
        self.assertEqual(
            self.addresses(output),
            [("output_0", "0x1000", "16B"), ("output_1", "0x2000", "16B")],
        )
        self.assertNotIn("ArgsInputsAddr", output)

    def test_explicit_zero_outputs_preserves_inputs(self):
        output = self.run_log(
            counts(inputs=2, outputs=0)
            + launch(addrs=("0x1000", "0x2000"), sizes=(16, 16))
        )
        self.assertEqual(
            self.addresses(output),
            [("input_0", "0x1000", "16B"), ("input_1", "0x2000", "16B")],
        )

    def test_count_record_after_launch_is_supported(self):
        output = self.run_log(launch() + counts())
        self.assertEqual(self.addresses(output)[-1], ("output_0", "0x3000", "16B"))
        self.assertNotIn("addr_", output)

    def test_other_operator_counts_cannot_supply_missing_roles(self):
        output = self.run_log(counts(op="Mul_2") + launch())
        self.assertEqual(
            [a[0] for a in self.addresses(output)], ["addr_0", "addr_1", "addr_2"]
        )

    def test_repeated_launches_preserve_steps_and_unknown_addresses(self):
        output = self.run_log(launch() + launch())
        self.assertIn("step 1", output)
        self.assertIn("step 2", output)
        self.assertEqual(
            [a[0] for a in self.addresses(output)], ["addr_0", "addr_1", "addr_2"] * 2
        )

    def test_mixed_metadata_preserves_each_operator_and_one_step(self):
        output = self.run_log(counts() + launch() + launch(op="Mul_2"))
        self.assertNotIn("step 2", output)
        self.assertEqual(
            [a[0] for a in self.addresses(output)],
            ["input_0", "input_1", "output_0", "addr_0", "addr_1", "addr_2"],
        )

    def test_sizes_and_address_order_are_unchanged(self):
        output = self.run_log(counts() + launch(sizes=(16, 1024, 1048576)))
        self.assertEqual(
            self.addresses(output),
            [
                ("input_0", "0x1000", "16B"),
                ("input_1", "0x2000", "1K"),
                ("output_0", "0x3000", "1M"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

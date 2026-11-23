# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise constant-input moment probes independently of a device kernel."""
from pathlib import Path
import runpy
import unittest

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[2] / "ops" / "ascendc-st-design"
          / "scripts" / "variance_probe.py")
find_pair = runpy.run_path(str(SCRIPT))["find_variance_pair"]


class VarianceProbeTests(unittest.TestCase):
    @staticmethod
    def probe(lengths, constants, output_dtype=np.float32):
        return find_pair(lengths, constants, input_dtype=np.float16,
                         accumulator_dtype=np.float32, output_dtype=output_dtype,
                         offset=0.0, matches=lambda actual, expected: abs(actual - expected) <= 1e-5)

    def test_pair_survives_storage_and_output_casts(self):
        for dtype in (np.float16, np.float32):
            lengths = range(2, 128)
            hit, control, stored = self.probe(lengths, [32.001], dtype)
            self.assertEqual(stored, float(np.float16(32.001)))
            self.assertEqual(abs(hit - control), 1)
            for n, should_pass in ((hit, False), (control, True)):
                x = np.full(n, stored, dtype=np.float32)
                scale = np.float32(1) / np.float32(n)
                mean = x.sum() * scale
                variance = dtype(np.square(x).sum() * scale - mean * mean)
                self.assertEqual(float(np.var(x, dtype=np.float64)), 0.0)
                self.assertEqual(abs(float(variance)) <= 1e-5, should_pass)

    def test_selects_stronger_finite_witness_in_either_order(self):
        for constants in ([32, 64], [64, 32]):
            _, _, value = self.probe(range(2, 128), constants)
            self.assertEqual(value, 64.0)

    def test_exact_scaling_and_zero_input(self):
        self.assertIsNone(self.probe([2, 4, 8, 16], [32.0]))
        self.assertIsNone(self.probe(range(2, 128), [0.0]))

    def test_disjoint_lengths_cannot_form_a_control(self):
        self.assertIsNone(self.probe(range(2, 200, 2), [32.0]))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check emitted ADD-chain semantics, including zero and aliasing hazards."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]


class PatternTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("c++"), "a C++ compiler is needed for the pure host model"
    )
    def test_stride_factors_and_input_handle_preservation(self):
        source = r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include "ccu-patterns.hpp"
struct Variable {
    std::shared_ptr<uint64_t> value = std::make_shared<uint64_t>(0);
    Variable() = default;
    Variable(const Variable& other) : value(other.value) {} // CCU handle alias
    void operator=(const Variable& other) const { *value = *other.value; }
    void operator=(uint64_t n) const { *value = n; }
    void operator+=(const Variable& other) const { *value += *other.value; }
};
int main() {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    for (uint64_t stride : {uint64_t{0}, uint64_t{1}, uint64_t{31}, uint64_t{1}<<32, max/65535}) {
        for (uint32_t factor = 0; factor <= 65535; ++factor) {
            Variable input; input = stride;
            Variable output;
            cannbot_ccu_example::EmitScaledStride(input, static_cast<uint16_t>(factor), output);
            assert(*input.value == stride);
            assert(*output.value == stride * factor);
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "patterns.cc"
            path.write_text(source)
            result = subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-I",
                    str(SKILL / "assets"),
                    str(path),
                    "-o",
                    str(root / "patterns"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            result = subprocess.run(
                [str(root / "patterns")], capture_output=True, text=True
            )
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()

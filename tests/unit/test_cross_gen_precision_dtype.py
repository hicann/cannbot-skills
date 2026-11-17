# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify named dtype conversion independently of enum ordering or NPU runtime."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = (
    REPO
    / "ops/ascendc-cross-gen-migration/routes/3510-to-920x/references/precision-dtype-map.h.template"
)


@unittest.skipUnless(shutil.which("c++"), "C++ compiler is required")
class PrecisionDtypeTests(unittest.TestCase):
    def test_fp8_conversion_and_unknown_type_rejection(self):
        # Header/API stand-ins deliberately use non-contiguous scalar enums.
        source = r"""
#include <cstdio>
#include <stdexcept>
namespace at {
enum class ScalarType { Float=6, Half=5, Float8_e5m2=23, Float8_e4m3fn=24, Unknown=99 };
}
enum aclDataType { ACL_DT_UNDEFINED=-1, ACL_FLOAT=0, ACL_FLOAT16=1,
                   ACL_FLOAT8_E5M2=35, ACL_FLOAT8_E4M3FN=36 };
#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
    _(at::ScalarType::Float, ACL_FLOAT) \
    _(at::ScalarType::Half, ACL_FLOAT16)
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::invalid_argument("unsupported"); } while (0)
#include "precision-dtype-map.h.template"
int main() {
    for (auto type : {at::ScalarType::Float, at::ScalarType::Half,
                      at::ScalarType::Float8_e5m2, at::ScalarType::Float8_e4m3fn}) {
        std::printf("%d\n", int(MigrationAclDataType(type)));
    }
    try {
        MigrationAclDataType(at::ScalarType::Unknown);
        return 1;
    } catch (const std::invalid_argument&) {
        std::puts("unsupported rejected");
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.cpp").write_text(source)
            binary = root / "test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    f"-I{TEMPLATE.parent}",
                    str(root / "test.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
            self.assertEqual(
                result.stdout.splitlines(),
                ["0", "1", "35", "36", "unsupported rejected"],
            )


if __name__ == "__main__":
    unittest.main()

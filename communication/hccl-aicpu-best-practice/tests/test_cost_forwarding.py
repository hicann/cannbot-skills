# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cost forwarding regression: omitted default fields must not silently pass."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_cost_forwarding as cf

HEADER = """struct CalcCostCoeffParam {
    u32 rankSize = 0; float dataRatio = 0.0f;
    CommTopo netType = CommTopo::COMM_TOPO_1DMESH;
    BufferType inputBuffer; BufferType outputBuffer; BufferType scratchBuffer;
    std::vector<u32> portNum; bool isPod = false;
    const char* algName = nullptr; HcclComm comm = nullptr;
    const TopoInfoWithNetLayerDetails* topoInfo = nullptr; u32 repeatednum = 1;
};"""
PREFIX = (
    "CalcCostCoeffParam{n, ratio, topo, in, out, scratch, std::vector<u32>{1, 2}, isPod"
)


class CostForwardingTests(unittest.TestCase):
    def test_four_stages_and_missing_single_stage(self):
        good = "Call(" + PREFIX + ", algName});\n"
        self.assertEqual(cf.inspect(good * 4, HEADER, 4)["status"], "PASS")
        result = cf.inspect(good * 3 + PREFIX + "};", HEADER, 4)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            [f["line"] for f in result["findings"] if f["status"] == "FAIL"], [4]
        )

    def test_null_and_unknown_expressions(self):
        for value in ("nullptr", "NULL", "0", "{}"):
            self.assertEqual(
                cf.inspect(PREFIX + ", " + value + "}", HEADER, 1)["status"], "FAIL"
            )
        self.assertEqual(
            cf.inspect(PREFIX + ", name}", HEADER, 1)["status"], "UNVERIFIED"
        )
        self.assertEqual(
            cf.inspect(PREFIX + ", param.algName}", HEADER, 1, "param.algName")[
                "status"
            ],
            "PASS",
        )

    def test_comments_and_strings_are_not_initializers(self):
        source = (
            "// " + PREFIX + "}\n/* " + PREFIX + "} */\n"
            'LOG("CalcCostCoeffParam{ignored}");\n'
            + PREFIX
            + ", /* forward */ algName,}"
        )
        self.assertEqual(cf.inspect(source, HEADER, 1)["status"], "PASS")

    def test_changed_field_order_detects_wrong_position(self):
        header = HEADER.replace(
            "const char* algName = nullptr; HcclComm comm = nullptr;",
            "HcclComm comm = nullptr; const char* algName = nullptr;",
        )
        self.assertEqual(cf.inspect(PREFIX + ", algName}", header, 1)["status"], "FAIL")
        self.assertEqual(
            cf.inspect(PREFIX + ", comm, algName}", header, 1)["status"], "PASS"
        )

    def test_uncertain_layout_or_call_count_never_passes(self):
        for header, source, expected in (
            (HEADER, PREFIX + ", algName}", 4),
            (
                HEADER.replace("const char* algName", "#ifdef X\nconst char* algName"),
                PREFIX + "}",
                1,
            ),
            (HEADER.replace("algName", "removed"), PREFIX + "}", 1),
            (HEADER, "CalcCostCoeffParam{.algName = algName}", 1),
            (HEADER, PREFIX + ", algName", 1),
            (HEADER, "CalcCostCoeffParam p; p.algName = algName;", 1),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    cf.inspect(source, header, expected)["status"], "UNVERIFIED"
                )

    def test_dependencies_are_selected_not_assumed(self):
        source = PREFIX + ", algName, nullptr, topoInfo}"
        self.assertEqual(cf.inspect(source, HEADER, 1)["status"], "PASS")
        required = dict(algName="algName", comm="comm", topoInfo="topoInfo")
        result = cf.inspect(source, HEADER, 1, required_fields=required)
        self.assertEqual(
            {f["field"]: f["status"] for f in result["findings"]},
            dict(algName="PASS", comm="FAIL", topoInfo="PASS"),
        )
        self.assertEqual(
            cf.inspect(
                source.replace("nullptr", "comm"), HEADER, 1, required_fields=required
            )["status"],
            "PASS",
        )

    def test_each_call_must_forward_each_requested_field(self):
        full = PREFIX + ", algName, comm, topoInfo};\n"
        missing = PREFIX + ", algName, comm};\n"
        result = cf.inspect(
            full * 3 + missing,
            HEADER,
            4,
            required_fields=dict(comm="comm", topoInfo="topoInfo"),
        )
        self.assertEqual(
            [
                (f["line"], f["field"])
                for f in result["findings"]
                if f["status"] == "FAIL"
            ],
            [(4, "topoInfo")],
        )
        unknown = cf.inspect(full, HEADER, 1, required_fields={"futureField": "value"})
        self.assertEqual(unknown["status"], "UNVERIFIED")

    def test_requested_expression_alias_and_layout_change(self):
        source = PREFIX + ", name, context.comm, topology}"
        requested = dict(comm="context.comm", topoInfo="topology")
        self.assertEqual(
            cf.inspect(source, HEADER, 1, required_fields=requested)["status"], "PASS"
        )
        header = HEADER.replace(
            "HcclComm comm = nullptr;", "u32 added = 0; HcclComm comm = nullptr;"
        )
        self.assertNotEqual(
            cf.inspect(source, header, 1, required_fields=requested)["status"], "PASS"
        )

    def test_invalid_requirements_are_rejected(self):
        for values in (
            ["comm"],
            ["comm="],
            ["comm=nullptr"],
            ["a.b=x"],
            ["comm=a", "comm=b"],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cf.requirements(values)
        self.assertEqual(
            cf.requirements([" comm = context.comm ", "topoInfo=topology"]),
            dict(comm="context.comm", topoInfo="topology"),
        )

    def test_cli_default_and_explicit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, header = Path(tmp) / "executor.cc", Path(tmp) / "cost.h"
            source.write_text(PREFIX + ", algName, nullptr, topoInfo}")
            header.write_text(HEADER)
            command = [
                sys.executable,
                str(Path(cf.__file__)),
                "--executor",
                str(source),
                "--cost-header",
                str(header),
                "--expected-calls",
                "1",
            ]
            default = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(default.returncode, 0)
            explicit = subprocess.run(
                command
                + [
                    "--require-field",
                    "comm=comm",
                    "--require-field",
                    "topoInfo=topoInfo",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(explicit.returncode, 1)
            self.assertEqual(
                json.loads(explicit.stdout)["required_fields"],
                dict(comm="comm", topoInfo="topoInfo"),
            )
            conflict = subprocess.run(
                command + ["--require-field", "comm=comm", "--name-expression", "name"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(conflict.returncode, 2)


if __name__ == "__main__":
    unittest.main()

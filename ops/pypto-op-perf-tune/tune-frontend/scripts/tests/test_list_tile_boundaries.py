# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0

import hashlib
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "list_tile_boundaries.py"


class TileBoundariesTest(unittest.TestCase):
    def analyze(self, source: str) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            impl = root / "sample_impl.py"
            report = root / "report.md"
            agenda = root / "agenda.md"
            impl.write_text(source, encoding="utf-8")
            subprocess.run(
                [sys.executable, str(SCRIPT), str(impl), "--out", str(report),
                 "--emit-agenda", str(agenda)],
                check=True,
                capture_output=True,
                text=True,
            )
            digest = hashlib.sha256(impl.read_bytes()).hexdigest()
            report_text = report.read_text(encoding="utf-8")
            agenda_text = agenda.read_text(encoding="utf-8")
            self.assertIn(digest, report_text)
            self.assertIn(digest, agenda_text)
            report_rows = re.findall(
                r"^\| \d+ \| (.+?) \| (.+?) \| .* \| (F-[^|]+) \|$", report_text, re.MULTILINE
            )
            agenda_rows = re.findall(
                r"^\| \d+ \| (.+?) \| (F-[^|]+) \| (?:PENDING \| ){4}PENDING \|$",
                agenda_text,
                re.MULTILINE,
            )
            expected = [(f"{kind} {location}", rule) for kind, location, rule in report_rows]
            self.assertEqual(agenda_rows, expected)
            return report_text, agenda_text

    def test_output_directory_is_reported_as_a_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            impl = root / "sample_impl.py"
            impl.write_text("import pypto\n", encoding="utf-8")
            output = root / "report.md"
            output.mkdir()

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(impl), "--out", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn(str(output), result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(set(root.iterdir()), {impl, output})

    def test_failed_publish_preserves_existing_report(self) -> None:
        main = runpy.run_path(str(SCRIPT))["main"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            impl = root / "sample_impl.py"
            impl.write_text("import pypto\n", encoding="utf-8")
            output = root / "report.md"
            output.write_text("previous report\n", encoding="utf-8")

            with patch.object(Path, "replace", side_effect=OSError("failed to publish")):
                status = main([str(impl), "--out", str(output)])

            self.assertEqual(status, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous report\n")
            self.assertEqual(set(root.iterdir()), {impl, output})

    def test_layered_jit_ignores_host_and_unused_functions(self) -> None:
        report, agenda = self.analyze("""\
import pypto as pto
import torch as th
from pypto.frontend import jit as compile_kernel

def layer_h(tensor):
    tile = tensor.view([4, 16], [0, 0])
    return pto.reshape(tile, [16, 4])

def other_layer_h(tensor):
    return pto.view(tensor, [16, 4], [0, 0])

_LAYERS = {"first": layer_h, "other": other_layer_h}

def factory(kind):
    selected_layer = _LAYERS[kind]

    def layer_i(tensor):
        return selected_layer(tensor)

    @compile_kernel()
    def kernel(tensor: pto.Tensor([4, 16], pto.DT_FP16)):
        return layer_i(tensor)

    def unused(tensor):
        return pto.view(tensor, [1, 64], [0, 0])

    return kernel

def host(tensor):
    tensor = tensor.view(4, 16).reshape(16, 4)
    return th.matmul(tensor, tensor)
""")
        for output in (report, agenda):
            for name in ("kernel", "layer_i", "layer_h", "other_layer_h"):
                self.assertIn(f"Function: `{name}`", output)
            for name in ("host", "factory", "unused"):
                self.assertNotIn(f"Function: `{name}`", output)
        self.assertIn("view shape [4, 16]", report)
        self.assertIn("F-20 R4", agenda)
        self.assertNotIn("pto.Tensor", report)

    def test_undecorated_fragment_resolves_pypto_imports(self) -> None:
        report, agenda = self.analyze("""\
from pypto import view as slice_tile, matmul as cube
import torch

def fragment(x, w):
    a = slice_tile(x, [4, 16], [0, 0])
    return cube(a, w, b_trans=True)

@torch.jit.script
def host(x):
    return x.view(4, 16)
""")
        self.assertIn("view 'a' at L5 feeds matmul at L6", report)
        self.assertIn("vector→cube", agenda)
        self.assertNotIn("Function: `host`", report)

    def test_inline_views_feed_both_cube_operands(self) -> None:
        report, agenda = self.analyze("""\
import pypto

@pypto.frontend.jit
def kernel(x, w):
    pypto.set_vec_tile_shapes(16, 16)
    product = pypto.matmul(pypto.view(x, [16, 16], [0, 0]), w.view([16, 16], [0, 0]),
                           pypto.DT_FP32, a_trans=False, b_trans=True)
    return pypto.sum(product, -1, keepdim=True)
""")
        self.assertIn("L6 feeds matmul at L6 (A operand, a_trans=False)", report)
        self.assertIn("L6 feeds matmul at L6 (B operand, b_trans=True)", report)
        self.assertIn("vec tile active before view: L5: (16, 16)", report)
        self.assertNotIn("consumer: pending", report)
        self.assertEqual(agenda.count("| vector→cube L6→L6 | F-20 R4 |"), 2)
        self.assertIn("'sum' at L8 (reduce output [M,N]→[M,1])", report)

    def test_non_view_reassignment_ends_the_view_binding(self) -> None:
        report, agenda = self.analyze("""\
import pypto

@pypto.frontend.jit
def kernel(x, w):
    a = pypto.view(x, [16, 16], [0, 0])
    first = pypto.matmul(a, w, pypto.DT_FP32)
    a = pypto.cast(w, pypto.DT_FP32)
    second = pypto.matmul(a, w, pypto.DT_FP32)
    a = pypto.view(x, [16, 16], [0, 0])
    a = w
    return pypto.matmul(a, w, pypto.DT_FP32)
""")
        self.assertIn("view 'a' at L5 feeds matmul at L6", report)
        self.assertIn("'cast' at L7 (dtype cast)", report)
        self.assertEqual(agenda.count("| vector→cube "), 1)

    def test_assemble_writes_prevent_hoisting_of_buffer_reads(self) -> None:
        report, agenda = self.analyze("""\
import pypto

@pypto.frontend.jit
def kernel(x, state, grouped, method):
    for i in pypto.loop(16):
        for j in pypto.loop(16):
            fixed = pypto.view(x, [1, 1], [0, 0])
            partial = pypto.view(x, [1, 1], [i, 0])
            value = pypto.view(x, [1, 1], [i, j])
            prior = pypto.view(state, [1, 1], [0, 0])
            prior_grouped = pypto.view(grouped, [1, 1], [0, 0])
            prior_method = pypto.view(method, [1, 1], [0, 0])
            pypto.assemble(value, [0, 0], state)
            pypto.assemble([(value, [0, 0])], grouped)
            method.assemble(value, [0, 0])
""")
        self.assertIn("'view' at L7 depends on 0/2 loop vars", report)
        self.assertIn("'view' at L8 depends on 1/2 loop vars", report)
        self.assertEqual(agenda.count("| F-21 hoist "), 2)


if __name__ == "__main__":
    unittest.main()

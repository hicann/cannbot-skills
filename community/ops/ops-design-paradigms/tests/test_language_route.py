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

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "references/paradigms/resolve_example.py"


class LanguageRouteTests(unittest.TestCase):
    def test_language_is_required_when_spec_omits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "spec.yaml"
            spec.write_text("op:\n  paradigms: [Broadcast]\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--spec", str(spec), "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("provide --language", result.stderr)

    def test_explicit_dsl_does_not_select_ascendc_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "spec.yaml"
            spec.write_text("op:\n  paradigms: [Broadcast]\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--spec",
                    str(spec),
                    "--language",
                    "dsl",
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["language"], "dsl")
            self.assertEqual(payload["examples"], {})


class YamlSpecTests(unittest.TestCase):
    def run_spec(self, text, *extra):
        with tempfile.TemporaryDirectory() as temp:
            spec = Path(temp) / "spec.yaml"
            spec.write_text(text, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--spec", str(spec), "--json", *extra],
                text=True,
                capture_output=True,
            )

    def test_equivalent_yaml_forms_resolve_same_existing_example(self):
        forms = [
            "op:\n  language: ascendc\n  paradigms: [Broadcast]\n",
            'op:\n  language: "ascendc" # comment\n  paradigms:\n    - Broadcast\n',
            "op: {language: 'ascendc', paradigms: [Broadcast]}\n",
            "defaults: &paradigms [Broadcast]\nop:\n  language: ascendc\n  paradigms: *paradigms\n",
        ]
        outputs = []
        for form in forms:
            with self.subTest(form=form):
                result = self.run_spec(form)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertTrue(payload["examples"]["Broadcast"])
                self.assertTrue(
                    all(Path(p).is_dir() for p in payload["examples"]["Broadcast"])
                )
                outputs.append(payload)
        self.assertTrue(all(output == outputs[0] for output in outputs))

    def test_invalid_fields_report_errors_instead_of_empty_success(self):
        forms = [
            "",
            "[]",
            "op: []",
            "op: null",
            "op: [",
            "op: {language: ascendc, paradigms: Broadcast}",
            "op: {language: ascendc, paradigms: [123]}",
            'op: {language: ascendc, paradigms: [""]}',
            "op: {language: ascendc, paradigms: null}",
            "op: {language: [ascendc], paradigms: [Broadcast]}",
            "op: {language: unknown, paradigms: [Broadcast]}",
            "op: {category: [], language: ascendc, paradigms: [Broadcast]}",
        ]
        for form in forms:
            with self.subTest(form=form):
                result = self.run_spec(form)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_language_override_preserves_isolation(self):
        result = self.run_spec(
            "op: {language: ascendc, paradigms: [Broadcast]}", "--language", "dsl"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"language": "dsl", "examples": {}})

    def test_duplicate_and_unregistered_paradigms(self):
        result = self.run_spec(
            "op: {language: ascendc, paradigms: [Broadcast, Broadcast, FutureParadigm]}"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)["examples"]["Broadcast"]
        self.assertEqual(len(paths), len(set(paths)))

    def test_reduction_is_registered_with_existing_entry(self):
        import yaml

        root = SCRIPT.parents[2]
        routes = yaml.safe_load((SCRIPT.parent / "routes.yaml").read_text())["routes"][
            "paradigms"
        ]
        self.assertEqual(
            routes["Reduction"], ["references/paradigms/reduction/patterns.md"]
        )
        for paths in routes.values():
            for path in paths:
                self.assertTrue((root / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()

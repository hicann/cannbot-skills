#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License in the root of the software repository for the full text of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_ROOT.parents[1]
SCRIPT = SKILL_ROOT / "scripts/analyze_stream_logs.py"
CASES = json.loads((SKILL_ROOT / "references/golden_cases.json").read_text())


def _configured_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


GE_REPO_ROOT = _configured_path("GE_REPO_ROOT")


def _case_root(case):
    """Resolve an explicitly configured external log root for a case.

    Golden logs live outside the migrated Skill.  A missing environment
    variable is an intentional skip, rather than a failure or a fallback to a
    path in the cannbot-skills repository.
    """
    root_env = case.get("root_env")
    if not root_env:
        return None
    root = _configured_path(root_env)
    if root is None:
        return None
    return root


def _check_models(data, expected):
    model_sessions = []
    for item in data.get("sessions", []):
        if item.get("session_type") == "model":
            model_sessions.append(item)
    checks = []
    for model in expected:
        matched = []
        for item in model_sessions:
            if (
                item.get("graph") == model.get("graph")
                and item.get("model_id") == model.get("model_id")
                and item.get("correlation") == model.get("correlation")
            ):
                matched.append(item)
        checks.append(len(matched) == 1)
        if matched and "inferred" in model:
            checks.append(
                (matched[0].get("correlation_inference") == "inferred") == model["inferred"]
            )
    return checks


def _check_compile_subgraphs(data, expected):
    compile_graphs = [
        item
        for item in data.get("sessions", [])
        if item.get("session_type") == "compile_graph"
    ]
    return [
        any(
            item.get("graph", "").startswith(graph.get("prefix", ""))
            and item.get("session_role") == graph.get("session_role")
            for item in compile_graphs
        )
        for graph in expected
    ]


def _evaluate_expected(data, expected):
    checks = []
    scalar_checks = {
        "shape_mode": data["mode"]["shape_mode"],
        "scenario_class": data["mode"].get("scenario_class"),
        "graph_form": data["mode"]["graph_form"],
        "logical_stream_count": data["compile"]["logical_stream_count"],
        "root_graph_class": data["mode"].get("root_graph_class"),
    }
    for key, actual in scalar_checks.items():
        if key in expected:
            checks.append(actual == expected[key])
    if "final_model_stream_count" in expected:
        checks.extend(
            (
                data["compile"]["final_model_stream_count"]
                == expected["final_model_stream_count"],
                data["compile"]["event_count"] == expected["event_count"],
            )
        )
    if "batch_mapping" in expected:
        checks.extend(
            stream_id in data["batch_mapping"][batch].get("logic_stream_ids", [])
            for batch, stream_id in expected["batch_mapping"].items()
        )
        checks.append(
            data["batch_mapping"]["active_branch"]["label"] == expected["active_branch"]
        )
    if "models" in expected:
        checks.extend(_check_models(data, expected["models"]))
    if "compile_subgraphs" in expected:
        checks.extend(_check_compile_subgraphs(data, expected["compile_subgraphs"]))
    return checks


def main():
    failed = 0
    for name, case in CASES.items():
        source_root = _case_root(case)
        if source_root is None:
            LOGGER.info("SKIP %s: set %s", name, case.get("root_env", "an external log root"))
            continue
        compile_log = source_root / case["compile"]
        runtime_log = source_root / case["runtime"]
        if not compile_log.is_file() or not runtime_log.is_file():
            LOGGER.info("SKIP %s: fixture log missing", name)
            continue
        command = [
            sys.executable,
            str(SCRIPT),
            "--compile",
            str(compile_log),
            "--runtime",
            str(runtime_log),
        ]
        if GE_REPO_ROOT is not None:
            command.extend(["--repo-root", str(GE_REPO_ROOT)])
        command.extend(["--format", "json"])
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            LOGGER.error("FAIL %s: %s", name, result.stderr)
            failed += 1
            continue
        data = json.loads(result.stdout)
        expected = case["expected"]
        checks = _evaluate_expected(data, expected)
        if all(checks):
            LOGGER.info("PASS %s", name)
        else:
            LOGGER.error("FAIL %s: expected values did not match", name)
            failed += 1
    return failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

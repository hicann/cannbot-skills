# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the shipped documentation against an installed wheel in a new user project.

Usage: python tests/smoke.py --work /absolute/new-directory-outside-tmp
Runs offline with the installed OpModel package; no compiler or device is required.
"""

import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def check_links():
    count = 0
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text()):
            if "://" in target or target.startswith("mailto:"):
                continue
            location, separator, anchor = target.partition("#")
            destination = (
                (path.parent / unquote(location)).resolve() if location else path
            )
            if not destination.is_relative_to(ROOT) or not destination.exists():
                raise AssertionError((str(path.relative_to(ROOT)), target))
            if separator and destination.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", destination.read_text(), re.M)
                anchors = [
                    re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
                    for h in headings
                ]
                if unquote(anchor) not in anchors:
                    raise AssertionError(
                        (str(path.relative_to(ROOT)), target, "missing anchor")
                    )
            count += 1
    return dict(local_links=count)


def check_identity():
    installed_version = version("opmodel")
    locations = {}
    for name in ("accel_model", "operators", "hardware"):
        locations[name] = list(importlib.import_module(name).__path__)
        if not all(
            Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())
            for p in locations[name]
        ):
            raise AssertionError(
                (name, "must load from the active environment", locations[name])
            )
    return dict(
        installed_version=installed_version,
        origins=locations,
        dependencies={n: version(n) for n in ("opmodel", "simpy", "PyYAML", "numpy")},
    )


def check_physical(result):
    required = (
        "capacity",
        "assignment_coverage",
        "region_coverage",
        "quantity",
        "runtime",
        "backend_agreement",
    )
    statuses = {name: result["validation"][name]["status"] for name in required}
    if not result["passed"] or any(status != "passed" for status in statuses.values()):
        raise AssertionError(statuses)
    if result["validation"]["assignment_coverage"]["tasks"] != 9:
        raise AssertionError("Expected 3 Cube and 6 AIV tasks in the physical example")
    if result["device_experiment"] or result["model_calibrated"]:
        raise AssertionError(
            "Synthetic example must not claim device or calibrated results"
        )
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, required=True, help="New project directory, outside /tmp"
    )
    args = parser.parse_args()
    project = args.work.resolve()
    if (
        project.is_relative_to("/tmp")
        or project.is_relative_to(ROOT)
        or ROOT.is_relative_to(project)
    ):
        parser.error("Choose a separate new user project outside /tmp and the skill")
    project.mkdir(parents=True, exist_ok=False)
    logs = project / "logs"
    logs.mkdir()
    (project / "work").mkdir()
    report = dict(status="running", checks=[])
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTHON", "OPMODEL"))
    }
    env.update(
        PATH=str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        OPMODEL_SKILL_DIR=str(ROOT),
        OPMODEL_WORKSPACE=str(project / "work"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )

    def command(name, argv):
        result = subprocess.run(
            argv, cwd=project, env=env, capture_output=True, text=True, timeout=180
        )
        (logs / (name + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise AssertionError(
                f"{name}: exit {result.returncode}; see {logs / (name + '.log')}"
            )
        return result

    def document(relative, select=lambda block: True, language="bash"):
        blocks = re.findall(
            rf"^```{language}\n(.*?)^```", (ROOT / relative).read_text(), re.M | re.S
        )
        blocks = [block for block in blocks if select(block)]
        if not blocks:
            raise AssertionError(f"No executable examples in {relative}")
        name = relative.replace("/", "-").replace(".md", "") + "-" + language
        for index, block in enumerate(blocks):
            argv = (
                [sys.executable, "-I", "-B", "-c", block]
                if language == "python"
                else ["/bin/bash", "-euo", "pipefail", "-c", block]
            )
            command(f"{name}-{index}", argv)
        return dict(language=language, executed_blocks=len(blocks))

    def outcomes():
        def load(rel):
            return json.loads((project / rel).read_text())

        baseline = load("work/baseline/result.json")
        split = load("work/with_split/result.json")
        tile = load("work/tile64/result.json")
        joint = load("work/joint_reevaluated/result.json")
        for result in (baseline, split, tile, joint):
            if result["total_cycles"] <= 0:
                raise AssertionError("Invalid full-case cycles")
        if baseline["summary"]["active_cores"] != 4:
            raise AssertionError(
                "Tutorial has four output tasks; expected four active cores"
            )
        search = load("work/joint_search/search.json")
        if not search["rows"] or not search["coverage"]["complete"]:
            raise AssertionError("Declared joint-search space was not covered")
        physical = check_physical(load("work/physical/result.json"))
        return dict(
            baseline_cycles=baseline["total_cycles"],
            split_cycles=split["total_cycles"],
            tile_cycles=tile["total_cycles"],
            joint_cycles=joint["total_cycles"],
            joint_coverage=search["coverage"],
            physical_checks=physical,
        )

    checks = [
        ("links", check_links),
        ("installed-wheel", check_identity),
        ("quickstart", lambda: document("references/quickstart.md")),
        (
            "dag-and-reevaluation",
            lambda: document(
                "references/optimization.md",
                lambda b: "app.cli.dag_run" in b or "case_tile64" in b,
            ),
        ),
        ("hardware-override", lambda: document("references/hardware.md")),
        ("physical", lambda: document("assets/examples/physical/README.md")),
        (
            "public-python-examples",
            lambda: [
                document(relative, language="python")
                for relative in (
                    "references/hardware.md",
                    "references/framework/numeric-execution-validation.md",
                    "references/framework/graph-search.md",
                )
            ],
        ),
        (
            "stage-and-numeric-contracts",
            lambda: command(
                "interfaces",
                [sys.executable, "-I", "-B", str(ROOT / "tests/interfaces.py")],
            ).stdout.strip(),
        ),
        (
            "joint-search-and-session",
            lambda: document("references/joint-optimization/wheel-commands.md"),
        ),
        ("outcomes", outcomes),
    ]
    for name, function in checks:
        start = time.monotonic()
        try:
            details = function()
            row = dict(name=name, status="passed", details=details)
        except Exception:
            row = dict(name=name, status="failed", error=traceback.format_exc())
        row["seconds"] = round(time.monotonic() - start, 3)
        report["checks"].append(row)
        print(name, row["status"], flush=True)
        if row["status"] == "failed":
            report["status"] = "failed"
        (project / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        if report["status"] == "failed":
            print(row["error"], file=sys.stderr)
            return 1
    report["status"] = "passed"
    (project / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print("Report:", project / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

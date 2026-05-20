# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
SKILLS_DIR = SCRIPTS_DIR.parent.parent


def check_plugin_installed(plugin_name: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", plugin_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def _build_pytest_args(args):
    pytest_args = [
        "-" + "v" * min(args.verbose, 3),
        "--tb=short",
        f"--rootdir={SCRIPTS_DIR}",
    ]
    if args.skill:
        skill_pattern = "|".join(args.skill)
        pytest_args.extend(["-k", skill_pattern])
    pytest_args.append(str(SCRIPTS_DIR / "test_skill_basic.py"))
    return pytest_args


def _configure_reports(pytest_args, args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.html_report:
        if check_plugin_installed("pytest-html"):
            pytest_args.append(f"--html={output_dir}/report.html")
            pytest_args.append("--self-contained-html")
        else:
            logger.warning("pytest-html not installed, skipping HTML report")
            logger.info("Install with: pip install pytest-html")
    if args.json_report:
        if check_plugin_installed("pytest-json-report"):
            pytest_args.append("--json-report")
            pytest_args.append(f"--json-report-file={output_dir}/report.json")
            pytest_args.append("--json-report-indent=2")
        else:
            logger.warning("pytest-json-report not installed, skipping JSON report")
            logger.info("Install with: pip install pytest-json-report")


def _run_pytest(pytest_args):
    logger.info("Running pytest with args: %s", pytest_args)
    logger.info("Skills directory: %s", SKILLS_DIR)
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + pytest_args,
        cwd=str(SCRIPTS_DIR),
        capture_output=False
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run pytest-based skill evaluation")
    parser.add_argument(
        "--skill",
        nargs="*",
        help="Specific skills to test (default: all skills with evals)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity level"
    )
    parser.add_argument(
        "--output",
        default=str(SKILLS_DIR / "skill-test-framework" / "results"),
        help="Output directory for test results"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report (requires pytest-html)"
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Generate JSON report (requires pytest-json-report)"
    )
    args = parser.parse_args()
    pytest_args = _build_pytest_args(args)
    _configure_reports(pytest_args, args)
    return _run_pytest(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
